"""Prática 3 (warm-starting HPO, Feurer et al. 2014): constroi a matriz de perdas
K configuracoes x n datasets para a MLP sobre embeddings MiniLM.

Espaco Theta (4 hiperparametros, escala log): h (neuronios, camada unica),
alpha (L2), learning_rate_init, batch_size. Config 0 = default da sklearn;
configs 1..K-1 = pontos Sobol embaralhados. loss = 1 - macro-F1 (5-fold CV,
mesmas folds para todas as configs de um dataset).

Saidas em --out-dir (default data_hs/hpo): configs.csv, loss_matrix.csv,
loss_folds.csv, build_cost.csv e cache/<dataset_id>.csv (retomavel).
"""
from __future__ import annotations

import argparse
import os
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import qmc
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

RANDOM_STATE = 42
N_SPLITS = 5
MAX_ITER = 200  # default da sklearn

# (low, high) de cada hiperparametro, amostrado em escala log
RANGES = {"h": (16, 512), "alpha": (1e-6, 1e-1), "lr": (1e-4, 1e-1), "batch_size": (16, 512)}
DEFAULT = {"h": 100, "alpha": 1e-4, "lr": 1e-3, "batch_size": 200}  # 'auto' = min(200, n)
COORDS = [("h", "u_h"), ("alpha", "u_alpha"), ("lr", "u_lr"), ("batch_size", "u_bs")]


def to_unit(name, value):
    lo, hi = RANGES[name]
    return (np.log(value) - np.log(lo)) / (np.log(hi) - np.log(lo))


def from_unit(name, u):
    lo, hi = RANGES[name]
    return np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo)))


def make_configs(K: int) -> pd.DataFrame:
    rows = [{"config_id": 0, **DEFAULT}]
    if K > 1:
        m = int(np.ceil(np.log2(K - 1))) if K > 2 else 0
        sob = qmc.Sobol(d=4, scramble=True, seed=RANDOM_STATE)
        pts = sob.random_base2(m)[: K - 1]
        for i, u in enumerate(pts, start=1):
            rows.append({
                "config_id": i,
                "h": int(round(from_unit("h", u[0]))),
                "alpha": float(from_unit("alpha", u[1])),
                "lr": float(from_unit("lr", u[2])),
                "batch_size": int(round(from_unit("batch_size", u[3]))),
            })
    df = pd.DataFrame(rows)
    # coordenadas normalizadas em [0,1] (log), calculadas dos valores ja arredondados
    for name, ucol in COORDS:
        df[ucol] = [to_unit(name, v) for v in df[name]]
    return df


def make_model(cfg, auto_batch: bool):
    clf = MLPClassifier(
        hidden_layer_sizes=(int(cfg["h"]),), alpha=float(cfg["alpha"]),
        learning_rate_init=float(cfg["lr"]),
        batch_size="auto" if auto_batch else int(cfg["batch_size"]),
        max_iter=MAX_ITER, random_state=RANDOM_STATE,
    )
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def run_config(cfg, emb, y, folds):
    """Retorna lista de (config_id, fold, loss, fit_seconds)."""
    out = []
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.filterwarnings("ignore", message=".*batch_size.*clipped.*")  # batch > n_train
        for f, (tr, te) in enumerate(folds):
            t0 = time.time()
            model = make_model(cfg, auto_batch=int(cfg["config_id"]) == 0)
            model.fit(emb[tr], y[tr])
            loss = 1.0 - f1_score(y[te], model.predict(emb[te]), average="macro")
            out.append((int(cfg["config_id"]), f, float(loss), time.time() - t0))
    return out


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=128)
    ap.add_argument("--datasets", nargs="*", default=None, help="subconjunto de dataset_ids")
    ap.add_argument("--n-jobs", type=int, default=12)
    ap.add_argument("--data-dir", default="data_hs")
    ap.add_argument("--out-dir", default="data_hs/hpo")
    return ap.parse_args()


def main():
    args = parse_args()
    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    configs = make_configs(args.K)
    configs.to_csv(out_dir / "configs.csv", index=False)
    print(f"[hpo] {len(configs)} configs -> {out_dir / 'configs.csv'}")

    manifest = pd.read_csv(data_dir / "datasets_manifest.csv")
    ids = args.datasets or manifest["dataset_id"].tolist()
    cfg_rows = configs.to_dict("records")

    with Parallel(n_jobs=args.n_jobs) as par:
        for n, d in enumerate(ids, 1):
            cpath = cache_dir / f"{d}.csv"
            if cpath.exists() and len(pd.read_csv(cpath)) == len(cfg_rows) * N_SPLITS:
                print(f"[{n}/{len(ids)}] {d}: cache ok, pulando")
                continue
            t0 = time.time()
            y = pd.read_parquet(data_dir / "datasets" / f"{d}.parquet")["label"].to_numpy()
            emb = np.load(data_dir / "embeddings" / f"{d}__minilm.npy")
            folds = list(StratifiedKFold(N_SPLITS, shuffle=True, random_state=RANDOM_STATE).split(emb, y))
            res = par(delayed(run_config)(c, emb, y, folds) for c in cfg_rows)
            df = pd.DataFrame([r for rs in res for r in rs],
                              columns=["config_id", "fold", "loss", "fit_seconds"])
            df.to_csv(cpath, index=False)
            print(f"[{n}/{len(ids)}] {d}: {len(df)} fits, {df.fit_seconds.mean():.2f}s/fit, "
                  f"wall {time.time() - t0:.1f}s, default loss {df[df.config_id == 0].loss.mean():.3f}, "
                  f"best {df.groupby('config_id').loss.mean().min():.3f}")

    # agregacao
    longs = []
    for d in ids:
        df = pd.read_csv(cache_dir / f"{d}.csv")
        df.insert(0, "dataset_id", d)
        longs.append(df)
    long = pd.concat(longs, ignore_index=True)
    long.to_csv(out_dir / "loss_folds.csv", index=False)
    mat = long.groupby(["config_id", "dataset_id"]).loss.mean().unstack("dataset_id")[ids]
    mat.to_csv(out_dir / "loss_matrix.csv")
    cost = long.groupby("dataset_id").fit_seconds.sum().reindex(ids).rename("total_fit_seconds")
    cost.to_csv(out_dir / "build_cost.csv")
    print(f"[hpo] loss_matrix {mat.shape} -> {out_dir / 'loss_matrix.csv'}; "
          f"custo total {cost.sum() / 3600:.2f} h de CPU")


if __name__ == "__main__":
    main()
