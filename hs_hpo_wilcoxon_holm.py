"""Pratica 3: testes de Wilcoxon pareados (50 datasets) com correcao de Holm
dentro de cada familia, sobre AUC e norm_regret (results_hpo/auc.csv, curves.csv)."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

WS = [f"ws_sim_{x}_k{k}" for x in ("base", "ext") for k in (1, 3, 5)]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results_hpo")
    return ap.parse_args()


def holm(p):
    """Holm step-down: ordena p, adj = acumulado maximo de min(1,(m-r)*p)."""
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.minimum(1.0, (m - np.arange(m)) * p[order])
    adj = np.maximum.accumulate(adj)
    out = np.empty(m)
    out[order] = adj
    return out


def test(x, y):
    d = x - y
    if np.all(d == 0):
        return 1.0
    return wilcoxon(x, y, zero_method="zsplit").pvalue


def main():
    out = Path(parse_args().out_dir)
    auc = pd.read_csv(out / "auc.csv", index_col=0)
    cur = pd.read_csv(out / "curves.csv")
    rows = []

    def family(name, pairs):
        fam = []
        for a, b, budget, xa, xb in pairs:
            xa, xb = xa.align(xb, join="inner")
            fam.append(dict(family=name, a=a, b=b, budget=budget,
                            mean_diff=float((xa - xb).mean()),
                            wins_a=int((xa < xb).sum()),
                            p_raw=float(test(xa.values, xb.values))))
        ph = holm([r["p_raw"] for r in fam])
        for r, p in zip(fam, ph):
            r["p_holm"] = float(p)
        rows.extend(fam)

    family("auc_vs_ws_rank", [(m, "ws_rank", "auc", auc[m], auc["ws_rank"])
                              for m in WS + ["ws_sim_randnn_k3"]])
    family("auc_vs_randnn", [(m, "ws_sim_randnn_k3", "auc", auc[m], auc["ws_sim_randnn_k3"])
                             for m in WS])
    family("auc_ext_vs_base", [(f"ws_sim_ext_k{k}", f"ws_sim_base_k{k}", "auc",
                                auc[f"ws_sim_ext_k{k}"], auc[f"ws_sim_base_k{k}"])
                               for k in (1, 3, 5)])
    pv = {i: cur[cur.i == i].groupby(["method", "dataset"]).norm_regret.mean().unstack(0)
          for i in (5, 15, 30)}
    family("warm_vs_cold", [(a, "smbo_cold", i, pv[i][a], pv[i]["smbo_cold"])
                            for a in ("ws_sim_ext_k3", "ws_rank") for i in (5, 15, 30)])

    df = pd.DataFrame(rows)
    df.to_csv(out / "wilcoxon_holm.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(df.to_string(index=False, float_format=lambda v: f"{v:.4g}"))


if __name__ == "__main__":
    main()
