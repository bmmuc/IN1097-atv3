"""Pratica 3 (warm-starting HPO, Feurer et al. 2014): busca simulada por consulta
a tabela (matriz de perdas K x n de hs_hpo_build_matrix.py), leave-one-dataset-out.

Metodos (todos sobre as K configs, sem reposicao, orcamento T):
  default       -- sempre a config 0 (default da sklearn)
  random        -- permutacao aleatoria
  smbo_cold     -- n_init configs aleatorias + SMBO (GP + EI)
  ws_sim_<X>_k<k> -- n_init = melhores configs dos k datasets mais proximos
                    (L1 em meta-features z-score; X in {base, ext}) + SMBO
  ws_sim_randnn_k3 -- controle: k=3 datasets de treino ALEATORIOS + SMBO
  ws_rank       -- n_init = configs com melhor rank medio nos datasets de treino + SMBO
Nota: as inicializacoes warm-start sao deterministicas; a seed afeta apenas
random, o init do cold, o controle randnn e os restarts do GP.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import norm, rankdata, spearmanr, wilcoxon
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

import evaluate as ev

U_COLS = ["u_h", "u_alpha", "u_lr", "u_bs"]
KS = (1, 3, 5)
XVARS = ("base", "ext")
MAIN_WS = "ws_sim_ext_k3"
MAIN5 = ["default", "random", "smbo_cold", MAIN_WS, "ws_rank"]
WS_VARIANTS = [f"ws_sim_{x}_k{k}" for x in XVARS for k in KS]
CONTROL = "ws_sim_randnn_k3"
ALL_METHODS = ["default", "random", "smbo_cold"] + WS_VARIANTS + [CONTROL, "ws_rank"]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=30)
    ap.add_argument("--n-init", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--n-jobs", type=int, default=12)
    ap.add_argument("--data-dir", default="data_hs")
    ap.add_argument("--hpo-dir", default="data_hs/hpo")
    ap.add_argument("--out-dir", default="results_hpo")
    return ap.parse_args()


# Meta-conhecimento (usa so datasets de treino)

def prep_meta(Xdf: pd.DataFrame, target: str):
    """z-score com media/desvio do treino; descarta colunas constantes/NaN no
    treino; imputa NaN com a mediana do treino. Retorna (Z_train, z_target, train_ids)."""
    Xn = Xdf.select_dtypes("number").replace([np.inf, -np.inf], np.nan)
    tr = Xn.drop(index=target)
    med = tr.median()
    tr = tr.fillna(med)
    keep = tr.columns[(tr.std(ddof=0) > 1e-12) & med.notna()]
    tr, tgt = tr[keep], Xn.loc[[target], keep].fillna(med[keep])
    mu, sd = tr.mean(), tr.std(ddof=0)
    return ((tr - mu) / sd).to_numpy(), ((tgt - mu) / sd).to_numpy()[0], list(tr.index)


def interleave(rank_lists, n_init):
    """rank 1 de nn1, rank 1 de nn2, ..., rank 2 de nn1 ... (dedupe)."""
    out, seen = [], set()
    for r in range(len(rank_lists[0])):
        for lst in rank_lists:
            c = int(lst[r])
            if c not in seen:
                seen.add(c)
                out.append(c)
                if len(out) == n_init:
                    return out
    return out


def knowledge(L: pd.DataFrame, X_variants: dict, target: str, n_init: int):
    """Tudo que e derivado so dos datasets de treino para o alvo `target`."""
    train = [d for d in L.columns if d != target]
    Ltr = L[train].to_numpy()
    ranked = {d: np.argsort(L[d].to_numpy(), kind="stable") for d in train}  # melhor -> pior
    mean_rank = np.mean([rankdata(Ltr[:, j]) for j in range(Ltr.shape[1])], axis=0)
    out = {"train": train, "ws_rank": [int(c) for c in np.argsort(mean_rank, kind="stable")[:n_init]],
           "ranked": ranked, "nn": {}}
    for xn, Xdf in X_variants.items():
        Z, z, ids = prep_meta(Xdf, target)
        dist = np.abs(Z - z).sum(axis=1)
        order = np.argsort(dist, kind="stable")
        out["nn"][xn] = ([ids[i] for i in order], dist[order])
    return out


def warm_init(kn, xn, k, n_init):
    nn_ids = kn["nn"][xn][0][:k]
    return interleave([kn["ranked"][d] for d in nn_ids], n_init)


# SMBO (GP + EI) sobre tabela

def smbo(init, loss, U, T, seed):
    K = len(loss)
    ev_idx = list(init)
    while len(ev_idx) < T:
        kernel = (ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=0.5, length_scale_bounds=(0.05, 10.0), nu=2.5)
                  + WhiteKernel(1e-2, (1e-6, 1.0)))
        gp = GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=2, random_state=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp.fit(U[ev_idx], loss[ev_idx])
            cand = np.setdiff1d(np.arange(K), ev_idx)
            mu, sd = gp.predict(U[cand], return_std=True)
        best = loss[ev_idx].min()
        imp = best - mu - 0.01
        z = imp / np.maximum(sd, 1e-12)
        ei = np.where(sd > 1e-12, imp * norm.cdf(z) + sd * norm.pdf(z), 0.0)
        ev_idx.append(int(cand[int(np.argmax(ei))]))  # empate -> menor indice
    return loss[ev_idx]


def run_target(L, X_variants, U, target, t_idx, T, n_init, seeds):
    loss = L[target].to_numpy()
    K = len(loss)
    lstar, lworst = loss.min(), loss.max()
    den = (lworst - lstar) or 1.0
    kn = knowledge(L, X_variants, target, n_init)
    rows = []

    def add(method, seed, vals):
        best = np.minimum.accumulate(vals[:T])
        nr = (best - lstar) / den
        for i in range(len(best)):
            rows.append((method, target, seed, i + 1, best[i], nr[i]))

    inits = {f"ws_sim_{x}_k{k}": warm_init(kn, x, k, n_init) for x in XVARS for k in KS}
    inits["ws_rank"] = kn["ws_rank"]
    for s in range(seeds):
        rng = np.random.default_rng([s, t_idx])
        add("default", s, np.full(T, loss[0]))
        add("random", s, loss[rng.permutation(K)])
        add("smbo_cold", s, smbo(rng.choice(K, n_init, replace=False), loss, U, T, s))
        for m, ini in inits.items():
            add(m, s, smbo(ini, loss, U, T, s))
        rnd = rng.choice(len(kn["train"]), 3, replace=False)
        add(CONTROL, s, smbo(interleave([kn["ranked"][kn["train"][j]] for j in rnd], n_init), loss, U, T, s))
    info = {"target": target, "inits": inits, "nn": {x: (kn["nn"][x][0][:5], kn["nn"][x][1][:5]) for x in XVARS},
            "ranked": {d: kn["ranked"][d][:5] for d in kn["train"]}}
    return rows, info


# Analises

def safe_wilcoxon(a, b):
    d = np.asarray(a) - np.asarray(b)
    if np.allclose(d, 0):
        return 1.0
    return float(wilcoxon(a, b, zero_method="zsplit").pvalue)


# Parametros do diagrama CD para impressao
CD_KW = dict(figsize=(6.4, 3.6), dpi=220, tick_fontsize=13, label_fontsize=13,
             cd_fontsize=13, title_fontsize=14, tight=True)


def friedman_block(df, name, out_txt, out_png, title):
    r = ev.friedman_nemenyi(df, higher_is_better=False)
    ev.plot_cd_diagram(r["mean_ranks"], r["cd"], title, str(out_png), **CD_KW)
    out_txt.append(f"== {name} (N={r['N']}, k={r['k']}) ==\nFriedman chi2={r['statistic']:.3f} p={r['pvalue']:.4g}\n"
                   f"CD(Nemenyi, 0.05)={r['cd']:.3f}\nranks medios (1=melhor):\n{r['mean_ranks'].sort_values().round(3).to_string()}\n"
                   f"pares significativos: {[(a, b, round(d, 2)) for a, b, d in r['significant_pairs']] or 'nenhum'}\n")
    return r


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data_dir, hpo_dir = Path(args.data_dir), Path(args.hpo_dir)

    L = pd.read_csv(hpo_dir / "loss_matrix.csv", index_col="config_id")
    cfgs = pd.read_csv(hpo_dir / "configs.csv", index_col="config_id").loc[L.index]
    ids = list(L.columns)
    X_variants = {"base": pd.read_csv(data_dir / "X.csv", index_col="dataset_id").loc[ids],
                  "ext": pd.read_csv(data_dir / "X_extended.csv", index_col="dataset_id").loc[ids]}
    U = cfgs[U_COLS].to_numpy()
    K, n = L.shape
    T = min(args.T, K)
    n_init = args.n_init
    budgets = [n_init, min(15, T), T]
    print(f"[hpo-ws] K={K} configs, n={n} datasets, T={T}, n_init={n_init}, seeds={args.seeds}")

    # rank do default por dataset
    drank = L.rank(axis=0, method="min").loc[0]
    drank.rename("default_rank").to_csv(out / "default_rank.csv", header=True)
    print(f"[hpo-ws] rank do default (de {K}): mediana {drank.median():.0f}, media {drank.mean():.1f}, "
          f"melhor-que-metade em {(drank <= K / 2).sum()}/{n} datasets")

    res = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(run_target)(L, X_variants, U, d, j, T, n_init, args.seeds) for j, d in enumerate(ids))
    curves = pd.DataFrame([r for rows, _ in res for r in rows],
                          columns=["method", "dataset", "seed", "i", "best_loss", "norm_regret"])
    curves.to_csv(out / "curves.csv", index=False)
    infos = {inf["target"]: inf for _, inf in res}

    # ---------- Analise 1
    seed_avg = curves.groupby(["method", "dataset", "i"]).norm_regret.mean().reset_index()
    mean_curve = seed_avg.groupby(["method", "i"]).norm_regret.mean().unstack("method")
    auc = seed_avg.groupby(["method", "dataset"]).norm_regret.mean().unstack("method")  # media sobre i=1..T
    auc.to_csv(out / "auc.csv")
    reg = {b: seed_avg[seed_avg.i == b].pivot(index="dataset", columns="method", values="norm_regret") for b in budgets}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for m in MAIN5:
        ax.plot(mean_curve.index, mean_curve[m].clip(lower=1e-3), label=m, lw=2 if m == MAIN_WS else 1.5)
    ax.axvline(n_init, color="gray", ls="--", lw=1)
    ax.set_yscale("log")
    ax.set_xlabel("Numero de avaliacoes")
    ax.set_ylabel("Regret normalizado medio (melhor ate agora)")
    ax.set_title("Curvas de busca: warm-start vs. cold-start (media sobre datasets e seeds)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "loss_curves.png", dpi=150)
    plt.close(fig)

    txt = []
    fr = {}
    for b in budgets:
        fr[f"b{b}"] = friedman_block(reg[b][MAIN5], f"orcamento {b} avaliacoes", txt, out / f"cd_b{b}.png",
                                     f"Regret normalizado com {b} avaliacoes")
    fr["auc"] = friedman_block(auc[MAIN5], "AUC (regret medio i=1..T)", txt, out / "cd_auc.png",
                               "AUC do regret normalizado")
    (out / "friedman.txt").write_text("\n".join(txt))

    # ---------- Analise 2
    # Colunas p_raw_* sao valores-p de Wilcoxon BRUTOS (sem correcao);
    # valores corrigidos por Holm: ver hs_hpo_wilcoxon_holm.py (wilcoxon_holm.csv).
    rows = []
    for m in WS_VARIANTS + [CONTROL, "ws_rank", "smbo_cold"]:
        row = {"method": m, "auc": auc[m].mean()}
        for b in budgets:
            row[f"regret@{b}"] = reg[b][m].mean()
        row["p_raw_auc_vs_ws_rank"] = safe_wilcoxon(auc[m], auc["ws_rank"]) if m != "ws_rank" else np.nan
        for b in budgets:
            row[f"p_raw@{b}_vs_ws_rank"] = safe_wilcoxon(reg[b][m], reg[b]["ws_rank"]) if m != "ws_rank" else np.nan
        row["p_raw_auc_vs_randnn"] = safe_wilcoxon(auc[m], auc[CONTROL]) if m in WS_VARIANTS else np.nan
        rows.append(row)
    mf = pd.DataFrame(rows)
    mf.to_csv(out / "metafeatures.csv", index=False)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
    sub = mf.set_index("method")
    for ax, col, ttl in [(axs[0], "auc", "AUC do regret normalizado"), (axs[1], f"regret@{n_init}", f"Regret apos {n_init} avaliacoes (so init)")]:
        cols = ["tab:gray" if m in ("smbo_cold", CONTROL) else "tab:green" if m == "ws_rank" else "tab:blue" for m in sub.index]
        ax.barh(sub.index, sub[col], color=cols)
        ax.set_xlabel(ttl + " (menor e melhor)")
        ax.invert_yaxis()
        ax.grid(alpha=0.3, axis="x")
    fig.suptitle("Variantes de warm-start por similaridade (base/ext x k) vs. ranking global e controle aleatorio")
    fig.tight_layout()
    fig.savefig(out / "metafeatures.png", dpi=150)
    plt.close(fig)

    # ---------- Analise 3
    fails = []
    for d in ids:
        nn_ids, nn_d = infos[d]["nn"]["ext"]
        warm = infos[d]["inits"][MAIN_WS]
        loss = L[d].to_numpy()
        trank = rankdata(loss, method="min")
        nn1_best = int(infos[d]["ranked"][nn_ids[0]][0])
        fails.append({
            "dataset": d, "nn1": nn_ids[0], "nn1_dist": nn_d[0],
            "auc_cold": auc.loc[d, "smbo_cold"], "auc_ws": auc.loc[d, MAIN_WS],
            "gain_auc": auc.loc[d, "smbo_cold"] - auc.loc[d, MAIN_WS],
            "gain_init": reg[n_init].loc[d, "smbo_cold"] - reg[n_init].loc[d, MAIN_WS],
            "warm_configs": " ".join(map(str, warm)),
            "warm_best_rank_in_target": int(min(trank[c] for c in warm)),
            "nn1_best_config": nn1_best, "nn1_best_rank_in_target": int(trank[nn1_best]),
            "target_oracle_config": int(np.argmin(loss)), "default_rank": int(drank[d]),
        })
    fd = pd.DataFrame(fails)
    fd.to_csv(out / "failure.csv", index=False)
    rho_auc = spearmanr(fd.nn1_dist, fd.gain_auc)
    rho_init = spearmanr(fd.nn1_dist, fd.gain_init)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(fd.nn1_dist, fd.gain_auc, c=np.where(fd.gain_auc < 0, "tab:red", "tab:blue"))
    for _, r in fd.iterrows():
        ax.annotate(r.dataset, (r.nn1_dist, r.gain_auc), fontsize=5, alpha=0.8)
    ax.axhline(0, color="gray", lw=1)
    ax.set_xlabel("Distancia L1 ao vizinho mais proximo (meta-features z-score, X_ext)")
    ax.set_ylabel(f"Ganho de AUC (cold - {MAIN_WS})")
    ax.set_title(f"Quando o warm-start falha? Spearman rho={rho_auc.statistic:.2f} (p={rho_auc.pvalue:.3f})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "failure_scatter.png", dpi=150)
    plt.close(fig)

    neg = fd[fd.gain_auc < 0]
    print(f"\n[falhas] {len(neg)}/{n} datasets com gain_auc<0 ({MAIN_WS} pior que cold)")
    for _, r in neg.iterrows():
        print(f"  {r.dataset}: gain_auc={r.gain_auc:+.4f} nn1={r.nn1} (dist {r.nn1_dist:.1f}) "
              f"melhor config do nn1={r.nn1_best_config} (rank {r.nn1_best_rank_in_target} no alvo); "
              f"warm={r.warm_configs}, melhor rank do warm no alvo={r.warm_best_rank_in_target}, "
              f"oraculo={r.target_oracle_config}")

    # ---------- summary.md
    L_ = [f"# Warm-starting HPO -- resumo\n",
          f"K={K} configs, n={n} datasets, T={T}, n_init={n_init}, seeds={args.seeds}, orcamentos={budgets}\n",
          f"- Rank do default entre {K} configs: mediana {drank.median():.0f}, media {drank.mean():.1f}; "
          f"melhor que a mediana em {(drank <= K / 2).sum()}/{n} datasets.",
          f"- Regret do default (norm.): media {np.mean([(L.loc[0, d] - L[d].min()) / ((L[d].max() - L[d].min()) or 1) for d in ids]):.4f}\n",
          "## AUC medio do regret normalizado (menor e melhor)\n", auc[ALL_METHODS].mean().sort_values().round(4).rename("auc").to_markdown(), "",
          "## Regret normalizado por orcamento\n"]
    for b in budgets:
        L_ += [f"Orcamento {b}:", reg[b][ALL_METHODS].mean().sort_values().round(4).rename("regret").to_markdown(), ""]
    L_ += ["## Friedman / Nemenyi (5 metodos principais)\n"]
    for k_, r in fr.items():
        best = r["mean_ranks"].idxmin()
        L_.append(f"- {k_}: p={r['pvalue']:.3g}, CD={r['cd']:.2f}, melhor rank medio: {best} ({r['mean_ranks'][best]:.2f})")
    L_ += ["", "## Analise 2 (meta-features)\n", mf.round(4).to_markdown(index=False), "",
           "## Analise 3 (falhas)\n",
           f"- Spearman(dist NN1, ganho AUC) = {rho_auc.statistic:.3f} (p={rho_auc.pvalue:.3g}); "
           f"(dist NN1, ganho@{n_init}) = {rho_init.statistic:.3f} (p={rho_init.pvalue:.3g})",
           f"- Ganho AUC medio {fd.gain_auc.mean():.4f}; datasets com ganho<0: {len(neg)}/{n}: {', '.join(neg.dataset) or '-'}"]
    (out / "summary.md").write_text("\n".join(L_) + "\n")
    print(f"[hpo-ws] saidas em {out}")
    print("\n".join(L_[6:12]))


if __name__ == "__main__":
    main()
