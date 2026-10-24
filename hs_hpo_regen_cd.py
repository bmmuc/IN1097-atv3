"""Regenera os diagramas CD de results_hpo/ (cd_b5, cd_b15, cd_b30, cd_auc) a partir
de curves.csv e auc.csv, sem rodar a busca; reusa friedman_block de hs_hpo_warmstart."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from hs_hpo_warmstart import MAIN5, friedman_block

HPO_OUT = Path(__file__).resolve().parent / "results_hpo"
BUDGETS = [5, 15, 30]


def main():
    curves = pd.read_csv(HPO_OUT / "curves.csv")
    # media sobre seeds, por (metodo, dataset, i)
    seed_avg = curves.groupby(["method", "dataset", "i"]).norm_regret.mean().reset_index()
    auc = pd.read_csv(HPO_OUT / "auc.csv", index_col="dataset")
    tabelas = [(f"orcamento {b} avaliacoes", seed_avg[seed_avg.i == b].pivot(
        index="dataset", columns="method", values="norm_regret")[MAIN5],
        f"cd_b{b}.png", f"Regret normalizado com {b} avaliacoes") for b in BUDGETS]
    tabelas.append(("AUC (regret medio i=1..T)", auc[MAIN5], "cd_auc.png", "AUC do regret normalizado"))

    txt = []  # friedman.txt nao e reescrito
    for nome, df, png, titulo in tabelas:
        r = friedman_block(df, nome, txt, HPO_OUT / png, titulo)
        print(f"{nome}: CD={r['cd']:.3f} ranks={r['mean_ranks'].sort_values().round(2).to_dict()}")


if __name__ == "__main__":
    main()
