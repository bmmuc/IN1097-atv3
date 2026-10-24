# Warm-starting HPO -- resumo

K=128 configs, n=50 datasets, T=30, n_init=5, seeds=20, orcamentos=[5, 15, 30]

- Rank do default entre 128 configs: mediana 42, media 45.7; melhor que a mediana em 40/50 datasets.
- Regret do default (norm.): media 0.2169

## AUC medio do regret normalizado (menor e melhor)

| method           |    auc |
|:-----------------|-------:|
| ws_rank          | 0.04   |
| ws_sim_base_k3   | 0.0421 |
| ws_sim_ext_k5    | 0.0439 |
| ws_sim_ext_k3    | 0.0451 |
| ws_sim_base_k5   | 0.0493 |
| ws_sim_base_k1   | 0.0531 |
| ws_sim_randnn_k3 | 0.0535 |
| ws_sim_ext_k1    | 0.0539 |
| smbo_cold        | 0.0782 |
| random           | 0.0927 |
| default          | 0.2169 |

## Regret normalizado por orcamento

Orcamento 5:
| method           |   regret |
|:-----------------|---------:|
| ws_rank          |   0.0482 |
| ws_sim_ext_k5    |   0.0578 |
| ws_sim_ext_k3    |   0.0625 |
| ws_sim_base_k3   |   0.064  |
| ws_sim_base_k5   |   0.0736 |
| ws_sim_randnn_k3 |   0.0792 |
| ws_sim_ext_k1    |   0.0846 |
| ws_sim_base_k1   |   0.085  |
| random           |   0.1347 |
| smbo_cold        |   0.1373 |
| default          |   0.2169 |

Orcamento 15:
| method           |   regret |
|:-----------------|---------:|
| ws_sim_base_k3   |   0.0261 |
| ws_sim_ext_k5    |   0.0333 |
| ws_rank          |   0.0334 |
| ws_sim_ext_k3    |   0.0341 |
| ws_sim_base_k1   |   0.0393 |
| ws_sim_base_k5   |   0.0412 |
| ws_sim_ext_k1    |   0.043  |
| ws_sim_randnn_k3 |   0.0431 |
| smbo_cold        |   0.0536 |
| random           |   0.0736 |
| default          |   0.2169 |

Orcamento 30:
| method           |   regret |
|:-----------------|---------:|
| ws_sim_base_k3   |   0.0132 |
| ws_sim_ext_k3    |   0.0171 |
| ws_rank          |   0.0173 |
| ws_sim_ext_k5    |   0.0182 |
| ws_sim_base_k5   |   0.0199 |
| ws_sim_randnn_k3 |   0.021  |
| ws_sim_ext_k1    |   0.022  |
| ws_sim_base_k1   |   0.023  |
| smbo_cold        |   0.0249 |
| random           |   0.0454 |
| default          |   0.2169 |

## Friedman / Nemenyi (5 metodos principais)

- b5: p=2.19e-27, CD=0.86, melhor rank medio: ws_rank (1.64)
- b15: p=9.24e-28, CD=0.86, melhor rank medio: ws_sim_ext_k3 (1.95)
- b30: p=1.04e-28, CD=0.86, melhor rank medio: ws_sim_ext_k3 (1.93)
- auc: p=1.33e-30, CD=0.86, melhor rank medio: ws_rank (1.72)

## Analise 2 (meta-features)

| method           |    auc |   regret@5 |   regret@15 |   regret@30 |   p_raw_auc_vs_ws_rank |   p_raw@5_vs_ws_rank |   p_raw@15_vs_ws_rank |   p_raw@30_vs_ws_rank |   p_raw_auc_vs_randnn |
|:-----------------|-------:|-----------:|------------:|------------:|-------------------:|-----------------:|------------------:|------------------:|------------------:|
| ws_sim_base_k1   | 0.0531 |     0.085  |      0.0393 |      0.023  |             0.0103 |           0.0003 |            0.4823 |            0.7954 |            0.9924 |
| ws_sim_base_k3   | 0.0421 |     0.064  |      0.0261 |      0.0132 |             0.5721 |           0.1203 |            0.0977 |            0.144  |            0.001  |
| ws_sim_base_k5   | 0.0493 |     0.0736 |      0.0412 |      0.0199 |             0.0329 |           0.0038 |            0.1457 |            0.751  |            0.2955 |
| ws_sim_ext_k1    | 0.0539 |     0.0846 |      0.043  |      0.022  |             0.0031 |           0.0003 |            0.1405 |            0.7803 |            0.9086 |
| ws_sim_ext_k3    | 0.0451 |     0.0625 |      0.0341 |      0.0171 |             0.3326 |           0.2301 |            0.911  |            0.7137 |            0.0212 |
| ws_sim_ext_k5    | 0.0439 |     0.0578 |      0.0333 |      0.0182 |             0.1719 |           0.1726 |            0.6444 |            0.1333 |            0.0002 |
| ws_sim_randnn_k3 | 0.0535 |     0.0792 |      0.0431 |      0.021  |             0      |           0      |            0.0061 |            0.0703 |          nan      |
| ws_rank          | 0.04   |     0.0482 |      0.0334 |      0.0173 |           nan      |         nan      |          nan      |          nan      |          nan      |
| smbo_cold        | 0.0782 |     0.1373 |      0.0536 |      0.0249 |             0      |           0      |            0.0001 |            0.0038 |          nan      |

Valores-p brutos (sem correção); corrigidos por Holm em wilcoxon_holm.csv.

## Analise 3 (falhas)

- Spearman(dist NN1, ganho AUC) = -0.121 (p=0.401); (dist NN1, ganho@5) = -0.084 (p=0.563)
- Ganho AUC medio 0.0331; datasets com ganho<0: 6/50: ethos_category_gender_vs_rest, ethos_category_religion_vs_rest, ihc_stg1_implicit_vs_not, ihc_stg2_threatening_vs_rest, ihc_target_lgbt_vs_not, mlma_fr_hate_vs_normal
