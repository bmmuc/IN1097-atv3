# Warm-starting da otimização de hiperparâmetros em detecção de discurso de ódio

Código da atividade prática 3 de IN1097 (Tópicos Avançados em Agentes
Inteligentes 2).

A pergunta deixa de ser *qual algoritmo usar* (Atividades 1 e 2) e passa a ser
**com qual configuração começar a busca de hiperparâmetros** de um algoritmo
fixo. Os 50 datasets de discurso de ódio e as meta-características `X` (base,
36 colunas) e `X_extended` (82 colunas) são os das atividades anteriores.

## Os meta-dados

| objeto | forma | conteúdo |
|---|---|---|
| `data_hs/hpo/loss_matrix.csv` | 128 × 50 | **matriz de perdas**: 1 − macro-F1 médio (CV 5 folds) por (configuração, dataset) |
| `data_hs/hpo/loss_folds.csv` | 32 000 linhas | perda e tempo de treino de cada fold |
| `data_hs/hpo/configs.csv` | 128 linhas | hiperparâmetros de cada configuração e coordenadas normalizadas `u_*` |
| `data_hs/hpo/build_cost.csv` | 50 linhas | segundos de CPU por dataset (total: 63,2 h) |
| `data_hs/X.csv`, `data_hs/X_extended.csv` | 50 × 36, 50 × 82 | meta-características |

- **Algoritmo**: `StandardScaler` + `MLPClassifier` (uma camada oculta) sobre
  embeddings congelados do `all-MiniLM-L6-v2` — o `minilm_mlp`, melhor pipeline
  da Atividade 1.
- **Espaço Θ** (escala log): neurônios `h ∈ [16, 512]`, `alpha ∈ [1e-6, 1e-1]`,
  `learning_rate_init ∈ [1e-4, 1e-1]`, `batch_size ∈ [16, 512]`; `max_iter=200`.
- **K = 128**: configuração 0 = padrão do scikit-learn; 1–127 = sequência de
  Sobol embaralhada (semente 42).
- **Protocolo**: `StratifiedKFold(5, shuffle=True, random_state=42)`, mesmos
  folds para todas as configurações de um dataset.

## Reprodução

```bash
pip install -r requirements.txt
```

**1. Matriz de perdas** (~5,5 h em 12 núcleos; retomável, cache por dataset em
`data_hs/hpo/cache/`):

```bash
python hs_hpo_build_matrix.py --K 128 --n-jobs 12
```

Exige os parquets em `data_hs/datasets/` e os embeddings MiniLM em cache
(`data_hs/embeddings/*__minilm.npy`), que não são versionados. A matriz
resultante está versionada, então é possível partir direto da etapa 2. Para
regerar as entradas do zero, ver *Reprodução integral* abaixo.

**2. Busca simulada e análises** (~30 min em 12 núcleos):

```bash
python hs_hpo_warmstart.py --T 30 --n-init 5 --seeds 20
python hs_hpo_wilcoxon_holm.py   # Wilcoxon pareado com correção de Holm (valores do relatório)
```

Cada busca consulta a matriz em vez de treinar. Protocolo leave-one-dataset-out:
o dataset alvo sai da matriz e da normalização das meta-características.
Métodos comparados (orçamento T = 30, n_init = 5, 20 sementes):

| método | pontos iniciais |
|---|---|
| `default` | só a configuração padrão |
| `random` | random search sem reposição |
| `smbo_cold` | 5 aleatórios + SMBO |
| `ws_sim_{base,ext}_k{1,3,5}` | melhores configurações dos k datasets mais próximos (L1 nas meta-características z-score) + SMBO |
| `ws_sim_randnn_k3` | controle: melhores configurações de 3 datasets sorteados + SMBO |
| `ws_rank` | 5 configurações de melhor rank médio nos datasets de treino + SMBO |

Os pontos iniciais de `ws_sim_*` e `ws_rank` são determinísticos por dataset alvo; as 20 sementes afetam apenas `random`, `smbo_cold`, o controle `ws_sim_randnn_k3` e os reinícios do otimizador do GP, logo as 20 execuções não são 20 inicializações independentes de warm-start.

SMBO idêntico para todos: processo gaussiano (Matérn 2,5 + ruído branco) e
Expected Improvement sobre as configurações ainda não avaliadas.

## Saídas (`results_hpo/`)

| arquivo | conteúdo |
|---|---|
| `curves.csv` | melhor perda e regret normalizado por (método, dataset, semente, avaliação) |
| `loss_curves.png`, `auc.csv` | análise 1: curvas de regret e AUC por dataset |
| `friedman.txt`, `cd_b5.png`, `cd_b15.png`, `cd_b30.png`, `cd_auc.png` | Friedman + Nemenyi em 5, 15 e 30 avaliações e na AUC |
| `metafeatures.csv`, `metafeatures.png` | análise 2: X_base vs X_extended, k ∈ {1, 3, 5}, contra `ws_rank` e o controle aleatório; colunas `p_raw_*` são valores-p brutos, sem correção (corrigidos em `wilcoxon_holm.csv`) |
| `failure.csv`, `failure_scatter.png` | análise 3: ganho do warm-start sobre o SMBO frio vs distância ao vizinho mais próximo |
| `wilcoxon_holm.csv` | Wilcoxon pareado com correção de Holm por família de comparações (os valores-p do relatório; os de `summary.md` não são corrigidos) |
| `default_rank.csv` | rank da configuração padrão entre as 128 em cada dataset |
| `summary.md` | resumo numérico |

## Resultado em uma linha

Os warm-starts reduzem o regret após os 5 pontos iniciais a menos da metade do
SMBO frio, e a vantagem persiste (menor) até T = 30. O controle independente
dos dados (`ws_rank`) empata ou supera o warm-start por similaridade: o ganho
vem principalmente de um portfólio de configurações boas em quase todo
dataset (taxa de aprendizado baixa, rede larga), não das meta-características.
Discussão completa no relatório.

## Reprodução integral (a partir das fontes)

Ordem de execução para regerar todas as entradas da etapa 1. Todas as saídas
abaixo, exceto parquets e embeddings, já estão versionadas.

| passo | comando | produz | tempo |
|---|---|---|---|
| 0 | `pip install -r requirements.txt` | ambiente | |
| 1 | `python build_datasets.py` | `data_hs/datasets/*.parquet`, `data_hs/datasets_manifest.csv` | ~30 s |
| 2 | `python hs_build_metadataset.py [--embed-python /caminho/python-com-cuda]` | `data_hs/embeddings/*.npy`, `X.csv`, `P.csv`, `P_folds.csv` | ~2 h (embeddings: ~4 min em GPU) |
| 3 | `python hs_build_extended_x.py` | `data_hs/X_extended.csv`, `meta_feature_catalog.csv` | ~7 min |
| 4 | `python hs_hpo_build_matrix.py --K 128 --n-jobs 12` | `data_hs/hpo/*` | ~5,5 h |
| 5 | `python hs_hpo_warmstart.py` | `results_hpo/*` | ~30 min |
| 6 | `python hs_hpo_wilcoxon_holm.py` | `results_hpo/wilcoxon_holm.csv` | segundos |

Observações:

- **Passo 1 — fontes externas.** 32 dos 50 datasets vêm do HuggingFace e
  exigem rede e um token do HuggingFace (`huggingface-cli login`). Os outros 18
  vêm de cópias locais de fontes com termos de uso próprios, que precisam ser
  obtidas junto aos autores originais. Elas ficam sob a raiz apontada pela
  variável `TCC_ROOT`:

  ```
  $TCC_ROOT/implicit-hate-corpus/   implicit_hate_v1_stg{1,2,3}_posts.tsv   (IHC)
  $TCC_ROOT/sbic/                   sbic_{dev,test}_filled_v*.csv            (SBIC)
  $TCC_ROOT/hatewic/                HateWiC_IndividualAnnos_with_def.csv     (HateWiC)
  $TCC_ROOT/hateval/                arquivos do SemEval-2019 Task 5          (HatEval)
  ```

  Sem elas, o script constrói só os 32 datasets do HuggingFace, registra quais
  foram descartados e não falha. Nesse caso a matriz resultante cobre esse
  subconjunto, e não a reportada. O script é determinístico.
- **Passo 2.** Este passo é o que gera os embeddings MiniLM usados aqui. Ele
  também refaz a CV dos 10 algoritmos da Atividade 1, que é a parte demorada.
  Os modelos (`all-MiniLM-L6-v2`, hateBERT, mBERT) são baixados do HuggingFace
  na primeira execução. `--embed-python` aponta para um interpretador com
  PyTorch+CUDA, caso o ambiente principal não tenha.
- **Passo 3.** Baixa o léxico HurtLex v1.2 (CC BY-SA 4.0) para
  `data_hs/lexica/`.
- **Diagramas.** `python hs_hpo_regen_cd.py` regera só os diagramas de
  diferença crítica a partir de `results_hpo/curves.csv` e `auc.csv`.

## Demais scripts

Os scripts das Atividades 1 e 2 necessários para regerar os dados de entrada
estão incluídos: `build_datasets.py`, `hs_build_metadataset.py`,
`hs_embed_worker.py`, `hs_meta_features.py`, `hs_algorithms.py`,
`hs_build_extended_x.py`, `hs_domain_meta_features.py` e `evaluate.py` (teste
de Friedman/Nemenyi e diagramas de diferença crítica).
