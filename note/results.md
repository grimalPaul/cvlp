# Results

All results below are on the test split (unless otherwise specified).

## miniviquae

### BM25

| # | Model | MRR@1 | MRR@5 | MRR@10 | MRR@20 | MRR@100 | P@1   | P@5   | P@10  | P@20  | P@100 | Hit_Rate@1 | Hit_Rate@5 | Hit_Rate@10 | Hit_Rate@20 | Hit_Rate@100 |
|---|-------|-------|-------|--------|--------|---------|-------|-------|-------|-------|-------|------------|------------|-------------|-------------|--------------|
| a | BM25  | 0.157 | 0.208 | 0.216  | 0.223  | 0.229   | 0.157 | 0.096 | 0.072 | 0.055 | 0.03  | 0.157      | 0.297      | 0.362       | 0.462       | 0.684        |

Avec Best hyperparameters: {'b': 0.2, 'k1': 0.4}

### Table

#### Zero shot T5

| # | Model              | MRR@1 | MRR@5  | MRR@10  | MRR@20  | MRR@100 | P@1   | P@5     | P@10    | P@20    | P@100   | Hit_Rate@1 | Hit_Rate@5 | Hit_Rate@10 | Hit_Rate@20 | Hit_Rate@100 |
|---|--------------------|-------|--------|---------|---------|---------|-------|---------|---------|---------|---------|------------|------------|-------------|-------------|--------------|
|a|vlt5_zs_1token |0.001|  0.001    |0.001     |0.001     |0.001      |0.001  |0.000     |0.001     |0.001     |0.001            |0.001  |0.002         |0.002          |0.006          |0.009|
|b|vlt5_zs_avg    |0.004|  0.008ᵃᶜ  |0.009ᵃᶜ   |0.010ᵃᶜ   |0.013ᵃᶜᵈ   |0.004  |0.003ᵃᶜ   |0.003ᵃᶜ   |0.002ᵃᶜ   |0.004ᵃᶜᵈ         |0.004  |0.014ᵃᶜ       |0.025ᵃᶜ        |0.040ᵃᶜ        |0.187ᵃᶜᵈ|
|c|vlt5_zs_vqa_1token|0    |  0.000    |0.000     |0.000     |0.001      |0      |0.000     |0.000     |0.000     |0.000            |0      |0.000         |0.000          |0.002          |0.020|
|d|vlt5_zs_imt_1token|0.002|  0.003ᶜ   |0.004ᶜ    |0.005ᶜ    |0.006ᵃᶜ    |0.002  |0.001ᶜ    |0.001ᶜ    |0.002ᶜ    |0.002ᵃᶜ          |0.002  |0.006         |0.013ᵃᶜ        |0.028ᵃᶜ        |0.091ᵃᶜ|
|e|vlt5_zs_vqa_avg|0.004|  0.008ᵃᶜ  |0.010ᵃᶜ   |0.011ᵃᶜᵈ  |0.014ᵃᶜᵈ   |0.004  |0.004ᵃᶜᵈ  |0.004ᵃᶜᵈ  |0.004ᵃᶜᵈ  |0.005ᵃᶜᵈ         |0.004  |0.017ᵃᶜᵈ      |0.027ᵃᶜ        |0.053ᵃᶜᵈ       |0.191ᵃᶜᵈ|
|f|vlt5_zs_imt_avg|0.001|  0.005ᵃᶜ  |0.006ᵃᶜ   |0.008ᵃᶜ   |0.011ᵃᶜᵈ   |0.001  |0.003ᵃᶜ   |0.003ᵃᶜ   |0.003ᵃᶜᵈ  |0.005ᵃᶜᵈ         |0.001  |0.014ᵃᶜ       |0.023ᵃᶜ        |0.051ᵃᶜᵈ       |0.198ᵃᶜᵈ|

#### Zero shot Bart

| # | Model              | MRR@1 | MRR@5  | MRR@10  | MRR@20  | MRR@100 | P@1   | P@5     | P@10    | P@20    | P@100   | Hit_Rate@1 | Hit_Rate@5 | Hit_Rate@10 | Hit_Rate@20 | Hit_Rate@100 |
|---|--------------------|-------|--------|---------|---------|---------|-------|---------|---------|---------|---------|------------|------------|-------------|-------------|--------------|
|a|    vlbart_zs_vqa_1token|        0|        0|         0 |        0|      0    |      0 |     0 |      0 |      0 |       0 |            0 |        0 |             0.001  |       0.001 |          0.006|
|b|    vlbart_zs_imt_1token|        0|        0|         0 |        0|      0    |      0 |     0 |      0 |      0 |       0 |            0 |        0 |             0      |       0     |          0.008|
|c|    vlbart_zs_vqa_avg   |        0|        0|         0 |        0|      0.001|      0 |     0 |      0 |      0 |       0 |            0 |        0.|001          0.002  |       0.002 |          0.002|
|d|    vlbart_zs_imt_avg   |        0|        0|         0 |        0|      0    |      0 |     0 |      0 |      0 |       0 |            0 |        0 |             0.001  |       0.001 |          0.002|
|e|    vlbart_zs_1token            |0    |0         |0         |0          |0          |0      |0       |0       |0        |0             |0         |0              |0              |0.001           |0.006|
|f|    vlbart_zs_avg               |0    |0.001     |0.001     |0.001      |0.001      |0      |0       |0       |0        |0             |0         |0.002          |0.002          |0.003           |0.005|
|g    |vlbart_zs_avg_dict          |0    |0.001 |0.002ᵃᶜᵈᵉᶠ  |0.003ᵃᵇᶜᵈᵉᶠ  |0.004ᵃᵇᶜᵈᵉᶠ      |0  |0.001  |0.001ᵃᵇᶜᵈᵉᶠ  |0.001ᵃᵇᶜᵈᵉᶠ  |0.001ᵃᵇᶜᵈᵉᶠ             |0         |0.004  |0.013ᵃᵇᶜᵈᵉᶠ   | 0.023ᵃᵇᶜᵈᵉᶠ  |  0.072ᵃᵇᶜᵈᵉᶠ|

On remarque bien pour la dernière ligne que suivant la facon dont je charge mon modèle on a des résultats différents. Je n'ai pas bien implémenté chargement du modèle.

#### Zero shot decoder

| # | Model              | MRR@1 | MRR@5  | MRR@10  | MRR@20  | MRR@100 | P@1   | P@5     | P@10    | P@20    | P@100   | Hit_Rate@1 | Hit_Rate@5 | Hit_Rate@10 | Hit_Rate@20 | Hit_Rate@100 |
|---|--------------------|-------|--------|---------|---------|---------|-------|---------|---------|---------|---------|------------|------------|-------------|-------------|--------------|
|a    |vlt5_zs_decoder        |0.001    |0.001     |0.001     |0.001      |0.001  |0.001      |0       |0       |0       | 0   |      0.001  |       0.001    |      0.001   |       0.001   |        0.018|
|b    |vlt5_zs_imt_decoder    |0        |0         |0         |0          |0      |0          |0       |0       |0       | 0   |      0      |       0        |      0.001   |       0.002   |        0.008|
|a    |vlbart_zs_decoder      |  0      |  0       |  0       |  0        |  0    |  0        |  0     |  0     |  0     |   0 |        0    |         0.001  |        0.001 |         0.002 |          0.005|
|b    |vlbart_zs_imt_decoder  |  0.001  |  0.001   |  0.001   |  0.001    |  0.001|  0.001    |  0     |  0     |  0     |   0 |        0.001|         0.002  |        0.002 |         0.003 |          0.005|

En zero shot vraiment pas bon

### vlt5_test1

VL T5 pretrained without further training for IR, no prefix, fastercnn
Cosine similarity on faiss

```json
{
    "mrr@1": 0.0008084074373484236,
    "mrr@5": 0.0010778765831312314,
    "mrr@10": 0.001193363359895292,
    "mrr@20": 0.0014307034921424574,
    "mrr@100": 0.00149402021792263,
    "precision@1": 0.0008084074373484236,
    "precision@5": 0.00048504446240905426,
    "precision@10": 0.0008084074373484236,
    "precision@20": 0.0008488278092158449,
    "precision@100": 0.0006224737267582862,
    "hit_rate@1": 0.0008084074373484236,
    "hit_rate@5": 0.0016168148746968471,
    "hit_rate@10": 0.002425222312045271,
    "hit_rate@20": 0.005658852061438965,
    "hit_rate@100": 0.00889248181083266
}

For less 1% of questions we retrieve at least one relevant passage in top 100.
This model doesnt work.

```

###  vlt5_zs_vqa_1token

```json
{
    "mrr@1": 0.0,
    "mrr@5": 0.0,
    "mrr@10": 0.0,
    "mrr@20": 0.00012551589158830786,
    "mrr@100": 0.0005402878547189864,
    "precision@1": 0.0,
    "precision@5": 0.0,
    "precision@10": 0.0,
    "precision@20": 0.00012126111560226356,
    "precision@100": 0.00028294260307194823,
    "hit_rate@1": 0.0,
    "hit_rate@5": 0.0,
    "hit_rate@10": 0.0,
    "hit_rate@20": 0.002425222312045271,
    "hit_rate@100": 0.02021018593371059
}
```

### vlt5_zs_imt_1token

```json
{
    "mrr@1": 0.0016168148746968471,
    "mrr@5": 0.0028967933171651847,
    "mrr@10": 0.003915964122108018,
    "mrr@20": 0.005053951518232674,
    "mrr@100": 0.006410926836064285,
    "precision@1": 0.0016168148746968471,
    "precision@5": 0.0011317704122877931,
    "precision@10": 0.0012934518997574777,
    "precision@20": 0.0016168148746968471,
    "precision@100": 0.002012934518997575,
    "hit_rate@1": 0.0016168148746968471,
    "hit_rate@5": 0.005658852061438965,
    "hit_rate@10": 0.012934518997574777,
    "hit_rate@20": 0.028294260307194827,
    "hit_rate@100": 0.09135004042037187
}
```

### vlt5_zs_vqa_avg

```json
{
    "mrr@1": 0.004042037186742118,
    "mrr@5": 0.008299649690110482,
    "mrr@10": 0.00957706175976184,
    "mrr@20": 0.011360972617670762,
    "mrr@100": 0.014197215128619618,
    "precision@1": 0.004042037186742118,
    "precision@5": 0.004203718674211802,
    "precision@10": 0.0037995149555375917,
    "precision@20": 0.0037995149555375913,
    "precision@100": 0.005125303152789006,
    "hit_rate@1": 0.004042037186742118,
    "hit_rate@5": 0.016976556184316895,
    "hit_rate@10": 0.02667744543249798,
    "hit_rate@20": 0.05335489086499596,
    "hit_rate@100": 0.19078415521422798
}
```

### vlt5_zs_imt_avg

```json
{
    "mrr@1": 0.0008084074373484236,
    "mrr@5": 0.0052681218000538935,
    "mrr@10": 0.006448974092466413,
    "mrr@20": 0.008213986086272728,
    "mrr@100": 0.011296138139737356,
    "precision@1": 0.0008084074373484236,
    "precision@5": 0.0029102667744543252,
    "precision@10": 0.0025869037995149557,
    "precision@20": 0.0030719482619240095,
    "precision@100": 0.004947453516572353,
    "hit_rate@1": 0.0008084074373484236,
    "hit_rate@5": 0.0137429264349232,
    "hit_rate@10": 0.023443815683104285,
    "hit_rate@20": 0.05092966855295069,
    "hit_rate@100": 0.19805982215036377
}
```

### vlbart_zs_vqa_1token

```json
{
    "mrr@1": 0.0,
    "mrr@5": 0.0,
    "mrr@10": 0.00010105092966855295,
    "mrr@20": 0.00010105092966855295,
    "mrr@100": 0.000194813080267826,
    "precision@1": 0.0,
    "precision@5": 0.0,
    "precision@10": 8.084074373484237e-05,
    "precision@20": 4.0420371867421184e-05,
    "precision@100": 5.658852061438966e-05,
    "hit_rate@1": 0.0,
    "hit_rate@5": 0.0,
    "hit_rate@10": 0.0008084074373484236,
    "hit_rate@20": 0.0008084074373484236,
    "hit_rate@100": 0.005658852061438965
}
```

### vlbart_zs_imt_1token

```json
{
    "mrr@1": 0.0,
    "mrr@5": 0.0,
    "mrr@10": 0.0,
    "mrr@20": 0.0,
    "mrr@100": 0.00016336181331850718,
    "precision@1": 0.0,
    "precision@5": 0.0,
    "precision@10": 0.0,
    "precision@20": 0.0,
    "precision@100": 8.084074373484237e-05,
    "hit_rate@1": 0.0,
    "hit_rate@5": 0.0,
    "hit_rate@10": 0.0,
    "hit_rate@20": 0.0,
    "hit_rate@100": 0.008084074373484237
}
```

### vlbart_zs_vqa_avg

```json
{
    "mrr@1": 0.0,
    "mrr@5": 0.0004042037186742118,
    "mrr@10": 0.00048504446240905415,
    "mrr@20": 0.00048504446240905415,
    "mrr@100": 0.000511991376987335,
    "precision@1": 0.0,
    "precision@5": 0.00032336297493936947,
    "precision@10": 0.00024252223120452713,
    "precision@20": 0.0002021018593371059,
    "precision@100": 0.00036378334680679065,
    "hit_rate@1": 0.0,
    "hit_rate@5": 0.0008084074373484236,
    "hit_rate@10": 0.0016168148746968471,
    "hit_rate@20": 0.0016168148746968471,
    "hit_rate@100": 0.002425222312045271
}
```

### vlbart_zs_imt_avg

```json
 {
    "mrr@1": 0.0,
    "mrr@5": 0.0,
    "mrr@10": 0.00011548677676406051,
    "mrr@20": 0.00011548677676406051,
    "mrr@100": 0.00012737512143094908,
    "precision@1": 0.0,
    "precision@5": 0.0,
    "precision@10": 8.084074373484237e-05,
    "precision@20": 0.00012126111560226354,
    "precision@100": 8.89248181083266e-05,
    "hit_rate@1": 0.0,
    "hit_rate@5": 0.0,
    "hit_rate@10": 0.0008084074373484236,
    "hit_rate@20": 0.0008084074373484236,
    "hit_rate@100": 0.0016168148746968471
}
```

## Reminder

- Mean Reciprocal Rank : The reciprocal rank of a query response is the multiplicative inverse of the rank of the first correct answer. The MRR is the average.
- Hits@K (here hit_rate@K) : proportion of questions for which IR retrieves at least one relevant passages in top-K
- Precision@K = Precision is the proportion of the retrieved documents that are relevant, $\frac{\text{\# of recommended items @K that are relevant}}{\text{\# of recommended items @K}}$


### pretrained tasks VLT5

avec best epoch 23. On obtient cela, on avait un batch size de 
4 * 8gpu soit 32 pendant l'entrainement.

    Model                         MRR@1    MRR@5    MRR@10    MRR@20    MRR@100    P@1    P@5    P@10    P@20    P@100    Hit_Rate@1    Hit_Rate@5    Hit_Rate@10    Hit_Rate@20    Hit_Rate@100
---  --------------------------  -------  -------  --------  --------  ---------  -----  -----  ------  ------  -------  ------------  ------------  -------------  -------------  --------------
a    multitask_resnet_embedding    0.003    0.007      0.01     0.011      0.014  0.003  0.004   0.005   0.004    0.006         0.003         0.016          0.036          0.057            0.16
