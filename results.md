# Results

All results below are on the test split (unless otherwise specified).

## miniviquae

### BM25

    Model      MRR@1    MRR@5    MRR@10    MRR@20    MRR@100    P@1    P@5    P@10    P@20    P@100    Hit_Rate@1    Hit_Rate@5    Hit_Rate@10    Hit_Rate@20    Hit_Rate@100
---  -------  -------  -------  --------  --------  ---------  -----  -----  ------  ------  -------  ------------  ------------  -------------  -------------  --------------
a    BM25       0.157    0.208     0.216     0.223      0.229  0.157  0.096   0.072   0.055     0.03         0.157         0.297          0.362          0.462           0.684

Avec Best hyperparameters: {'b': 0.2, 'k1': 0.4}

## 0 shots

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

## Reminder

- Mean Reciprocal Rank : The reciprocal rank of a query response is the multiplicative inverse of the rank of the first correct answer. The MRR is the average.
- Hits@K (here hit_rate@K) : proportion of questions for which IR retrieves at least one relevant passages in top-K
- Precision@K = Precision is the proportion of the retrieved documents that are relevant, $\frac{\text{\# of recommended items @K that are relevant}}{\text{\# of recommended items @K}}$
