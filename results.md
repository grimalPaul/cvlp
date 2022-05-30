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

## Reminder

- Mean Reciprocal Rank : The reciprocal rank of a query response is the multiplicative inverse of the rank of the first correct answer. The MRR is the average.
- Hits@K (here hit_rate@K) : proportion of questions for which IR retrieves at least one relevant passages in top-K
- Precision@K = Precision is the proportion of the retrieved documents that are relevant, $\frac{\text{\# of recommended items @K that are relevant}}{\text{\# of recommended items @K}}$
