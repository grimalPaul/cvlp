# Expe

## Prompt tuning

### 1 config
0.15 de hite rate 100 le reste en dessous de 1%

"share_vis_embedding":true,
"share_embedding":true
"encoder_prompt_len":40,
"mid_dim":800,

## Adapter

### 1 config

"share_vis_embedding":true,
"share_embedding":true
"reduction_factor":8,
"unfreeze_visual_embedding": true
"unfreeze_layer_norms": true

avec L2 norm lors de l'indexage
    Model                 MRR@1    MRR@5    MRR@10    MRR@20    MRR@100    P@1    P@5    P@10    P@20    P@100    Hit_Rate@1    Hit_Rate@5    Hit_Rate@10    Hit_Rate@20    Hit_Rate@100
---  ------------------  -------  -------  --------  --------  ---------  -----  -----  ------  ------  -------  ------------  ------------  -------------  -------------  --------------
a    adapter1_embedding    0.006    0.014     0.017     0.019      0.024  0.006  0.006   0.006   0.006     0.01         0.006         0.027          0.048          0.082           0.317

sans L2 norm (FLAT)
    Model                 MRR@1    MRR@5    MRR@10    MRR@20    MRR@100    P@1    P@5    P@10    P@20    P@100    Hit_Rate@1    HRate@5    Hit_Rate@10    Hit_Rate@20    Hit_Rate@100
---  ------------------  -------  -------  --------  --------  ---------  -----  -----  ------  ------  -------  ------------  ---------  -------------  -------------  --------------
a    adapter1_embedding    0.006    0.012     0.015     0.018      0.023  0.006  0.005   0.005   0.006    0.009         0.006      0.026          0.044          0.094           0.319


### Adapter avec projecction en fin d'encoder
    "share_vis_embedding":true,
    "share_embedding":true
    "reduction_factor":8,
"unfreeze_visual_embedding": true
"unfreeze_layer_norms": true
utilise normalization de Sentence T5

