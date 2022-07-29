# Expe

## Prompt tuning

### 1 config

0.15 de hite rate 100 le reste en dessous de 1%
    Model                      MRR@1    MRR@5    MRR@10    MRR@20    MRR@100    P@1    P@5    P@10    P@20    P@100    Hit_Rate@1    Hit_Rate@5    Hit_Rate@10    Hit_Rate@20    Hit_Rate@100
---  -----------------------  -------  -------  --------  --------  ---------  -----  -----  ------  ------  -------  ------------  ------------  -------------  -------------  --------------
a    prompt_tuning_embedding    0.003    0.006     0.007     0.008      0.011  0.003  0.002   0.003   0.002    0.004         0.003         0.011          0.023          0.036            0.15

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

Resultat catastrophique, très vite atteint un seuil
Peut être du au paramètre de training
ou alors le manque de données
voir pour learning rate peut être
    Model                           MRR@1    MRR@5    MRR@10    MRR@20    MRR@100    P@1    P@5    P@10    P@20    P@100    Hit_Rate@1    Hit_Rate@5    Hit_Rate@10    Hit_Rate@20    Hit_Rate@100
---  ----------------------------  -------  -------  --------  --------  ---------  -----  -----  ------  ------  -------  ------------  ------------  -------------  -------------  --------------
a    adapter_projection_embedding    0.001    0.001     0.001     0.001      0.001  0.001      0       0       0        0         0.001         0.001          0.002          0.002           0.011

## config batch size

sur les 16 go je passe un batch size de :
sur les 32 go je passe un batch size de 20



## pretrained tasks on VLT5 :


    "stat_test": "fisher",
    "metrics": [
        "mrr@1",
        "mrr@5",
        "mrr@10",
        "mrr@20",
        "mrr@100",
        "precision@1",
        "precision@5",
        "precision@10",
        "precision@20",
        "precision@100",
        "hit_rate@1",
        "hit_rate@5",
        "hit_rate@10",
        "hit_rate@20",
        "hit_rate@100"
    ],
    "model_names": [
        "multitask_resnet_embedding"
    ],
    "multitask_resnet_embedding": {
        "scores": {
            "mrr@1": 0.0064672594987873885,
            "mrr@5": 0.009835623821072488,
            "mrr@10": 0.012089540747584402,
            "mrr@20": 0.013363896935987857,
            "mrr@100": 0.015647390465009948,
            "precision@1": 0.0064672594987873885,
            "precision@5": 0.004203718674211803,
            "precision@10": 0.003880355699272433,
            "precision@20": 0.003920776071139855,
            "precision@100": 0.006273241713823768,
            "hit_rate@1": 0.0064672594987873885,
            "hit_rate@5": 0.018593371059013743,
            "hit_rate@10": 0.034761519805982216,
            "hit_rate@20": 0.05335489086499596,
            "hit_rate@100": 0.17057396928051738
        },


= zero shot .......


% Add in preamble
\usepackage{graphicx}
\usepackage{booktabs}
========================


% To change the table size, act on the resizebox argument `0.8`.
\begin{table*}[ht]
\centering
\caption{
Overall effectiveness of the models.
The best results are highlighted in boldface.
Superscripts denote significant differences in Fisher's randomization testwith $p \le 0.01$.
}
\resizebox{0.8\textwidth}{!}{
\begin{tabular}{c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c}
\toprule
\textbf{\#}
& \textbf{Model}
& \textbf{MRR@1}
& \textbf{MRR@5}
& \textbf{MRR@10}
& \textbf{MRR@20}
& \textbf{MRR@100}
& \textbf{P@1}
& \textbf{P@5}
& \textbf{P@10}
& \textbf{P@20}
& \textbf{P@100}
& \textbf{Hit_Rate@1}
& \textbf{Hit_Rate@5}
& \textbf{Hit_Rate@10}
& \textbf{Hit_Rate@20}
& \textbf{Hit_Rate@100} \\
\midrule
a &
multitask\_resnet\_embedding &
\textbf{0.006}\hphantom{} &
\textbf{0.010}\hphantom{} &
\textbf{0.012}\hphantom{} &
\textbf{0.013}\hphantom{} &
\textbf{0.016}\hphantom{} &
\textbf{0.006}\hphantom{} &
\textbf{0.004}\hphantom{} &
\textbf{0.004}\hphantom{} &
\textbf{0.004}\hphantom{} &
\textbf{0.006}\hphantom{} &
\textbf{0.006}\hphantom{} &
\textbf{0.019}\hphantom{} &
\textbf{0.035}\hphantom{} &
\textbf{0.053}\hphantom{} &
\textbf{0.171}\hphantom{} \\
\bottomrule
\end{tabular}
}
\label{tab:results}
\end{table*}
~                 