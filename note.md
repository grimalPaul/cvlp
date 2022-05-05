# Note

## A propos des hidden states

Strategies at the end of the encoder :

- pooling last hidden states
- concat hidden states
- sum some hidden states

récup dernier hidden states ou
faire une sorte de moyenne des hidden states (pooling, ...) ou somme
Voir ce qui est fait sur d'autres modèles comme Bert pour tester différentes choses. Suivant ce que je vais mettre en place comme traitement sur hidden states on aura des résultats différents. Peut être une autre chise à tester stratégiquement. [blog stratégies pour Bert](https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/#3-extracting-embeddings).

Dans CLIP la stratégie est la suivante : pooled output
`pooled_output = last_hidden_state[torch.arange(last_hidden_state.shape[0]), input_ids.argmax(dim=-1)]`

Mais on pourrait travailler sur la somme, de certains layers ou la concaténation.

Voir le test dans `test.ipynb`

Rappel :

past key values = cointaines precomputed key and value hidden states of the attention blocks.

hidden states = sorties des différentes couches

Quel layer fait quel hidden state

Dans VLT5

### VLT5

En comparant on voit que le code de VLT5 est complètement basé la dessus [ici](https://github.dev/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py)

### CLIP

Dans l'encoder de text ils utilisent des causals Mask je ne connais pas leur utilité mais à garder en tête

## Explication shared  Embed

[Tying the output and input embeddding](https://arxiv.org/pdf/1608.05859.pdf)

[Section 3.4 attention is all you need](https://arxiv.org/pdf/1706.03762.pdf)

dans VLT5 encoder et decoder partage même embedding

## A propos des tokenizer

```py
 """Note that this method behave differently between fast and slow tokenizers:
    
        - in fast tokenizers (instances of :class:`~transformers.PreTrainedTokenizerFast`), this method will
          replace the unknown tokens with the :obj:`unk_token`,
        - in slow tokenizers (instances of :class:`~transformers.PreTrainedTokenizer`), this method keep unknown
          tokens unchanged."""
```

Voir `test.ipynb:` on a pu encoder des entités nommées donc on peut considérer que les entités nommées sont dans le vocabulaire.

padding =  true permet de pouvoir faire du batch
