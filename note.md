# Note

## A propos des hidden states

Strategies at the end of the encoder :

- concat n hidden states
or
- sum n hidden states
then average

dans clip vision utilise le hidden states correspondant au premier token [ici](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPVisionModel)
dans cliptext model pareil retourne last hidden states du premier tokens [doc](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel)

 [blog stratégies pour Bert](https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/#3-extracting-embeddings).

Chercher ou essayer différentes façon et ainsi trouver la meilleure.

Dans CLIPText

Mais on pourrait travailler sur la somme, de certains layers ou la concaténation.

Voir le test dans `test.ipynb`

Rappel :

past key values = cointaines precomputed key and value hidden states of the attention blocks.

hidden states = sorties des différentes couches

Quel layer fait quel hidden state

On pourrait imaginer pouvoir choisir l'embedding voulu quand on appelle encoder

### VLT5

En comparant on voit que le code de VLT5 est complètement basé la dessus [ici](https://github.dev/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py)

### CLIP

Dans l'encoder de text ils utilisent des causals Mask je ne connais pas leur utilité mais à garder en tête

est ce que dans clip embedding image et test est partagé ? dans le cas ou ca ne l'es pas on ne devrait pas mettre un embedding commun entre les deux encoders même si on peut quand même le faire.

## Explication shared  Embed

[Tying the output and input embeddding](https://arxiv.org/pdf/1608.05859.pdf)

[Section 3.4 attention is all you need](https://arxiv.org/pdf/1706.03762.pdf)

dans VLT5 encoder et decoder partage même embedding = Normal

dans VLT5 visual et text embedding on le même embedding

Brt Without pretrain
Embedding dim 50465 768
Bart epoch 30
Embedding dim 50465 768
T5 without pretrain
Embedding dim 32200 768
T5 epoch 30
Embedding
32200 768

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

## accéder aux images

on enregistre le nom de l'image dans le dataset passage dans le cas ou on veut faire les embeddings des images à chaque fois

dans le cas ou on a besoin d'enregistrer les features des images. Faire embeddings de la kb. Et trouver un moyen de transférer les embeddings avec le passage index

## similarity search dense vector

[Faiss](https://github.com/facebookresearch/faiss)

[Exemples faciles ici](https://programmer.ink/think/faiss-dense-vector-retrieval-framework.html)

[Hugging face Faiss Index on dataset](https://huggingface.co/docs/datasets/faiss_es)

## Visual embedding de T5

Fait un embedding dans une certaine dimension. Je pense que chaque box a une dimension. Je pense qu'avec Clip on fournit une seule image et embedding d'une seule image.
