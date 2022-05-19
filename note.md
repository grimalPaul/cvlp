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

padding =  true, il y aura la même taille pout tous les éléments passés en paramêtre
padding='max_lenght' => prend la taille max du modèle.
attention mask met des 1 ou il y a des mots et 0 sinon.
Dans notres cas les masks sont gérés automatiquement on ne passe que les séquences de tokens.

## accéder aux images

on enregistre le nom de l'image dans le dataset passage dans le cas ou on veut faire les embeddings des images à chaque fois

dans le cas ou on a besoin d'enregistrer les features des images. Faire embeddings de la kb. Et trouver un moyen de transférer les embeddings avec le passage index

## similarity search dense vector

[Faiss](https://github.com/facebookresearch/faiss)

[Exemples faciles ici](https://programmer.ink/think/faiss-dense-vector-retrieval-framework.html)

[Hugging face Faiss Index on dataset](https://huggingface.co/docs/datasets/faiss_es)

How generate qrels and runs.

Dans le code de Paul on génére avec le searcher

es : false => dans Knowledge Base on va utiliser add_or_load_faiss_index
C'est ce qu'on fait pour DPR et pour les images

voir le code e Paul + regarder la bibliothèque [pyserini](https://github.com/castorini/pyserini).

Il y a beaucoup de chose à mettre en place pour automatiser gérer faiss index. Pas besoin de elastic search pour dense retrieval, unniquement avec faiss index. Donc à ettre en place.

## Visual embedding de T5

Fait un embedding dans une certaine dimension. Je pense que chaque box a une dimension. Je pense qu'avec Clip on fournit une seule image et embedding d'une seule image.

## Piste de Training

on peut faire une première partie avec entrainement uniquement sur du text comme dpr encoder puis ajouter les images.

Ou faire un mix constant de ça

Dans viquae, DPR d'abord pretrain sur toutes les questions de triviaqa et de la base de données kilt_wikipedia.

Puis fine tuning sur dataset.

On pourrait dans une première phase faire cela.

Dans une seconde phase juste sur les photos. contrastive learning sue les photos (il faudrait trouver des bons et des mauvais exemples). On pourrait faire tirage aléatoire mais prendre au moins des mauvais exemple du même type. Générer des irrelevant passages en minant des indices uniquement avec encoder des images pour avoir des bons et des mauvais exemples. On pourra mixer avec ou sans.

Utiliser irrelevant BM25 pour avoir des mauvaise exemple quand on fait du contrastive learning pour le text.

Et pour avoir des mauvais exemples généraux soit aléatoire soit utilisé BM25_irrelevant. Et aussi faire une recherche uniquement avec modèle image, on cherche image article les plus proches, et on fais le même traitement pour avoir les irrelevants et avoir des mauvais exemples pertinents.

VL adapter, ils ne mettent à jour qu'une petite quantité pour le finetuning.
Trouver grosses facon de pretrain.
Puis faire du finetuning avec vl adapter si cela est possible.

On pourrait utiliser DPR indices pour générer des irrelevants. On pourrait utiliser le modèle non préentrainés ca pourrait être intérressant dans un second temps. Facile à faire car on l'a déjà fait et on utilisera un modèle tel quel donc pourra etre considéré comme zero shot.

## dataset

quand on fait un set format seul les colonnes indiqués sont récupérables en utilisant les indices

## Text retrieval

TF-IDF algorithm compute similarity score between 2 pieces of text. TF refers to how many words in the query are found in the context. IDF is the inverse of the fraction of documents containing this word. BM25 is a variation of the TF-IDF.

DPR (Dense Retrieval Passage) use two BERT encoder models, one encode passage the other the question. The model is trained to minimized the cross-entropy between the similarity of the passage and the question.

### BM25

tune

on cherche meilleur paramètre pour bm25

normalement aura généré des indices pour des parties du dataset

puis on lance le search pour avoir ces indices partout

on génère irrelevant pour avoir irrelevant.

on pourra utiliser cela comme pour entrainer DPR mais à voir.

### Ranx

[Ranx](https://github.com/AmenRa/ranx)

[Ranx documentation](https://amenra.github.io/ranx/qrels_and_run/)

Library to evaluate information retrieval

ranx use qrels to compute information

Qrels =  query relevances, lists the relevance judgements for each query, stores the ground truth for conducting evaluations.

Runs = stores the relevance scores estimated by the model under evaluation.

on fait une comparaison par rapport à ce qu'on est censé trouver.

Le search défini par paul permet de créer les qurels et le run.

signification

les lettres en puissance à coté des performances :
Indiquer que meilleur que tel ou tel ligne de manière significative.
superscript denote sgnificant differences in Fisher's randomizaton test with p <= 0.01

### Relevant passages

On recherche dans viquae dataset les titres de l'article
puis on cherche dans les passages de cette article lesquels contiennent la réponse ou une réponse alternative