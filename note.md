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

Il faudra ajouter lors des entrainements la gestion du séparateur [SEP] ou le changer dans le dataset. Pour DPR le tokenizer est celui de BERT et SEP est un mot dans son vocabulaire.

Le sep bart est `</s>`
Le sep pour T5 est `?`. Il parle d'un sep_token

### Tokenizer T5

[doc hugging face](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Tokenizer)
No jobs required :-), activate a conda environment with tensorboard on the frontal server:

module load conda
conda activate myenv
tensorboard --logdir=. --port=6666  # change to any available port

From your local computer, redirect the port tensorboard is listening on:

ssh -L 6006:localhost:6006 -N username@factoryia

And open tensorboard in your browser at <http://localhost:6006>

## similarity search dense vector FAISS

[Faiss](https://github.com/facebookresearch/faiss)

[Exemples faciles ici](https://programmer.ink/think/faiss-dense-vector-retrieval-framework.html)

[Hugging face Faiss Index on dataset](https://huggingface.co/docs/datasets/faiss_es)

How generate qrels and runs.

Dans le code de Paul on génére avec le searcher

Lorsque que l'on créé les index on peut indiquer des traitements comme :

- `Flat` pour indiquer `the vectors are stored as is, without any encoding`.
- `L2norm`pour indiquer `L2-normalizes the input vectors`.

Innerproduct on normalized data = cosine similarity.

[GPU vs CPU](https://github.com/facebookresearch/faiss/wiki/Comparing-GPU-vs-CPU)

Paul a eu un problème pour faire L2norm avec GPU

[list metric](https://github.com/facebookresearch/faiss/wiki/The-index-factory)

voir le code e Paul + regarder la bibliothèque [pyserini](https://github.com/castorini/pyserini).

Il y a beaucoup de chose à mettre en place pour automatiser gérer faiss index. Pas besoin de elastic search pour dense retrieval, unniquement avec faiss index. Donc à ettre en place.

Il charge ou non sur le GPU en passant en indice le device.

Paul normalize le score avec normalize.

il me faut mes vecteurs au format numpy. Donc essayer d'enregistrer les embeddings en numpy.

ou juste numpy selon si j'utilise gpu ou non
tensor_array.cpu().detach().numpy()
ou juste utiliser.numpy()

## Visual embedding de T5

Fait un embedding dans une certaine dimension. Je pense que chaque box a une dimension. JeNo jobs required :-), activate a conda environment with tensorboard on the frontal server:

module load conda
conda activate myenv
tensorboard --logdir=. --port=6666  # change to any available port

From your local computer, redirect the port tensorboard is listening on:

ssh -L 6006:localhost:6006 -N username@factoryia

And open tensorboard in your browser at <http://localhost:6006>'image + on ajoute obj order embedding

```py
VisualEmbedding(
    (feat_embedding): Sequential(
      (0): Linear(in_features=2048, out_features=768, bias=True)
      (1): T5LayerNorm()
    )
    (absolute_vis_pos_embedding): Sequential(
      (0): Linear(in_features=5, out_features=768, bias=True)
      (1): T5LayerNorm()
    )
    (obj_order_embedding): Embedding(32200, 768)
    (img_order_embedding): Embedding(2, 768)
)
```

## Piste de Training

Déjà juste ajouter un préfix pour notre modèle et pour cette tâche.
`extract`.

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

**pretraining avec trivia qa en associant une image de wikipedia à la question et une autre à la base de données.**

### Training

Etape :

Commencer par VL-adapter directement !

1. mettre en place du continuous optimization pour essayer d'avoir les meilleurs performances possibles avec les données actuels. En supposant que le modèle est assez général. Voir si on peut faire du few shot en entrainant comme cela.

2. entrainer à l'aide d'une projection finale en gelant le modèle et en ajoutant une couche linéaire en fin de modèle. NON

3. mettre en place adapter et faire un finetuning sue le peu de donner que l'on a en utilisant juste les données ou +

4. pretraining et fine tuning (DPR + autres tâches possible de pretraining, et du coup utiliser CLIP)

- faire juste un finetuning sur le modèle actuel en modifiant seulement des parties du modèle.

- On part d'un modèle vierge avec adapter + on fait des tâches de vision de pretraining sur un encoder decoder. Puis on recup encoder, on duplique et pretarain si on trouve une tache ou fine tuning directement.

voir si pas un dataset question image et passage wikipedia.

- faire directement un training directement sur les encoders en partant des encoder pretrained

- faire un training sur les encoders sans entrainements

- Avec CLIP on va devoir faire quoi qu'il arrive des pretrainings donc à voir quels sont les pretraining qu'on va faire. Voir si dans VL adapter ca part uniquement de BART et T5 brute.

- on peut aussi pendant le pretraining ajouter une tâche de question réponse

Expliquer qu'utiliser encoder decoder pour faire le train généralise mieux.

Modèle génératif plus interessant mais utiliser dans un context discriminatif.

On peut essayer de partir du modèle préentrainé pour faire une optimization du prompting. (aussi commencer juste par le text puis ajouter image)

faire des tâches pour apprendre des personnalités de wikipedia ?

- match du text
- match des images
- math text + images

commencer par optimiser prompting
puis faire une projection à la fin
manières de gérer hidden states
entrainer un seul encoder sur des tâches et finetuner avec la tâche dont on a besoin plus tard

- pour apprendre la projection visuelle, on peut utiliser le dataset de CLIP (s'il est dispo), et juste entrainer cette projection (et Adapters ?) puis dans un second temps faire les autres choses ici article qui fournit dataset dans le style de CLIP [article](https://arxiv.org/abs/2111.02114).

- on peut toujours essayer de partager lors de l'apprentissage la proejction visuelle

- pour apprendre la projection visuelle on peut aussi utiliser  toutes les images de la KB, peut permettre de fixer les concepts

### Training encoder architecture

Learn visual projection with caption (maybe othe visual tasks) and freezing the model (not sure that freezing the whole model is a good idea).
Then DPR training (maybe coupled with visual task : roundrobin)
then finetuning

au vue du papier ST5, logique d'utiliser pooling ou encoder + decoder

Quand on utilisera clip
A voir mais je ne sais pas s'il y aura la notion des boxes, donc je pourrai peut être les virer de ce que je réalise actuelement.

### Training encoder decoder architecture

pretraining DPR or with dataset ST5 seems to be a good idea

### Contrastive learning

#### DPR

Explication de la gestion des exemples positives et des exemples négatives
Gestion quand relevant et irrelevant en commun entre passages ?

Va calculer une similarité entre une question et son relevant passage et tous les autres passages (reelavtn passages des autres question, son irrelevant passage, et irrelevant passages des autres questions).

puis LOGsoft max sur matrice de similarité PAS JUSTE SOFT MAX
[aide](https://stackoverflow.com/questions/65192475/pytorch-logsoftmax-vs-softmax-for-crossentropyloss)

[nllloss](https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html)

et enfin on utilise NLLLoss reduction = 'mean'

Utilise qu'un seul relevant passage. On fait la même chose pour l'instant.

## Dataset

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
superscript denote sgnificant differences in Fisher's randomizaton test with p <= 0.01.

### Relevant passages

On recherche dans viquae dataset les titres de l'article
puis on cherche dans les passages de cette article lesquels contiennent la réponse ou une réponse alternative

Remarque :
Vu un  passage recup comme une bonne réponse alors que ne parait pas pertinent. Je pense que le passage a été ajouté après avec find relevant dans le search. Si il trouve un passage qui peut quand même contenir la réponse il l'ajoute. peut être a ajouter à un moment pour comptabiliser ce genre de chose.

## Embedding

Ajouter du padding sur le text pour pouvoir faire du batch

embedding de T5

## VL Adapter

Appliquer Adapter to simplify the training, [github](https://github.com/ylsung/VL_adapter) [paper](https://arxiv.org/pdf/2112.06825.pdf)

ne font pas de pretrain, apprennent juste la projection pour être utilisé par le modèle de language.
Puis font uniquement un fine tuning avec les adapters pour avoir de bonnes performances en changeant très peu les paramètres.

Mais si aussi bien que ca en a l'air est une très bonne piste pour ce que je veux faire

Je pense que je peux aussi essayer d'optimiser mon prompting dans mon cas

Topk semble juste être une proportion des données

```py
 if isinstance(self.topk, float) and (0 < self.topk <= 1):
            used_samples = int(self.topk * len(data))
            data = random.sample(data, used_samples)
            if self.verbose:
                print(f"Use only {len(data)} data")

        elif self.topk > 0:
            data = data[:int(self.topk)]
            if self.verbose:
                print(f"Use only {len(data)} data")
```

### Dimension adapter

Dans multitask pour single adapter les dimensions sont d'après le paiper d=96
donc un facteur de reduction de 8

Chaque adapteur a :
couche dense avec biais de 768 vers 768/facteur de reduction
couche dense avec biais de 768/facteur de reduction

Deux adapters présents par block de transformers

Et 12 blocks

Donc

12 x 2 x (nb params des deux couches denses)


### Gestion des single adapters

Il y a un objet AdapterConfig(object)

Comment est géré l'entrainement des singles adapter ?

Etude du fichier multitask.py
roundrobin ordonnancement tourniquet
set_epoch est à utiliser pour distribuer les calculs.
avec roundrobin prends les tâches et les mélanges avec le seed `epoch` si `shuffle=True`
C'est le cas pour le training de single adapter
shuffle = True pour single adapter
nombre de workers 4

DataLoader
Dataset

Une bouche avec un nombre d'epoch
  Une boucle ou on réalise un certain nombre de tache
    À chaque fois on charge un tache différente et on fait la descente de gradient
    voir dans Multitask_dataLoader.set_epoch et **next** pour comprendre

### prompt multiple prompt

- si single prompt embedding sera partagé même si tâche différente.
- si multiple prompt une facon de faire embedding par tâche

avec Moduledict je peux passer différents modèles suivant ce que je veux mettre en place.

Comment est géré entrainement du prompting ?

Dans article ils précisent que promptig sur le decoder n'est pas efficace

### Prompt controller

quand on appelle le prompt module forward
on passe en argument la taille du batch size
la tache (en multi prompt, un prompt par tache)

si on remplit `use_tasks_prompt`, on aura le format suivant :
`{inputs prompt}{prompt de la task}{text ou question}`

## Features avec CLIP

Comment est géré extraction de CLIP.

voir clip_prepo_feats. py
aussi notion de box ?

dimension input visual projection

```note
feat dim 2048
dim model 768


if use_clip and clip_model_name == "ViT-B/32":
            self.visual_feat_dim = 768
        elif use_vit:
            self.visual_feat_dim = 768
        elif use_clip and clip_model_name == "RN50x4":
            self.visual_feat_dim = 2560
        else:
            self.visual_feat_dim = 2048
```

## Sortie de FasterCNN

Projection 2048 vers 768, bias = true

## Prefix disponible pour VLT5, VLBART

interessant :

- "image text match:"
- "vqa:"

Ne nous intéresse pas :

- "span prediction:"
- "ground caption:"
- "caption region:"
- "visual grounding:"

## Code

Nom des embedding :
nomModel_embedding_prefix_pooling

vlt5_embedding_vqa_1token
vlt5_embedding_vqa_avg
vlt5_embedding_imt_1token
vlt5_embedding_imt_avg

vlbart_embedding_vqa_1token
vlbart_embedding_vqa_avg
vlbart_embedding_imt_1token
vlbart_embedding_imt_avg

Nom des indices pour la recheche :
nomModel_technique_prefix_pooling

vlt5_zs_vqa_1token
vlt5_zs_vqa_avg
vlt5_zs_imt_1token
vlt5_zs_imt_avg

vlbart_zs_vqa_1token
vlbart_zs_vqa_avg
vlbart_zs_imt_1token
vlbart_zs_imt_avg

## Test a realiser

- refaire les tests avec les modèles chargés en state dict
- refaire les tests en ajoutant le token DE start de sequence ou CLS en début et utiliser ce token pour faire similarité
- utiliser token start avec le decodeur et refaire les experiences
- uniquement du prompt tuning (voir config à tester, encodeur w/o decodeur, traitement des hidden states
- puis essayer vl adapters sur les différentes configurations possibles (encoder w/o decodeur, juste fine tuning, puis essayer pretraining)

## Load model with state dict

[information state dict](https://pytorch.org/tutorials/recipes/recipes/what_is_state_dict.html)
[load and init some part of the model](https://discuss.pytorch.org/t/load-part-model-with-pretrained-weights-other-part-reinit-xavier/41310)
[load avec les keys du states dict](https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113)

## Distribuer et entrainement

[Lien utile](https://glassboxmedicine.com/2020/03/04/multi-gpu-training-in-pytorch-data-and-model-parallelism/)

[Distributed data vs distributeddataParallel](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)

If my model is too large to fit on a single GPU, I must use model parallel to split it

Dataloader  permet de charger les données pour le dataset

workers pour charger données pour le gpu
pin memory = True the data loader will copy Tensors
            into CUDA pinned memory before returning them.  If your data elements
            are a custom type, or your :attr:`collate_fn` returns a batch that is a custom type,

```py
# dans VL adapter voir CLT5.src.dist_utils.py
import torch.distributed as dist

distributed = ...

if distributed:
    torch.cuda.set_device(args.gpu)
    dist.init_process_group(backend='nccl')

if distributed:
  dist.barrier()

```

warmup ratio ou warmup_step, petit changement jusqu'à atteindre le learning rate

appliquer une normalisation

clip grad norm : garder le gradient dans certaine proportion.

Si je fais la validation uniquement sur un processus je dois mettre un dist barrier a la fin de validation pour que les aitres attendent le gpu qui gere la validation

Il y a possibilité d'ajoute un training à chaque fois

Si on a distributed data et qu'on met un batch size, ce sera le batch size par device donc si b_s = 16 on aura 16 * N gpu

## Arhictecture encoder decoder

J'ai mis en place très rapidement mon modèle. Peut être que j'auurai besoin de refactor à un moment.

Je pourrai juste charger une partie et pas les deux ocmme je fais à chaque fois pour faciliter l'inférence. A voir

Dépendra du test que je réalise

hidden states decoder 768

## Miniviquae dataset

pas d'indices de passage pour les passages suivants :
train 201, 388, 428, 677, 1077
val 43, 121, 473, 582, 688, 1075, 1099
test 578, 643, 904, 1056

par exemple dans test, index 578
answer thousand island
provenance title Saint Lawrence River
Saint Lawrence River, kb 2785,

La réponse n'est juste pas dans le text !

Pas trouvé de problème pour BM25 irrelevant, comprennent toujours des indices.

Mais y penser lorsqu'on utilisera d'autres provenances indices.

## ENCODAGE CLIP

pistes chercher vis_encoder et get vis encoder pour trouver des éléments de réponse.
Voir aussi vis forward
on voit comment sont créer les boxes
faire des tests avec vis_forward en chargeant le model de vision. regarder pour les embeddings.

## PARTAGE d'embedding, visual embedding, etc

Ce qu'on peut éventuellement partagé :

```py
# dans VLT5
self.shared = nn.Embedding(config.vocab_size, config.d_model)
# dans Joint encoder
self.visual_embedding = VisualEmbedding(self.config, embed_tokens)
```

si unfreeze visual embedding peut être interessant à partager
embedding du text aussi

Comment se déroule un entrainement quand on partage des embeddings ?

[why share embedding](https://arxiv.org/pdf/1608.05859.pdf)

Voir pour initialiser prompt et adapter par défaut
Mettre partage des infos

probablement couplé à du random search

<https://arxiv.org/pdf/1711.09846.pdf>
<https://medium.com/distributed-computing-with-ray/hyperparameter-optimization-for-transformers-a-guide-c4e32c6c989b>

## Normalization or not 

Ajouter ou non une normalization sur les embeddings pour faire un cosine similarity

ST5 ils font un cosine

DPR disent que c'est nul

Je pense que j'ai intérêt à normaliser

A la fin de l'encoder il y a une T5 layer norm qqui va juste diviser par la variance pas de soustraction du mean. 

Est ce que j'ajoute une L2 norm alors ? 

A voir pour l'instant je peux faire les tests comme cela

Peut être vaut le coup d'ajouter une couche de projection + normalisation comme CLIP et comme ST5 l'a fait !

Cela veut dire moyen hidden states et couche de projection

## Tensorboard

No jobs required :-), activate a conda environment with tensorboard on the frontal server:
```bash
module load conda
conda activate myenv
tensorboard --logdir=. --port=6666  # change to any available port

tensorboard --logdir=XXXXXXXX --port=AAAA
```

From your local computer, redirect the port tensorboard is listening on:

```bash
ssh -L AAAA:localhost:AAAA -N bergamote
ssh -L 6666:localhost:6666 -N username@factoryia
```

And open tensorboard in your browser at <http://localhost:6666>

## multi image

Un modele de vision qui prends plusieurs image en entrée

## A lire

<https://ai.googleblog.com/2021/05/kelm-integrating-knowledge-graphs-with.html>

- (IA)^3 : le travail que je mentionnais, conseillé par Julien Launay

Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning
<https://arxiv.org/pdf/2205.05638.pdf>

- une étude récente pas trop optimiste sur ces histoires de PET
Revisiting Parameter-Efficient Tuning: Are We Really There Yet?
<https://arxiv.org/abs/2202.07962>
