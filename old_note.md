# Title

## Model

As input text + an image

joint encoder image and text then an autoregressiv decoder

try with several ways to encode image

I can use T5 and BART

## Train

### Pretrain

Warm statrting encoder - decoder with corresponding model

Look at the paper, they explain some methods and refer to method used in T5

Pretrained tasks : ?

- followingg Raffel, we must mask 15% of input text
- find other ways

Maybe Use TriviQA

### Finetuning

Retrieve video to finetune with a new prefix (I think just add a prefix at the begining)

final objective : ? predict type of entities ? type of the pictures (painting, drawing, photo) and name of the entity (we supposed that CLIP knows the entities => idk if it works with T5 text encoder)

We dont have lot of picture in ViQuAE, maybe train with other things

fine tuning je dirai que c'est la qu'on introduit le ou les préfix pour le prompting

devra t  on utiliser de la data augmentation pour avoir un peu plus d'image, ou essayer dans un context de few shot ? Few shot me parait compliqué car nous avons quand même des layers a entrainer et entrainer T5

### TODO

- comprendre comment ils initialise la visual encoder avec faster cn : a voir mais je crois que toutes les images sont prétraités pour avoir tous les éléments sous
- get config to initialize models

## Experiment

## Benchmarks

How evaluate the model ? With Viquae

## Slurm help

## Analyse git

[Link to VL-T5](https://github.com/j-min/VL-T5/)

### VLT5

scripts : tous les scripts pour train les modèles sur les différents datasets
inference : inference on custom images
src : les différentes informations sont là

acces au model dans source dans :  

- modeling_bart.py
- modeling_T5.oy

### Notes about the model

```sh
src/
    modeling_t5.py modeling_bart.py                       <= VL-T5/VL-BART model classes
    pretrain.py, pretrain_data.py, pretrain_model.py      <= pretraining
    vqa.py, vqa_data.py vqa_model.py ...                  <= fine-tuning on downstream tasks (ex. VQA, GQA, NLVR2)
    multitask.py, multitask_data.py multiask_model.py     <= multitask learning on 7 downstream tasks
    param.py                                              <= (argparse) configuration
    tokenization.py                                       <= custom tokenizer
    utils.py, dist_utils.py                               <= utility functions
snap/                                                     <= store weight checkpoints
scripts/                                                  <= bash scripts for pretraining and finetuning
```

#### `modeling_T5.py`

##### `class VisualEmbedding`

Forward prends feats, pos; etc en input, donc tous l'encodage par fastercnn est fait avant.

Dans cette classe ils récups les infos qu'ils projettents ou transforment comme désiré

pour créer le model passe en paramètre config <= quelle config ? surement une config qu'on enregistre après training
nn.Sequential = créer un mini modele en donnant une liste de couche comme COnv2d, relu etc
On retrouve bien la somme dde 4 pour représenter image dans forward :  

```py
vis_embedding = feat_embedding + absolute_vis_pos_embedding + \
                img_order_embedding + obj_order_embedding
```

Dans notre cas on utiliserait directement un encoder a part peut être pour appliquer des transformations, nous n'avons pas besoin d'autant de chose.

On pourra peut être mettre tester plusieurs embedding en // avec les différents modèles utilisés utilisés dans ViQuAE

Je pense qu'il faudrait pour accélerer les choses, se créer une db de toutes les images qu'on utilisera pour l'entrainement qui seront déjà encodé pour gagner du temps

Peut  être utiliser le layer de projection de CLIP pour projeter si c'est dans la dimension voulue

##### `class JointEncoder`

T5stack impression que c'est le modèle modifiable
[isssue to use T5BLOCK](https://github.com/huggingface/transformers/issues/7360)
[Modulelist](https://pytorch.org/docs/stable/generated/torch.nn.ModuleList.html) créer une liste de Module (dans l'exemple font une liste de nn.Linear qu'on peut utiliser en itérant dessus)

_Remark_ : What’s the difference between a Sequential and a torch.nn. ModuleList? A ModuleList is exactly what it sounds like–a list for storing Module s! On the other hand, the layers in a Sequential are connected in a cascading way.

[transformers.models.t5.modeling_t5](https://huggingface.co/transformers/v4.11.3/_modules/transformers/models/t5/modeling_t5.html)

`self.final_layer_norm = T5LayerNorm`

in forward : torch.cat embdedding text and embedding img l208

##### `class VLT5`

based on `T5ForConditionalGeneration`
encoder used is `JointEncoder`
They extend vocab => need to do the same for prefix, instruction or other ideas. I can use appproximately the same methods

in forward after the output of the decoder :

```py
"Rescale output before projecting on vocab
# See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
```

then get lm logits from self.lm_head(sequence output)

They create a sequence thanks `VLSeq2SeqLMOutput`

##### `VLSeq2SeqLMOutput`

herite from ModelOutput
All models have outputs that are instances of subclasses of ModelOutput. Those are data structures containing all the information returned by the model, but that can also be used as tuples or dictionaries.

##### Bilan fonctionnement du model

extrait les 3 ou 4 features dont ils ont besoin
Les utilisent en entré du modèle =

puis visual embedding pour les transformer comme voulu
concat text embedding and image embedding

positon pour l'embedding semble dépendre des positions fournies par les modèles (x1, x2, y1,y2)

#### `trainer_base.py`

create config : récupère les config T5 et BART
ils chargent la config des modèles pretrained

ils vont créer un tokenizer pour le text
création de l'optimizer

A voir : init weight

il y a un train/test/val loader qui contient les données

#### `pretrain_model.py`

ici il y a la train step on peut y voir le batch de passer avec les features

dans output = self c'est pour le forward le text est input ids

#### `pretrain.py`

dans le main woorker endroit on sont chargés les loaders que je peux étudier

#### `pretrain_data.py`

Endroit ou sont cronstruits les loader

#### Bilan entrainement

on encode les images en dehors du modèle
on aura besoin d'une relative ou absolute position
Peut etre besoin d'appliquer une transo sur encodage de l'image

pour joint encoder on concat encodage image et text qu'on injecte dans T5
On peut appliquer à cette endroit là des masks

##### A comprendre

- les past keys
- comment bien utiliser les hidden states

### Note

- la position des éléments est elle utile ? pourrait l'être mais l'encoder doit pouvoir le faire ou alors utiliser un modèle qui nous donne des positions (comme dans l'article)
- Dans leur code semble avoir essayé plusieurs positions pour les objets
- [aide pour freeze paramètres](https://stackoverflow.com/questions/71048521/how-to-freeze-parts-of-t5-transformer-model) pas encore lu
- j'ai un doute concernant le fait d'associer un encoder text et un image, peuvent ils conduire à avoir des vocabulaires cohérents. CAD CLIP connait entité T5 non, vont ils réussir à mettre en commun info ? essayer d'y voir plus clair et construire qqch
- ils utilisent beaucoup de pretrained checkpoint [article about that on hugging face](https://huggingface.co/blog/warm-starting-encoder-decoder)
- wandb package python pour monitorer ses entrainements

## Ideas

encoder image avec CLIp pour les entités nommées
encoder avec plusieurs modèles différents
fine tuner avec des choses vue en prompting qui semble fonctionner
voir pour faire le moins de transfo possible pour évoter le pretrain
Peut etre interessant de projecter encoding de CLIP avec encoder d de CLIP

Pourquoi passer par dees modèles qui peuvent générer alors qu'on veut juste encoder nos modèles

## Ideas to go further

If it works, we can encode text + image and work with that

## O

<https://colab.research.google.com/github/j-min/VL-T5/blob/main/inference_example.ipynb>

## exp test

Démarche :

1. récupérer le modèle
2. faire du prompting  sur différentes images avec le modèle actuel
3. identifier des problèmes (par intuition les entités nommés)
4. garder en têtes mes idées

tester avec le modèle actuel ce qu'il se passe avec des images
voir comment je peux tester différentes formes de prompting avec ça pour voir ce que ça donne

en utilisant span prediction

utiliser en testant différentes hypothèses :

tester différents essaies pour analyser problème

Attention à ma démarche, voir comment réagi le moodèle avec différentes entités, voir si T5 arrive a reconnaitre l'image

C'est là ou il faut bidouiller

voir pour mettre les images que l'on veut

voir ce que ca donne sur du prompting

## problèmes identifiés

Utiliser token pour faire span prediction :
peut etre que si il faut utiliser <extra_id_0> is riding a <extra_id_1> nuemeroté de 0 à 99

Encodage de clip pourrait être intéressant car connait si c'est une peinture dessin etc ce qui permet de tirer + d'infos qu'autre chose
mais pourrait être interessant d'utiliser d'autres choses
