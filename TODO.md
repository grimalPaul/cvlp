# TODO et mise en place

Ne pas oublier comment fonctionne l'entrainement
Si aux boxes je ne dis pas à quoi ca correspond on va avoir un problème pour l'entrainement
Il faudrait plutot pouvoir donner 'tous l'embedding' de l'image avec clip directement
Pour  l'instant on commence comme cela et on pourra jouer sur lorsuqe l'on instancie le joint encoder pour changer cela

### pour charger joint encoder

créer un trainer pour juste charger le joint encoder en ajoutants les clés à ignorer
instantier le joint encoder à partir du checkpoint
le sauvegarder
et essayer de le recharger depuis le dictionnaire

Voir comment je peux faire pour charger un modèle directement depuis un modèle avec un ajout d'une fonction pour charger depuis. Je pense que c'est + logique avec le trainer que je veux mettre en place


## TODO

- [ ] charger Joint encoder uniquement Je pense qu'il faut ajouter les clés à ignorer puis sauvegarder à nouveau le modèle comme voulu
- [ ] Réussir à précharger le modèle avec T5
- [ ] Réussir à charger le modèle avec VL T5
- [] monter en compétence hugging face et architecture
  - [ ] hidden states, custom architecture
  - [ ] faire le learning
  - [ ] regarder github VL adaer comment mis en place pour construire architecture de mon model
  - [ ] réfléchir à la loss
  - [ ] comment on nourrit le model la question et image et des bons et mauvais exemple. Voir entrainement du DPR qui peut aider
- bien penser au vocab par rapport à mon dataset s'il y a des subtilités même si pour l'instant j'utilise le modèle tout fait
- [ ] réfléchir poru ajouter la projection a la fin, comment freeze plus haut, et changer encoder image
- [ ] torch topk
- [ ] voir comment faire pour charger une classe

## DONE

faire un semblant de la structure de la classe

## mise en place et choix

subtilité ici c'est qu'on veut le joint encoder. On s'en fout du reste mais on va le garder pour l'instant. Et on va voir pour ne charger que joint encoder et pas le reste
Partir du VLT5 de base

Passé par les Blocks etc pour construire mon modèle :

Traitement sur les images, on utilise ceux utiliser avec les modèles d'images donnés

au final on va vouloir essayer CLIP

VL BERT ?

VL adapter.

Comme pour VL T5 je dois pouvoir récupérer un pretrain du modèle pour essayer la version faster cnn et la version clip.
Pas mal de changement entre VL T5 et VL adapter. mais je dois pouvoir globalement récupérer les codes pour clip

On va partir de VLT5 quoi qu

contrastive pretraining

On sait que clip connait les concepts

pas d'intérêt à fine tuner le model de vision. + faster cnn pas stable a l'entrainement de cette facon la : source article => Ils en parlent dans VL adapter mais recup la source pour l'insérer.

Il faut que la recherche entre les vecteurs soient de la même facon que la cosine similarité entre modèle

De cette facon si on récup les même choses on peut imaginer essayer notre modèle avec d'autres types d'encoder

On peut facilement utiliser VL Bart aussi donc le faire aussi

### recup

on voit comment faire une architecture compatible clip et faster cnn
on récupère truc pour preprocess image avec clip
on voit si on utilise le meilleur modèle de VL T5 avec CLIP comme ca on pourra utiliser les deux en baseline.
Il y a des infos dans le trainer.py de VL adapter pour freeze le modèle

## ressource

multimodal représentation mais par région :

- [VLT5]()
- [VL BERT](https://github.com/jackroos/VL-BERT)
- [Unified framework vl bert](https://arxiv.org/pdf/2011.15124.pdf) [git](https://github.com/e-bug/volta)

voir si pas mieux de partir sur VL Bert car on s'en fout du decoder et VL T5 s'est son seul intérêt à part qu'on peut utiliser le pretrain. Mais je peux surement utiliser un pretrain de ce modèle.

<https://drive.google.com/drive/folders/1wLdUVd0zYFsrF0LQvAUCy5TnTGDW48Fo> pour recup les modèles

VL BART ?
VL T5 ?
VL T5 > VIL BERT

avec CLIP

- [VL Adapter](https://github.com/ylsung/VL_adapter/) [paper](https://arxiv.org/pdf/2112.06825.pdf)

- [Clip VIL](https://arxiv.org/pdf/2107.06383.pdf) Il reste plus intéressant d'utiliser VL T5 ou VL Bart je pense résultat meilleurs mais idées est la même

Au final il décide de freeze CLIP donc ca nous arrange

Comment features image de clip sont intégrés ?

dans CLIP VIL

clip_prepro_feats.py extract the clip features (both fc feature and last conv feature) of each image. The features are saved in data/cocotalk_clip_${model_type}_fc and data/cocotalk_clip_${model_type}_att
[fichier ou il font une prepro des images pour clip](https://github.com/clip-vil/CLIP-ViL/blob/master/CLIP-ViL-Direct/caption/scripts/clip_prepro_feats.py)

Dans Vl Adapter
dans VLT5/scripts/images on a les scripts bash pour faire les entrainements de VLT5

- [ViQuAE]
- [simple implementation de clip](https://towardsdatascience.com/simple-implementation-of-openai-clip-model-a-tutorial-ace6ff01d9f2)

Note sur VL T5 et Visual adapter

La philosophie c'est d'avoir un modeling un modeling_t5 qui est le modèle de base et dont pour chaque tahe particulière un modèle peut hériter

Dans classification.py on recup le clip et on le définit comme étant l'encoder du model. Da,s modeling on a qqch pour clip ligne 725 ou on voit un embedding. Ils utilisent pour les boxes ne sortent de matrice.
Je pense que pour les features ils prennent plusieurs aussi. Comprendre ce que renvoit exactement ce modèle.
