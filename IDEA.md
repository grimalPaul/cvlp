1. cosine similarité entre deux embeddings

CLIP avec deux encoder pour essayer de faire matcher question+image et passage + image

pour entrainement on pourrait utiliser TRIVIAQA pour avoir une plus grosse base d'entrainement ou quelquechose comme cela
Puis fine tuning sur le train pour enfin avoir les résultats.

2. Bonne representation texte et image

Utiliser T5 avec INSTRUCTION NER, faire un meilleur prompting

Penser aussi à la projection qu'il y a pour clip.

on pourrait juste essayer de projeter image et question dans le même espace mais je ne pense pas que ca va apprendre une bonne représentation

Il faut réussir à trouver une bonne représentation avec du transformers.



VLMo ressemble un peu mais pas le même objectif

Lire la survey sur Embedding je peux apprendre des choses je pense
