from datasets import load_from_disk, disable_caching
import ranx
import numpy as np


disable_caching()

def to_numpy():
    pass


dataset = load_from_disk('')
# doc : https://huggingface.co/docs/datasets/v1.0.1/faiss_and_ea.html

# add embedding to the column
dataset.add_faiss_index(column='embedding')

#  batch version to retrieve the scores and the ids of the examples
# batch version
# k number of example to retriev by queries
# output :
# total_scores (List[List[float]): The retrieval scores of the retrieved examples per query. 
# total_indices (List[List[int]]): The indices of the retrieved examples per query.
dataset.search_batch(index_name = '', queries = [], k='')
# scores_batch
# indices_batch

qrels = ranx.Qrels()
runs = {}
reference_kb_path='passages'
kb_path

"""
KnowledgeBase indexing only with faiss index 
index KB

est ce que je devrais faire une L2 norm
voir dans le code de Paul porur voir des versions pour le faire batcher
batcher et sur le gpu

utilise ce qu'il récupère de scores_batch, indices_batch

sauvegarde scores et indices dans le dataset line 357

map indices : je ne crois pas en avoir besoin, je crois que c'est utilisé
pour faire le mapping de la KB à passage. Donc ne pas se soucier de cela.

pour les runs ça:
format_run_indices : Identifiers in ranx should be str, not int. 
Also, list cannot be empty because of Numba
transforme indices batch en string


pour les qrels :
on donne les provenances indices
puis mets des confiances à 1 pour chaques indices de provenance

puis faire add run et qrels

Pour moi je peux direct compute les résultats vu que pas de notion de model différents

Voir si on peut ne rien récup et commment c'est géré dans le cas où cela peut arriver

Pas besoin de la classe search
En effet je ne vais pas avoir (enfin pour l'instant de méthodes différentes à utiliser)
Mais je serai surement amené à fuse des résultats

sauvegarde ou non des faiss index

on peut save index avec
save_faiss_index()

load_faiss_index('nom colonnes', 'nom de l'index')

Plusieurs types d'index possibles
https://github.com/facebookresearch/faiss/wiki/Faiss-indexes

je dois convertir mes listes pour numpy


On va commencer sans faire de normalization juste numpy 

"""