from datasets import load_from_disk, disable_caching
import ranx
import numpy as np
from cvlep.utils import device

disable_caching()

def normalize():
    # map embedding to numpy
    pass

def format_run_indices(indices_batch, scores_batch):
    """Identifiers in ranx should be str, not int. Also, list cannot be empty because of Numba"""
    str_indices_batch, non_empty_scores = [], []
    for indices, scores in zip(indices_batch, scores_batch):
        if len(indices) > 0:
            str_indices_batch.append(list(map(str, indices)))
            non_empty_scores.append(scores)
        else:
            str_indices_batch.append(["DUMMY_RUN"])
            non_empty_scores.append([0])
    return str_indices_batch, non_empty_scores


def format_qrels_indices(indices_batch):
    """Identifiers in ranx should be str, not int. Also, list cannot be empty because of Numba"""
    str_indices_batch, non_empty_scores = [], []
    for indices in indices_batch:
        if len(indices) > 0:
            str_indices_batch.append(list(map(str, indices)))
            # relevance score should always be 1, see https://github.com/AmenRa/ranx/issues/5
            non_empty_scores.append([1 for _ in indices])
        else:
            str_indices_batch.append(["DUMMY_QREL"])
            non_empty_scores.append([0])
    return str_indices_batch, non_empty_scores

class KnowledgeBase:
    """A KB can be indexed by several indexes."""
    def __init__(self, kb_path=None, **index_kwargs):
        # index kwargs = index_load = False, index_name = None,
        # index_path=None, key_kb = None, string_factory = None
        
        self.dataset = load_from_disk(kb_path)

        # to save multiple index of this dataset      
        self.index_names = {}
        for index_name, index_kwarg in index_kwargs.items():
            self.add_or_load_index(index_name=index_name,**index_kwarg)
    
    def search_batch(self, queries, k):
        # queries must be np.float32
        return self.dataset.search_batch(self.index_name, queries, k=k)

    def add_or_load_index(self, index_name = None, index_load = False, index_path=None, key_kb = None, string_factory = None):
        if index_load:
            self.dataset.load_faiss_index(index_name=index_name, file=index_path, device=device)
        else:
            self.dataset.add_faiss_index(column=key_kb,string_factory=string_factory,device=device,index_name=index_name)
            self.dataset.save_faiss_index(index_name=index_name,file=index_path)
        self.index_names[index_name] =  key_kb

class Searcher:
    # je veux pouvoir avoir plusieurs résultats en une fois
    # meme tableau avoir vlt5, clip t5, etc

    def __init__(self, k, **kb_kwargs) -> None:
        self.qrels = ranx.Qrels()
        self.runs = {}
        self.k = k
        self.kbs = {}
        for kb_path, kwargs in kb_kwargs:
            kb = KnowledgeBase(kb_path, **kwargs)
            # on a une kb indexés par plusieurs indexes
            # on doit enregistrés tout les runs possibles
            self.kbs[kb_path] = kb

            for index_name in kb.index_names.keys():
                run = ranx.Run()
                run.name = index_name
                self.runs[index_name] = run
        # metrics used for ranx
        ks = [1, 5, 10, 20, 100]
        default_metrics_kwargs = dict(metrics=[f"{m}@{k}" for m in ["mrr", "precision", "hit_rate"] for k in ks])
        self.metrics_kwargs = default_metrics_kwargs

    def __call__(self, batch):

        for kb in self.kbs.values():
            for index_name, key_kb in kb.index_names.items():
                queries = batch[key_kb]
    
                scores_batch, indices_batch = self.kb.search_batch(index_name, queries, k=self.k)
                
                # store result in the dataset
                batch[f'{index_name}_scores'] = scores_batch
                batch[f'{index_name}_indices'] = indices_batch

                # store results in the run
                str_indices_batch, non_empty_scores = format_run_indices(indices_batch, scores_batch)
                self.runs[index_name].add_multi(
                    q_ids=batch['id'],
                    doc_ids=str_indices_batch,
                    scores=non_empty_scores
                )

        # add Qrels scores depending on the documents retrieved by the systems
        str_indices_batch, non_empty_scores = format_qrels_indices(batch['provenance_indices'])
        self.qrels.add_multi(
            q_ids=batch['id'],
            doc_ids=str_indices_batch,
            scores=non_empty_scores
        )

        return batch


def dataset_search(dataset, k=100, metric_save_path=None, map_kwargs={}, **kwargs):
    searcher = Searcher(k=k, **kwargs)

    # search expects a batch as input
    dataset = dataset.map(searcher, batched=True, **map_kwargs)

    # compute metrics
    report = ranx.compare(
        searcher.qrels,
        # on peut passer plusieurx runs, c'est ce qu'on veut
        runs=searcher.runs.values(),
        **searcher.metrics_kwargs
    )

    print(report)
    # save qrels, metrics (in JSON and LaTeX), statistical tests, and runs.
    if metric_save_path is not None:
        metric_save_path.mkdir(exist_ok=True)
        searcher.qrels.save(metric_save_path/"qrels.trec", kind='trec')
        report.save(metric_save_path/"metrics.json")
        with open(metric_save_path/"metrics.tex", 'wt') as file:
            file.write(report.to_latex())
        for index_name, run in searcher.runs.items():
            run.save(metric_save_path/f"{index_name}.trec", kind='trec')

    return dataset    

dataset = load_from_disk('')
# doc : https://huggingface.co/docs/datasets/v1.0.1/faiss_and_ea.html

# add embedding to the column
dataset.add_faiss_index(column='embedding', index_name=None, device=None; string_factory=None)
dataset.save_faiss_index(index_name=None,file=None)


# si load = True on peut recharger index depuis sauvegarde :
dataset.load_faiss_index(index_name=None,file=None,device=None)

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

# pourquoi on normalize dans notre cas ??
dataset.map(lambda batch: {column : normalize(batch[column])}, batched = True)

# convert to numpy
passages = passages.map(lambda example : {'vlt5_embedding':np.array(example)},input_columns = 'vlt5_embedding')

"""
Mes besoins :
pouvoir faire mes tests sur plusieurs indices 
"vlt5 embedding, vlt5_embedding2, etc"

"""


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