"""

Usage :
python -m search --dataset=path_dataset --config=path/config.json \
    --metrics_path=path/to/save/metrics --k=<k nearest neighbors> \
    --batch_size=<batch size>
"""


import json
from datasets import load_from_disk, disable_caching
import ranx
import numpy as np
import argparse
import torch
import re
from pathlib import Path

disable_caching()


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return None


def L2norm(queries):
    """Normalize each query to have a unit-norm. Expects a batch of vectors of the same dimension"""
    norms = np.linalg.norm(queries, axis=1, keepdims=True)
    return queries/norms


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

    def __init__(self, kb_path=None, index_kwargs={},  device=None):
        # index kwargs = index_load = False, index_name = None,
        # index_path=None, key_kb = None, string_factory = None
        self.dataset = load_from_disk(kb_path)
        if device == "cpu":
            device = None
        else:
            device = get_device()
        # to save multiple index of this dataset
        self.index_names = {}
        self.index_doL2norm = {}
        for index_name, index_kwarg in index_kwargs.items():
            do_L2norm = self.add_or_load_index(index_name=index_name,
                                   device=device, **index_kwarg)
            self.index_doL2norm[index_name] = do_L2norm

    def search_batch(self, index_name, queries, k):
        # queries must be np.float32
        queries = np.array(queries, dtype=np.float32)
        if self.index_doL2norm[index_name]:
            queries = L2norm(queries)
        return self.dataset.search_batch(index_name, queries, k=k)

    def add_or_load_index(self, device=None, index_name=None, index_load=False, index_path=None, key_kb=None, string_factory=None):
        if string_factory is not None and 'L2norm' in string_factory:
            do_L2norm = True
        else:
            do_L2norm = False
        if index_load:
            self.dataset.load_faiss_index(
                index_name=index_name, file=index_path, device=device)
        else:
            # HACK: fix L2-normalisation on GPU https://github.com/facebookresearch/faiss/issues/2010
            if do_L2norm and device is not None:
                # normalize the vectors
                self.dataset = self.dataset.map(
                    lambda batch: {key_kb: L2norm(batch[key_kb])}, batched=True)
                # remove "L2norm" from string_factory
                string_factory = re.sub(
                    r"(,L2norm|L2norm[,]?)", "", string_factory)
                if not string_factory:
                    string_factory = None
            self.dataset.add_faiss_index(
                column=key_kb, string_factory=string_factory, device=device, index_name=index_name)
            if index_path is not None:
                self.dataset.save_faiss_index(
                    index_name=index_name, file=f'{index_path}/{index_name}.faiss')
        self.index_names[index_name] = key_kb
        return do_L2norm


class Searcher:

    def __init__(self, k=100, kb_kwargs={}) -> None:
        self.qrels = ranx.Qrels()
        self.runs = {}
        self.k = k
        self.kbs = {}
        for kb_path, kwargs in kb_kwargs.items():
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
        default_metrics_kwargs = dict(
            metrics=[f"{m}@{k}" for m in ["mrr", "precision", "hit_rate"] for k in ks])
        self.metrics_kwargs = default_metrics_kwargs

    def __call__(self, batch):

        for kb in self.kbs.values():
            for index_name, key_kb in kb.index_names.items():
                queries = batch[key_kb]

                scores_batch, indices_batch = kb.search_batch(
                    index_name, queries, k=self.k)

                # store result in the dataset
                batch[f'{index_name}_scores'] = scores_batch
                batch[f'{index_name}_indices'] = indices_batch

                # store results in the run
                str_indices_batch, non_empty_scores = format_run_indices(
                    indices_batch, scores_batch)
                self.runs[index_name].add_multi(
                    q_ids=batch['id'],
                    doc_ids=str_indices_batch,
                    scores=non_empty_scores
                )

        # add Qrels scores depending on the documents retrieved by the systems
        str_indices_batch, non_empty_scores = format_qrels_indices(
            batch['provenance_indices'])
        self.qrels.add_multi(
            q_ids=batch['id'],
            doc_ids=str_indices_batch,
            scores=non_empty_scores
        )

        return batch


def dataset_search(dataset, k=100, metric_save_path=None, batch_size = 1000,**kwargs):
    searcher = Searcher(k=k, **kwargs)
    # search expects a batch as input
    dataset = dataset.map(searcher, batched=True, batch_size = batch_size)

    # compute metrics
    report = ranx.compare(
        searcher.qrels,
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--metrics_path', type=str, required=True)
    parser.add_argument('--k', type=int, required=True)
    parser.add_argument('--batch_size', type=int, required=False, default=1000)
    arg = parser.parse_args()
    with open(arg.config, 'r') as file:
        config = json.load(file)
    dataset = load_from_disk(arg.dataset_path)
    dataset = dataset_search(
        dataset=dataset,
        k=arg.k,
        metric_save_path=Path(arg.metrics_path),
        batch_size = arg.batch_size, 
        **config
    )
    dataset.save_to_disk(arg.dataset_path)