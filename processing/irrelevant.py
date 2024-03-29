"""
Generate {key}_irrelevant_indices that can be used as hard negatives examples
In our experiments, we use irrelevant passages retrives by BM25 as irrelevant 
"""

from datasets import load_from_disk, disable_caching
from meerqat.ir.metrics import find_relevant
import argparse

disable_caching()

def keep_relevant_search_wrt_original_in_priority(item, passages, key = "BM25"):
    # used to train DPR or context question encoder
    indices = item[f'{key}_indices']
    relevant_indices, _ = find_relevant(indices, item['output']['original_answer'], [], passages)
    if relevant_indices:
        item[f'{key}_provenance_indices'] = relevant_indices
    else:
        item[f'{key}_provenance_indices'] = item['original_answer_provenance_indices']
    item[f'{key}_irrelevant_indices'] = list(set(indices) - set(relevant_indices))
    return item

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--passages_path', type=str, required=True)
    parser.add_argument('--indice', type=str, required=True)
    arg = parser.parse_args()

    passages = load_from_disk(arg.passages_path)
    dataset = load_from_disk(arg.dataset_path)
        
    dataset = dataset.map(keep_relevant_search_wrt_original_in_priority, fn_kwargs=dict(passages=passages, key=arg.indice))
    dataset.save_to_disk(arg.dataset_path)
