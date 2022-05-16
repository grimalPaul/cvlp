from datasets import load_from_disk, disable_caching
from meerqat.ir.metrics import find_relevant
import argparse

disable_caching()

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, required=True)
parser.add_argument('--kb_path', type=str, required=True)
arg = parser.parse_args()


pathKB = arg.dataset_path
pathDat = arg.kb_path
kb = load_from_disk(pathKB)
dataset = load_from_disk(pathDat)

def keep_relevant_search_wrt_original_in_priority(item, kb, key = "BM25"):
    # this contains the latest result of the fusion
    # to reproduce the results of the paper, use DPR+Image as IR
    indices = item[f'{key}_indices']
    relevant_indices, _ = find_relevant(indices, item['output']['original_answer'], [], kb)
    if relevant_indices:
        item[f'{key}_provenance_indices'] = relevant_indices
    else:
        item[f'{key}_provenance_indices'] = item['original_answer_provenance_indices']
    item[f'{key}_irrelevant_indices'] = list(set(indices) - set(relevant_indices))
    return item
    
dataset = dataset.map(keep_relevant_search_wrt_original_in_priority, fn_kwargs=dict(kb=kb))
dataset.save_to_disk(arg.dataset_path)