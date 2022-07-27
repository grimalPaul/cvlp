"""
Generate two news columns :
    - {key}_relevant_indices : indices that you retrieve with your system
    - {key}_irrelevant_indices : wrong indices (irrelevant passages) that you retrieve
"""

from datasets import load_from_disk, disable_caching
import argparse
import numpy as np

disable_caching()

def compare_relevant(item, key="vlt5"):
    # compare retrieve passage and passages containing answer
    indices = np.array(item[f'{key}_indices'], dtype=np.int64)
    relevant_indices = np.array(item['provenance_indices'], dtype=np.int64)
    item[f'{key}_relevant_indices'] = np.intersect1d(indices, relevant_indices)
    item[f'{key}_irrelevant_indices'] = np.setdiff1d(indices,relevant_indices)
    return item

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--indice', type=str, required=True)
    arg = parser.parse_args()
    dataset = load_from_disk(arg.dataset_path)
    dataset = dataset.map(compare_relevant, batched=False, fn_kwargs=dict(key=arg.indice))
    dataset.save_to_disk(arg.dataset_path)
