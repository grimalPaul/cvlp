from datasets import load_from_disk, disable_caching
import argparse
import json
from tqdm import tqdm
disable_caching()


def passage2text(passages):
    text = ''
    for passage in passages:
        text += passage + '\n'
    return text


def generate_doc(dataset, kb, passages, key, k=5):
    relevant_items = list()
    irrelevant_items = list()
    for i, row in enumerate(tqdm(dataset)):
        if len(row[f'{key}_relevant_indices']) > 0:
            indexes_passages = row[f'{key}_relevant_indices']
            relevant_items.append(
                {
                    'index': i,
                    'image_input': row['image'],
                    'image_passage': kb[passages[indexes_passages[0]]['index']]['image'],
                    'input': row['input'],
                    'passages': passages[indexes_passages]['passage'][0:k]
                })
        if len(row[f'{key}_irrelevant_indices']) > 0:
            indexes_passages = row[f'{key}_irrelevant_indices']
            irrelevant_items.append(
                {
                    'index': i,
                    'image_input': row['image'],
                    'image_passage': kb[passages[indexes_passages[0]]['index']]['image'],
                    'input': row['input'],
                    'passages': passages[indexes_passages]['passage'][0:k]
                })
    return relevant_items, irrelevant_items


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--passages_path', type=str, required=True)
    parser.add_argument('--kb_path', type=str, required=True)
    parser.add_argument('--key', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=False, default=None)
    parser.add_argument('--k', type=int, required=False, default=5)

    arg = parser.parse_args()

    passages = load_from_disk(arg.passages_path)
    kb = load_from_disk(arg.kb_path)
    dataset = load_from_disk(arg.dataset_path)
    relevant, irrelevant = generate_doc(dataset, kb, passages, arg.key, arg.k)
    if arg.save_path is not None:
        with open(f'{arg.save_path}/relevant.json', 'w') as f:
            json.dump(relevant, f, indent=4)
        with open(f'{arg.save_path}/irrelevant.json', 'w') as f:
            json.dump(irrelevant, f, indent=4)
