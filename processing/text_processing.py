import argparse
from datasets import load_from_disk, disable_caching

disable_caching()


def add_prefix(item, prefix, key, name):
    item[f'{key}_{name}'] = prefix + ':' + item[key]
    return item


def embed_dataset(dataset_path, key, prefix, name):
    dataset = load_from_disk(dataset_path)
    dataset = dataset.map(add_prefix, batched=False,
                          fn_kwargs=dict(prefix=prefix, key=key, name=name))
    dataset.save_to_disk(dataset_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--key', type=str, required=True)
    parser.add_argument('--prefix', type=str, required=True)
    parser.add_argument('--name', type=str, required=True)
    arg = parser.parse_args()

    embed_dataset(arg.dataset_path, arg.key, arg.prefix, arg.name)
