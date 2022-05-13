# CVLP

CVLP = Constrastive Visual Language Pre-Training

## Installation

I suggest to create a virtual environnement and pip install the `requirements.txt`.

```bash
conda create -n cvlp python=3.7
conda activate cvlp
pip install -r requirements.txt
```

Install elasticsearch. I worked with this version : [elasticsearch-7.17](https://www.elastic.co/guide/en/elasticsearch/reference/7.17/targz.html)

### Download tokenizer and model

use `python download_models` to download and save models T5, Bart in `data/`

To download fastercnn, click to the links below :

- [Configuration](https://s3.amazonaws.com/models.huggingface.co/bert/unc-nlp/rcnn-vg-finetuned/config.yaml)
- [Pytorch_model.bin](https://cdn.huggingface.co/unc-nlp/frcnn-vg-finetuned/pytorch_model.bin)

Place the downloaded files in `data/frcnn_model/`.

### Download visual model

`TODO`

### Download pretrained encoder

`TODO`

### Download and preprocess data

You can get the viquae_dataset and the knowldege base from Hugging face with

```py

from datasets import load_dataset
dataset = load_dataset('PaulLerner/viquae_dataset')
dataset.save_to_disk(PATH_DATASET)

kb = load_dataset('PaulLerner/viquae_wikipedia')
kb.save_to_disk(PATH_KB)
```

We do not want to work with split categories in the Knowledge base. We will then concat them. You can do that :

```py
from datasets import load_from_disk, disable_caching, concatenate_datasets

disable_caching()

kb = load_from_disk(PATH_KB)
humans_with_faces = kb['humans_with_faces']
non_humans = kb['non_humans']
humans_without_faces = kb['humans_without_faces']

# You must remove the columns that are not in common
humans_with_faces = humans_with_faces.remove_columns([...])
non_humans = non_humans.remove_columns([...])
humans_without_faces =humans_without_faces.remove_columns([...])

# If you need you can keep where the data come from
humans_with_faces = humans_with_faces.map(lambda x: {"type":"humans_with_faces"})
non_humans = non_humans.map(lambda x: {"type":"non_humans"})
humans_without_faces = humans_without_faces.map(lambda x: {"type":"humans_without_faces"})

# Concatenate and save
new_kb = concatenate_datasets([humans_with_faces,non_humans, humans_without_faces])
new_kb.save_to_disk(PATH_TO_SAVE_KB)
```

## Aknowledgments

Many thanks to following codes that help us a lot in building this codebase:

- Our model is based on this model [VL-T5](https://github.com/j-min/VL-T5)
- [VL adapter](https://github.com/ylsung/VL_adapter)
- [ViQuAE](https://github.com/PaulLerner/ViQuAE/)
