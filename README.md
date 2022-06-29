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

Install Faiss

```bash

pip install faiss-gpu
# CPU-only version
conda install -c pytorch faiss-cpu

# GPU(+CPU) version
conda install -c pytorch faiss-gpu

# or for a specific CUDA version
conda install -c pytorch faiss-gpu cudatoolkit=10.2 # for CUDA 10.2
```

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

#### Download dataset and knowledge base

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

#### Preprocess data

From [ViQuAE project](https://github.com/PaulLerner/ViQuAE)

Splitting articles in passages

```bash
python -m meerqat.data.loading passages path/to/kb path/to/save/passages experiments/passages/config.json
```

Then you can extract some columns from the dataset to allow quick (and string) indexing:

```bash
python -m meerqat.data.loading map path/to/kb wikipedia_title path/to/save/title2index.json --inverse

python -m meerqat.data.loading map path/to/kb passage_index path/to/save/article2passage.json
```

Find relevant passages in the linked wikipedia article

```bash
python -m meerqat.ir.metrics relevant path/to/viquae_dataset path/to/passages viquae_passages path/to/title2index.json path/to/article2passage.json
```

### Search relevant and irrelevant passages with BM25

Before running any of the commands below you should launch the [Elastic Search server](https://www.elastic.co/guide/en/elasticsearch/reference/7.17/targz.html)

Tune hyperparameters with the command :

```bash
python -m meerqat.ir.hp bm25 path/to/datasets/validation \
    experiments/ir/viquae/hp/bm25/config.json --k=100 \
    --test=path/to/dataset/test \
    --metrics=experiments/ir/viquae/hp/bm25/metrics
```

You will obtains some metrics obtains on the test after optimize bm25 on the split `validation`. Then apply this command to apply BM25 indices on the whole dataset (with the last command, you just generated `BM25_indices` for test, but we will need this indices in the whole dataset for futur training)

Thus, apply them on the whole dataset with :

```bash
python -m meerqat.ir.search \
    /scratch_global/stage_pgrimal/data/CVLP/data/datasets/vlt5_viquae_dataset \
    experiments/ir/viquae/bm25/config.json --k=100 \
    --metrics=experiments/ir/viquae/bm25/metrics
```

You will have now `BM25_indices`. Be careful, do not use the generated metrics. They have been computed on `train`, `validation` and `test`, not just on `test`.

Then generate irrelevant passages.

```bash
python -m processing.irrelevant \
    --indice=BM25 \
    --passages_path= path/to/passages \
    --dataset_path=path/to/dataset
```

### Search relevant and irrelevant passages with DPR

TODO: MAYBE do the same things than before with DPR zero shot

### Train VLT5

#### Visual Encoder Faster CNN

We train our model from the pretrain VLT5 model

##### Prompt tuning

TODO: add step
freeze or not visual encoder

##### Adapter

TODO: add step
freeze or not visual encoder

#### Visual Encoder : CLIP

We must pretrained the model on pretrained task(s). We use adapter for the pretraining.

##### Pretraining


###### Download and preprocess data

We need to download the kilt dataset filtered on viquae question.

```py
from datasets import load_dataset

dataset = load_dataset("PaulLerner/triviaqa_for_viquae")
```

We need to mine passage with bm25. We will use the dataset `kilt_wikipedia`. We need to preprocess the data like we do previously for BM25.
```py

```

And apply

```bash
python -m meerqat.data.loading passages path/to/kilt_wikipedia path/to/save/passages experiments/passages/config.json

# Extract some columns from the dataset to allow quick (and string) indexing:

python -m meerqat.data.loading map path/to/kilt_wikipedia wikipedia_title path/to/save/title2index.json --inverse

python -m meerqat.data.loading map path/to/kilt_wikipedia passage_index path/to/save/article2passage.json
```

Find relevant passages in the linked wikipedia article

```bash
python -m meerqat.ir.metrics relevant path/to/viquae_dataset path/to/passages viquae_passages path/to/title2index.json path/to/article2passage.json
```

After that you an run BM25 to get provenance indices. We do not tune this time and directly use paramter from this [study](https://github.com/castorini/pyserini/blob/master/docs/experiments-dpr.md). b=0.4 k=0.9 



###### Prompt tunning

TODO: add step
freeze or not visual encoder

###### Adapters

TODO: add step
freeze or not visual encoder

## Aknowledgments

Many thanks to following codes that help us a lot in building this codebase:

- Our model is based on this model [VL-T5](https://github.com/j-min/VL-T5)
- [VL adapter](https://github.com/ylsung/VL_adapter)
- [ViQuAE](https://github.com/PaulLerner/ViQuAE/)

## Miscellaneous

```py
column = 'input'
# Bart
# Replace sep
dataset = dataset.map(lambda x: {x[column]:x[column].replace('[SEP]','</s>')})
# add classification token
dataset = dataset.map(lambda x: {x[column]:'<s>'+x[column]})

# T5
# replace sep
dataset = dataset.map(lambda x: {x[column]:x[column].replace('[SEP]','.')})
# add classification token
dataset = dataset.map(lambda x: {x[column]:'<pad>+'x[column]})
```

cls_token bart : `<s>`

sep token bart : `</s>`

cls token T5, T5 use `<pad>` for classifier
sep token T5 : dont have, use `.`, `,`, or `;` instead ?. Explore that in further works.

## Methodology

Quand je referai mes expés sur gros dataset, réfléchir à bien avoir juste les passages contenant les réponses dans passages.
