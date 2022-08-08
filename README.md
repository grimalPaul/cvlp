# CVLP

CVLP = Constrastive Visual Language Pre-Training

## Table of contents

- [CVLP](#cvlp)
  - [Table of contents](#table-of-contents)
  - [Installation](#installation)
    - [Download tokenizer and model](#download-tokenizer-and-model)
    - [Download pretrained encoder](#download-pretrained-encoder)
    - [Download and preprocess data](#download-and-preprocess-data)
      - [Download dataset and knowledge base](#download-dataset-and-knowledge-base)
      - [Organization](#organization)
      - [Preprocess data](#preprocess-data)
    - [Search relevant and irrelevant passages with BM25](#search-relevant-and-irrelevant-passages-with-bm25)
  - [Train VLT5](#train-vlt5)
    - [Finetuning on the pretrained VLT5](#finetuning-on-the-pretrained-vlt5)
      - [Prompt tuning](#prompt-tuning)
      - [Adapter](#adapter)
    - [Pretraining](#pretraining)
    - [Multitask Pretraining](#multitask-pretraining)
      - [Kilt trivia](#kilt-trivia)
      - [Wikimage](#wikimage)
      - [Multimedia](#multimedia)
      - [Sentence T5](#sentence-t5)
      - [Pretraining with different configuration](#pretraining-with-different-configuration)
    - [Finetuning with different configuration](#finetuning-with-different-configuration)
  - [Embedding and research](#embedding-and-research)
  - [Analyse](#analyse)
  - [Note config files for research, multitask and finetuning](#note-config-files-for-research-multitask-and-finetuning)
    - [Research](#research)
    - [Multitask](#multitask)
    - [Fine-tuning](#fine-tuning)
  - [Aknowledgments](#aknowledgments)
  - [Miscellaneous](#miscellaneous)
  - [Methodology](#methodology)

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

use `python download_models` to download and save models T5, Bart and SentenceT5 in `data/`

To download fastercnn, click to the links below :

- [Configuration](https://s3.amazonaws.com/models.huggingface.co/bert/unc-nlp/rcnn-vg-finetuned/config.yaml)
- [Pytorch_model.bin](https://cdn.huggingface.co/unc-nlp/frcnn-vg-finetuned/pytorch_model.bin)

Place the downloaded files in `data/frcnn_model/`.

To download the visual encoder of CLIP, click [here](https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt) and place the downloaded file in `data/clip/`

### Download pretrained encoder

To download pretrained VLT5, click [here](https://drive.google.com/drive/folders/1_SBj4sZ0gUqfBon1gFBiNRAmfHv5w_ph?usp=sharing) and place the downloaded model in `data/VLT5`.

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

#### Organization

You should have this organization :

```tree
.
├── clip_features.ipynb
├── cvlep
│   ├── CLIPT5
│   └── VLT5
├── data
│   ├── clip
│   │   └── RN101.pt
│   ├── frcnn_model
│   │   ├── config.yaml
│   │   └── pytorch_model.bin
│   ├── sentenceT5
│   │   └── pytorch_model.bin
│   ├── t5_pretrained
│   ├── tokenizer
│   │   └── t5-base
│   └── VLT5
│       └── VLT5Epoch30.pth
├── DATASETS.md
├── download_models.py
├── experiments
│   ├── ir
│   │   ├── triviaqa_for_viquae
│   │   ├── viquae
│   │   └── VL
│   └── passages
├── snap
├── tensorboard
├── README.md
└── requirements.txt
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
python -m meerqat.ir.metrics relevant path/to/viquae_dataset path/to/passages path/to/title2index.json path/to/article2passage.json
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
    path/to/viquae_dataset \
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

## Train VLT5

### Finetuning on the pretrained VLT5

We train our model from the pretrain VLT5 model

#### Prompt tuning

```bash
encoder_question_path=experiments/config_vladapter/bergamote/prompt/encoder_prompting.json
encoder_passage_path=experiments/config_vladapter/bergamote/prompt/encoder_prompting.json
model_path=experiments/config_vladapter/bergamote/prompt/config_model.json
training_path=experiments/config_vladapter/bergamote/prompt/training_prompt.json

echo "Prompt tuning"
srun --kill-on-bad-exit=1 python -m cvlep.trainer_base_vladapter \
    --encoder_question_path=${encoder_question_path} \
    --encoder_passage_path=${encoder_passage_path} \
    --model_path=${model_path} \
    --training_path=${training_path}
```

#### Adapter

```bash
encoder_question_path=experiments/config_vladapter/bergamote/adapter_projection/encoder_simple_adapter.json
encoder_passage_path=experiments/config_vladapter/bergamote/adapter_projection/encoder_simple_adapter.json
model_path=experiments/config_vladapter/bergamote/adapter_projection/config_model.json
training_path=experiments/config_vladapter/bergamote/adapter_projection/training_simple_adapter.json

echo "Projection et adapter"
srun --kill-on-bad-exit=1 python -m cvlep.trainer_base_vladapter \
    --encoder_question_path=${encoder_question_path} \
    --encoder_passage_path=${encoder_passage_path} \
    --model_path=${model_path} \
    --training_path=${training_path}

echo "The End"
```

### Pretraining

We must pretrained the model on pretrained task(s). We use adapter for the pretraining.

### Multitask Pretraining

#### Kilt trivia

We need to download the kilt dataset filtered on viquae question.

```py
from datasets import load_dataset

dataset = load_dataset("PaulLerner/triviaqa_for_viquae")
dataset.save_to_disk("path/triviaqa_for_viquae")
```

We need to mine passage with bm25. We will use the dataset `kilt_wikipedia`. We need to preprocess the data like we did previously.

```bash
python -m meerqat.data.loading passages path/to/kilt_wikipedia path/to/save/passages experiments/passages/config.json

# Extract some columns from the dataset to allow quick (and string) indexing:

python -m meerqat.data.loading map path/to/kilt_wikipedia wikipedia_title path/to/save/title2index.json --inverse

python -m meerqat.data.loading map path/to/kilt_wikipedia passage_index path/to/save/article2passage.json
```

Find relevant passages in the linked wikipedia article

```bash
python -m meerqat.ir.metrics relevant path/to/triviaqa_for_viquae path/to/passages path/to/title2index.json path/to/article2passage.json
```

After that you an run BM25 to get provenance indices. We do not tune this time and directly use parameter from DPR[LINK] (b =0.4, k1=0.9).

```bash
python -m meerqat.ir.search \
    path/to/triviaqa_for_viquae \
    experiments/ir/triviaqa_for_viquae/config.json \
    --k=100 \
    --metrics=experiments/ir/triviaqa_for_viquae/metrics
```

Then, generate irrelevant passage indices

```bash
python -m processing.irrelevant \
    --indice=BM25 \
    --passages_path= path/to/passages \
    --dataset_path=path/to/triviaqa_for_viquae
```

#### Wikimage

To develop entities representation of our model, we create a dataset from the knowledge base of viquae. We take `wikidata_id` and get from wikidata image's entity for entities who have more than one image.

TODO :We release the dataset and the image.

We removed entities present in Viquae (train, validation and test)

#### Multimedia

We developed a multimedia training which consist to match differents passages of a same article with different illustrative images. We take the above dataset to create this one.

TODO: We release the dataset, images and the passage

We removed entities present in Viquae (train, validation and test)

#### Sentence T5

As we are going to use the T5 sentence to initialize the model, we have to download it and calculate a score with :

We create a new Conda environment just to embed the dataset due to a dependency conflict

```bash
conda create -n sentenceT5 python

pip install sentence_transformers datasets
```

```py
from sentence_transformers import SentenceTransformer
import torch

model = SentenceTransformer("sentence-transformers/sentence-t5-base")
model.save("path/to/sentence_T5") # save the model for the embedding
torch.save(model.state_dict(), "path/to/save/state_dict") # save the weight of the model to use it when initializing the model
```

Embedding and research
```bash
source activate sentenceT5

echo "Passage"
python -m processing.embedding_sentence_T5_only \
    --batch_size=${batch_size} \
    --model_path=${model_path} \
    --key_text=passage \
    --key_embedding=${key_embedding} \
    --dataset_path=${passages}

echo "dataset"
python -m processing.embedding_sentence_T5_only \
    --batch_size=${batch_size} \
    --model_path=${model_path} \
    --key_text=input \
    --key_embedding=${key_embedding} \
    --dataset_path=${dataset}
    
source activate cvlp

echo "research"

python -m search \
    --dataset_path=/scratch_global/stage_pgrimal/data/CVLP/data/datasets/miniviquae/test \
    --config=experiments/ir/sentence-T5/sentence_T5.json \
    --metrics_path=experiments/ir/sentence-T5 \
    --k=100 \
    --batch_size=128

```

#### Pretraining with different configuration

FasterRCNN

```py
pass
```

CLIP

```py
pass
```

### Finetuning with different configuration

FasterRCNN

```py
pass
```

CLIP

```py
pass
```

## Embedding and research

Embed dataset and KB

```bash
echo "embedding passage"
python -m processing.embedding_dataset \
    --dataset_path=${passages} \
    --type=passage \
    --config_question_path=${config_question_path} \
    --config_passage_path=${config_passage_path} \
    --config_model_path=${config_model_path} \
    --key_boxes=vision_boxes \
    --key_vision_features=vision_features \
    --key_text=passage \
    --key_embedding=${key_in_passage} \
    --kb_path=${kb} \
    --batch_size=${batch_size}

echo "embedding dataset"

python -m processing.embedding_dataset \
    --dataset_path=${dataset} \
    --type=question \
    --config_question_path=${config_question_path} \
    --config_passage_path=${config_passage_path} \
    --config_model_path=${config_model_path} \
    --key_boxes=vision_boxes \
    --key_vision_features=vision_features \
    --key_text=input \
    --key_embedding=${key_in_dataset} \
    --batch_size=${batch_size}
```

And then compute the different metrics :

```bash
python -m search \
    --dataset_path=viquae/test \
    --config=experiments/ir/VL/experiments/clip_multitask/multitask.json \
    --metrics_path=experiments/ir/VL/clip_multitask/ \
    --k=100 \
    --batch_size=64
```

## Analyse

You can use the following command to generate 2 json to see manually where the model is wrong or correct :

```bash
python -m processing.compare_relevant \
    --indice=${indice} \
    --dataset_path=${dataset}


echo "create json"
python -m processing.analyse_result \
    --key=${indice} \
    --dataset_path=${dataset} \
    --kb_path=${kb} \
    --passages_path=${passages} \
    --k=${k} \ # numeber
    --save_path=${save_path} # where you save the json file

```

## Note config files for research, multitask and finetuning

### Research

```json
{
    "kb_kwargs": {
        "path/to/kb  where you search": {
            "device": "indicate cpu or gpu",
            "index_kwargs": {
                "name_of_your_index_1": {
                    "key_kb": "column to index to do the search",
                    "index_load": "true or false if you want to load index from index path",
                    "index_path": "path where you want to load or save your index",
                    "string_factory": "https://github.com/facebookresearch/faiss/wiki/The-index-factory"
                },
                "name_of_your_index_2": {
                    "key_kb": "column to index to do the search",
                    "index_load": "true or false if you want to load index from index path",
                    "index_path": "path where you want to load or save your index",
                    "string_factory": "https://github.com/facebookresearch/faiss/wiki/The-index-factory"
                }
            }
        }
    }
}
```

### Multitask

Config for training_path argument.

```bash
python -m cvlep.trainer_multitask \
    --encoder_question_path=... \
    --encoder_passage_path=... \
    --model_path=... \
    --training_path=#THIS FILE
```

Please precise in `dataset` each dataset you want to use.
In `split` add `train` and/or `validation`.
File example :

```json
{
    "adam_eps": 1e-06,
    "clip_grad_norm": 5,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "epochs": 40,
    "fp16": true,
    "gradient_accumulation_steps": 1,
    "lr": 0.0003,
    "num_workers": 2,
    "val_workers":0,
    "optim": "adamw",
    "output": "path to save encoders",
    "seed": 0,
    "train": true,
    "log_tensorboard_path": "path to log",
    # config of each dataset
    "datasets": {
        "triviaqa": {
            "batch_size": 20,
            "tokenizer_path": "TokenizerConfig.json",
            "passages_path": "/passages",
            "dataset_path": "triviaqa_for_viquae",
            "key_relevant": "provenance_indices",
            "key_irrelevant": "BM25_irrelevant_indices",
            "key_text_question": "input",
            "key_text_passage": "passage",
            "topk": -1,
            "split":"train"
        },
        "match_image": {
            "batch_size": 20,
            "dataset_path": "/wikimage",
            "topk":-1,
            "key_image":"list_images",
            "key_vision_features":"clip_features",
            "key_vision_boxes":null,
            "split":"train"
        },
        "match_article": {
            "batch_size": 20,
            "kb_path": "multimedia",
            "passages_path": "passages",
            "tokenizer_path": "TokenizerConfig.json",
            "key_passage_index": "passage_index",
            "key_text_passage": "passage",
            "key_list_images": "list_images",
            "key_vision_features": "clip_features",
            "key_vision_boxes": null,
            "topk": -1,
            "split":"train"
        },
        "viquae":{
            "batch_size":16,
            "tokenizer_path": "TokenizerConfig.json",
            "dataset_path": "miniviquae",
            "kb_path": "kb",
            "passages_path": "passages",
            "key_relevant": "provenance_indices",
            "key_text_question": "input",
            "key_text_passage": "passage",
            "key_vision_features": "clip_features",
            "key_vision_boxes": null,
            "key_irrelevant": "BM25_irrelevant_indices",
            "split":"validation"
        }
    }
}
```

Text embedding is always freeze currently

### Fine-tuning

```bash
python -m cvlep.trainer_base_vladapter \
    --encoder_question_path=... \
    --encoder_passage_path=... \
    --model_path=... \
    --training_path=#THIS FILE
```

Here we pass only one dataset :

```json
{
    "adam_eps": 1e-06,
    "batch_size": 4,
    "clip_grad_norm": 5,
    "warmup_ratio": 0.2,
    "weight_decay": 0.01,
    "epochs": 200,
    "fp16": true,
    "gradient_accumulation_steps": 1,
    "lr": 0.0001,
    "num_workers": 3,
    "optim": "adamw",
    "output": "path to save encoders",
    "seed": 0,
    "valid_batch_size": 4,
    "train":true,
    "log_tensorboard_path":"log of the training",
    "tokenizer_path": "TokenizerConfig.json",
    "dataset_path": "miniviquae",
    "kb_path": "kb",
    "passages_path": "passages",
    "key_relevant": "provenance_indices",
    "key_text_question": "input",
    "key_text_passage": "passage",
    "key_vision_features": "fastrcnn_features",
    "key_vision_boxes": "fastrcnn_boxes",
    "key_irrelevant": "BM25_irrelevant_indices"
}
```

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
