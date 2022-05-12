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

## Aknowledgments

Many thanks to following codes that help us a lot in building this codebase:

- Our model is based on this model [VL-T5](https://github.com/j-min/VL-T5)
- [VL adapter](https://github.com/ylsung/VL_adapter)
- [ViQuAE](https://github.com/PaulLerner/ViQuAE/)
