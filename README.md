# CVLEP

CVLEP = Constrastive Visual Language Embedding Pre-Training

## Installation

I suggest to create a virtual environnement and pip install the `requirements.txt`.
install elasticsearch. I worked with this version : [elasticsearch-7.17](https://www.elastic.co/guide/en/elasticsearch/reference/7.17/targz.html)

### Download tokenizer and model

```py
from transformers import BartTokenizer, T5Tokenizer

bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
bart_tokenizer.save_pretrained('data/tokenizer/bart-base')

t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')
t5_tokenizer.save_pretrained('data/tokenizer/t5-base')
```

### Download visual model

`TODO`

### Download pretrained encoder

`TODO`

## Aknowledgments

Many thanks to following codes that help us a lot in building this codebase:

- Our model is based on this model [VL-T5](https://github.com/j-min/VL-T5)
- [VL adapter ](https://github.com/ylsung/VL_adapter)
- [ViQuAE](https://github.com/PaulLerner/ViQuAE/)
