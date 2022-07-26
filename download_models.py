from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import BartForConditionalGeneration, BartTokenizer

if __name__ == '__main__':
    
    #download and save T5
    model = T5ForConditionalGeneration.from_pretrained('t5-base')
    model.save_pretrained('data/t5_pretrained')
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    tokenizer.save_pretrained('data/tokenizer/t5-base')

    #download and save Bart
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
    model.save_pretrained('data/bart_pretrained')
    tokenizer.save_pretrained('data/bart_pretrained')

    # download and save faster CNN
    # click on this link to download
    # https://cdn.huggingface.co/unc-nlp/frcnn-vg-finetuned/pytorch_model.bin
    # https://s3.amazonaws.com/models.huggingface.co/bert/unc-nlp/frcnn-vg-finetuned/config.yaml
    
    #download and save CLIP
    #TODO:dl clip