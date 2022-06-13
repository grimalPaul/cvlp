from cvlep.VLT5.param import Config
from cvlep.VLT5.tokenization import VLT5Tokenizer, VLT5TokenizerFast
from transformers import T5Tokenizer, BartTokenizer, T5TokenizerFast, BartTokenizerFast
from multiprocessing import context
import warnings
from datasets import disable_caching, load_from_disk
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
disable_caching()


"""----------------------------------------"""

# https://arxiv.org/abs/2004.04906
# Loss with relevant passage and irrelevant passages
# can used random irrelevant or irrelant mined by sime method to define (BM25, ...)

"""We use the same hyperparameters as Karpukinh et al.. 
We train DPR using 4 V100 GPUs of 32GB, allowing a total batch size of 256 
(32 questions * 2 passages each * 4 GPUs). 
This is crucial because each question uses all passages paired with 
other questions in the batch as negative examples. Each question is paired 
with 1 relevant passage and 1 irrelevant passage mined with BM25."""


class DPRDataset(Dataset):
    def __init__(
            self,
            passages_path,
            dataset_path,
            kb_path,
            tokenizer_path,
            n_irrelevant_passages=1,
            key_relevant='provenance_indices',
            key_irrelevant='BM25_irrelevant_indices',
            key_text_question='input',
            key_text_passage='passage',
            key_vision_features='fastrcnn_features',
            key_vision_boxes='fastrcnn_boxes',
            split='train',
            verbose=True
    ) -> None:
        super().__init__()
        self.verbose = verbose
        self.split = split
        self.n_relevant_passages = 1
        self.n_irrelevant_passages = n_irrelevant_passages
        if self.verbose:
            print('Data sources: ', self.split)
            print(
                f'Number of passages per question in a batch :\
                {self.n_relevant_passages + self.n_irrelevant_passages} \
                nb relevants : {self.n_relevant_passages} \
                nb irrelevants : {self.n_irrelevant_passages}'
            )
        self.passages = load_from_disk(passages_path)
        self.dataset = load_from_disk(dataset_path)[self.split]
        self.kb = load_from_disk(kb_path)
        self.key_index_relevant_passages = key_relevant
        self.key_index_irrelevant_passages = key_irrelevant
        self.key_text_question = key_text_question
        self.key_text_passage = key_text_passage
        self.key_vision_features = key_vision_features
        self.key_boxes = key_vision_boxes
        # we use the same tokenizer for question and passage
        TokenizerConfig = Config.load_json(tokenizer_path)
        if 't5' in TokenizerConfig.tokenizer:
            if TokenizerConfig.use_vision:
                # tokenizer_class = VLT5Tokenizer
                tokenizer_class = VLT5TokenizerFast
            else:
                # tokenizer_class = T5Tokenizer
                tokenizer_class = T5TokenizerFast
        elif 'bart' in TokenizerConfig.tokenizer:
            tokenizer_class = BartTokenizer
            # tokenizer_class = BartTokenizerFast
        else:
            raise ValueError('This type of tokenizer is not implemented')

        self.tokenizer = tokenizer_class.from_pretrained(
            TokenizerConfig.tokenizer,
            max_length=TokenizerConfig.max_text_length,
            do_lower_case=TokenizerConfig.do_lower_case,
        )
        self.tokenization_kwargs = TokenizerConfig.tokenization_kwargs

    def __len__(self):
        return self.dataset.num_rows

    def __getitem__(self, index):
        item = {}
        # question features
        item['question_text'] = self.dataset[index][self.key_text_question]
        question_image_features = torch.Tensor(
            self.dataset[index][self.key_vision_features])
        question_image_boxes = torch.Tensor(
            self.dataset[index][self.key_boxes])
        item['question_image_features'] = torch.squeeze(
            question_image_features, dim=1)
        item['question_image_boxes'] = torch.squeeze(
            question_image_boxes, dim=1)

        # relevant and irrelevant passage features
        relevants_index = self.dataset[index][self.key_index_relevant_passages]
        irrelevant_index = self.dataset[index][self.key_index_irrelevant_passages]
        if len(relevants_index) < 1:
            warnings.warn(
                f"Didn't find any relevant passage for question {self.dataset[index]['id']}")
            item[f'passage_relevant_text'] = None
            item[f'passage_relevant_image_boxes'] = None
            item[f'passage_relevant_image_features'] = None
        else:
            relevant_index = np.random.choice(relevants_index)
            kb_index = self.passages[relevant_index]['index']
            item[f'passage_relevant_text'] = self.passages[relevant_index][self.key_text_passage]
            passage_image_features = torch.Tensor(
                self.kb[kb_index][self.key_vision_features])
            passage_image_boxes = torch.Tensor(
                self.kb[kb_index][self.key_boxes])
            item[f'passage_relevant_image_boxes'] = torch.squeeze(
                passage_image_boxes, dim=1)
            item[f'passage_relevant_image_features'] = torch.squeeze(
                passage_image_features, dim=1)
        if len(irrelevant_index) < 1:
            warnings.warn(
                f"Didn't find any irrelevant passage for question {self.dataset[index]['id']}")
            item[f'passage_irrelevant_text'] = None
            item[f'passage_irrelevant_image_boxes'] = None
            item[f'passage_irrelevant_image_features'] = None
        else:
            irrelevant_index = np.random.choice(irrelevant_index)
            kb_index = self.passages[irrelevant_index]['index']
            item[f'passage_irrelevant_text'] = self.passages[irrelevant_index][self.key_text_passage]
            passage_image_features = torch.Tensor(
                self.kb[kb_index][self.key_vision_features])
            passage_image_boxes = torch.Tensor(
                self.kb[kb_index][self.key_boxes])
            item[f'passage_irrelevant_image_boxes'] = torch.squeeze(
                passage_image_boxes, dim=1)
            item[f'passage_irrelevant_image_features'] = torch.squeeze(
                passage_image_features, dim=1)

        return item

    def collate_fn(self, batch):
        labels = []
        for target, relevant_text in enumerate(batch['passage_relevant_text']):
            # for now, we consider that we always have relevant and ireeelevant passage
            # but we should replace None with something if it happen
            if relevant_text is None:
                labels.append(-100)  # ignore index when computing the loss
            else:
                labels.append(target)
            """ 
            if item['passage_irrelevant_text'] is None:
                # pas de irrelevant
                # mettre du vent à la place et le bon nombre
                self.n_irrelevant_passages
            """
        question_inputs = self.tokenizer(
            batch['question_text'], **self.tokenization_kwargs)
        context_inputs = self.tokenizer(
            batch['passage_relevant_text'] + batch['passage_irrelevant_text'], **self.tokenization_kwargs)
        labels = torch.tensor(labels)
        visual_feats_context = torch.concat(
            [batch['passage_relevant_image_features'], batch['passage_irrelevant_image_features']])
        context_image_boxes = torch.concat(
            [batch['passage_relevant_image_boxes'], batch['passage_irrelevant_image_boxes']])
        return {
            "input_ids_question": question_inputs,
            "input_ids_context": context_inputs,
            "labels": labels,
            "visual_feats_question": batch['question_image_features'],
            "visual_feats_context": visual_feats_context,
            "question_image_boxes": batch['question_image_boxes'],
            "context_image_boxes": context_image_boxes
        }


#                        |  passage   |              |  question  |
# je dois gérer à chaque |text + image| associé à du |text + image|
# example positif prendre question réponse, et négatif on utilise juste le batch
class CLIPlikeDataset(Dataset):
    def __init__(
            self,
            passages_path,
            dataset_path,
            kb_path,
            tokenizer_path,
            key_relevant='provenance_indices',
            key_text_question='input',
            key_text_passage='passage',
            key_vision_features='fastrcnn_features',
            key_vision_boxes='fastrcnn_boxes',
            split='train',
            verbose=True
    ) -> None:
        super().__init__()
        self.verbose = verbose
        self.split = split
        if self.verbose:
            print('Data sources: ', self.split)
        self.passages = load_from_disk(passages_path)
        self.dataset = load_from_disk(dataset_path)[self.split]
        self.kb = load_from_disk(kb_path)
        self.key_index_relevant_passages = key_relevant
        self.key_text_question = key_text_question
        self.key_text_passage = key_text_passage
        self.key_vision_features = key_vision_features
        self.key_boxes = key_vision_boxes

        # we use the same tokenizer for question and passage
        TokenizerConfig = Config.load_json(tokenizer_path)
        if 't5' in TokenizerConfig.tokenizer:
            if TokenizerConfig.use_vision:
                # tokenizer_class = VLT5Tokenizer
                tokenizer_class = VLT5TokenizerFast
            else:
                # tokenizer_class = T5Tokenizer
                tokenizer_class = T5TokenizerFast
        elif 'bart' in TokenizerConfig.tokenizer:
            tokenizer_class = BartTokenizer
            # tokenizer_class = BartTokenizerFast
        else:
            raise ValueError('This type of tokenizer is not implemented')

        self.tokenizer = tokenizer_class.from_pretrained(
            TokenizerConfig.tokenizer,
            max_length=TokenizerConfig.max_text_length,
            do_lower_case=TokenizerConfig.do_lower_case,
        )
        self.tokenization_kwargs = TokenizerConfig.tokenization_kwargs

    def __len__(self):
        return self.dataset.num_rows

    def __getitem__(self, index):
        item = {}
        # question features
        item['question_text'] = self.dataset[index][self.key_text_question]
        question_image_features = torch.Tensor(
            self.dataset[index][self.key_vision_features])
        question_image_boxes = torch.Tensor(
            self.dataset[index][self.key_boxes])
        item['question_image_features'] = torch.squeeze(
            question_image_features, dim=1)
        item['question_image_boxes'] = torch.squeeze(
            question_image_boxes, dim=1)

        # passage features
        relevants_index = self.dataset[index][self.key_index_relevant_passages]
        if len(relevants_index) < 1:
            # gerer dans collate fn
            # si pas de relevant passage on peut prendre un random
            # et le mettre à 0 ?
            pass
        # select just one relevant passage
        relevant_index = np.random.choice(relevants_index)
        kb_index = self.passages[relevant_index]['index']
        item['passage_text'] = self.passages[relevant_index][self.key_text_passage]
        passage_image_features = torch.Tensor(
            self.kb[kb_index][self.key_vision_features])
        passage_image_boxes = torch.Tensor(self.kb[kb_index][self.key_boxes])
        item['passage_image_boxes'] = torch.squeeze(passage_image_boxes, dim=1)
        item['passage_image_features'] = torch.squeeze(
            passage_image_features, dim=1)
        return item

    def collate_fn(self, batch):
        question_inputs = self.tokenizer(
            batch['question_text'], **self.tokenization_kwargs)
        context_inputs = self.tokenizer(
            batch['passage_relevant_text'], **self.tokenization_kwargs)
        return {
            "input_ids_question": question_inputs,
            "input_ids_context": context_inputs,
            "visual_feats_question": batch['question_image_features'],
            "visual_feats_context": batch['passage_relevant_image_features'],
            "question_image_boxes": batch['question_image_boxes'],
            "context_image_boxes": batch['passage_relevant_image_boxes']
        }


def get_dataloader():
    kwargs_clip = {
        "tokenizer_path": "experiments/configEncoder/bergamote/TokenizerConfig.json",
        "dataset_path": "/scratch_global/stage_pgrimal/data/CVLP/data/datasets/vlt5_dataset",
        "kb_path": "/scratch_global/stage_pgrimal/data/CVLP/data/datasets/kb",
        "passages_path": "/scratch_global/stage_pgrimal/data/CVLP/data/datasets/vlt5_passages",
        "key_relevant": 'provenance_indices',
        "key_text_question": 'input',
        "key_text_passage": 'passage',
        "key_vision_features": 'vlt5_features',
        "key_vision_boxes": 'vlt5_normalized_boxes',
        "split": 'train'
    }
    kwargs_dpr = {
        "tokenizer_path": "experiments/configEncoder/bergamote/TokenizerConfig.json",
        "dataset_path": "/scratch_global/stage_pgrimal/data/CVLP/data/datasets/vlt5_dataset",
        "kb_path": "/scratch_global/stage_pgrimal/data/CVLP/data/datasets/kb",
        "passages_path": "/scratch_global/stage_pgrimal/data/CVLP/data/datasets/vlt5_passages",
        "key_relevant": 'provenance_indices',
        "key_text_question": 'input',
        "key_text_passage": 'passage',
        "key_vision_features": 'vlt5_features',
        "key_vision_boxes": 'vlt5_normalized_boxes',
        "split": 'train',
        "key_irrelevant" : 'BM25_irrelevant_indices'
    }

    dataset_clip = CLIPlikeDataset(**kwargs_clip)
    dataloader_clip = DataLoader(dataset_clip, batch_size=2, collate_fn=dataset_clip.collate_fn)
    dataset_dpr = DPRDataset(**kwargs_dpr)
    dataloader_dpr = DataLoader(dataset_dpr, batch_size=2, collate_fn=dataset_dpr.collate_fn)
    return dataloader_clip,dataloader_dpr


if __name__ == '__main__':
    pass

    """
    data = next(iter)
    passage_embeddings = encoder_passage(batch_passage)
    question_embeddings = encoder_question(batch_question)

    # TODO : peut être normaliser ?
    logits = (passage_embeddings @ question_embeddings.T) / self.temperature
    questions_similarity = question_embeddings @ question_embeddings.T
    passages_similarity = passage_embeddings @ passage_embeddings.T
    targets = F.softmax(
        (questions_similarity + passages_similarity) / 2 * self.temperature, dim=-1
    )
    questions_loss = cross_entropy(logits, targets, reduction='none')
    passages_loss = cross_entropy(logits.T, targets.T, reduction='none')
    loss = (questions_loss + passages_loss) / 2.0  # shape: (batch_size)
    return loss.mean()


    def cross_entropy(preds, targets, reduction='none'):
        log_softmax = nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(preds)).sum(1)
        if reduction == "none":
            return loss
        elif reduction == "mean":
            return loss.mean()

    # DataLoader


    # loss

    # deux facon de faire le training avec hard examples et negative ou à la facon de clip
    # mettre en place les deux
    """
