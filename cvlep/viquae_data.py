from cvlep.VLT5.param import Config
from cvlep.VLT5.tokenization import VLT5Tokenizer, VLT5TokenizerFast
from transformers import T5Tokenizer, BartTokenizer, T5TokenizerFast, BartTokenizerFast
from multiprocessing import context
import warnings
from datasets import disable_caching, load_from_disk
import random
import torch
from torch.utils.data import Dataset, DataLoader
disable_caching()


def remove_lines(n_rows, split):
            index = [_ for _  in range(n_rows)]
            if split == 'train':
                idx2remove = [201, 388, 428, 677, 1077]            
            elif split == 'validation':
                idx2remove = [43, 121, 473, 582, 688, 1075, 1099]
            elif split == 'test':
                idx2remove = [578, 643, 904, 1056]
            # reverse the list to pop without shift index
            idx2remove = idx2remove[::-1]
            for i in idx2remove:
                index.pop(i)
            return index

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
        index = remove_lines(self.dataset.num_rows, self.split)
        self.dataset = self.dataset.select(index)
        self.kb = load_from_disk(kb_path)
        self.key_index_relevant_passages = key_relevant
        self.key_index_irrelevant_passages = key_irrelevant
        self.key_text_question = key_text_question
        self.key_text_passage = key_text_passage
        self.key_vision_features = key_vision_features
        self.key_boxes = key_vision_boxes
        # we use the same tokenizer for question and passage
        self.TokenizerConfig = Config.load_json(tokenizer_path)
        if 't5' in self.TokenizerConfig.tokenizer:
            if self.TokenizerConfig.use_vision:
                # tokenizer_class = VLT5Tokenizer
                tokenizer_class = VLT5TokenizerFast
            else:
                # tokenizer_class = T5Tokenizer
                tokenizer_class = T5TokenizerFast
        elif 'bart' in self.TokenizerConfig.tokenizer:
            tokenizer_class = BartTokenizer
            # tokenizer_class = BartTokenizerFast
        else:
            raise ValueError('This type of tokenizer is not implemented')

        self.tokenizer = tokenizer_class.from_pretrained(
            self.TokenizerConfig.tokenizer,
            do_lower_case=self.TokenizerConfig.do_lower_case,
        )

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
            question_image_features, dim=0)
        item['question_image_boxes'] = torch.squeeze(
            question_image_boxes, dim=0)
        item['n_boxes_question'] = item['question_image_boxes'].size()[0]

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
            relevant_index = random.choice(relevants_index)
            kb_index = self.passages[relevant_index]['index']
            item[f'passage_relevant_text'] = self.passages[relevant_index][self.key_text_passage]
            passage_image_features = torch.Tensor(
                self.kb[kb_index][self.key_vision_features])
            passage_image_boxes = torch.Tensor(
                self.kb[kb_index][self.key_boxes])
            item[f'passage_relevant_image_boxes'] = torch.squeeze(
                passage_image_boxes, dim=0)
            item[f'passage_relevant_image_features'] = torch.squeeze(
                passage_image_features, dim=0)
            item['n_boxes_passage_relevant'] = item['passage_relevant_image_boxes'].size()[
                0]

        if len(irrelevant_index) < 1:
            warnings.warn(
                f"Didn't find any irrelevant passage for question {self.dataset[index]['id']}")
            item[f'passage_irrelevant_text'] = None
            item[f'passage_irrelevant_image_boxes'] = None
            item[f'passage_irrelevant_image_features'] = None
        else:
            irrelevant_index = random.choice(irrelevant_index)
            kb_index = self.passages[irrelevant_index]['index']
            item[f'passage_irrelevant_text'] = self.passages[irrelevant_index][self.key_text_passage]
            passage_image_features = torch.Tensor(
                self.kb[kb_index][self.key_vision_features])
            passage_image_boxes = torch.Tensor(
                self.kb[kb_index][self.key_boxes])
            item[f'passage_irrelevant_image_boxes'] = torch.squeeze(
                passage_image_boxes, dim=0)
            item[f'passage_irrelevant_image_features'] = torch.squeeze(
                passage_image_features, dim=0)
            item['n_boxes_passage_irrelevant'] = item['passage_irrelevant_image_boxes'].size()[
                0]

        return item

        # add n_boxes

    def collate_fn(self, batch):
        B = len(batch)
        if self.TokenizerConfig.use_vision:

            V_L_question = max(item['n_boxes_question'] for item in batch)
            V_L_context = max(max(item['n_boxes_passage_relevant'],
                              item['n_boxes_passage_irrelevant']) for item in batch)
            feat_dim = batch[0]['question_image_features'].shape[-1]
            # boxes are represented by 4 points
            question_boxes = torch.zeros(B, V_L_question, 4, dtype=torch.float)
            question_vis_feats = torch.zeros(
                B, V_L_question, feat_dim, dtype=torch.float)
            relevant_boxes = torch.zeros(B, V_L_context, 4, dtype=torch.float)
            relevant_vis_feats = torch.zeros(
                B, V_L_context, feat_dim, dtype=torch.float)
            irrelevant_boxes = torch.zeros(
                B, V_L_context, 4, dtype=torch.float)
            irrelevant_vis_feats = torch.zeros(
                B, V_L_context, feat_dim, dtype=torch.float)

        relevant_text, irrelevant_text, question_text, labels = list(), list(), list(), list()
        for i, item in enumerate(batch):
            question_text.append(item['question_text'])
            relevant_text.append(item['passage_relevant_text'])
            irrelevant_text.append(item['passage_irrelevant_text'])
            if self.TokenizerConfig.use_vision:
                n_boxes_relevant = item['n_boxes_passage_relevant']
                n_boxes_irrelevant = item['n_boxes_passage_irrelevant']
                n_boxes_question = item['n_boxes_question']
                question_boxes[i,
                               :n_boxes_question] = item['question_image_boxes']
                question_vis_feats[i,
                                   :n_boxes_question] = item['question_image_features']
                relevant_boxes[i,
                               :n_boxes_relevant] = item['passage_relevant_image_boxes']
                relevant_vis_feats[i,
                                   :n_boxes_relevant] = item['passage_relevant_image_features']
                irrelevant_boxes[i,
                                 :n_boxes_irrelevant] = item['passage_irrelevant_image_boxes']
                irrelevant_vis_feats[i,
                                     :n_boxes_irrelevant] = item['passage_irrelevant_image_features']
            if item['passage_relevant_text'] is None:
                labels.append(-100)  # ignore index when computing the loss
            else:
                labels.append(i)
                # for now, we consider that we always have relevant and ireeelevant passage
                # but we should replace None with something if it happen

                """ 
                if item['passage_irrelevant_text'] is None:
                    # pas de irrelevant
                    # mettre du vent à la place et le bon nombre
                    self.n_irrelevant_passages
                """
        question_inputs = self.tokenizer(
            question_text, padding='max_length', truncation=True, return_tensors="pt")
        context_inputs = self.tokenizer(
            relevant_text + irrelevant_text, padding='max_length', truncation=True, return_tensors="pt")
        labels = torch.tensor(labels)
        visual_feats_context = torch.concat(
            [relevant_vis_feats, irrelevant_vis_feats])
        context_image_boxes = torch.concat(
            [relevant_boxes, irrelevant_boxes])
        return {
            "input_ids_question": question_inputs,
            "input_ids_context": context_inputs,
            "labels": labels,
            "visual_feats_question": question_vis_feats,
            "visual_feats_context": visual_feats_context,
            "question_image_boxes": question_boxes,
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
        index = remove_lines(self.dataset.num_rows, self.split)
        self.dataset = self.dataset.select(index)
        self.kb = load_from_disk(kb_path)
        self.key_index_relevant_passages = key_relevant
        self.key_text_question = key_text_question
        self.key_text_passage = key_text_passage
        self.key_vision_features = key_vision_features
        self.key_boxes = key_vision_boxes

        # we use the same tokenizer for question and passage
        self.TokenizerConfig = Config.load_json(tokenizer_path)
        if 't5' in self.TokenizerConfig.tokenizer:
            if self.TokenizerConfig.use_vision:
                # tokenizer_class = VLT5Tokenizer
                tokenizer_class = VLT5TokenizerFast
            else:
                # tokenizer_class = T5Tokenizer
                tokenizer_class = T5TokenizerFast
        elif 'bart' in self.TokenizerConfig.tokenizer:
            tokenizer_class = BartTokenizer
            # tokenizer_class = BartTokenizerFast
        else:
            raise ValueError('This type of tokenizer is not implemented')

        self.tokenizer = tokenizer_class.from_pretrained(
            self.TokenizerConfig.tokenizer,
            do_lower_case=self.TokenizerConfig.do_lower_case,
        )

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
            question_image_features, dim=0)
        item['question_image_boxes'] = torch.squeeze(
            question_image_boxes, dim=0)
        item['n_boxes_question'] = item['question_image_boxes'].size()[0]

        # passage features
        relevants_index = self.dataset[index][self.key_index_relevant_passages]
        if len(relevants_index) < 1:
            warnings.warn(
                f"Didn't find any relevant passage for question {self.dataset[index]['id']}")
        # select just one relevant passage
        relevant_index = random.choice(relevants_index)
        kb_index = self.passages[relevant_index]['index']
        item['passage_text'] = self.passages[relevant_index][self.key_text_passage]
        passage_image_features = torch.Tensor(
            self.kb[kb_index][self.key_vision_features])
        passage_image_boxes = torch.Tensor(self.kb[kb_index][self.key_boxes])
        item['passage_image_boxes'] = torch.squeeze(passage_image_boxes, dim=0)
        item['passage_image_features'] = torch.squeeze(
            passage_image_features, dim=0)
        item['n_boxes_passage'] = item['passage_image_boxes'].size()[0]

        return item

    def collate_fn(self, batch):
        B = len(batch)
        if self.TokenizerConfig.use_vision:
            V_L_question = max(item['n_boxes_question'] for item in batch)
            V_L_context = max(item['n_boxes_passage'] for item in batch)
            feat_dim = batch[0]['question_image_features'].shape[-1]
            # boxes are represented by 4 points
            question_boxes = torch.zeros(B, V_L_question, 4, dtype=torch.float)
            question_vis_feats = torch.zeros(
                B, V_L_question, feat_dim, dtype=torch.float)
            context_boxes = torch.zeros(B, V_L_context, 4, dtype=torch.float)
            context_vis_feats = torch.zeros(
                B, V_L_context, feat_dim, dtype=torch.float)

        context_text, question_text = list(), list()
        for i, item in enumerate(batch):
            question_text.append(item['question_text'])
            context_text.append(item['passage_text'])
            if self.TokenizerConfig.use_vision:
                n_boxes_context = item['n_boxes_passage']
                n_boxes_question = item['n_boxes_question']
                question_boxes[i,
                               :n_boxes_question] = item['question_image_boxes']
                question_vis_feats[i,
                                   :n_boxes_question] = item['question_image_features']
                context_boxes[i,
                              :n_boxes_context] = item['passage_image_boxes']
                context_vis_feats[i,
                                  :n_boxes_context] = item['passage_image_features']
        question_inputs = self.tokenizer(
            question_text, padding='max_length', truncation=True, return_tensors="pt")
        context_inputs = self.tokenizer(
            context_text, padding='max_length', truncation=True, return_tensors="pt")
        return {
            "input_ids_question": question_inputs,
            "input_ids_context": context_inputs,
            "visual_feats_question": question_vis_feats,
            "visual_feats_context": context_vis_feats,
            "question_image_boxes": question_boxes,
            "context_image_boxes": context_boxes
        }


def test_dataloader():
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
        "key_irrelevant": 'BM25_irrelevant_indices'
    }
    batch_size = 2
    dataset_clip = CLIPlikeDataset(**kwargs_clip)
    dataloader_clip = DataLoader(
        dataset_clip, batch_size=batch_size, collate_fn=dataset_clip.collate_fn)
    dataset_dpr = DPRDataset(**kwargs_dpr)
    dataloader_dpr = DataLoader(
        dataset_dpr, batch_size=batch_size, collate_fn=dataset_dpr.collate_fn)
    return dataloader_clip, dataloader_dpr


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
