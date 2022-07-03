import math
import numpy as np
from cvlep.VLT5.param import Config
from cvlep.CLIPT5.tokenization import VLT5Tokenizer, VLT5TokenizerFast
from transformers import T5Tokenizer, BartTokenizer, T5TokenizerFast, BartTokenizerFast
from multiprocessing import context
import warnings
from datasets import disable_caching, load_from_disk
import random
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

disable_caching()


def remove_lines(n_rows, split):
    index = [_ for _ in range(n_rows)]
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


# Kilt dataset
# "train_dataset": "data/triviaqa/without_viquae"
# "eval_dataset": "data/triviaqa/with_viquae_validation"
# "kb": "data/kilt_passages"
# # Entity image wikipedia image

# même chose que DPR sauf qu'on ne passera pas d'image
class KiltDataset(Dataset):
    def __init__(
            self,
            passages_path,
            dataset_path,
            tokenizer_path,
            n_irrelevant_passages=1,
            key_relevant='provenance_indices',
            key_irrelevant='BM25_irrelevant_indices',
            key_text_question='input',
            key_text_passage='passage',
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
        # TODO:check si besoin d'enlever des lignes sans réponses
        index = remove_lines(self.dataset.num_rows, self.split)
        self.dataset = self.dataset.select(index)
        self.key_index_relevant_passages = key_relevant
        self.key_index_irrelevant_passages = key_irrelevant
        self.key_text_question = key_text_question
        self.key_text_passage = key_text_passage
        # we use the same tokenizer for question and passage
        self.TokenizerConfig = Config.load_json(tokenizer_path)
        if 't5' in self.TokenizerConfig.tokenizer:
            if self.TokenizerConfig.use_vision:
                # tokenizer_class = VLT5Tokenizer
                tokenizer_class = VLT5TokenizerFast
            else:
                # tokenizer_class = T5Tokenizer
                tokenizer_class = T5TokenizerFast
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
    
        # relevant and irrelevant passage features
        relevants_index = self.dataset[index][self.key_index_relevant_passages]
        irrelevant_index = self.dataset[index][self.key_index_irrelevant_passages]
        if len(relevants_index) < 1:
            warnings.warn(
                f"Didn't find any relevant passage for question {self.dataset[index]['id']}")
            item[f'passage_relevant_text'] = None
        else:
            relevant_index = random.choice(relevants_index)
            item[f'passage_relevant_text'] = self.passages[relevant_index][self.key_text_passage]

        if len(irrelevant_index) < 1:
            warnings.warn(
                f"Didn't find any irrelevant passage for question {self.dataset[index]['id']}")
            item[f'passage_irrelevant_text'] = None
        else:
            irrelevant_index = random.choice(irrelevant_index)
            item[f'passage_irrelevant_text'] = self.passages[irrelevant_index][self.key_text_passage]
        return item

    def collate_fn(self, batch):
        B = len(batch)
        relevant_text, irrelevant_text, question_text, labels = list(), list(), list(), list()
        for i, item in enumerate(batch):
            # TODO: voir si besoin de changer gestion pour le text, car a un impact sur attention mask
            
            if item['passage_relevant_text'] is None:
                labels.append(-100)  # ignore index when computing the loss
                relevant_text.append('')
            else:
                labels.append(i)
                relevant_text.append(item['passage_relevant_text'])
            if item['passage_irrelevant_text'] is None:
                irrelevant_text.append('')
            else:
                irrelevant_text.append(item['passage_irrelevant_text']) 
            question_text.append(item['question_text'])
        question_input = self.tokenizer(
            question_text, padding='max_length', truncation=True, return_tensors="pt")
        context_input = self.tokenizer(
            relevant_text + irrelevant_text, padding='max_length', truncation=True, return_tensors="pt")
        labels = torch.tensor(labels)
        return {
            "input_ids_question": question_input.input_ids,
            "attention_mask_question": question_input.attention_mask,
            "input_ids_context": context_input.input_ids,
            "attention_mask_context": context_input.attention_mask,
            "labels": labels
        }


class wikiImage(Dataset):
    def __init__(
        self,
        dataset_path
        ):
        super().__init__()
        self.dataset = load_from_disk(dataset_path)
    def __len__(self):
        self.dataset.num_rows

    def __getitem__(self, index):
        pass

    def collate_fn(self, batch):
        B = len(batch)

        return {
            "image_question":None,
            "image_passage":None
        }


# passage avec relevant passage 
# juste relevant passage pour l'instant
# pas d'irrelevant
class multimedia(Dataset):
    def __init__(
        self,
        passages_path,
        topk,
        kb_path,
        tokenizer_path,
        key_index_article='index',
        key_text_passage='passage',
        key_vision_features='fastrcnn_features',
        key_vision_boxes='fastrcnn_boxes',
        verbose=True
    ):
        super().__init__()
        self.dataset = load_from_disk(passages_path)
        self.tokp = topk
        self.verbose = verbose
        if isinstance(self.topk, float) and (0 < self.topk <= 1):
            used_samples = int(self.topk * len(data))
            data = random.sample(data, used_samples)
            if self.verbose:
                print(f"Use only {len(data)} data")
        self.kb = load_from_disk(kb_path)
        self.key_index_article = key_index_article
        self.key_text_passage = key_text_passage
        self.key_vision_features = key_vision_features
        self.key_vision_boxes = key_vision_boxes
        # we use the same tokenizer for question and passage
        self.TokenizerConfig = Config.load_json(tokenizer_path)
        if 't5' in self.TokenizerConfig.tokenizer:
            if self.TokenizerConfig.use_vision:
                # tokenizer_class = VLT5Tokenizer
                tokenizer_class = VLT5TokenizerFast
            else:
                # tokenizer_class = T5Tokenizer
                tokenizer_class = T5TokenizerFast
        else:
            raise ValueError('This type of tokenizer is not implemented')

        self.tokenizer = tokenizer_class.from_pretrained(
            self.TokenizerConfig.tokenizer,
            do_lower_case=self.TokenizerConfig.do_lower_case,
        )

    def __len__(self):
        return self.dataset.num_rows

    def __getitem__(self, index):
        pass

    def collate_fn(self, batch):
        return {
            
        }

class ImageCaption(Dataset):
    # return image embedding and a link caption
    # new loss : we dont have any irrelevant passage
    # just kind of cosine similarity
    def __init__(
        self,

    ):
        super().__init__()
        

    def __len__(self):
        return self
    
    def __getitem__(self, index):
        pass

    def collate_fn(self, batch):
        return{

        }

"""

Utiliser même chose que DPR avec les targets pour plus de simplicités
def compute_loss_like_clip(model1, model2, data, temperature):
    # fake function to implement after in the model

    def cross_entropy(preds, targets, reduction='none'):
        log_softmax = nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(preds)).sum(1)
        if reduction == "none":
            return loss
        elif reduction == "mean":
            return loss.mean()

    question_embeddings = model1(
        data['input_ids_question'],
        (data['visual_feats_question'],data['question_image_boxes'])
    )
    passage_embeddings = model2(
        data['input_ids_context'],
        (data['visual_feats_context'],data['context_image_boxes'])
        )


    # TODO : peut être normaliser ? Voir la gueule de la training loop dans clip
    # normalized features from https://github.com/openai/CLIP/blob/b46f5ac7587d2e1862f8b7b1573179d80dcdd620/clip/model.py#L363
    # image_features = image_features / image_features.norm(dim=1, keepdim=True)
    # text_features = text_features / text_features.norm(dim=1, keepdim=True)
    logits = (passage_embeddings @ question_embeddings.T) / temperature
    questions_similarity = question_embeddings @ question_embeddings.T
    passages_similarity = passage_embeddings @ passage_embeddings.T
    targets = F.softmax(
        (questions_similarity + passages_similarity) / 2 * temperature, dim=-1
    )
    questions_loss = cross_entropy(logits, targets, reduction='none')
    passages_loss = cross_entropy(logits.T, targets.T, reduction='none')
    loss = (questions_loss + passages_loss) / 2.0  # shape: (batch_size)
    return loss.mean()
"""


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_loader(
        cls,
        mode,
        batch_size,
        seed,
        distributed,
        workers,
        tokenizer_path,
        dataset_path,
        kb_path,
        passages_path,
        key_relevant,
        key_text_question,
        key_text_passage,
        key_vision_features,
        key_vision_boxes,
        split,
        verbose=False,
        key_irrelevant=None
):

    g = torch.Generator()
    g.manual_seed(seed)

    if cls == "":
        dataset = KiltDataset(
            passages_path=passages_path,
            dataset_path=dataset_path,
            kb_path=kb_path,
            tokenizer_path=tokenizer_path,
            key_relevant=key_relevant,
            key_irrelevant=key_irrelevant,
            key_text_question=key_text_question,
            key_text_passage=key_text_passage,
            key_vision_features=key_vision_features,
            key_vision_boxes=key_vision_boxes,
            split=split,
            verbose=verbose
        )
    elif cls == "clip":
        dataset = SimpleContrastiveDataset(passages_path,
                                           dataset_path=dataset_path,
                                           kb_path=kb_path,
                                           tokenizer_path=tokenizer_path,
                                           key_relevant=key_relevant,
                                           key_text_question=key_text_question,
                                           key_text_passage=key_text_passage,
                                           key_vision_features=key_vision_features,
                                           key_vision_boxes=key_vision_boxes,
                                           split=split,
                                           verbose=verbose
                                           )
    else:
        raise NotImplementedError("This dataset is not implemented")
    # we want datasampler to avoid to have duplicate question
    if distributed:
        sampler = DistributedSampler(dataset, drop_last=True, seed=seed)
    else:
        sampler = None

    if mode == 'train':
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=(sampler is None),
            num_workers=workers,
            pin_memory=True,
            sampler=sampler,
            collate_fn=dataset.collate_fn,
            worker_init_fn=seed_worker,
            generator=g
        )
    elif mode == 'eval':
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
            sampler=sampler,
            collate_fn=dataset.collate_fn,
            worker_init_fn=seed_worker,
            generator=g
        )
    elif mode == 'test':
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
            sampler=sampler,
            collate_fn=dataset.collate_fn,
            worker_init_fn=seed_worker,
            generator=g
        )
    else:
        raise NotImplementedError('this mode is not implemented')
    return loader

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
    dataset_clip = SimpleContrastiveDataset(**kwargs_clip)
    dataloader_clip = DataLoader(
        dataset_clip, batch_size=batch_size, collate_fn=dataset_clip.collate_fn)
    dataset_dpr = DPRDataset(**kwargs_dpr)
    dataloader_dpr = DataLoader(
        dataset_dpr, batch_size=batch_size, collate_fn=dataset_dpr.collate_fn)
    return dataloader_clip, dataloader_dpr


if __name__ == '__main__':
    pass
