import json
import numpy as np
from cvlep.VLT5.param import Config
from cvlep.CLIPT5.tokenization import VLT5TokenizerFast
from transformers import T5TokenizerFast
from multiprocessing import context
import warnings
from datasets import disable_caching, load_from_disk
import random
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from cvlep.viquae_data import Viquae

disable_caching()

# Kilt dataset
# "train_dataset": "data/triviaqa/without_viquae"
# "eval_dataset": "data/triviaqa/with_viquae_validation"
# "kb": "data/kilt_passages"
# # Entity image wikipedia image


class KiltDataset(Dataset):
    def __init__(
            self,
            passages_path,
            dataset_path,
            tokenizer_path,
            key_relevant='provenance_indices',
            key_irrelevant='BM25_irrelevant_indices',
            key_text_question='input',
            key_text_passage='passage',
            split='train',
            topk=-1,
            verbose=True
    ) -> None:
        super().__init__()
        self.verbose = verbose
        if split == "validation":
            self.split = "with_viquae_validation"
        elif split == "train":
            self.split = "without_viquae"
        else:
            raise NotImplementedError("This split is not implemented")
        self.n_relevant_passages = 1
        self.n_irrelevant_passages = 1
        self.topk = topk
        if self.verbose:
            print('Data sources: ', self.split)
            print(
                f'Number of passages per question in a batch :\
                {self.n_relevant_passages + self.n_irrelevant_passages} \
                \nnb relevants : {self.n_relevant_passages} \
                \nnb irrelevants : {self.n_irrelevant_passages}'
            )
        self.passages = load_from_disk(passages_path)
        self.dataset = load_from_disk(dataset_path)[self.split]
        if self.topk != -1 and 0 < self.topk <= 1:
            self.len = int(self.dataset.num_rows * self.topk)
        else:
            self.len = self.dataset.num_rows
        self.key_index_relevant_passages = key_relevant
        self.key_index_irrelevant_passages = key_irrelevant
        self.key_text_question = key_text_question
        self.key_text_passage = key_text_passage
        # we use the same tokenizer for question and passage
        self.TokenizerConfig = Config.load_json(tokenizer_path)
        if 't5' in self.TokenizerConfig.tokenizer:
            if self.TokenizerConfig.use_vision:
                tokenizer_class = VLT5TokenizerFast
            else:
                tokenizer_class = T5TokenizerFast
        else:
            raise ValueError('This type of tokenizer is not implemented')

        self.tokenizer = tokenizer_class.from_pretrained(
            self.TokenizerConfig.tokenizer,
            do_lower_case=self.TokenizerConfig.do_lower_case,
        )

    def __len__(self):
        return self.len

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
            # TODO: voir si besoin de changer gesti/scratch_global/stage_pgrimal/data/CVLP/data/datasets/kilt/trivaqa_for_viquaen pour le text, car a un impact sur attention mask

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
            "labels": labels,
            "visual_feats_question": None,
            "visual_feats_context": None,
            "question_image_boxes": None,
            "context_image_boxes": None
        }


# TODO:split possibility
# TODO: image embedding
class WikiImage(Dataset):
    def __init__(
        self,
        dataset_path,
        key_image
    ):
        super().__init__()
        self.dataset = load_from_disk(dataset_path)
        self.key_image = key_image

    def __len__(self):
        self.dataset.num_rows

    def __getitem__(self, index):
        item = {}
        list_images = self.dataset[index][self.key_image]
        # tirer aléatoirement deux index de features d'images
        if True:
            # avec remise
            fct = random.choices
        else:
            # sans remise
            fct = random.sample
        index_images = fct(range(len(list_images)), k=2)
        # TODO:define when I ll be sure of the structure
        boxes_question = None
        boxes_context = None
        item['image_features_question'] = list_images[index_images[0]]
        item['image_features_context'] = list_images[index_images[1]]
        item['image_boxes_question'] = boxes_question[index_images[0]]
        item['image_boxes_context'] = boxes_context[index_images[1]]
        item['n_boxes_question'] = item[''].size()[0]
        item['n_boxes_context'] = item[''].size()[0]
        return item

    def collate_fn(self, batch):
        B = len(batch)
        V_L_question = max(item['n_boxes_question'] for item in batch)
        V_L_context = max(item['n_boxes_context'] for item in batch)
        feat_dim = batch[0]['image_features_question'].shape[-1]
        # boxes are represented by 4 points
        question_boxes = torch.zeros(B, V_L_question, 4, dtype=torch.float)
        question_vis_feats = torch.zeros(
            B, V_L_question, feat_dim, dtype=torch.float)
        context_boxes = torch.zeros(B, V_L_context, 4, dtype=torch.float)
        context_vis_feats = torch.zeros(
            B, V_L_context, feat_dim, dtype=torch.float)
        labels = list()
        for i, item in enumerate(batch):
            n_boxes_context = item['n_boxes_context']
            n_boxes_question = item['n_boxes_question']
            question_boxes[i,
                           :n_boxes_question] = item['question_image_boxes']
            question_vis_feats[i,
                               :n_boxes_question] = item['image_features_question']
            context_boxes[i,
                          :n_boxes_context] = item['image_boxes_context']
            context_vis_feats[i,
                              :n_boxes_context] = item['image_features_context']
            labels.append(i)
        labels = torch.tensor(labels)
        return {
            "input_ids_question": None,
            "attention_mask_question": None,
            "input_ids_context": None,
            "attention_mask_context": None,
            "labels": labels,
            "visual_feats_question": question_vis_feats,
            "visual_feats_context": context_vis_feats,
            "question_image_boxes": question_boxes,
            "context_image_boxes": context_boxes
        }

# TODO : use with image embedding


class MultimediaDataset(Dataset):
    # match differents passages paired with illustrative image
    #  of one article
    def __init__(
        self,
        passages_path,
        topk,
        kb_path,

        tokenizer_path,
        key_passage_index='passage_index',
        key_text_passage='passage',
        key_list_images="list_images",
        key_vision_features='fastrcnn_features',
        key_vision_boxes='fastrcnn_boxes',
        verbose=True
    ):
        super().__init__()
        self.passages = load_from_disk(passages_path)
        self.tokp = topk
        self.verbose = verbose
        self.kb = load_from_disk(kb_path)
        if isinstance(self.topk, float) and (0 < self.topk <= 1):
            used_samples = int(self.topk * self.kb.num_rows)
            self.kb = self.kb[:used_samples]
            if self.verbose:
                print(f"Use only {used_samples} data")

        self.key_passage_index = key_passage_index
        self.key_list_images = key_list_images
        self.key_text_passage = key_text_passage
        self.key_vision_features = key_vision_features
        self.key_vision_boxes = key_vision_boxes

        # we use the same tokenizer for question and passage
        self.TokenizerConfig = Config.load_json(tokenizer_path)
        if 't5' in self.TokenizerConfig.tokenizer:
            if self.TokenizerConfig.use_vision:
                tokenizer_class = VLT5TokenizerFast
            else:
                tokenizer_class = T5TokenizerFast
        else:
            raise ValueError('This type of tokenizer is not implemented')

        self.tokenizer = tokenizer_class.from_pretrained(
            self.TokenizerConfig.tokenizer,
            do_lower_case=self.TokenizerConfig.do_lower_case,
        )

    def __len__(self):
        return self.kb.num_rows

    def __getitem__(self, index):
        # We use 'question' and 'context'
        # 'question' is intented for encoder_question
        # 'context' is intented for encoder_passage

        list_passages = self.kb[str(index)][self.key_passage_index]
        index_passages = random.sample(len(list_passages), k=2)
        list_images = self.kb[index][self.key_list_images]
        # TODO:pour index images avec ou sans remise ?
        index_images = random.sample(range(len(list_images)), k=2)
        item = {}

        # passage for question encoder
        item['question_text'] = self.passages[index_passages[0]
                                              ][self.key_text_passage]
        # passage for passage encoder
        item['passage_text'] = self.passages[index_passages[1]
                                             ][self.key_text_passage]

        boxes_question = None
        boxes_context = None
        item['image_features_question'] = list_images[index_images[0]]
        item['image_features_context'] = list_images[index_images[1]]
        item['image_boxes_question'] = boxes_question[index_images[0]]
        item['image_boxes_context'] = boxes_context[index_images[1]]
        item['n_boxes_question'] = item[''].size()[0]
        item['n_boxes_context'] = item[''].size()[0]
        return item

    def collate_fn(self, batch):

        B = len(batch)

        V_L_question = max(item['n_boxes_question'] for item in batch)
        V_L_context = max(item['n_boxes_context'] for item in batch)
        feat_dim = batch[0]['image_features_question'].shape[-1]
        # boxes are represented by 4 points
        question_boxes = torch.zeros(B, V_L_question, 4, dtype=torch.float)
        question_vis_feats = torch.zeros(
            B, V_L_question, feat_dim, dtype=torch.float)
        context_boxes = torch.zeros(B, V_L_context, 4, dtype=torch.float)
        context_vis_feats = torch.zeros(
            B, V_L_context, feat_dim, dtype=torch.float)

        question_text, context_text, labels = list(), list(), list()
        for i, item in enumerate(batch):
            question_text.append(item['question_text'])
            context_text.append(item['passage_text'])
            n_boxes_context = item['n_boxes_context']
            n_boxes_question = item['n_boxes_question']
            question_boxes[i,
                           :n_boxes_question] = item['image_boxes_question']
            question_vis_feats[i,
                               :n_boxes_question] = item['image_features_question']
            context_boxes[i,
                          :n_boxes_context] = item['image_boxes_context']
            context_vis_feats[i,
                              :n_boxes_context] = item['image_features_context']

            labels.append(i)

        question_input = self.tokenizer(
            question_text, padding='max_length', truncation=True, return_tensors="pt")
        context_input = self.tokenizer(
            context_text, padding='max_length', truncation=True, return_tensors="pt")
        labels = torch.tensor(labels)

        return {
            "input_ids_question": question_input.input_ids,
            "attention_mask_question": question_input.attention_mask,
            "input_ids_context": context_input.inputs_ids,
            "attention_mask_context": context_input.attention_mask,
            "labels": labels,
            "visual_feats_question": question_vis_feats,
            "visual_feats_context": context_vis_feats,
            "question_image_boxes": question_boxes,
            "context_image_boxes": context_boxes
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
        task,
        mode,
        # tokenizer_path,
        # dataset_path,
        # kb_path,
        # passages_path,
        # key_relevant,
        # key_text_question,
        # key_text_passage,
        # key_vision_features,
        # key_vision_boxes,
        # batch_size
        # split,
        seed,
        distributed,
        workers,
        verbose=False,
        #key_irrelevant=None
        **dataset_args,
):
    # TODO: ajouter loader.task
    g = torch.Generator()
    g.manual_seed(seed)

    if task == "triviaqa":
        dataset_class = KiltDataset
    elif task == "match_image":
        dataset_class = WikiImage
    elif task == "match_article":
        dataset_class = MultimediaDataset
    elif task == "viquae":
        dataset_class == Viquae
    else:
        raise NotImplementedError("dataset about this task is not implemented")
    dataset = dataset_class(
        verbose=verbose,
        **dataset_args
    )
    if distributed:
        sampler = DistributedSampler(dataset, drop_last=True, seed=seed)
    else:
        sampler = None
    dataset_args['batch_size']
    if mode == 'train':
        loader = DataLoader(
            dataset=dataset,
            batch_size=dataset_args['batch_size'],
            shuffle=(sampler is None),
            num_workers=workers,
            pin_memory=(sampler is not None),
            sampler=sampler,
            collate_fn=dataset.collate_fn,
            worker_init_fn=seed_worker,
            generator=g
        )
    elif mode == 'eval':
        loader = DataLoader(
            dataset=dataset,
            batch_size=dataset_args['batch_size'],
            shuffle=False,
            num_workers=workers,
            pin_memory=(sampler is not None),
            sampler=sampler,
            collate_fn=dataset.collate_fn,
            worker_init_fn=seed_worker,
            generator=g
        )
    elif mode == 'test':
        loader = DataLoader(
            dataset=dataset,
            batch_size=dataset_args['batch_size'],
            shuffle=False,
            num_workers=workers,
            pin_memory=(sampler is not None),
            sampler=sampler,
            collate_fn=dataset.collate_fn,
            worker_init_fn=seed_worker,
            generator=g
        )
    else:
        raise NotImplementedError('this mode is not implemented')
    loader.task = task
    return loader


def test_dataloader():
    kwargs_triviaqa = {
        "tokenizer_path": "experiments/configEncoder/bergamote/TokenizerConfig.json",
        "dataset_path": "/scratch_global/stage_pgrimal/data/CVLP/data/datasets/kilt/triviaqa_for_viquae",
        "passages_path": "/scratch_global/stage_pgrimal/data/CVLP/data/datasets/kilt/passages",
        "key_relevant": 'provenance_indices',
        "key_text_question": 'input',
        "key_text_passage": 'passage',
        "split": 'train',
        "topk": 0.8
    }
    batch_size = 4
    dataset_kilt = KiltDataset(**kwargs_triviaqa)
    dataloader_kilt = DataLoader(
        dataset_kilt, batch_size=batch_size, collate_fn=dataset_kilt.collate_fn)

    return dataloader_kilt


if __name__ == '__main__':
    pass
