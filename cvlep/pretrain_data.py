import numpy as np
from cvlep.VLT5.param import Config
from cvlep.CLIPT5.tokenization import VLT5TokenizerFast
from transformers import T5TokenizerFast
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
        if isinstance(self.topk, float) and (0 < self.topk <= 1):
            used_samples = int(self.topk * self.dataset.num_rows)
            self.dataset = self.dataset.select(range(used_samples))
            if self.verbose:
                print(f"Use only {used_samples} data")
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
        return self.dataset.num_rows

    def __getitem__(self, index):
        item = {}
        # question features
        item['question_text'] = self.dataset[index][self.key_text_question]

        # relevant and irrelevant passage features
        relevants_index = self.dataset[index][self.key_index_relevant_passages]
        irrelevant_index = self.dataset[index][self.key_index_irrelevant_passages]
        if len(relevants_index) < 1:
            # warnings.warn(
            #     f"Didn't find any relevant passage for question {self.dataset[index]['id']}")
            item[f'passage_relevant_text'] = None
        else:
            relevant_index = random.choice(relevants_index)
            item[f'passage_relevant_text'] = self.passages[relevant_index][self.key_text_passage]

        if len(irrelevant_index) < 1:
            # warnings.warn(
            #     f"Didn't find any irrelevant passage for question {self.dataset[index]['id']}")
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
            "context_image_boxes": None,
            "task":"triviaqa"
        }


class WikiImage(Dataset):
    def __init__(
        self,
        dataset_path,
        key_image,
        key_vision_features='fastrcnn_features',
        key_vision_boxes='fastrcnn_boxes',
        split = 'train',
        topk=-1,
        sampling_with_replacement = False,
        verbose=True
    ):
        super().__init__()
        if split == "validation":
            self.split = "validation"
        elif split == "train":
            self.split = "train"
        else:
            raise NotImplementedError("This split is not implemented")
        self.dataset = load_from_disk(dataset_path)[self.split]
        self.topk = topk
        self.verbose = verbose
        if isinstance(self.topk, float) and (0 < self.topk <= 1):
            used_samples = int(self.topk * self.dataset.num_rows)
            self.dataset = self.dataset.select(range(used_samples))
            if self.verbose:
                print(f"Use only {used_samples} data")
        self.key_image = key_image
        self.key_vision_features=key_vision_features
        if "clip" in key_vision_features.lower():
            self.key_vision_boxes = None
        else:
            self.key_vision_boxes = key_vision_boxes
        self.sampling_with_replacement = sampling_with_replacement

    def __len__(self):
        return self.dataset.num_rows

    def __getitem__(self, index):
        item = {}
        list_images = self.dataset[index][self.key_image]
        # tirer aléatoirement deux index de features d'images
        if self.sampling_with_replacement:
            # with replacement
            fct = random.choices
        else:
            fct = random.sample
        index_images = fct(range(len(list_images)), k=2)
        try:
            item['image_features_question'] = torch.Tensor(self.dataset[index][self.key_vision_features][index_images[0]])
            item['image_features_context'] = torch.Tensor(self.dataset[index][self.key_vision_features][index_images[1]])
        except IndexError as e:
            print(index)
            error = f'WIKIMAGE index : {index}, {self.dataset[index]["wikipedia_title"]} {e}'
            print(error)
            index_images = [0,1]
            index = 0
            item['image_features_question'] = torch.Tensor(self.dataset[index][self.key_vision_features][index_images[0]])
            item['image_features_context'] = torch.Tensor(self.dataset[index][self.key_vision_features][index_images[1]])
        if self.key_vision_boxes is not None:
            boxes_question = torch.Tensor(self.dataset[index][self.key_vision_boxes][index_images[0]])
            boxes_context = torch.Tensor(self.dataset[index][self.key_vision_boxes][index_images[1]])
        else:
            boxes_question =  torch.zeros(item['image_features_question'].shape[0], 4) # (L, 4)
            boxes_context = torch.zeros( item['image_features_context'].shape[0], 4) # (L, 4)
        item['image_boxes_question'] = boxes_question
        item['image_boxes_context'] = boxes_context
        item['n_boxes_question'] = boxes_question.size()[0]
        item['n_boxes_context'] = boxes_context.size()[0]
        return item
# 32313, Martin Chilcott
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
                           :n_boxes_question] = item['image_boxes_question']
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
            "context_image_boxes": context_boxes,
            "task":"match_image"
        }

class MultimediaDataset(Dataset):
    # match differents passages paired with illustrative image
    #  of one article
    def __init__(
        self,
        passages_path,
        kb_path,
        tokenizer_path,
        split='train',
        key_passage_index='passage_index',
        key_text_passage='passage',
        key_list_images="list_images",
        key_vision_features='fastrcnn_features',
        key_vision_boxes='fastrcnn_boxes',
        topk=-1,
        verbose=True,
        sampling_with_replacement = False
    ):
        super().__init__()
        self.passages = load_from_disk(passages_path)
        self.topk = topk
        self.verbose = verbose
        if split == "validation":
            self.split = "validation"
        elif split == "train":
            self.split = "train"
        else:
            raise NotImplementedError("This split is not implemented")
        self.kb = load_from_disk(kb_path)[self.split]
        if isinstance(self.topk, float) and (0 < self.topk <= 1):
            used_samples = int(self.topk * self.kb.num_rows)
            self.kb = self.kb.select(range(used_samples))
            if self.verbose:
                print(f"Use only {used_samples} data")

        self.key_passage_index = key_passage_index
        self.key_list_images = key_list_images
        self.key_text_passage = key_text_passage
        self.key_vision_features = key_vision_features
        if "clip" in self.key_vision_features.lower():
            self.key_vision_boxes = None
        else:
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
        self.sampling_with_replacement = sampling_with_replacement

    def __len__(self):
        return self.kb.num_rows

    def __getitem__(self, index):
        # We use 'question' and 'context'
        # 'question' is intented for encoder_question
        # 'context' is intented for encoder_passage

        # list_passages = self.kb[index][self.key_passage_index]
        # index_passages = random.sample(range(len(list_passages)), k=2)
        # list_images = self.kb[index][self.key_list_images]
        # if self.sampling_with_replacement:
        #     # with replacement
        #     fct = random.choices
        # else:
        #     fct = random.sample
        # index_images = fct(range(len(list_images)), k=2)
        # item = {}

        # passage for question encoder
        # item['question_text'] = self.passages[index_passages[0]
        #                                       ][self.key_text_passage]
        # # passage for passage encoder
        # item['passage_text'] = self.passages[index_passages[1]
        #                                      ][self.key_text_passage]
        if self.sampling_with_replacement:
            # with replacement
            fct = random.choices
        else:
            fct = random.sample
        list_images = self.kb[index][self.key_list_images]

        index_images = fct(range(len(list_images)), k=2)
        try:
            item['image_features_question'] = torch.Tensor(self.kb[index][self.key_vision_features][index_images[0]])
            item['image_features_context'] = torch.Tensor(self.kb[index][self.key_vision_features][index_images[1]])
        except IndexError as e:
            print(index)
            error = f'MULTIMEDIA index : {index}, {self.dataset[index]["wikipedia_title"]} {e}'
            print(error)
            index_images = [0,1]
            index = 0
            item['image_features_question'] = torch.Tensor(self.dataset[index][self.key_vision_features][index_images[0]])
            item['image_features_context'] = torch.Tensor(self.dataset[index][self.key_vision_features][index_images[1]])
        # temp
        list_passages = self.kb[index][self.key_passage_index]
        index_passages = random.sample(range(len(list_passages)), k=2)
        list_images = self.kb[index][self.key_list_images]
        if self.sampling_with_replacement:
            # with replacement
            fct = random.choices
        else:
            fct = random.sample
        item = {}
        item['question_text'] = self.passages[index_passages[0]
                                              ][self.key_text_passage]
        # passage for passage encoder
        item['passage_text'] = self.passages[index_passages[1]
                                             ][self.key_text_passage]
        
        if self.key_vision_boxes is not None:
            boxes_question = torch.Tensor(self.kb[index][self.key_vision_boxes][index_images[0]])
            boxes_context = torch.Tensor(self.kb[index][self.key_vision_boxes][index_images[1]])
        else:
            boxes_question =  torch.zeros(item['image_features_question'].shape[0], 4) # (L, 4)
            boxes_context = torch.zeros( item['image_features_context'].shape[0], 4) # (L, 4)
        item['image_boxes_question'] = boxes_question
        item['image_boxes_context'] = boxes_context
        item['n_boxes_question'] = boxes_question.size()[0]
        item['n_boxes_context'] = boxes_context.size()[0]
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
            "input_ids_context": context_input.input_ids,
            "attention_mask_context": context_input.attention_mask,
            "labels": labels,
            "visual_feats_question": question_vis_feats,
            "visual_feats_context": context_vis_feats,
            "question_image_boxes": question_boxes,
            "context_image_boxes": context_boxes,
            "task":"match_article"
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

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_loader(
        task,
        mode,
        batch_size,
        seed,
        distributed,
        split,
        workers,
        verbose=False,
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
        dataset_class = Viquae
    else:
        raise NotImplementedError("dataset about this task is not implemented")
    dataset = dataset_class(
        verbose=verbose,
        split=split,
        **dataset_args
    )
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
            pin_memory=(sampler is not None),
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
            pin_memory=(sampler is not None),
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


def test_dataloader(task):
    kwargs_triviaqa = {
        "tokenizer_path": "experiments/configEncoder/bergamote/TokenizerConfig.json",
        "dataset_path": "/scratch_global/stage_pgrimal/data/CVLP/data/datasets/kilt/triviaqa_for_viquae",
        "passages_path": "/scratch_global/stage_pgrimal/data/CVLP/data/datasets/kilt/passages",
        "key_relevant": 'provenance_indices',
        "key_text_question": 'input',
        "key_text_passage": 'passage',
        "split": 'train',
        "topk":-1
    }
    kwargs_wikimage={
        "dataset_path": "/scratch_global/stage_pgrimal/data/CVLP/data/datasets/wikimage/wikimage_train_val_filter",
        "topk":-1,
        "key_image":"list_images",
        "key_vision_features":"clip_features",
        "key_vision_boxes":None
    }
    kwargs_multimedia={
        "kb_path": "/scratch_global/stage_pgrimal/data/CVLP/data/datasets/multimedia/filtered/multimedia_train_val_filter",
        "passages_path": "/scratch_global/stage_pgrimal/data/CVLP/data/datasets/multimedia/filtered/passages",
        "tokenizer_path": "experiments/config_vladapter/bergamote/adapter/TokenizerConfig.json",
        "key_passage_index": "passage_index",
        "key_text_passage": "passage",
        "key_list_images": "list_images",
        "key_vision_features": "clip_features",
        "key_vision_boxes": None,
        "topk": -1,
    }
    kwargs_viquae = {
        "tokenizer_path": "experiments/config_vladapter/bergamote/adapter/TokenizerConfig.json",
        "dataset_path": "/scratch_global/stage_pgrimal/data/CVLP/data/datasets/miniviquae",
        "kb_path": "/scratch_global/stage_pgrimal/data/CVLP/data/datasets/kb",
        "passages_path": "/scratch_global/stage_pgrimal/data/CVLP/data/datasets/passages",
        "key_relevant": "provenance_indices",
        "key_text_question": "input",
        "key_text_passage": "passage",
        "key_vision_features": "clip_features",
        "key_vision_boxes": None,
        "key_irrelevant": "BM25_irrelevant_indices",
    }
    batch_size = 2
    if task == "triviaqa":
        dataset_class = KiltDataset
        args = kwargs_triviaqa
    elif task == "match_image":
        dataset_class = WikiImage
        args=kwargs_wikimage
    elif task == "match_article":
        dataset_class = MultimediaDataset
        args = kwargs_multimedia
    elif task == "viquae":
        dataset_class = Viquae
        args = kwargs_viquae
    args['verbose']=True
    args['split'] = 'train'
    dataset =dataset_class(**args)
    print(dataset)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, collate_fn=dataset.collate_fn)
    return dataloader

if __name__ == '__main__':
    pass