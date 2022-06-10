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

    def __len__(self):
        return self.dataset.num_rows

    def __getitem__(self, index):
        item = {}
        # question features
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
            item[f'passage_relevant_text'] = None
            item[f'passage_relevant_image_boxes'] = None
            item[f'passage_relevant_image_features'] = None
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
        questions, relevant_passages, irrelevant_passages, labels = [], [], [], []
        for target, item in enumerate(batch):
            if item['passage_irrelevant_text'] is None:
                # pas de irrelevant
                # mettre du vent à la place et le bon nombre
                self.n_irrelevant_passages
            else:
                pass
            if item['passage_relevant_text'] is None:
                # pas de relevant
                # to ignore index, target should be equal to -100
                # mettre du vent et le bon nombre
                labels.append(-100)
                # add random
            else:
                labels.append(target)

            if len(relevant_passage) < 1:
                relevant_passage = ['']
                labels.append(self.loss_fct.ignore_index)
            else:
                labels.append(i)
            
            questions.append(item['input'])
            relevant_passages.extend(relevant_passage)
            irrelevant_passages.extend(irrelevant_passage)

        question_inputs = self.tokenizer(questions, **self.tokenization_kwargs)
        context_inputs = self.tokenizer(
            relevant_passages + irrelevant_passages, **self.tokenization_kwargs)
        labels = torch.tensor(labels)
        batch = dict(question_inputs=question_inputs,
                     context_inputs=context_inputs, labels=labels)


#                        |  passage   |              |  question  |
# je dois gérer à chaque |text + image| associé à du |text + image|
# example positif prendre question réponse, et négatif on utilise juste le batch
class CLIPlikeDataset(Dataset):
    def __init__(
            self,
            passages_path,
            dataset_path,
            kb_path,
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
        self.tokenizer = 'None'

    def __len__(self):
        return self.dataset.num_rows

    def __getitem__(self, index):
        # The __getitem__ function loads and returns a sample.
        item = {}
        # on associe un passage pertinent random à la question

        # voir si à ce niveau là qu'on doit mettre les éléments to device
        # voir si à ce niveau lçà qu'on fait les transformations ?
        # je pense que oui car  permet de charger les données

        # question features
        relevants_index = self.dataset[index][self.key_index_relevant_passages]

        if len(relevants_index) < 1:
            # gerer dans collate fn
            # si pas de relevant passage on peut prendre un random
            # et le mettre à 0 ?
            pass

        # select just one relevant passage
        relevant_index = np.random.choice(relevants_index)

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
        # collate fn permet de préciser comment on va charger le batch
        # TODO: add tokenizer to have less computation in the cpu
        self.tokenizer()


def get_dataloader():
    kwargs = {
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

    dataset = CLIPlikeDataset(**kwargs)
    dataloader = DataLoader(dataset, batch_size=2)
    return dataloader


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
