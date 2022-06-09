from datasets import disable_caching, load_from_disk
import random
import torch
from torch.utils.data import Dataset, DataLoader
disable_caching()


"""----------------------------------------"""

# https://arxiv.org/abs/2004.04906
# Loss with relevant passage and irrelevant passages
# can used random irrelevant or irrelant mined by sime method to define (BM25, ...)


class DPRDataset(Dataset):
    def __init__(self):
        super().__init__()


#                        |  passage   |              |  question  |
# je dois gérer à chaque |text + image| associé à du |text + image|
# example positif prendre question réponse, et négatif on utilise juste le batch
class CLIPlikeDataset(Dataset):
    def __init__(
            self,
            passage_path,
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
        self.sources = split.split(',')
        if self.verbose:
            print('Data sources: ', self.sources)
        self.passages = load_from_disk(passage_path)
        self.dataset = load_from_disk(dataset_path)[self.sources]
        self.kb = load_from_disk(kb_path)
        self.key_index_relevant_passages = key_relevant
        self.key_text_question = key_text_question
        self.key_text_passage = key_text_passage
        self.key_vision_features = key_vision_features
        self.key_boxes = key_vision_boxes

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
        item['question_text'] = self.dataset[index][self.key_text_question]
        question_image_features = torch.Tensor(
            self.dataset[index][self.key_vision_features])
        question_image_boxes = torch.Tensor(
            self.dataset[index][self.key_boxes])
        item['question_image_features'] = torch.squeeze(
            question_image_features, dim=1)
        item['question_image_boxes'] = torch.squeeze(
            question_image_boxes, dim=1)

        relevants_index = self.dataset[index][self.key_index_relevant_passages]
        if len(relevants_index) < 1:
            pass

        # select just one relevant passage
        relevant_index = random.choice(relevant_index)

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

        # to device est fait dans le forward
        # ici on peut tokenizer le batch
        # et convertir au bon type (numpy)

        # Calculating the Loss
        # tokenize dans le forward du modèle ?
        pass


if __name__ == '__main__':
    kwargs = {
        "dataset_path": "/scratch_global/stage_pgrimal/data/CVLP/data/datasets/vlt5_dataset",
        "kb_path": "/scratch_global/stage_pgrimal/data/CVLP/data/datasets/vlt5_dataset/kb",
        "passages_path": "/scratch_global/stage_pgrimal/data/CVLP/data/datasets/vlt5_passages",
        "key_relevant": 'provenance_indices',
        "key_text_question": 'input',
        "key_text_passage": 'passage',
        "key_vision_features": 'vlt5_features',
        "key_vision_boxes": 'vlt5_normalized_boxes',
        "split": 'train'
    }

    dataset = CLIPlikeDataset(**kwargs)


"""
passage_embeddings = encoder_passage(texts, images)
question_embeddings = encoder_question(texts, images)

# TODO : peut être normaliser

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
