# coding: utf-8

from dataclasses import dataclass
import torch
from torch import nn
from transformers import T5Config
from transformers.modeling_outputs import ModelOutput, BaseModelOutputWithPooling
from cvlep.VLT5.modeling_t5 import JointEncoder as encoderT5
from cvlep.VLT5.modeling_bart import JointEncoder as encoderBart
from typing import Optional, Tuple, Any


class CVLEP(nn.Module):
    def __init__(self, config, image_question_encoder, image_passage_encoder, embedding_question=None, embedding_passage=None):

        super().__init__()
        self.config = config

        self.image_question_encoder = image_question_encoder
        self.image_passage_encoder = image_passage_encoder
        print(config.config_str)
        if config.backbone_question == "t5":
            # TODO: modify we want the same visual projection and the same embedding
            if config.shared_embedding:
                if embedding_passage is None and embedding_question is None:
                    # check if when we train it will shared the same embedding
                    # and not independant copy
                    self.embedding_encoder_question = nn.Embedding(
                        config.num_embeddings, config.embedding_dim)
                    # we want they share the same parameters chack if it is
                    self.embedding_encoder_passage = self.embedding_encoder_question
                elif embedding_passage is None:
                    self.embedding_encoder_question = embedding_question
                    # we want they share the same parameters chack if it is
                    self.embedding_encoder_passage = self.embedding_encoder_question
                else:
                    self.embedding_encoder_passage = embedding_passage
                    # we want they share the same parameters chack if it is
                    self.embedding_encoder_question = self.embedding_encoder_passage
            else:
                if embedding_passage is None and embedding_question is None:
                    self.embedding_encoder_question = nn.Embedding(
                        config.num_embeddings, config.embedding_dim)
                    self.embedding_encoder_passage = nn.Embedding(
                        config.num_embeddings, config.embedding_dim)
                elif embedding_passage is None:
                    self.embedding_encoder_question = embedding_question
                    self.embedding_encoder_passage = nn.Embedding(
                        config.num_embeddings, config.embedding_dim)
                else:
                    self.embedding_encoder_passage = embedding_passage
                    self.embedding_encoder_question = embedding_question
            self.image_passage_encoder.set_input_embeddings(
                self.embedding_encoder_passage)
            self.image_question_encoder.set_input_embeddings(
                self.embedding_encoder_question)
        # according to the config we add or not a projection
        # We beginn with a simple linear projection
        # We can maybe after create a Projection Head like here
        # https://towardsdatascience.com/simple-implementation-of-openai-clip-model-a-tutorial-ace6ff01d9f2
        if config.use_projection:
            self.image_question_projection = nn.Linear(
                config.embedding_dim_question, config.projection_dim_question)
            self.image_passage_projection = nn.Linear(
                config.embedding_dim_passage, config.projection_dim_passage)
        else:
            self.image_question_projection = None
            self.image_passage_projection = None

        # we want to be able to freeze or not the model with the config

        # init projection weights
        # self.apply(self._init_weights)

    def forward(
        self
    ):
        image_question_output = self.image_question_encoder()
        image_passage_output = self.image_passage_encoder()

        # TODO: get the output
        image_question_embeds = image_question_output
        image_passage_embeds = image_passage_output
        if self.config.use_projection:
            image_question_embeds = self.image_question_projection(
                image_question_embeds)
            image_passage_embeds = self.image_passage_projection(
                image_passage_embeds)
        # normalized features
        # https://github.com/huggingface/transformers/blob/bdd690a74da5283cbc893dfd79e1c7c72ec1bcfa/src/transformers/models/clip/modeling_clip.py#L1038)
        image_question_embeds = self.image_question_embeds / \
            image_question_embeds.norm(dim=-1, keepdim=True)
        image_passage_embeds = self.image_passage_embeds / \
            image_passage_embeds.norm(dim=-1, keepdim=True)

        # cosine similarity

        raise NotImplementedError()

    def train_step(self, batch):
        raise NotImplementedError()

    @torch.no_grad()
    def test_step(self, batch):
        raise NotImplementedError()

    @torch.no_grad()
    def _init_weights(self, module):
        raise NotImplementedError()

    @torch.no_grad()
    def embed_image_passage(self, **kwargs):
        return self.image_passage_encoder(**kwargs)

    @torch.no_grad()
    def embed_image_question(self, **kwargs):
        return self.image_passage_encoder(**kwargs)


def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    # https://github.com/huggingface/transformers/blob/bdd690a74da5283cbc893dfd79e1c7c72ec1bcfa/src/transformers/models/clip/modeling_clip.py#L65

    return nn.functionnal.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    # https://github.com/huggingface/transformers/blob/bdd690a74da5283cbc893dfd79e1c7c72ec1bcfa/src/transformers/models/clip/modeling_clip.py#L69
    image_question_loss = contrastive_loss(similarity)
    image_passage_loss = contrastive_loss(similarity.T)
    return (image_question_loss, image_passage_loss)


# Think about that and define
@dataclass
class CVLEP_output(ModelOutput):
    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            Contrastive loss for image-text similarity.
        logits_per_image:(`torch.FloatTensor` of shape `(image_batch_size, text_batch_size)`):
            The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
            similarity scores.
        logits_per_text:(`torch.FloatTensor` of shape `(text_batch_size, image_batch_size)`):
            The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
            similarity scores.
        text_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of [`CLIPTextModel`].
        image_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The image embeddings obtained by applying the projection layer to the pooled output of [`CLIPVisionModel`].
        text_model_output(`BaseModelOutputWithPooling`):
            The output of the [`CLIPTextModel`].
        vision_model_output(`BaseModelOutputWithPooling`):
            The output of the [`CLIPVisionModel`].
    """

    loss: Optional[torch.FloatTensor] = None
    logits_per_image: torch.FloatTensor = None
    logits_per_text: torch.FloatTensor = None
    text_embeds: torch.FloatTensor = None
    image_embeds: torch.FloatTensor = None
    text_model_output: BaseModelOutputWithPooling = None
    vision_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k] if k not in ["text_model_output",
                                 "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )


class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim,
        dropout
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x
