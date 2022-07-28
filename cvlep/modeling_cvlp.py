# coding: utf-8

from dataclasses import dataclass
import torch
from torch import nn
from transformers import T5Config
from transformers.modeling_outputs import ModelOutput, BaseModelOutputWithPooling
from typing import Optional, Tuple, Any


class CVLEP(nn.Module):
    def __init__(self, config, image_question_encoder, image_passage_encoder):

        super().__init__()
        self.config = config
        self.share_visual_embedding = None
        self.share_embedding = None
        if config.share_vis_embedding:
            self.share_visual_embedding = image_question_encoder.visual_embedding
            image_question_encoder.set_vis_embedding(
                self.share_visual_embedding)
            image_passage_encoder.set_vis_embedding(
                self.share_visual_embedding)
        if config.share_embedding:
            self.share_embedding = image_question_encoder.embed_tokens
            image_question_encoder.set_input_embeddings(
                self.share_embedding)
            image_passage_encoder.set_input_embeddings(
                self.share_embedding)

        self.image_question_encoder = image_question_encoder
        self.image_passage_encoder = image_passage_encoder

    def forward(
        self,
        question_input_ids=None,
        question_attention_mask=None,
        question_vis_inputs=None,
        passage_input_ids=None,
        passage_attention_mask=None,
        passage_vis_inputs=None,
        task="IR",
    ):
        outputs_question = self.image_question_encoder(
            input_ids=question_input_ids,
            attention_mask=question_attention_mask,
            vis_inputs=question_vis_inputs,
            task=task,
            return_pooled_output=True,
            pool_strategy="avg"
        ).pooler_output
        output_passage = self.image_passage_encoder(
            input_ids=passage_input_ids,
            attention_mask=passage_attention_mask,
            vis_inputs=passage_vis_inputs,
            task=task,
            return_pooled_output=True,
            pool_strategy="avg"
        ).pooler_output

        return outputs_question, output_passage

    def to_device(self, input, device):
        if input is None:
            return None
        else:
            return input.to(device)

    def train_step(self, batch):
        device = next(self.parameters()).device

        outputs_question, output_passage = self.forward(
            question_input_ids=self.to_device(
                batch["input_ids_question"], device),
            question_attention_mask=self.to_device(
                batch["attention_mask_question"], device),
            question_vis_inputs=(self.to_device(batch["visual_feats_question"], device), self.to_device(
                batch["question_image_boxes"], device)),
            passage_input_ids=self.to_device(
                batch["input_ids_context"], device),
            passage_attention_mask=self.to_device(
                batch["attention_mask_context"], device),
            passage_vis_inputs=(self.to_device(batch["visual_feats_context"], device), self.to_device(
                batch["context_image_boxes"], device))
        )
        labels = batch['labels'].to(device)
        return outputs_question, output_passage, labels

    @torch.no_grad()
    def embed_image_passage(self, batch):
        device = next(self.parameters()).device

        return self.image_passage_encoder(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            vis_inputs=(batch['vision_features'].to(
                device), batch['boxes'].to(device)),
            task=batch["task"],
            return_pooled_output=True,
            pool_strategy="avg"
        ).pooler_output

    @torch.no_grad()
    def embed_image_question(self, batch):
        device = next(self.parameters()).device

        return self.image_question_encoder(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            vis_inputs=(batch['vision_features'].to(
                device), batch['boxes'].to(device)),
            task=batch["task"],
            return_pooled_output=True,
            pool_strategy="avg"
        ).pooler_output


def compute_loss_like_dpr(model1, model2, data, temperature):
    # fake function to implement after in the model
    log_softmax = nn.LogSoftmax(1)
    loss_fct = nn.NLLLoss(reduction='mean')

    question_embeddings = model1(
        data['input_ids_question'],
        (data['visual_feats_question'], data['question_image_boxes'])
    )
    context_embeddings = model2(
        data['input_ids_context'],
        (data['visual_feats_context'], data['context_image_boxes'])
    )

    similarities = question_embeddings @ context_embeddings.T
    log_probs = log_softmax(similarities)
    loss = loss_fct(log_probs, data['labels'])
    return loss


# for clip loss
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    # https://github.com/huggingface/transformers/blob/bdd690a74da5283cbc893dfd79e1c7c72ec1bcfa/src/transformers/models/clip/modeling_clip.py#L65

    return nn.functionnal.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    # https://github.com/huggingface/transformers/blob/bdd690a74da5283cbc893dfd79e1c7c72ec1bcfa/src/transformers/models/clip/modeling_clip.py#L69
    image_question_loss = contrastive_loss(similarity)
    image_passage_loss = contrastive_loss(similarity.T)
    return (image_question_loss + image_passage_loss) / 2.0