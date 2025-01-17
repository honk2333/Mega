# Assumption: *get_visual_embeddings(image)* gets the visual embeddings of the image.
from multiprocessing import pool
from transformers import BertTokenizer, VisualBertModel
import torch
import torch.nn as nn
import torch.utils.data as data
import os, random, json, logging, sys
import argparse
import numpy as np
from opennre.framework.data_loader import SentenceREDataset
from .base_encoder import BaseEncoder
import math
from torch.nn import functional as F

class VisualBertEncoder(nn.Module):
    def __init__(self, max_length, pretrain_path, blank_padding=True, mask_entity=False):
        """
        Args:
            max_length: max length of sentence
            pretrain_path: path of pretrain model
        """
        super().__init__()
        self.max_length = max_length
        self.blank_padding = blank_padding
        self.word_size = 50
        self.hidden_size = 768*2
        self.pic_feat = 4096
        
        self.mask_entity = mask_entity
        logging.info('Loading BERT pre-trained checkpoint.')
        # self.bert = BertModel.from_pretrained(bert_pretrain_path)
        self.tokenizer = BertTokenizer.from_pretrained(pretrain_path)
        self.linear_pic = nn.Linear(self.pic_feat, 2048)
        self.linear_pool = nn.Linear(self.hidden_size//2, self.hidden_size)
        
        self.model = VisualBertModel.from_pretrained("/home/wanghk/Mega/visualbert-vqa-coco-pre")
    def forward(self, token, att_mask, pos1, pos2, pic, rel):
        """
        Args:
            token: (B, L), index of tokens
            att_mask: (B, L), attention mask (1 for contents and 0 for padding)
            pos1: (B, 1), position of the head entity starter
            pos2: (B, 1), position of the tail entity starter
            pic: (B, N*H), pre-extracted visual object representation for multimodal semantics alignment
            rel: (B, N*L), structural alignment weight between tokens and visual objects
        Return:
            (B, 2H), representations for sentences
        """
        inputs = {}
        token_type_ids=torch.zeros_like(token,dtype=torch.long, device=token.device)
        # print(token.shape)
        inputs.update({
            'input_ids': token,
            'token_type_ids':token_type_ids,
            'attention_mask': att_mask
        })
        # print(pic.shape)
        pic = pic.view(-1, 10, self.pic_feat)
        visual_embeds = self.linear_pic(pic)
        # print(visual_embeds.shape)
        visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long, device=token.device)
        visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float, device=token.device)
        # print(visual_token_type_ids.shape)
        inputs.update(
            {
                "visual_embeds": visual_embeds,
                "visual_token_type_ids": visual_token_type_ids,
                "visual_attention_mask": visual_attention_mask,
            }
        )
        output = self.model(**inputs)
        pooler_output  = output.pooler_output 
        x = self.linear_pool(pooler_output)

        # last_hidden_states = output.last_hidden_state
        # text_embedding = last_hidden_states[:,:token.shape[1],:].transpose(1, 2)
        # vis_embedding = last_hidden_states[:,-10:,:].transpose(1, 2)
        # # print(last_hidden_states.shape)
        # # print(text_embedding.shape, vis_embedding.shape)
        # text_avg = torch.avg_pool1d(text_embedding, kernel_size=text_embedding.shape[-1]).squeeze(-1)  # [batch, 768]
        # vis_avg = torch.avg_pool1d(vis_embedding, kernel_size=vis_embedding.shape[-1]).squeeze(-1)  # [batch, 768]
        # # print(text_avg.shape, vis_avg.shape)
        # # print(text_avg.shape, vis_avg.shape)
        # x = torch.cat([text_avg, vis_avg], dim=-1)
        # print(x.shape)     
        return x

    def tokenize(self, item):
        """
        Args:
            item: data instance containing 'text' / 'token', 'h' and 't'
        Return:
            Name of the relation of the sentence
        """
        # Sentence -> token
        if 'text' in item:
            sentence = item['text']
            is_token = False
        else:
            sentence = item['token']
            is_token = True
        pos_head = item['h']['pos']
        pos_tail = item['t']['pos']

        pos_min = pos_head
        pos_max = pos_tail
        if pos_head[0] > pos_tail[0]:
            pos_min = pos_tail
            pos_max = pos_head
            rev = True
        else:
            rev = False

        if not is_token:
            sent0 = self.tokenizer.tokenize(sentence[:pos_min[0]])
            ent0 = self.tokenizer.tokenize(sentence[pos_min[0]:pos_min[1]])
            sent1 = self.tokenizer.tokenize(sentence[pos_min[1]:pos_max[0]])
            ent1 = self.tokenizer.tokenize(sentence[pos_max[0]:pos_max[1]])
            sent2 = self.tokenizer.tokenize(sentence[pos_max[1]:])
        else:
            sent0 = self.tokenizer.tokenize(' '.join(sentence[:pos_min[0]]))
            ent0 = self.tokenizer.tokenize(' '.join(sentence[pos_min[0]:pos_min[1]]))
            sent1 = self.tokenizer.tokenize(' '.join(sentence[pos_min[1]:pos_max[0]]))
            ent1 = self.tokenizer.tokenize(' '.join(sentence[pos_max[0]:pos_max[1]]))
            sent2 = self.tokenizer.tokenize(' '.join(sentence[pos_max[1]:]))

        if self.mask_entity:
            ent0 = ['[unused4]'] if not rev else ['[unused5]']
            ent1 = ['[unused5]'] if not rev else ['[unused4]']
        else:
            ent0 = ['[unused0]'] + ent0 + ['[unused1]'] if not rev else ['[unused2]'] + ent0 + ['[unused3]']
            ent1 = ['[unused2]'] + ent1 + ['[unused3]'] if not rev else ['[unused0]'] + ent1 + ['[unused1]']

        re_tokens = ['[CLS]'] + sent0 + ent0 + sent1 + ent1 + sent2 + ['[SEP]']
        pos1 = 1 + len(sent0) if not rev else 1 + len(sent0 + ent0 + sent1)
        pos2 = 1 + len(sent0 + ent0 + sent1) if not rev else 1 + len(sent0)
        pos1 = min(self.max_length - 1, pos1)
        pos2 = min(self.max_length - 1, pos2)

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(re_tokens)
        avai_len = len(indexed_tokens)

        # Position
        pos1 = torch.tensor([[pos1]]).long()
        pos2 = torch.tensor([[pos2]]).long()

        # Padding
        if self.blank_padding:
            while len(indexed_tokens) < self.max_length:
                indexed_tokens.append(0)  # 0 is id for [PAD]
            indexed_tokens = indexed_tokens[:self.max_length]
        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0)  # (1, L)

        # Attention mask
        att_mask = torch.zeros(indexed_tokens.size()).long()  # (1, L)
        att_mask[0, :avai_len] = 1

        return indexed_tokens, att_mask, pos1, pos2