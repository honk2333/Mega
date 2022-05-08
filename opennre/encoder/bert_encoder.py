import imp
import logging
from xml.dom import HierarchyRequestErr
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from .base_encoder import BaseEncoder
import math
from torch.nn import functional as F

import sys
import os
sys.path.append('./opennre/')
from models.layers.layer_norm import LayerNorm
from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.position_wise_feed_forward import PositionwiseFeedForward
from models.blocks.encoder_layer import EncoderLayer

from visualizer import get_local
class BERTEncoder(nn.Module):
    def __init__(self, max_length, pretrain_path, blank_padding=True, mask_entity=False):
        """
        Args:
            max_length: max length of sentence
            pretrain_path: path of pretrain model
        """
        super().__init__()
        self.max_length = max_length
        self.blank_padding = blank_padding
        self.hidden_size = 768
        self.pic_feat = 1024
        self.mask_entity = mask_entity
        logging.info('Loading BERT pre-trained checkpoint.')
        self.bert = BertModel.from_pretrained(pretrain_path)
        self.tokenizer = BertTokenizer.from_pretrained(pretrain_path)
        # self.word_embedding = BaseEncoder.word_embedding(self.num_token, self.word_size)

    def forward(self, token, att_mask):
        """
        Args:
            token: (B, L), index of tokens
            att_mask: (B, L), attention mask (1 for contents and 0 for padding)
        Return:
            (B, H), representations for sentences
        """
        _, x = self.bert(token, attention_mask=att_mask)

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

        return indexed_tokens, att_mask



class BERTEntityEncoder(nn.Module):
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
        self.hidden_size = 1536
        self.pic_feat = 2048
        self.obj_num = 10
        self.mask_entity = mask_entity
        logging.info('Loading BERT pre-trained checkpoint.')
        self.bert = BertModel.from_pretrained(pretrain_path)
        self.tokenizer = BertTokenizer.from_pretrained(pretrain_path)
        self.linear_pic = nn.Linear(self.pic_feat, self.hidden_size)
        self.linear_hidden = nn.Linear(self.hidden_size // 2, self.hidden_size)
        # guided-attention
        self.linear_q = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_k = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_v = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_final1 = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.linear_final = nn.Linear(self.hidden_size * 3, self.hidden_size)
        self.linear_merge = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, self.hidden_size)

        self.linear_q2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_k2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_v2 = nn.Linear(self.hidden_size, self.hidden_size)

        self.linear_q3 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_k3 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_v3 = nn.Linear(self.hidden_size, self.hidden_size)


        self.EncoderLayer1 = EncoderLayer(d_model=self.hidden_size,
                                        ffn_hidden=self.hidden_size,
                                        n_head=8,
                                        drop_prob=0.1)
        self.EncoderLayer2 = EncoderLayer(d_model=self.hidden_size,
                                        ffn_hidden=self.hidden_size,
                                        n_head=8,
                                        drop_prob=0.1)
        self.EncoderLayer3 = EncoderLayer(d_model=self.hidden_size,
                                        ffn_hidden=self.hidden_size,
                                        n_head=8,
                                        drop_prob=0.1)
        self.EncoderLayer4 = EncoderLayer(d_model=self.hidden_size,
                                        ffn_hidden=self.hidden_size,
                                        n_head=8,
                                        drop_prob=0.1)
        # self.attention = MultiHeadAttention(d_model=self.hidden_size, n_head=8)
        # self.norm1 = LayerNorm(d_model=self.hidden_size)
        # self.dropout1 = nn.Dropout(p=0.1)

        # self.ffn = PositionwiseFeedForward(d_model=self.hidden_size, hidden=self.hidden_size, drop_prob=0.1)
        # self.norm2 = LayerNorm(d_model=self.hidden_size)
        # self.dropout2 = nn.Dropout(p=0.1)

        # self.attention2 = MultiHeadAttention(d_model=self.hidden_size, n_head=8)
        # self.norm3 = LayerNorm(d_model=self.hidden_size)
        # self.dropout3 = nn.Dropout(p=0.1)

        # self.ffn2 = PositionwiseFeedForward(d_model=self.hidden_size, hidden=self.hidden_size, drop_prob=0.1)
        # self.norm4 = LayerNorm(d_model=self.hidden_size)
        # self.dropout4 = nn.Dropout(p=0.1)


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
        token_type_ids=torch.zeros_like(token,dtype=torch.long, device=token.device)
        out = self.bert(input_ids=token, attention_mask=att_mask,token_type_ids=token_type_ids)
        hidden = out.last_hidden_state  
        # print(type(hidden), type(_))
        # Get entity start hidden state
        onehot_head = torch.zeros(hidden.size()[:2]).float().to(hidden.device)  # (B, L)
        onehot_tail = torch.zeros(hidden.size()[:2]).float().to(hidden.device)  # (B, L)
        onehot_head = onehot_head.scatter_(1, pos1, 1)
        onehot_tail = onehot_tail.scatter_(1, pos2, 1)
        head_hidden = (onehot_head.unsqueeze(2) * hidden).sum(1)  # (B, H)
        tail_hidden = (onehot_tail.unsqueeze(2) * hidden).sum(1)  # (B, H)
        x = torch.cat([head_hidden, tail_hidden], -1)  # (B, 2H)
        hidden_rel = self.linear_hidden(hidden)

        # visual feature

        # pic = pic.view(-1, 10, self.pic_feat)
        pic = pic.view(-1, self.obj_num, self.pic_feat)
        pic = self.linear_pic(pic)

       
        # # semantics alignment by attention
        # x_k = self.linear_k(hidden_rel)
        # x_v = self.linear_v(hidden_rel)
        # pic_q = self.linear_q(pic)
        # pic_att = torch.sigmoid(self.att(pic_q, x_k, x_v))

        # # structural alignment and the combination of semantic graph alignment
        # rel = rel.view(-1, 10, self.max_length)
        # pic_rel = torch.matmul(rel, hidden_rel)
        # pic_out = torch.cat([pic, pic_rel], dim=-1)
        # pic_out = self.linear_final1(pic_out)
        # pic_out = torch.sum(pic_out, dim=1)


        # add
        # x = hidden[:,1:-1,:].transpose(1,2)
        # x = torch.avg_pool1d(x, kernel_size=x.shape[-1]).squeeze(-1)
        # x = self.linear_hidden(x)

        # pic_k = self.linear_k2(pic)
        # pic_v = self.linear_v2(pic)
        # head_hidden_q = self.linear_q2(head_hidden)
        # head_hidden_att = torch.sigmoid(self.att(head_hidden_q,pic_k,pic_v))

        # pic_k = self.linear_k2(pic)
        # pic_v = self.linear_v2(pic)
        # tail_hidden_q = self.linear_q2(tail_hidden)
        # tail_hidden_att = torch.sigmoid(self.att(tail_hidden_q,pic_k,pic_v))


        att_mask = att_mask.unsqueeze(1).repeat(1,self.obj_num,1)
        pic_att = self.EncoderLayer1(x=pic, y=hidden_rel, s_mask=att_mask.unsqueeze(1))
        hidden_rel_att = self.EncoderLayer2(x=hidden_rel, y=pic, s_mask=att_mask.transpose(1,2).unsqueeze(1))


        # hidden_rel_att = hidden_rel
        hidden_rel_att = hidden_rel_att.transpose(1,2)
        hidden_rel = torch.avg_pool1d(hidden_rel_att, kernel_size=hidden_rel_att.shape[-1]).squeeze(-1) 
        pic_att = pic.transpose(1,2)
        pic_out = torch.avg_pool1d(pic_att, kernel_size=pic_att.shape[-1]).squeeze(-1) 


        # fusion and final output
        x = torch.cat([x, pic_out, hidden_rel], dim=-1)
        x = self.linear_final(x)

        return x

    def tokenize(self, item, objs):
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
    
    @get_local('att_map')
    def att(self, query, key, value):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)
        att_map = F.softmax(scores, dim=-1)
        return torch.matmul(att_map, value)
