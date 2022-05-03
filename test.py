# # coding:utf-8
# import sys
# import os

# sys.path.append('./')
# import torch
# import numpy as np
# import json
# import opennre.model,opennre.framework,opennre.encoder

# import argparse
# import logging

# if __name__ == '__main__':
#     text_path = '/home/data_ti4_d/wanghk/MEGA/benchmark/ours/txt/ours_test.txt'
#     f = open(text_path, encoding='UTF-8')
#     data = []
#     f_lines = f.readlines()
#     for i1 in range(len(f_lines)):
#         line = f_lines[i1].rstrip()
#         if len(line) > 0:
#             dic1 = eval(line)
#             # print(dic1)
#             data.append(dic1)
#     f.close()
#     print(data[:5])
    
# Assumption: *get_visual_embeddings(image)* gets the visual embeddings of the image.
from transformers import BertTokenizer, VisualBertModel,BertModel
import torch

tokenizer = BertTokenizer.from_pretrained('/home/data_ti6_c/wanghk/bert_model/bert-base-uncased')
model = VisualBertModel.from_pretrained("/home/wanghk/Mega/visualbert-vqa-coco-pre")
bert = BertModel.from_pretrained('/home/data_ti6_c/wanghk/bert_model/bert-base-uncased')
inputs = tokenizer("The capital of France is Paris.", return_tensors="pt")
print(inputs)
out = bert(**inputs)
print(out)
print(out.size())

visual_embeds = torch.zeros(10, 4096).unsqueeze(0)
visual_embeds = torch.nn.Linear(4096,2048)(visual_embeds)
print(visual_embeds.shape)
# visual_embeds = get_visual_embeddings('/home/data_ti4_d/wanghk/MEGA/benchmark/ours/img_org/test/twitter_stream_2019_04_30_2_3_12.jpg').unsqueeze(0)
# print(visual_embeds.shape)
visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)

inputs.update(
    {
        "visual_embeds": visual_embeds,
        "visual_token_type_ids": visual_token_type_ids,
        "visual_attention_mask": visual_attention_mask,
    }
)

outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state
text_embedding = last_hidden_states[:,1:8,:].transpose(1, 2)
vis_embedding = last_hidden_states[:,-10:,:].transpose(1, 2)
print(last_hidden_states.shape)
print(text_embedding.shape, vis_embedding.shape)
text_avg = torch.avg_pool1d(text_embedding, kernel_size=text_embedding.shape[-1]).squeeze(-1)  # [batch, 768]
vis_avg = torch.avg_pool1d(vis_embedding, kernel_size=vis_embedding.shape[-1]).squeeze(-1)  # [batch, 768]
print(text_avg.shape, vis_avg.shape)
print(text_avg.shape, vis_avg.shape)
x = torch.cat([text_avg, vis_avg], dim=-1)
print(x.shape)    
