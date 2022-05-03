import pickle
import sys
import os
import matplotlib.pyplot as plt
sys.path.append('../')
import torch
import numpy as np
import json
import opennre.model,opennre.framework,opennre.encoder

import argparse
import logging


sentence_encoder = opennre.encoder.BERTEntityEncoder(
        max_length=128,
        pretrain_path='/home/data_ti6_c/wanghk/bert_model/bert-base-uncased',
        mask_entity=False
)
cache = pickle.load(open('./att_map.pkl','rb'))
print(list(cache.keys()))
attention_maps = cache['BERTEntityEncoder.att']
print(len(attention_maps))
# print(attention_maps[0].shape)
def visualize_head(att_map, id):
    text_data = []
    text_path = '/home/data_ti4_d/wanghk/MEGA/benchmark/ours/txt/ours_test.txt'
    with open(text_path, encoding='UTF-8') as f:
        f_lines = f.readlines()
        for i1 in range(len(f_lines)):
            line = f_lines[i1].rstrip()
            if len(line) > 0:
                dic1 = eval(line)
                text_data.append(dic1)
    item = text_data[id]
    print(item)
    # Sentence -> token
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
    sent0 = sentence_encoder.tokenizer.tokenize(' '.join(sentence[:pos_min[0]]))
    ent0 = sentence_encoder.tokenizer.tokenize(' '.join(sentence[pos_min[0]:pos_min[1]]))
    sent1 = sentence_encoder.tokenizer.tokenize(' '.join(sentence[pos_min[1]:pos_max[0]]))
    ent1 = sentence_encoder.tokenizer.tokenize(' '.join(sentence[pos_max[0]:pos_max[1]]))
    sent2 = sentence_encoder.tokenizer.tokenize(' '.join(sentence[pos_max[1]:]))

    ent0 = ['[unused0]'] + ent0 + ['[unused1]'] if not rev else ['[unused2]'] + ent0 + ['[unused3]']
    ent1 = ['[unused2]'] + ent1 + ['[unused3]'] if not rev else ['[unused0]'] + ent1 + ['[unused1]']
    re_tokens = ['[CLS]'] + sent0 + ent0 + sent1 + ent1 + sent2 + ['[SEP]']
    
    img_id = item['img_id']
    pic_path = '/home/data_ti4_d/wanghk/MEGA/benchmark/ours/imgSG/test'
    with open(os.path.join(pic_path,img_id),'r',encoding='utf-8') as f:
        line_list = f.readlines()
        objects = line_list[1].strip().split('\t')
    
    
    print(re_tokens)
    print(objects)
    att_map = att_map[0][:len(objects),:len(re_tokens)]
    plt.figure(dpi=300)
    fig, ax = plt.subplots()
    im = ax.imshow(att_map)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(re_tokens)))
    ax.set_yticks(np.arange(len(objects)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(re_tokens)
    ax.set_yticklabels(objects)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(att_map.shape[0]):
        for j in range(att_map.shape[1]):
            text = ax.text(j, i, str(att_map[i, j])[:5],
                        ha="center", va="center", color="w", size=3)
    fig.tight_layout()
    plt.savefig('/home/wanghk/Mega/ckpt/attention_map_'+str(id)+'.png',dpi=300)
    # plt.show()
id = 2
visualize_head(attention_maps[id], id=id)