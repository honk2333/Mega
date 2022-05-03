# Assumption: *get_visual_embeddings(image)* gets the visual embeddings of the image.
from transformers import BertTokenizer, VisualBertModel
import torch
import torch.utils.data as data
import os, random, json, logging, sys
import argparse
import numpy as np
from opennre.framework.data_loader import SentenceREDataset

tokenizer = BertTokenizer.from_pretrained("/home/data_ti6_c/wanghk/bert_model/bert-base-uncased")
model = VisualBertModel.from_pretrained("uclanlp/visualbert-vqa-coco-pre")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='ours', choices=['none', 'semeval', 'wiki80', 'tacred', 'nyt10', 'ours'],
                    help='Dataset. If not none, the following args can be ignored')

# Hyper-parameters
parser.add_argument('-b', '--batch_size', default=64, type=int,
                    help='Batch size')
parser.add_argument('--lr', default=2e-5, type=float,
                    help='Learning rate')
parser.add_argument('--max_length', default=128
                    , type=int,
                    help='Maximum sentence length')
parser.add_argument('--max_epoch', default=8, type=int,
                    help='Max number of training epochs')
args = parser.parse_args()

root_path = '/home/data_ti4_d/wanghk/MEGA'
sys.path.append(root_path)
args.train_file = os.path.join(root_path, 'benchmark', args.dataset,'txt/{}_train.txt'.format(args.dataset))
args.val_file = os.path.join(root_path, 'benchmark', args.dataset,'txt/{}_val.txt'.format(args.dataset))
args.test_file = os.path.join(root_path, 'benchmark', args.dataset,'txt/{}_test.txt'.format(args.dataset))
args.pic_train_file = os.path.join(root_path, 'benchmark', args.dataset, 'imgSG/train')
args.pic_val_file = os.path.join(root_path, 'benchmark', args.dataset, 'imgSG/val')
args.pic_test_file = os.path.join(root_path, 'benchmark', args.dataset, 'imgSG/test')
args.rel_train_file = os.path.join(root_path, 'benchmark', args.dataset,'rel_{}/train').format(args.rel_num)
args.rel_val_file = os.path.join(root_path, 'benchmark', args.dataset, 'rel_{}/val').format(args.rel_num)
args.rel_test_file = os.path.join(root_path, 'benchmark', args.dataset, 'rel_{}/test').format(args.rel_num)
args.rel2id_file = os.path.join(root_path, 'benchmark', args.dataset, '{}_rel2id.json'.format(args.dataset))

logging.info('Arguments:')
for arg in vars(args):
    logging.info('    {}: {}'.format(arg, getattr(args, arg)))


rel2id = json.load(open(args.rel2id_file))
dataset = SentenceREDataset(text_path=args.text_path, rel_path=args.rel_path, pic_path=args.pic_path,
                                rel2id=rel2id,
                                tokenizer=tokenizer
                            )
data_loader = data.DataLoader(dataset=dataset,
                                batch_size=args.batch_size,
                                shuffle=True,
                                pin_memory=True,
                                num_workers=8,
                                collate_fn=SentenceREDataset.collate_fn)

for epoch in range(args.max_epoch):
    model.train()
    logging.info("=== Epoch %d train ===" % epoch)
    avg_loss = AverageMeter()
    avg_acc = AverageMeter()
    avg_f1 = AverageMeter()
    t = tqdm(self.train_loader, ncols=110)
    for iter, data in enumerate(t):
        if torch.cuda.is_available():
            for i in range(len(data)):
                try:
                    data[i] = data[i].cuda()
                except:
                    pass

        label = data[0]
        args = data[1:]
        logits = self.parallel_model(*args)
        loss = self.criterion(logits, label)
        # loss = (loss-0.02).abs()+0.02
        score, pred = logits.max(-1)  # (B)
        acc = float((pred == label).long().sum()) / label.size(0)
        f1 = metrics.f1_score(pred.cpu(), label.cpu(), average='macro')
        # Log
        avg_loss.update(loss.item(), 1)
        avg_acc.update(acc, 1)
        avg_f1.update(f1, 1)
        t.set_postfix(loss=avg_loss.avg, acc=avg_acc.avg, f1=avg_f1.avg)
        # Optimize
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.optimizer.zero_grad()
        global_step += 1
    # Val
    logging.info("=== Epoch %d val ===" % epoch)
    result = self.eval_model(self.val_loader)
    logging.info('Metric {} current / best: {} / {}'.format(metric, result[metric], best_metric))
    if result[metric] > best_metric:
        logging.info("Best ckpt and saved.")
        folder_path = '/'.join(self.ckpt.split('/')[:-1])
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        torch.save({'state_dict': self.model.state_dict()}, self.ckpt)
        best_metric = result[metric]
logging.info("Best %s on val set: %f" % (metric, best_metric))


inputs = tokenizer("The capital of France is Paris.", return_tensors="pt")
visual_embeds = get_visual_embeddings(image).unsqueeze(0)
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