import os, logging, json
from tqdm import tqdm
import torch
from torch import nn, optim
from .data_loader import SentenceRELoader
from .utils import AverageMeter
import numpy as np
from sklearn import metrics
from torch.utils.tensorboard import SummaryWriter   

class SentenceRE(nn.Module):

    def __init__(self,
                 model,
                 train_path,
                 train_rel_path,
                 train_pic_path,
                 val_path,
                 val_rel_path,
                 val_pic_path,
                 test_path,
                 test_rel_path,
                 test_pic_path,
                 ckpt,
                 batch_size=64,
                 max_epoch=100,
                 lr=0.1,
                 weight_decay=1e-5,
                 warmup_step=300,
                 opt='sgd',
                 exp_name ='default'
                 ):

        super().__init__()
        self.max_epoch = max_epoch
        # Load data
        if train_path != None:
            self.train_loader = SentenceRELoader(
                train_path,
                train_rel_path,
                train_pic_path,
                model.rel2id,
                model.sentence_encoder.tokenize,
                batch_size,
                True)

        if val_path != None:
            self.val_loader = SentenceRELoader(
                val_path,
                val_rel_path,
                val_pic_path,
                model.rel2id,
                model.sentence_encoder.tokenize,
                batch_size,
                False)

        if test_path != None:
            self.test_loader = SentenceRELoader(
                test_path,
                test_rel_path,
                test_pic_path,
                model.rel2id,
                model.sentence_encoder.tokenize,
                batch_size,
                False
            )
        # Model
        self.model = model
        self.parallel_model = nn.DataParallel(self.model)
        # Criterion
        self.criterion = nn.CrossEntropyLoss()
        # Params and optimizer
        params = self.parameters()
        self.lr = lr
        if opt == 'sgd':
            self.optimizer = optim.SGD(params, lr, weight_decay=weight_decay)
        elif opt == 'adam':
            self.optimizer = optim.Adam(params, lr, weight_decay=weight_decay)
        elif opt == 'adamw':  # Optimizer for BERT
            from transformers import AdamW
            params = list(self.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            grouped_params = [
                {
                    'params': [p for n, p in params if not any(nd in n for nd in no_decay)],
                    'weight_decay': 0.01,
                    'lr': lr,
                    'ori_lr': lr
                },
                {
                    'params': [p for n, p in params if any(nd in n for nd in no_decay)],
                    'weight_decay': 0.0,
                    'lr': lr,
                    'ori_lr': lr
                }
            ]
            self.optimizer = AdamW(grouped_params, correct_bias=False)
        else:
            raise Exception("Invalid optimizer. Must be 'sgd' or 'adam' or 'adamw'.")
        # Warmup
        if warmup_step > 0:
            from transformers import get_linear_schedule_with_warmup
            training_steps = self.train_loader.dataset.__len__() // batch_size * self.max_epoch
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_step,
                                                             num_training_steps=training_steps)
        else:
            self.scheduler = None
        # Cuda
        if torch.cuda.is_available():
            self.cuda()
        # Ckpt
        self.ckpt = ckpt
        self.writer = SummaryWriter('/home/wanghk/Mega/ckpt/log/'+ exp_name)

    def train_model(self, metric='acc'):
        best_metric = 0
        global_step = 0
        # loader = [self.train_loader, self.val_loader]
        for epoch in range(self.max_epoch):
            self.train()
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
                self.writer.add_scalar('train_loss', loss.item(), iter)
                self.writer.add_scalar('train_acc', acc, iter)
                self.writer.add_scalar('train_f1', f1, iter)
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
            self.writer.add_scalar('valid_acc', result['acc'], epoch)
            self.writer.add_scalar('valid_f1', result['micro_f1'], epoch)

            logging.info('Metric {} current / best: {} / {}'.format(metric, result[metric], best_metric))
            if result[metric] > best_metric:
                logging.info("Best ckpt and saved.")
                folder_path = '/'.join(self.ckpt.split('/')[:-1])
                if not os.path.exists(folder_path):
                    os.mkdir(folder_path)
                torch.save({'state_dict': self.model.state_dict()}, self.ckpt)
                best_metric = result[metric]


            # 测试集指标
            result = self.eval_model(self.test_loader)
            # Print the result
            logging.info('Test set results:\n')
            logging.info('Accuracy: {}\n'.format(result['acc']))
            logging.info('Micro precision: {}\n'.format(result['micro_p']))
            logging.info('Micro recall: {}\n'.format(result['micro_r']))
            logging.info('Micro F1: {}'.format(result['micro_f1']))

        logging.info("Best %s on val set: %f" % (metric, best_metric))

    def eval_model(self, eval_loader):
        self.eval()
        avg_acc = AverageMeter()
        avg_loss = AverageMeter()
        pred_result = []
        with torch.no_grad():
            t = tqdm(eval_loader, ncols=110)
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
                score, pred = logits.max(-1)  # (B)
                # Save result
                for i in range(pred.size(0)):
                    pred_result.append(pred[i].item())
                # Log
                acc = float((pred == label).long().sum()) / label.size(0)
                avg_acc.update(acc, pred.size(0))
                avg_loss.update(loss.item(), 1)
                t.set_postfix(loss=avg_loss.avg,acc=avg_acc.avg)
        result = eval_loader.dataset.eval(pred_result)

        return result

    def load_state_dict(self, state_dict, strict=True):
        self.model.load_state_dict(state_dict, strict)
