import torch
from torch import nn, optim
import json
from .data_loader import SentenceRELoader, BagRELoader
from .utils import AverageMeter
from tqdm import tqdm
import os
import numpy as np

#  Custom Loss Function

def reconstruction(assignment, labels):
    """
    reconstruction function(Task specific function)

    Args:
        assignment:(B, n_sbags, n_bags) assignment of superbag
        labels: (B, C, n_bags) labels of bags
    """
    labels = labels.permute(0, 2, 1).contiguous()

    spixel_mean = torch.bmm(assignment, labels) / (assignment.sum(2, keepdim=True) + 1e-16)
    permuted_assignment = assignment.permute(0, 2, 1).contiguous()
    reconstructed_labels = torch.bmm(permuted_assignment, spixel_mean)

    return reconstructed_labels.permute(0, 2, 1).contiguous()

def reconstruact_loss_with_loss_entropy(assignment, labels):
    """
    reconstruction loss(Task specific function)

    Args:
        assignment:(B, n_sbags, n_bags) assignment of superbag
        labels: (B, C, n_bags) labels of bags
    """
    reconstructed_labels = reconstruction(assignment, labels)
    reconstructed_labels = reconstructed_labels / (1e-16 + reconstructed_labels.sum(1, keepdim=True))
    mask = labels > 0
    return -(reconstructed_labels[mask] + 1e-16).log().mean()


def compact_loss(assignment, labels):
    """
    compact loss encourage superbag to have lower spatial variance

    Args:
        assignment: (B, n_sbags, n_bags)
        labels:(B, C, n_bags)
    """
    reconstructed_labels = reconstruction(assignment, labels)
    return nn.functional.mse_loss(reconstructed_labels, labels)

class SuperBagRE(nn.Module):

    def __init__(self, 
                 model,
                 bag_encoder,
                 train_path, 
                 val_path, 
                 test_path,
                 ckpt, 
                 batch_size=32, 
                 max_epoch=100, 
                 lr=0.1, 
                 weight_decay=1e-5, 
                 opt='sgd',
                 bag_size=0,
                 loss_weight=False):
    
        super().__init__()
        self.max_epoch = max_epoch
        self.bag_size = bag_size
        self.bag_encoder = bag_encoder
        # Load data
        if train_path != None:
            self.train_loader = BagRELoader(
                train_path,
                model.rel2id,
                model.bag_encoder.sentence_encoder.tokenize,
                batch_size,
                True,
                bag_size=self.bag_size,
                entpair_as_bag=False)

        if val_path != None:
            self.val_loader = BagRELoader(
                val_path,
                model.rel2id,
                model.bag_encoder.sentence_encoder.tokenize,
                batch_size,
                False,
                bag_size=self.bag_size,
                entpair_as_bag=True)
        
        if test_path != None:
            self.test_loader = BagRELoader(
                test_path,
                model.rel2id,
                model.bag_encoder.sentence_encoder.tokenize,
                batch_size,
                False,
                bag_size=self.bag_size,
                entpair_as_bag=True
            )
        # Model
        self.model = nn.DataParallel(model)
        # Criterion
        if loss_weight:
            self.criterion = nn.CrossEntropyLoss(weight=self.train_loader.dataset.weight)
        else:
            self.criterion = nn.CrossEntropyLoss()
        # Params and optimizer
        params = self.model.parameters()
        self.lr = lr
        if opt == 'sgd':
            self.optimizer = optim.SGD(params, lr, weight_decay=weight_decay)
        elif opt == 'adam':
            self.optimizer = optim.Adam(params, lr, weight_decay=weight_decay)
        elif opt == 'adamw':
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
            raise Exception("Invalid optimizer. Must be 'sgd' or 'adam' or 'bert_adam'.")
        # Cuda
        if torch.cuda.is_available():
            self.cuda()
        # Ckpt
        self.ckpt = ckpt

    def train_model(self, metric='auc', weight=1.0, train_as_bag=True):
        best_metric = 0
        best_epoch = 0
        min_loss = 1
        min_epoch = 0
        if train_as_bag:
            print('=== train unit is bag level===')
        else:
            print('=== train unit is superbag level===')
        for epoch in range(self.max_epoch):
            # Train
            self.train()
            print("=== Epoch %d train ===" % epoch)
            avg_loss = AverageMeter()
            avg_acc = AverageMeter()
            avg_pos_acc = AverageMeter()
            t = tqdm(self.train_loader)
            for iter, data in enumerate(t):
                if torch.cuda.is_available():
                    for i in range(len(data)):
                        try:
                            data[i] = data[i].cuda()
                        except:
                            pass
                label = data[0]
                bag_name = data[1]
                scope = data[2]
                args = data[3:]
                loss1, cluster_label, logits = self.model(label, scope, args, bag_size=self.bag_size, train_as_bag=train_as_bag)
                if not train_as_bag:
                    label = torch.Tensor(cluster_label).long().cuda()
                loss2 = self.criterion(logits, label)
                loss = weight * loss1 + loss2
                score, pred = logits.max(-1) # (D)
                acc = float((pred == label).long().sum()) / label.size(0)
                pos_total = (label != 0).long().sum()
                pos_correct = ((pred == label).long() * (label != 0).long()).sum()
                if pos_total > 0:
                    pos_acc = float(pos_correct) / float(pos_total)
                else:
                    pos_acc = 0

                # Log
                avg_loss.update(loss.item(), 1)
                avg_acc.update(acc, 1)
                avg_pos_acc.update(pos_acc, 1)
                t.set_postfix(loss=avg_loss.avg, acc=avg_acc.avg, pos_acc=avg_pos_acc.avg)
                # Optimize
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            # loss最小的epoch
            if avg_loss.avg < min_loss:
                min_loss = avg_loss.avg
                min_epoch = epoch
                print("Minimum Loss Updated: %.4f" % min_loss)
                
            # Val 
            print("=== Epoch %d val ===" % epoch)
            result, _ = self.eval_model(self.val_loader)
            print("AUC: %.4f" % result['auc'])
            print("Micro F1: %.4f" % (result['micro_f1']))
            print("P@100: %.4f" % result['P100'])
            print("P@200: %.4f" % result['P200'])
            print("P@300: %.4f" % result['P300'])
            if result[metric] > best_metric:
                print("Best ckpt and saved.")
                torch.save({'state_dict': self.model.module.state_dict()}, self.ckpt)
                best_metric = result[metric]
                best_epoch = epoch
            # 连续6个epoch损失函数没有继续下降，则判断为收敛
            if epoch - min_epoch >= 6:
                break
        print("Best %s on val set: %f" % (metric, best_metric))

    def eval_model(self, eval_loader):
        self.model.eval()
        with torch.no_grad():
            t = tqdm(eval_loader)
            pred_result = []
            hits_result = []
            for iter, data in enumerate(t):
                if torch.cuda.is_available():
                    for i in range(len(data)):
                        try:
                            data[i] = data[i].cuda()
                        except:
                            pass
                label = data[0]
                bag_name = data[1]
                scope = data[2]
                args = data[3:]
                _, _, logits = self.model(label, scope, args, bag_size=self.bag_size)
                logits = logits.cpu().numpy()
                for i in range(len(logits)):
                    for relid in range(self.model.module.num_class):
                        if self.model.module.id2rel[relid] != 'NA':
                            pred_result.append({
                                'entpair': bag_name[i][:2], 
                                'relation': self.model.module.id2rel[relid], 
                                'score': logits[i][relid]
                            })
                    
                    hits_result.append({
                        'entpair': bag_name[i][:2],
                        'logits': logits[i]
                    })
            result = eval_loader.dataset.eval(pred_result)
            hits_result = eval_loader.dataset.eval_hits(hits_result, mode="hits100")
        return result, hits_result

    def load_state_dict(self, state_dict):
        self.model.module.load_state_dict(state_dict)
