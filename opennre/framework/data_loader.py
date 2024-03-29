import torch
import torch.utils.data as data
import os, random, json, logging
import numpy as np
import sklearn.metrics
from .utils import getArrayFromFile
import json

class SentenceREDataset(data.Dataset):
    """
    Sentence-level relation extraction dataset
    """
    def __init__(self, path, rel2id, tokenizer, kwargs):
        """
        Args:
            path: path of the input file
            rel2id: dictionary of relation->id mapping
            tokenizer: function of tokenizing
        """
        super().__init__()
        self.path = path
        self.tokenizer = tokenizer
        self.rel2id = rel2id
        self.kwargs = kwargs

        # Load the file
        f = open(path)
        self.data = []
        for line in f.readlines():
            line = line.rstrip()
            if len(line) > 0:
                self.data.append(eval(line))
        f.close()
        logging.info("Loaded sentence RE dataset {} with {} lines and {} relations.".format(path, len(self.data), len(self.rel2id)))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        seq = list(self.tokenizer(item, **self.kwargs))
        res = [self.rel2id[item['relation']]] + seq
        return [self.rel2id[item['relation']]] + seq # label, seq1, seq2, ...
    
    def collate_fn(data):
        data = list(zip(*data))
        labels = data[0]
        seqs = data[1:]
        batch_labels = torch.tensor(labels).long() # (B)
        batch_seqs = []
        for seq in seqs:
            batch_seqs.append(torch.cat(seq, 0)) # (B, L)
        return [batch_labels] + batch_seqs
    
    def eval(self, pred_result, use_name=False):
        """
        Args:
            pred_result: a list of predicted label (id)
                Make sure that the `shuffle` param is set to `False` when getting the loader.
            use_name: if True, `pred_result` contains predicted relation names instead of ids
        Return:
            {'acc': xx}
        """
        correct = 0
        total = len(self.data)
        correct_positive = 0
        pred_positive = 0
        gold_positive = 0
        neg = -1
        for name in ['NA', 'na', 'no_relation', 'Other', 'Others']:
            if name in self.rel2id:
                if use_name:
                    neg = name
                else:
                    neg = self.rel2id[name]
                break
        for i in range(total):
            if use_name:
                golden = self.data[i]['relation']
            else:
                golden = self.rel2id[self.data[i]['relation']]
            if golden == pred_result[i]:
                correct += 1
                if golden != neg:
                    correct_positive += 1
            if golden != neg:
                gold_positive +=1
            if pred_result[i] != neg:
                pred_positive += 1
        acc = float(correct) / float(total)
        try:
            micro_p = float(correct_positive) / float(pred_positive)
        except:
            micro_p = 0
        try:
            micro_r = float(correct_positive) / float(gold_positive)
        except:
            micro_r = 0
        try:
            micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r)
        except:
            micro_f1 = 0
        result = {'acc': acc, 'micro_p': micro_p, 'micro_r': micro_r, 'micro_f1': micro_f1}
        logging.info('Evaluation result: {}.'.format(result))
        return result
    
def SentenceRELoader(path, rel2id, tokenizer, batch_size, 
        shuffle, num_workers=8, collate_fn=SentenceREDataset.collate_fn, **kwargs):
    dataset = SentenceREDataset(path = path, rel2id = rel2id, tokenizer = tokenizer, kwargs=kwargs)
    data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=collate_fn)
    return data_loader

class BagREDataset(data.Dataset):
    """
    Bag-level relation extraction dataset. Note that relation of NA should be named as 'NA'.
    """
    def __init__(self, path, rel2id, tokenizer, entpair_as_bag=False, bag_size=0, mode=None):
        """
        Args:
            path: path of the input file
            rel2id: dictionary of relation->id mapping
            tokenizer: function of tokenizing
            entpair_as_bag: if True, bags are constructed based on same
                entity pairs instead of same relation facts (ignoring 
                relation labels)
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.rel2id = rel2id
        self.entpair_as_bag = entpair_as_bag
        self.bag_size = bag_size

        # Load the file
        f = open(path)
        self.data = []
        for line in f:
            line = line.rstrip()
            if len(line) > 0:
                self.data.append(eval(line))
        f.close()

        # Construct bag-level dataset (a bag contains instances sharing the same relation fact)
        if mode == None:
            self.weight = np.ones((len(self.rel2id)), dtype=np.float32)
            self.bag_scope = []
            self.name2id = {}
            self.bag_name = []
            self.facts = {}
            for idx, item in enumerate(self.data):
                fact = (item['h']['id'], item['t']['id'], item['relation'])
                if item['relation'] != 'NA':
                    self.facts[fact] = 1
                if entpair_as_bag:
                    name = (item['h']['id'], item['t']['id'])
                else:
                    name = fact
                if name not in self.name2id:
                    self.name2id[name] = len(self.name2id)
                    self.bag_scope.append([])
                    self.bag_name.append(name)
                self.bag_scope[self.name2id[name]].append(idx)
                self.weight[self.rel2id[item['relation']]] += 1.0
            self.weight = 1.0 / (self.weight ** 0.05)
            self.weight = torch.from_numpy(self.weight)
        else:
            pass
  
    def __len__(self):
        return len(self.bag_scope)

    def __getitem__(self, index):
        bag = self.bag_scope[index]
        if self.bag_size > 0:
            resize_bag = []
            if self.bag_size <= len(bag):
                resize_bag = random.sample(bag, self.bag_size)
            # 前：如果bag_size > bag的大小，则进行重采样
            # 后：如果bag_size > bag的大小，则使用整包
            else:
                resize_bag = bag + list(np.random.choice(bag, self.bag_size - len(bag)))
            bag = resize_bag
        max_bag = 2000
        if len(bag) > max_bag:
            bag = random.sample(bag, max_bag)
        seqs = None
        rel = self.rel2id[self.data[bag[0]]['relation']]
        for sent_id in bag:
            item = self.data[sent_id]
            seq = list(self.tokenizer(item))
            if seqs is None:
                seqs = []
                for i in range(len(seq)):
                    seqs.append([])
            for i in range(len(seq)):
                seqs[i].append(seq[i])
        for i in range(len(seqs)):
            seqs[i] = torch.cat(seqs[i], 0) # (n, L), n is the size of bag
        return [rel, self.bag_name[index], len(bag)] + seqs
  
    def collate_fn(data):
        data = list(zip(*data))
        label, bag_name, count = data[:3]
        seqs = data[3:]
        for i in range(len(seqs)):
            seqs[i] = torch.cat(seqs[i], 0) # (sumn, L)
            seqs[i] = seqs[i].expand((torch.cuda.device_count() if torch.cuda.device_count() > 0 else 1, ) + seqs[i].size())
        scope = [] # (B, 2)
        start = 0
        for c in count:
            scope.append((start, start + c))
            start += c
        assert(start == seqs[0].size(1))
        scope = torch.tensor(scope).long()
        label = torch.tensor(label).long() # (B)
        return [label, bag_name, scope] + seqs

    def collate_bag_size_fn(data):
        data = list(zip(*data))
        label, bag_name, count = data[:3]
        seqs = data[3:]
        for i in range(len(seqs)):
            seqs[i] = torch.stack(seqs[i], 0) # (batch, bag, L)
        scope = [] # (B, 2)
        start = 0
        for c in count:
            scope.append((start, start + c))
            start += c
        label = torch.tensor(label).long() # (B)
        return [label, bag_name, scope] + seqs

    def eval(self, pred_result, hits=False):
        """
        Args:
            pred_result: a list with dict {'entpair': (head_id, tail_id), 'relation': rel, 'score': score}.
                Note that relation of NA should be excluded.
        Return:
            {'prec': narray[...], 'rec': narray[...], 'mean_prec': xx, 'f1': xx, 'auc': xx}
                prec (precision) and rec (recall) are in micro style.
                prec (precision) and rec (recall) are sorted in the decreasing order of the score.
                f1 is the max f1 score of those precison-recall points
        """
        sorted_pred_result = sorted(pred_result, key=lambda x: x['score'], reverse=True)
        prec = []
        rec = []
        correct = 0
        total = len(self.facts)
        y_true = []
        y_logits = []
        for i, item in enumerate(sorted_pred_result):
            if (item['entpair'][0], item['entpair'][1], item['relation']) in self.facts:
                correct += 1
                y_true.append(1)
            else:
                y_true.append(0)
            y_logits.append(item['score'])
            prec.append(float(correct) / float(i + 1))
            rec.append(float(correct) / float(total))
        np_prec = np.array(prec)
        np_rec = np.array(rec) 
        f1 = (2 * np_prec * np_rec / (np_prec + np_rec + 1e-20)).max()
        mean_prec = np_prec.mean()
        auc = sklearn.metrics.auc(x=rec, y=prec)
        return {'micro_p': np_prec, 'micro_r': np_rec, 'micro_p_mean': mean_prec, 'micro_f1': f1, 'auc': auc, 'P100': prec[99], 'P200': prec[199], 'P300': prec[299], 'y_true': y_true, 'y_logits': y_logits}

    
    def eval_hits(self, pred_result, mode="hits100"):
        
        filename = ''
        if mode == 'hits100':
            filename = 'benchmark/nyt10-aug/rel100.txt'
        elif mode == 'hits200':
            filename = 'benchmark/nyt10-aug/rel200.txt'
        # 获取示例数量少于100、200、300的关系名称
        fewrel = getArrayFromFile(filename)
        
        ss = 0
        ss10 = 0
        ss15 = 0
        ss20 = 0

        ss_rel = {}
        ss10_rel = {}
        ss15_rel = {}
        ss20_rel = {}

        for i, item in enumerate(pred_result):
            entitypair = item['entpair']
            scores = item['logits']
            relation = ''
            rel_id = 0
            score = 0
            for rel in self.rel2id:
                # 找到该实体对的label
                if (entitypair[0], entitypair[1], rel) in self.facts:
                    relation = rel
                    rel_id = self.rel2id[rel]
                    score = scores[rel_id]
                    break
            # 该实体对对应的是长尾关系
            if relation in fewrel:
                ss += 1
                mx = 0
                # 获取比gold label评分高的示例
                for sc in scores:
                    if sc > score:
                        mx += 1
                if not relation in ss_rel:
                    ss_rel[relation] = 0
                    ss10_rel[relation] = 0
                    ss15_rel[relation] = 0
                    ss20_rel[relation] = 0
                ss_rel[relation] += 1.0
                if mx < 10:
                    ss10 += 1.0
                    ss10_rel[relation] += 1.0
                if mx < 15:
                    ss15 += 1.0
                    ss15_rel[relation] += 1.0
                if mx < 20:
                    ss20 += 1.0
                    ss20_rel[relation] += 1.0

        return {
            'micro': {
                'H10': ss10 / ss, 
                'H15': ss15 / ss, 
                'H20': ss20 / ss
                }, 
            'macro': {
                'H10': np.array([ss10_rel[i]/ss_rel[i]  for i in ss_rel]).mean(), 
                'H15': np.array([ss15_rel[i]/ss_rel[i]  for i in ss_rel]).mean(), 
                'H20': np.array([ss20_rel[i]/ss_rel[i]  for i in ss_rel]).mean()
                }
            }
    def predict_case(self, batch_idx, bag_names, final_assignments, prefix=''):
        one_batch_sents = []
        for (idx, name) in enumerate(bag_names):
            bag_id = self.name2id[name]
            bag_scope = self.bag_scope[bag_id]
            bag_sent = [self.data[i] for i in bag_scope]
            for sen in bag_sent:
                sen['assignment'] = final_assignments[idx]
            one_batch_sents += bag_sent
        f = open('./predict_result/' + prefix + '_batch' + str(batch_idx) + '.txt', 'w+')
        f.write('\n'.join(json.dumps(i) for i in one_batch_sents))
        f.close()
    
    def predict_case_LT(self, batch_idx, bag_names, rel_num, pred_result, prefix=''):
        one_batch_sents = []
        hits10 = 0
        hits15 = 0
        hits20 = 0
        for (idx, name) in enumerate(bag_names):
            bag_id = self.name2id[name]
            bag_scope = self.bag_scope[bag_id]
            bag_sent = [self.data[i] for i in bag_scope]
            results = pred_result[idx * rel_num: (idx + 1) * rel_num - 1]
            results = sorted(results, key=lambda x:x['score'], reverse=True)
            for sen in bag_sent:
                sen['top10'] = [res['relation'] for res in results[:10]]
                sen['top15'] = [res['relation'] for res in results[:15]]
                sen['top20'] = [res['relation'] for res in results[:20]]
                if sen['relation'] in sen['top10']:
                   hits10 += 1
                   hits15 += 1
                   hits20 += 1
                elif sen['relation'] in sen['top15']:
                    hits15 += 1
                    hits20 += 1
                elif sen['relation'] in sen['top20']:
                    hits20 += 1
            one_batch_sents += bag_sent
        f = open('./predict_result_lt/' + prefix + '_batch' + str(batch_idx) + '.txt', 'w+')
        f.write('\n'.join(json.dumps(i) for i in one_batch_sents))
        f.close()
        print('batch' + str(batch_idx) + 'Hits@K:')
        print('Hits10:' + str(hits10))
        print('Hits15:' + str(hits15))
        print('Hits20:' + str(hits20))
        


def BagRELoader(path, rel2id, tokenizer, batch_size, 
        shuffle, entpair_as_bag=False, bag_size=0, num_workers=8, 
        collate_fn=BagREDataset.collate_fn):
    if bag_size == 0:
        collate_fn = BagREDataset.collate_fn
    else:
        collate_fn = BagREDataset.collate_bag_size_fn
    dataset = BagREDataset(path, rel2id, tokenizer, entpair_as_bag=entpair_as_bag, bag_size=bag_size)
    data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=False,
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn)
    return data_loader
