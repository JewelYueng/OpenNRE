# coding:utf-8
import sys, json
import torch
import os
import numpy as np
import opennre
from opennre import encoder, model, framework
import sys
import os
import argparse
import logging

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', default='', 
        help='Checkpoint name')
parser.add_argument('--only_test', action='store_true', 
        help='Only run test')
# BERT的参数设置
parser.add_argument('--mask_entity', action="store_true", help="Mask entity mentions")
parser.add_argument('--pretrain_path', default='bert-base-uncased', help=' Pretrained ckpt path')

# Data
parser.add_argument('--metric', default='auc', choices=['micro_f1', 'auc'],
        help='Metric for picking up best checkpoint')
parser.add_argument('--dataset', default='none', choices=['none', 'wiki_distant', 'nyt10'],
        help='Dataset. If not none, the following args can be ignored')
parser.add_argument('--train_file', default='', type=str,
        help='Training data file')
parser.add_argument('--val_file', default='', type=str,
        help='Validation data file')
parser.add_argument('--test_file', default='', type=str,
        help='Test data file')
parser.add_argument('--rel2id_file', default='', type=str,
        help='Relation to ID file')

# Bag related
parser.add_argument('--bag_size', type=int, default=0,
        help='Fixed bag size. If set to 0, use original bag sizes')

# Hyper-parameters
parser.add_argument('--batch_size', default=160, type=int,
        help='Batch size')
parser.add_argument('--lr', default=0.1, type=float,
        help='Learning rate')
parser.add_argument('--optim', default='sgd', type=str,
        help='Optimizer')
parser.add_argument('--weight_decay', default=1e-5, type=float,
        help='Weight decay')
parser.add_argument('--max_length', default=120, type=int,
        help='Maximum sentence length')
parser.add_argument('--max_epoch', default=100, type=int,
        help='Max number of training epochs')

args = parser.parse_args()

# Some basic settings
root_path = '.'
sys.path.append(root_path)
if not os.path.exists('ckpt'):
    os.mkdir('ckpt')
if len(args.ckpt) == 0:
    args.ckpt = '{}_{}'.format(args.dataset, 'pcnn_att')
ckpt = 'ckpt/{}.pth.tar'.format(args.ckpt)

if args.dataset != 'none':
    opennre.download(args.dataset, root_path=root_path)
    args.train_file = os.path.join(root_path, 'benchmark', args.dataset, '{}_train.txt'.format(args.dataset))
    args.val_file = os.path.join(root_path, 'benchmark', args.dataset, '{}_val.txt'.format(args.dataset))
    args.test_file = os.path.join(root_path, 'benchmark', args.dataset, '{}_test.txt'.format(args.dataset))
    args.rel2id_file = os.path.join(root_path, 'benchmark', args.dataset, '{}_rel2id.json'.format(args.dataset))
else:
    if not (os.path.exists(args.train_file) and os.path.exists(args.val_file) and os.path.exists(args.test_file) and os.path.exists(args.rel2id_file)):
        raise Exception('--train_file, --val_file, --test_file and --rel2id_file are not specified or files do not exist. Or specify --dataset')

logging.info('Arguments:')
for arg in vars(args):
    logging.info('    {}: {}'.format(arg, getattr(args, arg)))

rel2id = json.load(open(args.rel2id_file))

# Download glove
# opennre.download('glove', root_path=root_path)
# word2id = json.load(open(os.path.join(root_path, 'pretrain/glove/glove.6B.50d_word2id.json')))
# word2vec = np.load(os.path.join(root_path, 'pretrain/glove/glove.6B.50d_mat.npy'))

# Define the sentence encoder
sentence_encoder = opennre.encoder.BERTEntityEncoder(
    max_length=args.max_length,
    pretrain_path=args.pretrain_path,
    mask_entity=args.mask_entity
)

# Define the model
model = opennre.model.BagAttention(sentence_encoder, len(rel2id), rel2id)

# Define the whole training framework
framework = opennre.framework.BagRE(
    train_path=args.train_file,
    val_path=args.val_file,
    test_path=args.test_file,
    model=model,
    ckpt=ckpt,
    batch_size=args.batch_size,
    max_epoch=args.max_epoch,
    lr=args.lr,
    weight_decay=args.weight_decay,
    opt=args.optim,
    bag_size=args.bag_size)

# Train the model
if not args.only_test:
    framework.train_model(args.metric)

# Test the model
framework.load_state_dict(torch.load(ckpt)['state_dict'])
result, hits_result = framework.eval_model(framework.test_loader)
# 存下y_true y_logits
np.save('./result'+ args.ckpt + '_true.npy', result['y_true'])
np.save('./result' + args.ckpt + '_scores.npy', result['y_logits'])

# Print the result
logging.info('Test set results:')
logging.info('AUC: {}'.format(result['auc']))
logging.info('Micro F1: {}'.format(result['micro_f1']))
logging.info('P@100: {}'.format(result['P100']))
logging.info('P@200: {}'.format(result['P200']))
logging.info('P@300: {}'.format(result['P300']))
logging.info('Micro Hits@K:')
logging.info('Hits@10: {}'.format(hits_result['micro']['H10']))
logging.info('Hits@15: {}'.format(hits_result['micro']['H15']))
logging.info('Hits@20: {}'.format(hits_result['micro']['H20']))
logging.info('Mscro Hits@K:')
logging.info('Hits@10: {}'.format(hits_result['macro']['H10']))
logging.info('Hits@15: {}'.format(hits_result['macro']['H15']))
logging.info('Hits@20: {}'.format(hits_result['macro']['H20']))
