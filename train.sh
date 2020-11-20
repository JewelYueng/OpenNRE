#!/bin/sh
echo "start to train, the scipt is ${$0}"
CUDA_VISIBLE_DEVICES=2 python example/train_bag_pcnn_att.py --train_file=benchmark/nyt10-aug/nyt10_train.txt --val_file=benchmark/nyt10-aug/nyt10_val.txt --test_file=benchmark/nyt10-aug/few_rel_val_300.txt --rel2id_file=benchmark/nyt10-aug/nyt10_rel2id.json --only_test