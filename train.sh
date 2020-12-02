#!/bin/sh
echo "start to train, the scipt is ${$0}"
# 训练PCNN+ATT+NYT-CEDA
CUDA_VISIBLE_DEVICES=2 python example/train_bag_pcnn_att.py --train_file=benchmark/nyt10-aug/nyt10_train.txt --val_file=benchmark/nyt10-aug/nyt10_val.txt --test_file=benchmark/nyt10-aug/nyt10_test.txt --rel2id_file=benchmark/nyt10-aug/nyt10_rel2id.json --ckpt=pcnn_with_nyt_augmented

# 训练PCNN+ATT+NYT
CUDA_VISIBLE_DEVICES=2 python example/train_bag_pcnn_att.py --train_file=benchmark/nyt10/nyt10_train.txt --val_file=benchmark/nyt10/nyt10_val.txt --test_file=benchmark/nyt10/nyt10_test.txt --rel2id_file=benchmark/nyt10-aug/nyt10_rel2id.json 

# 训练CNN+ATT+NYT-CEDA
CUDA_VISIBLE_DEVICES=0 python example/train_bag_cnn_att.py --train_file=benchmark/nyt10-aug/nyt10_train.txt --val_file=benchmark/nyt10-aug/nyt10_val.txt --test_file=benchmark/nyt10-aug/nyt10_test.txt --rel2id_file=benchmark/nyt10-aug/nyt10_rel2id.json --ckpt=cnn_with_nyt_augmented

# 训练CNN+ATT+NYT
CUDA_VISIBLE_DEVICES=0 python example/train_bag_cnn_att.py --train_file=benchmark/nyt10/nyt10_train.txt --val_file=benchmark/nyt10/nyt10_val.txt --test_file=benchmark/nyt10/nyt10_test.txt --rel2id_file=benchmark/nyt10/nyt10_rel2id.json --ckpt=cnn_with_nyt

# 训练BERT+ATT+NYT
CUDA_VISIBLE_DEVICES=0 python example/train_bag_bert_att.py --train_file=benchmark/nyt10/nyt10_train.txt --val_file=benchmark/nyt10/nyt10_val.txt --test_file=benchmark/nyt10/nyt10_test.txt --rel2id_file=benchmark/nyt10/nyt10_rel2id.json --ckpt=bert_with_nyt
