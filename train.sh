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

# 训练BERT+ATT+NYT(tiny)
CUDA_VISIBLE_DEVICES=0 python example/train_bag_bert_att.py --train_file=benchmark/nyt10/nyt10_train.txt --val_file=benchmark/nyt10/nyt10_val.txt --test_file=benchmark/nyt10/nyt10_test.txt --rel2id_file=benchmark/nyt10/nyt10_rel2id.json --ckpt=bert_with_nyt --pretrain_path=prajjwal1/bert-tiny

# 训练BERT+ATT+NYT-CEDA(tiny)
CUDA_VISIBLE_DEVICES=0,2,3 python example/train_bag_bert_att.py --train_file=benchmark/nyt10-aug/nyt_train_aug8.txt --val_file=benchmark/nyt10-aug/nyt10_val.txt --test_file=benchmark/nyt10-aug/nyt10_test.txt --rel2id_file=benchmark/nyt10-aug/nyt10_rel2id.json --ckpt=bert_with_nyt_augmented --pretrain_path=prajjwal1/bert-tiny

CUDA_VISIBLE_DEVICES=3 python example/train_bag_bert_att.py --train_file=benchmark/nyt10-aug/nyt_train_aug4.txt --val_file=benchmark/nyt10-aug/nyt10_val.txt --test_file=benchmark/nyt10-aug/nyt10_test.txt --rel2id_file=benchmark/nyt10-aug/nyt10_rel2id.json --ckpt=bert_with_nyt_augmented_4 --pretrain_path=prajjwal1/bert-tiny

# 训练BERT+ATT+NYT-CEDA(mini)(8)
CUDA_VISIBLE_DEVICES=3 python example/train_bag_bert_att.py --train_file=benchmark/nyt10-aug/nyt_train_aug8.txt --val_file=benchmark/nyt10-aug/nyt10_val.txt --test_file=benchmark/nyt10-aug/nyt10_test.txt --rel2id_file=benchmark/nyt10-aug/nyt10_rel2id.json --ckpt=bert_mini_with_nyt_augmented_8 --batch_size=160 --pretrain_path=prajjwal1/bert-mini

# 训练BERT+ATT+NYT-CEDA(small)(8)
CUDA_VISIBLE_DEVICES=0,3 python example/train_bag_bert_att.py --train_file=benchmark/nyt10-aug/nyt_train_aug8.txt --val_file=benchmark/nyt10-aug/nyt10_val.txt --test_file=benchmark/nyt10-aug/nyt10_test.txt --rel2id_file=benchmark/nyt10-aug/nyt10_rel2id.json --ckpt=bert_small_with_nyt_augmented_8 --batch_size=160 --pretrain_path=nreimers/BERT-Small-L-4_H-512_A-8

# 训练SuperBag+ATT+NYT(8)
CUDA_VISIBLE_DEVICES=2 python example/train_superbag_pcnn.py --train_file=benchmark/nyt10-aug/nyt_train_aug8.txt --val_file=benchmark/nyt10-aug/nyt10_val.txt --test_file=benchmark/nyt10-aug/nyt10_test.txt --rel2id_file=benchmark/nyt10-aug/nyt10_rel2id.json --ckpt=pcnn_superbag_with_nyt_weight0.5 --batch_size=160 --cluster_size=100 --num_iter=50
