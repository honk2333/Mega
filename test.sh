CUDA_VISIBLE_DEVICES=0 python example/train.py \
--dataset ours \
--batch_size 1 \
--metric micro_f1 \
--only_test \
--ckpt MEGA_ckpt \
--pretrain_path '/home/data_ti6_c/wanghk/bert_model/bert-base-uncased'
# --pretrain_path './bert-base-uncased'
