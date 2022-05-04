CUDA_VISIBLE_DEVICES=2 python example/train.py \
--dataset ours \
--max_epoch 10 \
--batch_size 32 \
--metric micro_f1 \
--lr 2e-5 \
--ckpt MEGA \
--pretrain_path '/home/data_ti6_c/wanghk/bert_model/bert-base-uncased' \
--pooler entity \
--exp_name BERT+SG+Att+Att
