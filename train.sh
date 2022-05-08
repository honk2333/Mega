CUDA_VISIBLE_DEVICES=0 python example/train.py \
--dataset ours \
--max_epoch 10 \
--batch_size 32 \
--metric micro_f1 \
--lr 1e-5 \
--ckpt MEGA \
--pretrain_path '/home/data_ti6_c/wanghk/bert_model/bert-base-uncased' \
--pooler entity \
--exp_name bup_10