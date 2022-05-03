CUDA_VISIBLE_DEVICES=0 python example/train.py \
--dataset ours \
--max_epoch 15 \
--batch_size 32 \
--metric micro_f1 \
--lr 2e-5 \
--ckpt MEGA \
--pretrain_path '/home/data_ti6_c/wanghk/bert_model/bert-base-uncased' \
# --pooler visualbert
--pooler entity
