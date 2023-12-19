CUDA_VISIBLE_DEVICES=0 python  main.py \
	--cuda 0 \
        --model SaCon \
	--lr 3e-5 --batch_size_per_gpu 128 --max_epoch 20 \
	--gradient_accumulation_steps 2 \
	--max_length 64 \
	--save_step 10000 \
	--alpha 0.3 \
	--temperature 0.07 \
	--train_sample \
	--save_dir ckpt_sacon \