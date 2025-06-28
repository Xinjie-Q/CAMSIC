


for lambda in 256 512 1024 2048 4096 8192
do
	CUDA_VISIBLE_DEVICES=0 python eval.py -im CAMSIC -d /home/xzhangga/dataset/cityscapes --data-name cityscapes --cuda --entropy-estimation \
	--net_path ./checkpoints/cityscapes/mse/CAMSIC/lamda$lambda$tail/train-run1/ckpt.pth.tar \
	--output ./bpp_estimate/cityscapes/CAMSIC/mse/lambda$lambda 
done


for lambda in 256 512 1024 2048 4096 8192
do
	CUDA_VISIBLE_DEVICES=0 python eval.py -im CAMSIC -d /home/xzhangga/dataset/instereo2k --data-name instereo2k --cuda --entropy-estimation \
	--net_path ./checkpoints/instereo2k/mse/CAMSIC/lamda$lambda$tail/train-run1/ckpt.pth.tar \
	--output ./bpp_estimate/instereo2k/CAMSIC/mse/lambda$lambda 
done

for lambda in 8 16 32 64 128 256
do
	CUDA_VISIBLE_DEVICES=0 python eval.py -im CAMSIC -d /home/xzhangga/dataset/cityscapes --data-name cityscapes --cuda --entropy-estimation \
	--net_path ./checkpoints/cityscapes/ms_ssim/CAMSIC/lamda$lambda$tail/train-run1/ckpt.pth.tar \
	--output ./bpp_estimate/cityscapes/CAMSIC/ms_ssim/lambda$lambda
done

for lambda in 8 16 32 64 128 256
do
	CUDA_VISIBLE_DEVICES=0 python eval.py -im CAMSIC -d /home/xzhangga/dataset/instereo2k --data-name instereo2k --cuda --entropy-estimation \
	--net_path ./checkpoints/instereo2k/ms_ssim/CAMSIC/lamda$lambda$tail/train-run1/ckpt.pth.tar \
	--output ./bpp_estimate/instereo2k/CAMSIC/ms_ssim/lambda$lambda
done


