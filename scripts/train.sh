for lambda in 256 512 1024 2048 4096 8192
do
python train.py --model-name CAMSIC -d /home/xzhangga/dataset/cityscapes --batch-size 4 \
    --data-name cityscapes --lambda $lambda --test-batch-size 1 --save --cuda --patch-size 256 256 --epochs 400 --sche_name steplr \
    --i_model_path ./pretrained_ckpt/ELIC_0450_ft_3980_Plateau.pth.tar
done

for lambda in 256 512 1024 2048 4096 8192
do
python train.py --model-name CAMSIC -d /home/xzhangga/dataset/instereo2k --batch-size 4 \
    --data-name instereo2k --lambda $lambda --test-batch-size 1 --save --cuda --patch-size 256 256 --epochs 400 --sche_name steplr \
    --i_model_path ./pretrained_ckpt/ELIC_0450_ft_3980_Plateau.pth.tar
done

for lambda in 8 16 32 64 128 256
do
python train.py --model-name CAMSIC -d /home/xzhangga/dataset/cityscapes --batch-size 4 \
    --data-name cityscapes --lambda $lambda --test-batch-size 1 --save --cuda --patch-size 256 256 --epochs 300 --sche_name steplr \
    --i_model_path ./checkpoints/instereo2k/mse/CAMSIC/lamda$lambda/train-run1/ckpt.pth.tar --learning-rate 5e-5
done

for lambda in 8 16 32 64 128 256
do
python train.py --model-name CAMSIC -d /home/xzhangga/dataset/instereo2k --batch-size 4 \
    --data-name instereo2k --lambda $lambda --test-batch-size 1 --save --cuda --patch-size 256 256 --epochs 300 --sche_name steplr \
    --i_model_path ./checkpoints/instereo2k/mse/CAMSIC/lamda$lambda/train-run1/ckpt.pth.tar --learning-rate 5e-5
done