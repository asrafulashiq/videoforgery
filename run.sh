

set -gx CUDA_VISIBLE_DEVICES 2,3

# test on coco pretrained, with and without boundary

python main.py --videoset youtube --model unet  --epoch 20 --lr 1e-4 \
           --ckpt ckpt/base_unet_coco_bce.pkl >> tmp/coco_pre_utube_unet.txt

python main.py --videoset davis --model unet --epoch 20 --lr 1e-4 \
           --ckpt ckpt/base_unet_coco_bce.pkl >> tmp/coco_pre_davis_unet.txt


python main.py --videoset all --model unet --epoch 20 --lr 1e-4 \
           --ckpt ckpt/base_unet_coco_bce.pkl >> tmp/coco_pre_all_unet.txt


# test with lr
python main.py --videoset youtube --model unet  --epoch 10 --lr 1e-5 \
           --ckpt ckpt/base_unet_coco_bce.pkl >> tmp/utube_lr_1.txt

python main.py --videoset youtube --model unet  --epoch 10 --lr 1e-2 \
           --ckpt ckpt/base_unet_coco_bce.pkl >> tmp/utube_lr_2.txt

python main.py --videoset youtube --model unet  --epoch 10 --lr 0.1 \
           --ckpt ckpt/base_unet_coco_bce.pkl >> tmp/utube_lr_3.txt


# test with batch-size
python main.py --videoset youtube --batch-size 10 --model unet  --epoch 10 --lr 1e-4 \
           --ckpt ckpt/base_unet_coco_bce.pkl >> tmp/utube_bs_1.txt

python main.py --videoset youtube --batch-size 60 --model unet  --epoch 10 --lr 1e-4 \
           --ckpt ckpt/base_unet_coco_bce.pkl >> tmp/utube_bs_2.txt



## DAVIS
# python main.py --videoset davis --model unet --epoch 10 --lr 1e-4 >> tmp/davis_unet.txt


# python main.py --videoset davis --model albunet --size 256 --epoch 10 --lr 1e-4 >> tmp/davis_rnet.txt


# python main.py --videoset davis --loss-type dice --model unet --epoch 10 --lr 1e-4 >> tmp/davis_unet2.txt


# python main.py --videoset davis --model albunet --loss-type dice --size 256 --epoch 10 --lr 1e-4 >> tmp/davis_rnet2.txt


# ## YOUTUBE

# python main.py --videoset youtube --model unet --epoch 10 --lr 1e-4 >> tmp/utube_unet.txt


# python main.py --videoset youtube --model albunet --size 256 --epoch 10 --lr 1e-4 >> tmp/utube_rnet.txt


# python main.py --videoset youtube --model unet --loss-type dice  --epoch 10 --lr 1e-4 >> tmp/utube_unet2.txt


# python main.py --videoset youtube --model albunet --loss-type dice  --size 256 --epoch 10 --lr 1e-4 >> tmp/utube_rnet2.txt
