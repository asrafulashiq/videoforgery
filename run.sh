## DAVIS
python main.py --videoset davis --model unet --epoch 10 --lr 1e-4 >> tmp/davis_unet.txt


python main.py --videoset davis --model albunet --size 256 --epoch 10 --lr 1e-4 >> tmp/davis_rnet.txt


python main.py --videoset davis --loss-type dice --model unet --epoch 10 --lr 1e-4 >> tmp/davis_unet2.txt


python main.py --videoset davis --model albunet --loss-type dice --size 256 --epoch 10 --lr 1e-4 >> tmp/davis_rnet2.txt


## YOUTUBE

python main.py --videoset youtube --model unet --epoch 10 --lr 1e-4 >> tmp/utube_unet.txt


python main.py --videoset youtube --model albunet --size 256 --epoch 10 --lr 1e-4 >> tmp/utube_rnet.txt


python main.py --videoset youtube --model unet --loss-type dice  --epoch 10 --lr 1e-4 >> tmp/utube_unet2.txt


python main.py --videoset youtube --model albunet --loss-type dice  --size 256 --epoch 10 --lr 1e-4 >> tmp/utube_rnet2.txt
