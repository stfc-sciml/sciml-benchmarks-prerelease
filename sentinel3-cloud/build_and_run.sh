sudo rm -Rf results
sudo docker build -t unet_tf . 
sudo docker run --gpus all --rm -it --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -v ~/git/ml-cloud/data/pixbox-2/:/workspace/unet/dataset -v $PWD/results/:/workspace/unet/results unet_tf:latest
