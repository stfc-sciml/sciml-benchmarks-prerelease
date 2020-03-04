sudo rm -Rf results
sudo docker build -t samueljackson/unet_tf . 
sudo docker run --gpus all --rm -it --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
    -v $PWD/dataset:/workspace/unet/dataset \
    -v $PWD/results/:/workspace/unet/results \
    --workdir /workspace/unet \
    samueljackson/unet_tf:latest \
        nsys profile -t cuda,osrt,nvtx,cudnn,cublas -y 20 -d 120 -o results/baseline -f true python main.py --data_dir dataset --model_dir results --epochs 3 --batch_size 4
    # python main.py \
    #          --data_dir dataset \
    #          --model_dir results \
    #          --epochs 2 \
    #          --batch_size 4 \
    #          --learning_rate 0.001 \
    #          --exec_mode train_and_predict \





