sudo rm -Rf results
sudo docker build -t samueljackson/unet_tf . 
sudo docker run --gpus all --rm -it --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
    -v ~/git/benchmarks/sentinel3-cloud/dataset:/workspace/unet/dataset \
    -v $PWD/results/:/workspace/unet/results \
    samueljackson/unet_tf:latest python main.py \
         --data_dir dataset \
         --model_dir results \
         --epochs 2 \
         --batch_size 8 \
         --learning_rate 0.001 \
         --exec_mode train_and_predict \
