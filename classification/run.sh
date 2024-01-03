NUM_GPUS=8
./dist_train.sh $NUM_GPUS -c\
    /workspace/akane/NAT/classification/configs/lowrank_dinat_s_tiny.yml \
    /workspace/datasets/ImageNet