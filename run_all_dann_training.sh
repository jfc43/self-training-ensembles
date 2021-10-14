#!/bin/bash

gpu_id=0

for source in mnist mnist-m svhn usps
do
for target in mnist mnist-m svhn usps
do
    CUDA_VISIBLE_DEVICES=$gpu_id python train_dann.py --source-dataset $source --target-dataset $target
done
done

for source in mnist mnist-m svhn usps
do
for target in mnist mnist-m svhn usps
do
    CUDA_VISIBLE_DEVICES=$gpu_id python train_dann.py --source-dataset $source --target-dataset $target --test-time
done
done