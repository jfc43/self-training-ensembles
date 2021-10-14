#!/bin/bash

gpu_id=0

for source in mnist mnist-m svhn usps
do
    CUDA_VISIBLE_DEVICES=$gpu_id python train_model.py --source-dataset $source --model-type dann_arch --base-dir ./checkpoints/dann_arch_source_models/
done

for source in mnist mnist-m svhn usps
do
    CUDA_VISIBLE_DEVICES=$gpu_id python train_model.py --source-dataset $source --model-type typical_dnn --base-dir ./checkpoints/typical_dnn_source_models/
done