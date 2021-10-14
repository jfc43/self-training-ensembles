#!/bin/bash

gpu_id=0

for model_type in typical_dnn dann_arch
do
for method in conf_avg ensemble_conf_avg conf trust_score proxy_risk our_ri our_rm
do
    CUDA_VISIBLE_DEVICES=$gpu_id python eval_pipeline.py --model-type $model_type --method $method
done
done
