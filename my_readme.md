```shell
pip install -e ".[torch,metrics]" --no-build-isolation
pip install 'deepspeed>=0.10.0,<=0.16.9'
pip install wandb
```

```shell
NAME=aigise-gemini-Qwen3-8B
LR=2.0e-6

# local
CUDA_VISIBLE_DEVICES=0,1,2,3 \
llamafactory-cli train examples/train_full/qwen3_8b.yaml \
output_dir=./results/${NAME}-lr${LR}-sft \
run_name=${NAME}-lr${LR}-sft \
push_to_hub_model_id=${NAME}-lr${LR}-sft \
learning_rate=${LR}

# multi node
cd /wekafs/kazhu/yz-LLaMA-Factory && \
pip install -e ".[torch,metrics]" --no-build-isolation && \
FORCE_TORCHRUN=1 \
NNODES=${PET_NNODES} \
NODE_RANK=${PET_NODE_RANK} \
llamafactory-cli train examples/train_full/qwen3_32b.yaml \
output_dir=./results/${NAME}-lr${LR}-sft \
run_name=${NAME}-lr${LR}-sft \
push_to_hub_model_id=${NAME}-lr${LR}-sft \
learning_rate=${LR} \
2>&1 | tee -a output/${WORKLOAD_ID}.k8s-job.log
```