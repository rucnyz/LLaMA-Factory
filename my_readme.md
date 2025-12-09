```shell
pip install -e ".[torch,metrics]" --no-build-isolation
pip install 'deepspeed>=0.10.0,<=0.16.9'
pip install wandb
```

```shell
cd /wekafs/kazhu/yz-LLaMA-Factory && pip install -e ".[torch,metrics]" --no-build-isolation && llamafactory-cli train examples/train_full/qwen3_14b.yaml
```


```shell
CUDA_VISIBLE_DEVICES=2,3,4,5 llamafactory-cli train examples/train_full/qwen3_14b.yaml output_dir=./results/aigise-gemini-Qwen3-14B-sft

CUDA_VISIBLE_DEVICES=2,3,4,5 llamafactory-cli train examples/train_full/qwen3_32b.yaml output_dir=./results/aigise-gemini-Qwen3-32B-sft

CUDA_VISIBLE_DEVICES=2,3,4,5 llamafactory-cli train examples/train_full/qwen3_8b.yaml output_dir=./results/aigise-gemini-Qwen3-8B-sft


# multi node
cd /wekafs/kazhu/yz-LLaMA-Factory && pip install -e ".[torch,metrics]" --no-build-isolation && FORCE_TORCHRUN=1 NNODES=$PET_NNODES NODE_RANK=$PET_NODE_RANK llamafactory-cli train examples/train_full/qwen3_32b.yaml output_dir=./results/aigise-gemini-Qwen3-32B-sft 2>&1 | tee -a output/$WORKLOAD_ID.k8s-job.log
```