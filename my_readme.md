```shell
pip install -e ".[torch,metrics]" --no-build-isolation
pip install 'deepspeed>=0.10.0,<=0.16.9'
pip install wandb
```

```shell
cd /home/kazhu@amd.com/yz-LLaMA-Factory && pip install -e ".[torch,metrics]" --no-build-isolation && llamafactory-cli train examples/train_full/qwen3_14b.yaml
```


```shell
CUDA_VISIBLE_DEVICES=2,3,4,5 llamafactory-cli train examples/train_full/qwen3_14b.yaml output_dir=./results/aigise-gemini-Qwen3-14B-sft
```