import json

import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer

name = "251208_132307"
# 配置
INPUT_FILE = f"data/{name}.jsonl"
OUTPUT_FILE = f"data/{name}_filtered.jsonl"
MODEL_NAME = "Qwen/Qwen3-8B"  # 根据需要修改模型名
MAX_TOKENS = 70000  # 100k tokens

def count_tokens(tokenizer, messages, system=None, tools=None):
    """按照 LlamaFactory 的逻辑计算 token 数量
    参考: src/llamafactory/data/processor/supervised.py L47
    这里简化处理，直接对所有消息内容进行编码并统计总token数
    """
    total_tokens = 0

    # 编码 system
    if system:
        total_tokens += len(tokenizer.encode(system, add_special_tokens=False))

    # 编码 tools
    if tools:
        tools_str = json.dumps(tools, ensure_ascii=False) if isinstance(tools, list) else str(tools)
        total_tokens += len(tokenizer.encode(tools_str, add_special_tokens=False))

    # 编码每条消息
    for msg in messages:
        content = msg.get("content", "")
        if content:
            total_tokens += len(tokenizer.encode(content, add_special_tokens=False))

    return total_tokens


def plot_distribution(token_lengths, max_tokens, output_path="token_distribution.png"):
    """绘制token长度分布图"""
    plt.figure(figsize=(12, 6))

    # 直方图
    plt.subplot(1, 2, 1)
    plt.hist(token_lengths, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(x=max_tokens, color='r', linestyle='--', label=f'Threshold: {max_tokens}')
    plt.xlabel('Token Length')
    plt.ylabel('Count')
    plt.title('Token Length Distribution')
    plt.legend()

    # 累积分布图
    plt.subplot(1, 2, 2)
    sorted_lengths = np.sort(token_lengths)
    cumulative = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths)
    plt.plot(sorted_lengths, cumulative)
    plt.axvline(x=max_tokens, color='r', linestyle='--', label=f'Threshold: {max_tokens}')
    plt.xlabel('Token Length')
    plt.ylabel('Cumulative Proportion')
    plt.title('Cumulative Distribution')
    plt.legend()

    plt.tight_layout()
    plt.show()
    plt.close()


def main():
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    total_count = 0
    kept_count = 0
    dropped_count = 0
    token_lengths = []  # 记录所有样本的token长度

    print(f"Processing {INPUT_FILE}...")

    with open(INPUT_FILE, encoding="utf-8") as fin, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as fout:

        for line in fin:
            total_count += 1
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Line {total_count}: JSON decode error - {e}")
                dropped_count += 1
                continue

            # 提取字段
            messages = data.get("conversations", data.get("messages", []))
            system = data.get("system", None)
            tools = data.get("tools", None)

            # 计算 token 数量
            token_count = count_tokens(tokenizer, messages, system, tools)
            token_lengths.append(token_count)

            if token_count <= MAX_TOKENS:
                fout.write(json.dumps(data, ensure_ascii=False) + "\n")
                kept_count += 1
            else:
                dropped_count += 1
                if dropped_count <= 10:  # 只打印前10条被丢弃的
                    print(f"Dropped line {total_count}: {token_count} tokens")

    print("\n=== Summary ===")
    print(f"Total samples: {total_count}")
    print(f"Kept samples: {kept_count}")
    print(f"Dropped samples: {dropped_count}")
    print(f"Output saved to: {OUTPUT_FILE}")

    # 统计信息
    if token_lengths:
        token_lengths_arr = np.array(token_lengths)
        max_length = int(np.max(token_lengths_arr))
        max_idx = int(np.argmax(token_lengths_arr)) + 1  # 转为1-based行号

        print("\n=== Token Length Statistics (All Data) ===")
        print(f"Min: {int(np.min(token_lengths_arr))}")
        print(f"Max: {max_length} (Line {max_idx})")
        print(f"Mean: {np.mean(token_lengths_arr):.2f}")
        print(f"Median: {int(np.median(token_lengths_arr))}")
        print(f"Std: {np.std(token_lengths_arr):.2f}")

        # 过滤后的统计
        kept_lengths = token_lengths_arr[token_lengths_arr <= MAX_TOKENS]
        if len(kept_lengths) > 0:
            print("\n=== Token Length Statistics (After Filtering) ===")
            print(f"Min: {int(np.min(kept_lengths))}")
            print(f"Max: {int(np.max(kept_lengths))}")
            print(f"Mean: {np.mean(kept_lengths):.2f}")
            print(f"Median: {int(np.median(kept_lengths))}")
            print(f"Std: {np.std(kept_lengths):.2f}")

        # 分位数
        percentiles = [25, 50, 75, 90, 95, 99]
        print("\n=== Percentiles (After Filtering) ===")
        for p in percentiles:
            print(f"P{p}: {int(np.percentile(kept_lengths, p))}")

        # 绘制分布图
        plot_distribution(token_lengths, MAX_TOKENS)


if __name__ == "__main__":
    main()
