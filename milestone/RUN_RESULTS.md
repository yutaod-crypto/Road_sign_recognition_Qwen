# GTSRB × Qwen3-VL-2B 运行结果汇总

本文档记录在本仓库中实际跑通基线与小样本 LoRA 后的**环境、命令、指标与产物路径**。测试集为分层采样的 **120 张**图（6 类 × 20 张），与 `00_prepare_gtsrb_grouped.py` 一致。

---

## 1. 环境

| 项 | 值 |
|----|-----|
| 项目路径 | `/sgl-workspace/intro2dl-proj/Road_sign_recognition_Qwen` |
| Python | 3.12.3（venv：`.venv`） |
| PyTorch | 2.6.0+cu124 |
| transformers | 5.5.4 |
| GPU使用方式 | `export CUDA_VISIBLE_DEVICES=1`（物理 **GPU 1**，进程内为 `cuda:0`） |
| 基座模型 | `Qwen/Qwen3-VL-2B-Instruct`（首次运行从 Hugging Face 自动下载） |

---

## 2. 数据准备

- GTSRB 通过 `torchvision.datasets.GTSRB(root="datasets", split=..., download=True)` 下载至 `datasets/gtsrb/`。
- 与脚本期望的布局对齐：
  - `datasets/gtsrb/train` → 软链到 `GTSRB/Training`
  - `datasets/gtsrb/test_images` → 软链到 `GTSRB/Final_Test/Images`
- 测试列表：`python milestone/00_prepare_gtsrb_grouped.py` → `artifacts/gtsrb_test_120.jsonl`

---

## 3. 代码修复（训练可跑）

`transformers` 5.x 下 Qwen3-VL 在传入 `image_grid_thw` 时必须同时传入 **`mm_token_type_ids`**，否则训练报错：

> `Multimodal data was passed ... but mm_token_type_ids is missing`

已在以下文件的 `GTSRBCollator` 中拼接 prompt 的 `mm_token_type_ids` 与答案段的全零向量，并在截断/批填充时与 `input_ids` 对齐：

- `04_train_gtsrb_qlora_small.py`
- `04_train_gtsrb_lora_full.py`

---

## 4. 执行的命令（可复现）

```bash
cd /sgl-workspace/intro2dl-proj/Road_sign_recognition_Qwen
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=1

python milestone/00_prepare_gtsrb_grouped.py

python milestone/01_run_gtsrb_qwen3vl.py \
  --eval_jsonl artifacts/gtsrb_test_120.jsonl \
  --out_preds artifacts/gtsrb_preds_baseline.jsonl

python milestone/02_eval_gtsrb.py \
  --preds_jsonl artifacts/gtsrb_preds_baseline.jsonl \
  --out_confusion_csv artifacts/gtsrb_confusion_baseline.csv

python milestone/03_prepare_gtsrb_train_small.py
python milestone/04_train_gtsrb_qlora_small.py

python milestone/05_run_gtsrb_qwen3vl_with_adapter.py \
  --eval_jsonl artifacts/gtsrb_test_120.jsonl \
  --out_preds artifacts/gtsrb_preds_lora_small.jsonl \
  --adapter_dir artifacts/gtsrb_qwen3vl_lora_small

python milestone/02_eval_gtsrb.py \
  --preds_jsonl artifacts/gtsrb_preds_lora_small.jsonl \
  --out_confusion_csv artifacts/gtsrb_confusion_lora_small.csv
```

**小样本训练配置（脚本内）：** 每类训练 10 条、验证 5 条 → 共 **60** 训练 / **30** 验证；LoRA `r=4, alpha=8`；2 epoch；fp16 基座 + LoRA（脚本名为 qlora_small，实际为 **fp16 全模型上的 LoRA**，非 4bit 训练）。

**训练日志摘要：** `train_runtime` = **29.05 s**，`train_loss` = **0.4143**，`epoch` = **2**。

---

## 5. 评估结果

### 5.1 Zero-shot 基线（4-bit 推理，`max_image_side=168`）

| 指标 | 值 |
|------|-----|
| 样本数 | 120 |
| 准确率 | **16.67%** (20/120) |
| 有效 JSON 率 | **100%** |

**按类准确率（基线）**

| 类 | 准确率 |
|----|--------|
| speed_limit | 0% |
| stop | **100%** |
| yield | 0% |
| no_entry | 0% |
| warning | 0% |
| direction | 0% |

现象与里程报告一致：模型几乎总预测为提示中的示例类 **stop**（prompt anchoring）。

**混淆矩阵（行=真实，列=预测）** — 见 `artifacts/gtsrb_confusion_baseline.csv`，摘要如下：

| gt \\ pred | speed_limit | stop | yield | no_entry | warning | direction |
|------------|-------------|------|-------|----------|---------|-----------|
| speed_limit | 0 | 20 | 0 | 0 | 0 | 0 |
| stop | 0 | 20 | 0 | 0 | 0 | 0 |
| yield | 0 | 20 | 0 | 0 | 0 | 0 |
| no_entry | 0 | 20 | 0 | 0 | 0 | 0 |
| warning | 0 | 20 | 0 | 0 | 0 | 0 |
| direction | 0 | 20 | 0 | 0 | 0 | 0 |

---

### 5.2 小样本 LoRA + 4-bit 推理（`05_run_...`，`max_image_side=384`）

| 指标 | 值 |
|------|-----|
| 样本数 | 120 |
| 准确率 | **86.67%** (104/120) |
| 有效 JSON 率 | **100%** |

**按类准确率（LoRA 小样本）**

| 类 | 准确率 |
|----|--------|
| speed_limit | **70%** (14/20) |
| stop | **100%** (20/20) |
| yield | **90%** (18/20) |
| no_entry | **95%** (19/20) |
| warning | **90%** (18/20) |
| direction | **75%** (15/20) |

**混淆矩阵** — 见 `artifacts/gtsrb_confusion_lora_small.csv`：

| gt \\ pred | speed_limit | stop | yield | no_entry | warning | direction |
|------------|-------------|------|-------|----------|---------|-----------|
| speed_limit | 14 | 0 | 6 | 0 | 0 | 0 |
| stop | 0 | 20 | 0 | 0 | 0 | 0 |
| yield | 0 | 0 | 18 | 0 | 2 | 0 |
| no_entry | 0 | 0 | 1 | 19 | 0 | 0 |
| warning | 0 | 0 | 1 | 0 | 18 | 1 |
| direction | 0 | 0 | 1 | 3 | 1 | 15 |

与课程里程 PDF 中「约 83.3%（60 条训练）」同量级；差异来自随机种子、库版本与基座（PDF 为 Qwen2.5-VL，本仓库为 Qwen3-VL）。

---

## 6. 产出文件一览

| 路径 | 说明 |
|------|------|
| `artifacts/gtsrb_test_120.jsonl` | 测试集索引 |
| `artifacts/gtsrb_preds_baseline.jsonl` | 基线预测 |
| `artifacts/gtsrb_confusion_baseline.csv` | 基线混淆矩阵 |
| `artifacts/gtsrb_train_small.jsonl` / `gtsrb_val_small.jsonl` | 小样本训练/验证 |
| `artifacts/gtsrb_qwen3vl_lora_small/` | LoRA adapter 与 processor |
| `artifacts/gtsrb_preds_lora_small.jsonl` | LoRA 推理预测 |
| `artifacts/gtsrb_confusion_lora_small.csv` | LoRA 混淆矩阵 |

---

## 7. 全量 LoRA 与加速（双卡 / 更大 batch）

全量数据：`03_prepare_gtsrb_train_full.py` 已生成约 **23976** 条训练、**2664** 条验证。

`04_train_gtsrb_lora_full.py` 已支持：

- **多卡 DDP**：勿使用 `device_map="auto"` 跨进程；用 `torchrun` 启动时脚本会自动每张卡一份模型并 `NCCL` 同步。
- **默认更大吞吐**：`per_device_train_batch_size=2`、`gradient_accumulation_steps=1`（单卡时每步见 2 条样本；与原先「batch1×累积2」的全局 batch 仍接近）。
- **bf16**：在支持 bf16 的 GPU上默认启用（H100/H200），通常比 fp16 更快；可传 `--fp16` 强制半精度。
- **TF32**：已打开 `matmul` / `cudnn` 的 TF32。
- **DataLoader**：默认 `dataloader_num_workers=4` + `pin_memory`，减轻 CPU 解码图像瓶颈。
- **显存极大时**：可试 `--no_gradient_checkpointing` 换速度（OOM 则去掉该参数）。

**双卡示例（推荐，约减半 wall-clock）：**

```bash
cd /sgl-workspace/intro2dl-proj/Road_sign_recognition_Qwen
source .venv/bin/activate
# 任选两张卡，例如 0 和 1
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 milestone/04_train_gtsrb_lora_full.py \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 1
```

**单卡加快速度（例如单张 H200）：**

```bash
CUDA_VISIBLE_DEVICES=0 python milestone/04_train_gtsrb_lora_full.py \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --no_gradient_checkpointing
```

（若 OOM，把 batch 改回 `2` 或去掉 `--no_gradient_checkpointing`。）

---

## 8. 历史说明

- 早期在本环境的「未跑完全量」仅因耗时长；可按第 7 节命令重跑并更新本节与评估产物。

---

*生成说明：由自动化实验流程整理；若需与论文表格逐字一致，请固定随机种子、模型 ID 与 `transformers` 版本后重跑。*
