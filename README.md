# Road_sign_recognition_Qwen
Here is the **entire README as one continuous write-up in the same style you showed**. You can copy and paste it **directly into GitHub**.

---

# Traffic Sign Recognition with Qwen3-VL and LoRA Fine-Tuning

This project explores using **Qwen3-VL-2B**, a vision-language model, for **traffic sign recognition** using the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset. The objective is to evaluate how well a large vision-language model performs on a structured perception task and then improve its performance through **LoRA (Low-Rank Adaptation) fine-tuning**.

The workflow of the project consists of four main stages. First, the pretrained Qwen3-VL model is used to run **baseline inference** on traffic sign images without any additional training. Second, a **small LoRA training run** is performed using a small subset of the dataset to verify that the training pipeline works correctly. Third, a **full LoRA fine-tuning run** is performed using the entire training dataset. Finally, the results from the baseline model and the fine-tuned model are compared using a consistent evaluation set.

The project is intended to simulate a simplified **autonomous-driving perception scenario**, where a model must recognize road signs in a real-world driving environment.

---

# Dataset

This project uses the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset.

The dataset can be downloaded from the official website:

[https://benchmark.ini.rub.de/gtsrb_news.html](https://benchmark.ini.rub.de/gtsrb_news.html)

Download the following files:

• **GTSRB_Final_Training_Images.zip**
• **GTSRB_Final_Test_Images.zip**
• **GTSRB_Final_Test_GT.zip**

After downloading, extract the files and place them inside the repository.

---

# Dataset Structure

After extracting the dataset, the repository should contain the following structure:

```
datasets/
└── gtsrb
    ├── train
    │   ├── 00000
    │   ├── 00001
    │   ├── 00002
    │   ├── ...
    │   └── 00041
    │
    ├── test_images
    │
    └── GT-final_test.csv
```

Each folder inside `train` corresponds to one traffic sign class and contains images along with a label file.

Example:

```
datasets/gtsrb/train/00014/
    00000.ppm
    00001.ppm
    00002.ppm
    ...
    GT-00014.csv
```

The CSV file contains labels in the format:

```
Filename;Width;Height;Roi.X1;Roi.Y1;Roi.X2;Roi.Y2;ClassId
00000.ppm;29;30;5;5;24;25;14
```

The `ClassId` corresponds to one of the original **43 traffic sign classes**.

---

# Label Grouping

Instead of predicting all 43 classes directly, the classes are grouped into **six higher-level categories** to simplify classification for the vision-language model.

The grouped labels are:

```
speed_limit
stop
yield
no_entry
warning
direction
```

This grouping reduces ambiguity and improves prediction stability when using a general-purpose vision-language model.

---

# Installation

Create a Python environment using Conda:

```
conda create -n qwen-vl python=3.10
conda activate qwen-vl
```

Install the required dependencies:

```
pip install torch transformers datasets accelerate peft pillow tqdm numpy pandas
```

---

# Baseline Evaluation

The baseline evaluation runs the pretrained **Qwen3-VL model** without any additional training.

Run the baseline inference:

```
python 01_run_gtsrb_qwen3vl.py \
--eval_jsonl artifacts/gtsrb_test_120.jsonl \
--out_preds artifacts/gtsrb_preds_baseline.jsonl
```

This produces predictions for a fixed test subset of images.

Evaluate the predictions using:

```
python 02_eval_gtsrb.py \
--preds_jsonl artifacts/gtsrb_preds_baseline.jsonl
```

The evaluation script reports classification accuracy and generates a confusion matrix.

---

# Small Training (Debug Version)

Before running full training, a **small LoRA training run** can be used to verify that the training pipeline works correctly.

This version trains on a small subset of the dataset (approximately 200 images).

Prepare the small dataset:

```
python 03_prepare_gtsrb_train_small.py
```

This step generates:

```
artifacts/gtsrb_train_small.jsonl
artifacts/gtsrb_val_small.jsonl
```

Train the LoRA adapter:

```
python 04_train_gtsrb_qlora_small.py
```

The trained adapter will be saved to:

```
artifacts/gtsrb_qwen3vl_lora_small
```

Run inference with the trained adapter:

```
python 05_run_gtsrb_qwen3vl_with_adapter.py \
--eval_jsonl artifacts/gtsrb_test_120.jsonl \
--out_preds artifacts/gtsrb_preds_lora_small.jsonl \
--adapter_dir artifacts/gtsrb_qwen3vl_lora_small
```

This step confirms that LoRA training and inference both function correctly.

---

# Full Training

After verifying that the pipeline works correctly, the LoRA adapter can be trained using the **entire GTSRB training dataset**.

Prepare the full training dataset:

```
python 03_prepare_gtsrb_train_full.py
```

This generates:

```
artifacts/gtsrb_train_full.jsonl
artifacts/gtsrb_val_full.jsonl
```

Train the LoRA adapter on the full dataset:

```
python 04_train_gtsrb_lora_full.py
```

The trained adapter will be saved to:

```
artifacts/gtsrb_qwen3vl_lora_full
```

Run inference with the trained model:

```
python 05_run_gtsrb_qwen3vl_with_adapter.py \
--eval_jsonl artifacts/gtsrb_test_120.jsonl \
--out_preds artifacts/gtsrb_preds_lora_full.jsonl \
--adapter_dir artifacts/gtsrb_qwen3vl_lora_full
```

---

# Evaluation

Evaluate the predictions from the fine-tuned model using:

```
python 02_eval_gtsrb.py \
--preds_jsonl artifacts/gtsrb_preds_lora_full.jsonl
```

The evaluation script outputs:

• classification accuracy
• confusion matrix
• prediction statistics

These results can then be compared with the baseline model.

---

# Hardware

This project was tested using:

```
NVIDIA RTX 4060 (8GB VRAM)
CUDA
Python + Conda environment
```

Training uses **LoRA with small batch size and gradient accumulation** to fit within limited GPU memory.

---

# Future Improvements

Possible future extensions:

• integrate **BDD100K driving dataset**
• evaluate **DriveLM reasoning tasks**
• multi-object traffic scene understanding
• traffic sign OCR tasks

---

# License

This project is intended for **research and educational purposes**.
