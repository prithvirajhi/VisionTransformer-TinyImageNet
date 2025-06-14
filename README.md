# 🔍 Vision Transformer on TinyImageNet

A pure PyTorch implementation of a Vision Transformer (ViT) trained from scratch on the TinyImageNet dataset using a single Jupyter notebook. This project documents the full training process, multiple optimization techniques, and key learnings.

---

## 📘 Notebook

Main notebook:  
[`FunViTImplementation_TinyImageNet_30_accuracy.ipynb`](https://github.com/prithvirajhi/VisionTransformer-TinyImageNet/blob/main/FunViTImplementation_TinyImageNet_30_accuracy.ipynb)

---

## 📊 Dataset: TinyImageNet

- **200 classes**, 64×64 images  
- 100,000 training images (500 per class)  
- Download: [http://cs231n.stanford.edu/tiny-imagenet-200.zip](http://cs231n.stanford.edu/tiny-imagenet-200.zip)

⚠️ **Important**: Before using `ImageFolder`, fix `val/` structure using `val_annotations.txt`.

---

## ⚙️ Model Configuration

| Hyperparameter       | Value       |
|----------------------|-------------|
| Image Size           | 64×64       |
| Patch Size           | 8           |
| Embedding Dim        | 512         |
| Depth (Transformer)  | 10          |
| Attention Heads      | 8           |
| MLP Dim              | 1024        |
| Dropout              | 0.1         |
| Classes              | 200         |

---

## 🚀 Training Enhancements

- ⚡ **AMP (Automatic Mixed Precision)**
- 📉 **Label Smoothing Loss**
- 🔄 **Exponential Moving Average (EMA)**
- 🌀 **Cosine LR Scheduler with Warm-up**
- ⏹️ **Early Stopping**
- 💾 **Checkpointing (model + EMA + optimizer + scaler)**

---

## 📈 Sample Results

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc |
|-------|------------|----------|-----------|---------|
| 30    | 3.91       | 3.83     | 17.6%     | 18.5%   |
| 84    | 3.27       | 3.32     | 29.3%     | 27.4%   |
| 192   | 2.57       | 3.14     | 42.8%     | 32.1%   |

> ⏹️ Training stopped early around epoch 192 due to validation loss plateau.

---

## 🧠 Learnings

- Training ViTs from scratch on small datasets like TinyImageNet is possible with stabilizations.
- AMP and label smoothing improve training speed and generalization.
- EMA stabilizes evaluation and improves final accuracy.
- Cosine LR scheduling with warm-up avoids early divergence.
- Weight initialization is important—can be integrated into model class.

---

## 🔧 Setup

Training and experimentation were done on:

- 🖥️ **RunPod** (RTX 4000 Ada)
- 🧪 **Google Colab** (for prototyping)

Ensure sufficient disk I/O performance for zip extraction and dataset operations.

---

## 🧭 Future Directions

- 🧪 Self-Supervised Learning (e.g., MAE, DINO)
- 🎓 Distillation to compact ViT or CNN
- 🧊 Post-training Quantization
- 📊 Visual debugging with Grad-CAM and confusion matrix

---

> 🔁 For suggestions or contributions, feel free to open an issue or PR.
