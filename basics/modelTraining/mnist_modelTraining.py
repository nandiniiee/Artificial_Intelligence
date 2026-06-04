import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import struct
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import time
from sklearn.svm import SVC

# parsing the raw idx binary files


# reads IDX3 file
def load_images(path):
    with open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
    return images.reshape(n, rows * cols)


# reads IDX1 file
def load_labels(path):
    with open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels


# loading all 4 files

print("LOADING RAW IDX BINARY FILES\n")

X_train = load_images("mnist_raw_dataset/train-images.idx3-ubyte")
y_train = load_labels("mnist_raw_dataset/train-labels.idx1-ubyte")
X_test = load_images("mnist_raw_dataset/t10k-images.idx3-ubyte")
y_test = load_labels("mnist_raw_dataset/t10k-labels.idx1-ubyte")

print(f"Training images: {X_train.shape} -> 60,000 images, each 784 pixels")
print(f"Training labels: {y_train.shape}")
print(f"Testing images: {X_test.shape} -> 10,000 images, each 784 pixels")
print(f"Testing labels: {y_test.shape}")
print(f"Pixel range : {X_train.min()} - {X_train.max()}")
print(f"Classes: {np.unique(y_train)}\n")

# normalizing pixel values from [0,255] to [0,1]
print("NORMALIZING PIXEL VALUES\n")

X_train = X_train / 255.0
X_test = X_test / 255.0
print(f"Pixel range after normalization: {X_train.min():.1f} - {X_train.max():.1f}\n")

# subsample training data for faster training
print("SUBSAMPLING TRAINING DATA\n")

SAMPLE_SIZE = 60000
# np.random.seed(42)
# idx = np.random.choice(len(X_train), size=SAMPLE_SIZE, replace=False)
X_train_sub = X_train
y_train_sub = y_train

print(f"using {SAMPLE_SIZE} of 60,000 training samples\n")
print(f"Shape: {X_train_sub.shape}\n")

# standardize features to have mean=0 and std=1 using StandardScaler
print("STANDARDIZING FEATURES\n")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_sub)
X_test_scaled = scaler.transform(X_test)
print("Featured scaled mean: mean nearly 0 and std nearly 1\n")
print(f"Train mean: {X_train_scaled.mean():.2f}, Train std: {X_train_scaled.std():.2f}\n")
print(f"Test mean: {X_test_scaled.mean():.2f}, Test std: {X_test_scaled.std():.2f}\n")

# Training SVM with RBF kernel
print("TRAINING SVM WITH RBF KERNEL (Kernel=rbf, C=5, gamma=scale)\n")

svm = SVC(kernel="rbf", C=5, gamma="scale", random_state=42)

# t0 = time.time()
svm.fit(X_train_scaled, y_train_sub)
# elapsed = time.time() - t0
# print(f"Training done in {elapsed:.1f} seconds\n")
print(f"Number of support vectors: {svm.support_vectors_.shape[0]}\n")

# predicting and evaluating on test set
print("PREDICTING AND EVALUATING ON TEST SET (Evaluating on 10,000 test images)\n")

y_pred = svm.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f" Test Accuracy: {acc*100:.2f}%\n")
print("Detailed Per-class report: \n")
print(classification_report(y_test, y_pred, target_names=[str(i) for i in range(10)]))

# confusion matrix and simple predictions
print("CONFUSION MATRIX AND SIMPLE PREDICTIONS: GENERATING PLOTS\n")

cm = confusion_matrix(y_test, y_pred)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle(
    f"SVM on MIST Results- Accuracy:{acc*100:.2f}%", fontsize=15, fontweight="bold"
)

# Plot 1 : Confusion matrix
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=range(10),
    yticklabels=range(10),
    ax=axes[0],
)
axes[0].set_title("Confusion Matrix", fontsize=13)
axes[0].set_xlabel("Predicted Label1")
axes[0].set_ylabel("True Label")

# Plot 2: Sampel predictions
axes[1].set_title("Sample Predictions (green = correct, red = wrong)", fontsize=13)
axes[1].axis("off")

sample_idx = np.random.choice(len(X_test), size=20, replace=False)
for i, si in enumerate(sample_idx):
    ax_sub = fig.add_axes(
        [0.52 + (i % 10) * 0.047, 0.12 if i >= 10 else 0.52, 0.04, 0.36]
    )
    ax_sub.imshow(X_test[si].reshape(28, 28), cmap="gray")
    color = 'green' if y_pred[si] == y_test[si] else 'red'
    ax_sub.set_title(f'P:{y_pred[si]}\nT:{y_test[si]}', color=color, fontsize=6, pad=1)
    ax_sub.axis("off")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('mnist_svm_results.png', dpi=150, bbox_inches='tight')
plt.show()
# print("Plots saved as mnist_svm_results.png\n")
# print("END OF SCRIPT\n")