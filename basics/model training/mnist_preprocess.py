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
print("=" * 60)
print("LOADING RAW IDX BINARY FILES")
print("=" * 60)

X_train = load_images("train-images.idx3-ubyte")
y_train = load_labels("train-labels.idx1-ubyte")
X_test = load_images("t10k-images.idx3-ubyte")
y_test = load_labels("t10k-labels.idx1-ubyte")

print(f"Training images: {X_train.shape} -> 60,000 images, each 784 pixels")
print(f"Training labels: {y_train.shape}")
print(f"Testing images: {X_test.shape} -> 10,000 images, each 784 pixels")
print(f"Testing labels: {y_test.shape}")
print(f"Pixel range : {X_train.min()} - {X_train.max()}")
print(f"Classes: {np.unique(y_train)}")

# normalizing pixel values from [0,255] to [0,1]
print("\n" + "=" * 60)
print("NORMALIZING PIXEL VALUES")
print("=" * 60)

X_train = X_train / 255.0
X_test = X_test / 255.0
print(f"Pixel range after normalization: {X_train.min():.1f} - {X_train.max():.1f}")

# subsample training data for faster training
print("\n" + "=" * 60)
print("SUBSAMPLING TRAINING DATA")
print("=" * 60)

SAMPLE_SIZE = 10000
np.random.seed(42)
idx = np.random.choice(len(X_train), size=SAMPLE_SIZE, replace=False)
X_train_sub = X_train[idx]
y_train_sub = y_train[idx]

print(f"using {SAMPLE_SIZE} of 60,000 training samples")
print(f"Shape: {X_train_sub.shape}")

# standardize features to have mean=0 and std=1 using StandardScaler
print("\n" + "=" * 60)
print("STANDARDIZING FEATURES")
print("=" * 60)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_sub)
X_test_scaled = scaler.transform(X_test)
print("Featured scaled mean: mean nearly 0 and std nearly 1")
print(f"Train mean: {X_train_scaled.mean():.2f}, Train std: {X_train_scaled.std():.2f}")
print(f"Test mean: {X_test_scaled.mean():.2f}, Test std: {X_test_scaled.std():.2f}")

# Training SVM with RBF kernel
print("\n" + "=" * 60)
print("TRAINING SVM WITH RBF KERNEL (Kernel=rbf, C=5, gamma=scale)")
print("=" * 60)

svm = SVC(kernel="rbf", C=5, gamma="scale", random_state=42)

t0 = time.time()
svm.fit(X_train_scaled, y_train_sub)
elapsed = time.time() - t0
print(f"Training done in {elapsed:.1f} seconds")
print(f"Number of support vectors: {svm.support_vectors_.shape[0]}")

# predicting and evaluating on test set
print("\n" + "=" * 60)
print("PREDICTING AND EVALUATING ON TEST SET (Evaluating on 10,000 test images)")
print("=" * 60)

y_pred = svm.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"\n Test Accuracy: {acc*100:.2f}%")
print("\nDetaield Per-class report: ")
print(classification_report(y_test, y_pred, target_names=[str(i) for i in range(10)]))

# confusion matrix and simple predictions
print("\n" + "=" * 60)
print("CONFUSION MATRIX AND SIMPLE PREDICTIONS: GENERATING PLOTS")
print("=" * 60)

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
    cmap="Blues",
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
print("Plots saved as mnist_svm_results.png")