import os
import random
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageEnhance
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


DATA_DIR = 'ear_data'
IMG_SIZE = 128 
CLASSES = ['Acute Otitis Media', 'Cerumen Impaction', 'Chronic Otitis Media', 'Myringosclerosis', 'Normal']
K_FOLDS = 5
EPOCHS_PER_FOLD = 120 

def augment_image(img):
    if random.random() > 0.5: img = ImageOps.mirror(img)
    img = img.rotate(random.uniform(-25, 25))
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.3))
    return img

def load_data():
    X, y = [], []
    print(f"📥 Loading dataset: {IMG_SIZE}x{IMG_SIZE}...")
    for idx, label in enumerate(CLASSES):
        path = os.path.join(DATA_DIR, label)
        if not os.path.exists(path): continue
        files = os.listdir(path)
        target = 1000 
        reps = max(1, target // len(files)) if len(files) > 0 else 1
        for img_name in files:
            try:
                raw_img = Image.open(os.path.join(path, img_name)).convert('L').resize((IMG_SIZE, IMG_SIZE))
                for _ in range(reps):
                    aug = augment_image(raw_img) if _ > 0 else raw_img
                    X.append(np.array(aug).flatten() / 255.0)
                    oh = np.zeros(len(CLASSES)); oh[idx] = 1
                    y.append(oh)
            except: continue
    X_arr = np.array(X)
    X_arr -= np.mean(X_arr, axis=0) 
    return X_arr, np.array(y)

class MedicalNN_Adam:
    def __init__(self, dims):
        self.W1 = np.random.randn(dims[0], dims[1]) * np.sqrt(2./dims[0])
        self.b1 = np.zeros((1, dims[1]))
        self.W2 = np.random.randn(dims[1], dims[2]) * np.sqrt(2./dims[1])
        self.b2 = np.zeros((1, dims[2]))
        self.mW1, self.vW1 = np.zeros_like(self.W1), np.zeros_like(self.W1)
        self.mW2, self.vW2 = np.zeros_like(self.W2), np.zeros_like(self.W2)
        self.t = 0

    def forward(self, X, training=True):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = np.where(self.z1 > 0, self.z1, self.z1 * 0.01)
        if training:
            self.drop_mask = (np.random.rand(*self.a1.shape) > 0.3)
            self.a1 *= self.drop_mask
        else:
            self.a1 *= 0.7
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        exp = np.exp(self.z2 - np.max(self.z2, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)

    def train_fold(self, X_t, y_t, X_v, y_v, f_idx):
        lr = 0.001
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        for i in range(EPOCHS_PER_FOLD):
            self.t += 1
            p = self.forward(X_t, training=True)
            dz2 = (p - y_t) / X_t.shape[0]
            dw2 = np.dot(self.a1.T, dz2)
            db2 = np.sum(dz2, axis=0, keepdims=True)
            da1 = np.dot(dz2, self.W2.T) * (self.drop_mask if hasattr(self, 'drop_mask') else 1)
            dz1 = da1 * np.where(self.z1 > 0, 1, 0.01)
            dw1 = np.dot(X_t.T, dz1)
            db1 = np.sum(dz1, axis=0, keepdims=True)
            for param, grad, m, v in [(self.W1, dw1, self.mW1, self.vW1), (self.W2, dw2, self.mW2, self.vW2)]:
                m[:] = beta1 * m + (1 - beta1) * grad
                v[:] = beta2 * v + (1 - beta2) * (grad**2)
                m_hat = m / (1 - beta1**self.t)
                v_hat = v / (1 - beta2**self.t)
                param -= lr * m_hat / (np.sqrt(v_hat) + eps)
            if i % 40 == 0:
                acc = np.mean(np.argmax(p, axis=1) == np.argmax(y_t, axis=1))
                print(f"Fold {f_idx+1} | Epoch {i:3} | Acc: {acc*100:.2f}%")
        val_p = self.forward(X_v, training=False)
        acc = np.mean(np.argmax(val_p, axis=1) == np.argmax(y_v, axis=1))
        return float(acc), np.argmax(y_v, axis=1), np.argmax(val_p, axis=1)

def save_clinical_report(y_true, y_pred, classes):
    
    report_dict = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    df = pd.DataFrame(report_dict).iloc[:-1, :5].T 
    
    plt.figure(figsize=(12, 7))
    sns.heatmap(df[['precision', 'recall', 'f1-score']], annot=True, cmap="YlGnBu", fmt=".3f", annot_kws={"size": 12})
    plt.title("Clinical Performance Analysis (Classification Report)", fontsize=16, fontweight='bold', pad=20)
    plt.ylabel("Condition", fontsize=12)
    plt.xlabel("Metric Score", fontsize=12)
    plt.tight_layout()
    plt.savefig('classification_report.png', dpi=300)
    print("✅ classification_report.png saved.")

if __name__ == "__main__":
    X, y = load_data()
    indices = np.arange(len(X)); np.random.shuffle(indices)
    X, y = X[indices], y[indices]
    
    fold_size = len(X) // K_FOLDS
    accuracies, truths, preds = [], [], []

    for k in range(K_FOLDS):
        print(f"\n--- 🧠 Fold {k+1}/{K_FOLDS} ---")
        v_idx = slice(k * fold_size, (k + 1) * fold_size)
        X_val, y_val = X[v_idx], y[v_idx]
        X_train = np.concatenate([X[:k * fold_size], X[(k + 1) * fold_size:]], axis=0)
        y_train = np.concatenate([y[:k * fold_size], y[(k + 1) * fold_size:]], axis=0)
        
        model = MedicalNN_Adam([IMG_SIZE*IMG_SIZE, 1024, len(CLASSES)])
        a, yt, yp = model.train_fold(X_train, y_train, X_val, y_val, k)
        accuracies.append(a); truths.extend(yt); preds.extend(yp)

   
    print("\n" + "="*60)
    print("📋 FINAL CLASSIFICATION REPORT (SKLEARN STYLE)")
    print("="*60)
    print(classification_report(truths, preds, target_names=CLASSES))
    print(f"🏆 Overall CV Accuracy: {np.mean(accuracies)*100:.2f}%")
    print("="*60)

  
    save_clinical_report(truths, preds, CLASSES)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix(truths, preds), annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title(f'Final Confusion Matrix (Acc: {np.mean(accuracies)*100:.1f}%)')
    plt.savefig('final_confusion_matrix.png')

    
    os.makedirs('weights', exist_ok=True)
    np.save('weights/W1.npy', model.W1); np.save('weights/b1.npy', model.b1)
    np.save('weights/W2.npy', model.W2); np.save('weights/b2.npy', model.b2)
