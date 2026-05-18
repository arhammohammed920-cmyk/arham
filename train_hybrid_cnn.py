import os
import random
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageEnhance
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras import layers, models

DATA_DIR = 'ear_data'
IMG_SIZE = 128
CLASSES = ['Acute Otitis Media', 'Cerumen Impaction', 'Chronic Otitis Media', 'Myringosclerosis', 'Normal']
EPOCHS = 30
BATCH_SIZE = 16

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
                raw_img = Image.open(os.path.join(path, img_name)).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
                for _ in range(reps):
                    aug = augment_image(raw_img) if _ > 0 else raw_img
                    X.append(np.array(aug) / 255.0)
                    oh = np.zeros(len(CLASSES)); oh[idx] = 1
                    y.append(oh)
            except: continue
    X_arr = np.array(X)
    return X_arr, np.array(y)

def create_hybrid_cnn(input_shape=(128, 128, 3), num_classes=5):
    inputs = layers.Input(shape=input_shape)
    
    # Branch 1: standard conv with 3x3 kernels
    x1 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x1 = layers.MaxPooling2D(2, 2)(x1)
    x1 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x1)
    x1 = layers.MaxPooling2D(2, 2)(x1)
    x1 = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x1)
    x1 = layers.GlobalAveragePooling2D()(x1)

    # Branch 2: broader context with 5x5 kernels
    x2 = layers.Conv2D(32, (5, 5), padding='same', activation='relu')(inputs)
    x2 = layers.MaxPooling2D(2, 2)(x2)
    x2 = layers.Conv2D(64, (5, 5), padding='same', activation='relu')(x2)
    x2 = layers.MaxPooling2D(2, 2)(x2)
    x2 = layers.Conv2D(128, (5, 5), padding='same', activation='relu')(x2)
    x2 = layers.GlobalAveragePooling2D()(x2)
    
    # Combine branches (Hybrid representation)
    combined = layers.Concatenate()([x1, x2])
    
    # Recurrent aspect (Sequence from features, optional in Hybrid, but let's use Dense for strong features)
    # Alternatively expand dims to sequence for LSTM/GRU
    x_seq = layers.Reshape((1, 256))(combined)
    x_rnn = layers.LSTM(64, return_sequences=False)(x_seq)
    
    x = layers.Dense(128, activation='relu')(x_rnn)
    x = layers.Dropout(0.5)(x)
    logits = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, logits)
    return model

def main():
    X, y = load_data()
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X, y = X[indices], y[indices]
    
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    model = create_hybrid_cnn()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("🚀 Training Hybrid CNN...")
    model.fit(
        x=X_train, y=y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val)
    )

    print("📊 Generating predictions and metrics...")
    val_preds = model.predict(X_val)
    y_pred_classes = np.argmax(val_preds, axis=1)
    y_true_classes = np.argmax(y_val, axis=1)

    print("\n" + "="*60)
    print("📋 HYBRID CNN CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_true_classes, y_pred_classes, target_names=CLASSES))
    print("="*60)
    
    report_dict = classification_report(y_true_classes, y_pred_classes, target_names=CLASSES, output_dict=True)
    df = pd.DataFrame(report_dict).iloc[:-1, :5].T 
    plt.figure(figsize=(12, 7))
    sns.heatmap(df[['precision', 'recall', 'f1-score']], annot=True, cmap="YlGnBu", fmt=".3f", annot_kws={"size": 12})
    plt.title("Hybrid CNN - Classification Report", fontsize=16, fontweight='bold', pad=20)
    plt.ylabel("Condition", fontsize=12)
    plt.xlabel("Metric Score", fontsize=12)
    plt.tight_layout()
    plt.savefig('hybrid_cnn_classification_report.png', dpi=300)
    print("✅ hybrid_cnn_classification_report.png saved.")

    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix(y_true_classes, y_pred_classes), annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title('Hybrid CNN - Confusion Matrix')
    plt.tight_layout()
    plt.savefig('hybrid_cnn_confusion_matrix.png', dpi=300)
    print("✅ hybrid_cnn_confusion_matrix.png saved.")
    
    os.makedirs('weights', exist_ok=True)
    model.save('weights/hybrid_cnn_model.h5')
    print("✅ Model weights saved to weights/hybrid_cnn_model.h5")

if __name__ == "__main__":
    main()
