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

def squeeze_excite_block(input_tensor, ratio=8):
    filters = input_tensor.shape[-1]
    se = layers.GlobalAveragePooling2D()(input_tensor)
    se = layers.Reshape((1, 1, filters))(se)
    se = layers.Dense(filters // ratio, activation='relu', use_bias=False)(se)
    se = layers.Dense(filters, activation='sigmoid', use_bias=False)(se)
    return layers.Multiply()([input_tensor, se])

def conv_block(input_tensor, num_filters):
    x = layers.Conv2D(num_filters, (3, 3), padding="same")(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    
    x = layers.Conv2D(num_filters, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    
    # Channel Attention
    x = squeeze_excite_block(x)
    return x

def create_chale_unet_classifier(input_shape=(128, 128, 3), num_classes=5):
    inputs = layers.Input(shape=input_shape)
    
    # UNet Encoder with Channel Attention (CHALE)
    c1 = conv_block(inputs, 32)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    
    c2 = conv_block(p1, 64)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    
    c3 = conv_block(p2, 128)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    
    c4 = conv_block(p3, 256)
    
    # Instead of UNet decoder, we use the bottleneck features for classification
    x = layers.GlobalAveragePooling2D()(c4)
    x = layers.Dense(128, activation='relu')(x)
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

    model = create_chale_unet_classifier()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("🚀 Training CHALE UNet...")
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
    print("📋 CHALE UNet CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_true_classes, y_pred_classes, target_names=CLASSES))
    print("="*60)
    
    report_dict = classification_report(y_true_classes, y_pred_classes, target_names=CLASSES, output_dict=True)
    df = pd.DataFrame(report_dict).iloc[:-1, :5].T 
    plt.figure(figsize=(12, 7))
    sns.heatmap(df[['precision', 'recall', 'f1-score']], annot=True, cmap="YlGnBu", fmt=".3f", annot_kws={"size": 12})
    plt.title("CHALE UNet - Classification Report", fontsize=16, fontweight='bold', pad=20)
    plt.ylabel("Condition", fontsize=12)
    plt.xlabel("Metric Score", fontsize=12)
    plt.tight_layout()
    plt.savefig('chale_unet_classification_report.png', dpi=300)
    print("✅ clahe_unet_classification_report.png saved.")

    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix(y_true_classes, y_pred_classes), annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title('CLAHE UNet - Confusion Matrix')
    plt.tight_layout()
    plt.savefig('chale_unet_confusion_matrix.png', dpi=300)
    print("✅ clahe_unet_confusion_matrix.png saved.")
    
    os.makedirs('weights', exist_ok=True)
    model.save('weights/chale_unet_model.h5')
    print("✅ Model weights saved to weights/chale_unet_model.h5")

if __name__ == "__main__":
    main()
