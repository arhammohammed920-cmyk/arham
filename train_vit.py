import os
import random
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageEnhance
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras import layers

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
                # Load as RGB
                raw_img = Image.open(os.path.join(path, img_name)).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
                for _ in range(reps):
                    aug = augment_image(raw_img) if _ > 0 else raw_img
                    X.append(np.array(aug) / 255.0)
                    oh = np.zeros(len(CLASSES)); oh[idx] = 1
                    y.append(oh)
            except: continue
    X_arr = np.array(X)
    return X_arr, np.array(y)

# ViT Architecture
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

@tf.keras.utils.register_keras_serializable()
class Patches(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super(Patches, self).__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

    def get_config(self):
        config = super(Patches, self).get_config()
        config.update({"patch_size": self.patch_size})
        return config

@tf.keras.utils.register_keras_serializable()
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super(PatchEncoder, self).__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

    def get_config(self):
        config = super(PatchEncoder, self).get_config()
        config.update({"num_patches": self.num_patches, "projection_dim": self.projection_dim})
        return config

def create_vit_classifier(input_shape=(128, 128, 3), num_classes=5):
    data_augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(factor=0.1),
            layers.RandomZoom(height_factor=0.1, width_factor=0.1),
        ],
        name="data_augmentation",
    )

    patch_size = 16
    num_patches = (input_shape[0] // patch_size) ** 2
    projection_dim = 128
    num_heads = 8
    transformer_units = [projection_dim * 2, projection_dim]
    transformer_layers = 8
    mlp_head_units = [512, 256]

    inputs = layers.Input(shape=input_shape)
    augmented = data_augmentation(inputs)
    patches = Patches(patch_size)(augmented)
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    for _ in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.2
        )(x1, x1)
        x2 = layers.Add()([attention_output, encoded_patches])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.2)
        encoded_patches = layers.Add()([x3, x2])

    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.GlobalAveragePooling1D()(representation)
    representation = layers.Dropout(0.5)(representation)
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    logits = layers.Dense(num_classes, activation="softmax")(features)

    model = tf.keras.Model(inputs=inputs, outputs=logits)
    return model

def main():
    X, y = load_data()
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X, y = X[indices], y[indices]
    
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    model = create_vit_classifier()
    
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.001, decay_steps=50 * max(1, len(X_train) // BATCH_SIZE)
    )
    optimizer = tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=0.0001) if hasattr(tf.keras.optimizers, 'AdamW') else tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
    ]

    print("🚀 Training Vision Transformer...")
    model.fit(
        x=X_train, y=y_train,
        batch_size=BATCH_SIZE,
        epochs=50,
        validation_data=(X_val, y_val),
        callbacks=callbacks
    )

    print("📊 Generating predictions and metrics...")
    val_preds = model.predict(X_val)
    y_pred_classes = np.argmax(val_preds, axis=1)
    y_true_classes = np.argmax(y_val, axis=1)

    print("\n" + "="*60)
    print("📋 ViT CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_true_classes, y_pred_classes, target_names=CLASSES))
    print("="*60)
    
    report_dict = classification_report(y_true_classes, y_pred_classes, target_names=CLASSES, output_dict=True)
    df = pd.DataFrame(report_dict).iloc[:-1, :5].T 
    plt.figure(figsize=(12, 7))
    sns.heatmap(df[['precision', 'recall', 'f1-score']], annot=True, cmap="YlGnBu", fmt=".3f", annot_kws={"size": 12})
    plt.title("ViT - Classification Report", fontsize=16, fontweight='bold', pad=20)
    plt.ylabel("Condition", fontsize=12)
    plt.xlabel("Metric Score", fontsize=12)
    plt.tight_layout()
    plt.savefig('vit_classification_report.png', dpi=300)
    print("✅ vit_classification_report.png saved.")

    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix(y_true_classes, y_pred_classes), annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title('ViT - Confusion Matrix')
    plt.tight_layout()
    plt.savefig('vit_confusion_matrix.png', dpi=300)
    print("✅ vit_confusion_matrix.png saved.")
    
    os.makedirs('weights', exist_ok=True)
    model.save('weights/vit_model.h5')
    print("✅ Model weights saved to weights/vit_model.h5")

if __name__ == "__main__":
    main()
