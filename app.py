import os
import io
import numpy as np
from PIL import Image
from flask import Flask, render_template, request

app = Flask(__name__)

import tensorflow as tf
from train_vit import Patches, PatchEncoder

try:
    model = tf.keras.models.load_model('weights/vit_model.h5', custom_objects={'Patches': Patches, 'PatchEncoder': PatchEncoder})
    print("✅ ViT Model Loaded: Ready for Inference.")
except Exception as e:
    print(f"❌ ERROR: Missing or invalid weight files in /weights folder. {e}")
    model = None

CLASSES = ['Acute Otitis Media', 'Cerumen Impaction', 'Chronic Otitis Media', 'Myringosclerosis', 'Normal']

def predict_image(img_bytes):
    if model is None:
        return "Model not loaded", 0.0
    
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB').resize((128, 128))
    x = np.array(img) / 255.0
    x = np.expand_dims(x, axis=0) # Add batch dimension
    
    preds = model.predict(x)
    idx = np.argmax(preds[0])
    conf = float(preds[0][idx]) * 100
    
    return CLASSES[idx], round(conf, 2)

@app.route('/')
def index(): return render_template('index.html')

@app.route('/diseases')
def diseases(): return render_template('diseases.html')

@app.route('/diagnose', methods=['GET', 'POST'])
def diagnose():
    result = None
    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            label, conf = predict_image(file.read())
            result = {"label": label, "conf": conf}
    return render_template('diagnose.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
