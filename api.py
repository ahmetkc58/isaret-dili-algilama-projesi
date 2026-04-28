import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
import difflib
from fastapi import FastAPI, File, UploadFile
import uvicorn
import io

# --- MODEL VE YARDIMCI SINIFLAR ---
# (Senin kodundaki squash, PrimaryCapsule ve CapsuleLayer sınıflarını buraya aynen ekliyoruz)
K = keras.backend

def squash(vectors, axis=-1):
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm)
    return scale * vectors / K.sqrt(s_squared_norm + K.epsilon())

class PrimaryCapsule(keras.layers.Layer):
    def __init__(self, dim_capsule, n_channels, kernel_size, strides, padding, **kwargs):
        super(PrimaryCapsule, self).__init__(**kwargs)
        self.dim_capsule, self.n_channels = dim_capsule, n_channels
        self.kernel_size, self.strides, self.padding = kernel_size, strides, padding
        self.conv = keras.layers.Conv2D(filters=dim_capsule * n_channels, kernel_size=kernel_size, strides=strides, padding=padding)
    def call(self, inputs):
        outputs = self.conv(inputs)
        batch_size = tf.shape(outputs)[0]
        outputs = K.reshape(outputs, [batch_size, -1, self.dim_capsule])
        return squash(outputs)

class CapsuleLayer(keras.layers.Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule, self.dim_capsule, self.routings = num_capsule, dim_capsule, routings
    def build(self, input_shape):
        self.W = self.add_weight(shape=[self.num_capsule, 144, self.dim_capsule, 8], initializer='glorot_uniform', name='W')
        self.built = True
    def call(self, inputs):
        inputs_expand = K.expand_dims(inputs, 1)
        inputs_tile = K.tile(inputs_expand, [1, self.num_capsule, 1, 1])
        inputs_hat = tf.einsum('abci,bcji->abcj', inputs_tile, self.W)
        b = tf.zeros([tf.shape(inputs_hat)[0], self.num_capsule, tf.shape(inputs)[1]])
        for i in range(self.routings):
            c = tf.nn.softmax(b, axis=1)
            s = tf.einsum('abc,abcd->abd', c, inputs_hat)
            outputs = squash(s)
            if i < self.routings - 1: b += tf.einsum('abd,abcd->abc', outputs, inputs_hat)
        return outputs

# --- API BAŞLATMA VE MODEL YÜKLEME ---
app = FastAPI()

classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'Y', 'Z', 'del', 'nothing', 'space']

def load_sign_model():
    input_layer = keras.layers.Input(shape=(224, 224, 3))
    base_model = keras.applications.MobileNetV2(weights=None, include_top=False, input_tensor=input_layer)
    x = base_model.output
    x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    primary_caps = PrimaryCapsule(8, 16, 3, 2, 'valid')(x) 
    digit_caps = CapsuleLayer(len(classes), 16, 3)(primary_caps)
    out_caps = keras.layers.Lambda(lambda z: K.sqrt(K.sum(K.square(z), axis=-1) + K.epsilon()))(digit_caps)
    model = keras.models.Model(inputs=input_layer, outputs=out_caps)
    model.load_weights('best_model.h5')
    return model

# Global model değişkeni (Sunucu açıldığında bir kez yüklenir)
model = load_sign_model()

def get_suggestion(text):
    if not text: return ""
    last_word = text.split()[-1] if text.split() else ""
    try:
        with open('turizm_sozluk.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        matches = difflib.get_close_matches(last_word.upper(), data["kelimeler"], n=1, cutoff=0.4)
        return matches[0] if matches else ""
    except: return ""

# --- ENDPOINT (İstek Noktası) ---
@app.post("/predict")
async def predict(file: UploadFile = File(...), current_text: str = ""):
    """
    Mobil uygulamadan gelen resmi alır ve tahmini döndürür.
    current_text: O ana kadar yazılmış olan metin (kelime önerisi için).
    """
    # 1. Gelen resmi oku
    request_object_content = await file.read()
    img = cv2.imdecode(np.frombuffer(request_object_content, np.uint8), cv2.IMREAD_COLOR)
    
    # 2. Ön işleme
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_final = np.expand_dims(img_resized.astype('float32') / 255.0, axis=0)

    # 3. Model Tahmini
    preds = model.predict(img_final, verbose=0)
    idx = np.argmax(preds[0])
    label = classes[idx]
    confidence = float(np.max(preds[0]))

    # 4. Kelime Önerisi (Eğer bir metin gelmişse)
    suggestion = get_suggestion(current_text + (label if label not in ['nothing', 'del', 'space'] else ""))

    return {
        "label": label,
        "confidence": confidence,
        "suggestion": suggestion
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)