import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
import difflib
import pyttsx3 

# TensorFlow/Keras Bileşenleri
layers = keras.layers
models = keras.models
K = keras.backend
MobileNetV2 = keras.applications.MobileNetV2

# --- 1. SESLENDİRME (TTS) FONKSİYONU ---
def seslendir(metin):
    """Her çağrıda motoru yeniden başlatarak kilitlenmeyi önler."""
    if metin and metin.strip():
        print(f"Seslendiriliyor: {metin}")
        try:
            # Motoru yerel olarak başlat (Kilitlenmeyi önlemek için en güvenli yol)
            engine = pyttsx3.init()
            engine.setProperty('rate', 150) # Okuma hızı
            
            # Türkçe ses ayarını dene
            voices = engine.getProperty('voices')
            for voice in voices:
                if "turkish" in voice.name.lower():
                    engine.setProperty('voice', voice.id)
                    break
            
            engine.say(metin)
            engine.runAndWait()
            
            # Kaynakları serbest bırak
            engine.stop()
            del engine 
        except Exception as e:
            print(f"Seslendirme hatası: {e}")

# --- 2. ÖZEL KATMANLAR (Notebook Uyumluluğu) ---
def squash(vectors, axis=-1):
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm)
    return scale * vectors / K.sqrt(s_squared_norm + K.epsilon())

class PrimaryCapsule(layers.Layer):
    def __init__(self, dim_capsule, n_channels, kernel_size, strides, padding, **kwargs):
        super(PrimaryCapsule, self).__init__(**kwargs)
        self.dim_capsule, self.n_channels = dim_capsule, n_channels
        self.kernel_size, self.strides, self.padding = kernel_size, strides, padding
        self.conv = layers.Conv2D(filters=dim_capsule * n_channels, kernel_size=kernel_size, strides=strides, padding=padding)

    def call(self, inputs):
        outputs = self.conv(inputs)
        batch_size = tf.shape(outputs)[0]
        outputs = K.reshape(outputs, [batch_size, -1, self.dim_capsule])
        return squash(outputs)

class CapsuleLayer(layers.Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule, self.dim_capsule, self.routings = num_capsule, dim_capsule, routings

    def build(self, input_shape):
        input_num_capsule = 144 
        self.W = self.add_weight(shape=[self.num_capsule, input_num_capsule, self.dim_capsule, 8],
                                 initializer='glorot_uniform', name='W')
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

# --- 3. MODEL KURULUMU ---
def build_model(num_classes=26):
    input_layer = layers.Input(shape=(224, 224, 3))
    base_model = MobileNetV2(weights=None, include_top=False, input_tensor=input_layer)
    x = base_model.output
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    primary_caps = PrimaryCapsule(8, 16, 3, 2, 'valid')(x) 
    digit_caps = CapsuleLayer(num_classes, 16, 3)(primary_caps)
    out_caps = layers.Lambda(lambda z: K.sqrt(K.sum(K.square(z), axis=-1) + K.epsilon()))(digit_caps)
    model = models.Model(inputs=input_layer, outputs=out_caps)
    print("Ağırlıklar yükleniyor...")
    model.load_weights('best_model.h5') 
    return model

def get_suggestion(text):
    if not text or text.strip() == "": return ""
    last_word = text.split()[-1] if text.split() else ""
    try:
        with open('turizm_sozluk.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        matches = difflib.get_close_matches(last_word.upper(), data["kelimeler"], n=1, cutoff=0.4)
        return matches[0] if matches else ""
    except: return ""

# --- ANA PROGRAM ---
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'Y', 'Z', 'del', 'nothing', 'space']
model = build_model()
accumulated_text = ""
cap = cv2.VideoCapture(0)

print("\n--- SİSTEM HAZIR ---")
print("A: Harf Ekle | S: Öneriyi Onayla ve Seslendir | V: Cümleyi Seslendir | Q: Çıkış")

while True:
    ret, frame = cap.read()
    if not ret: break

    roi = frame[100:400, 100:400]
    cv2.rectangle(frame, (100, 100), (400, 400), (0, 255, 0), 2)
    
    img = cv2.resize(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB), (224, 224))
    img = np.expand_dims(img.astype('float32') / 255.0, axis=0)

    preds = model.predict(img, verbose=0)
    label = classes[np.argmax(preds[0])]
    suggestion = get_suggestion(accumulated_text)
    
    cv2.putText(frame, f"Harf: {label}", (110, 80), 1, 1.8, (255, 0, 0), 2)
    cv2.putText(frame, f"Cumle: {accumulated_text}", (30, 430), 1, 1.2, (255, 255, 255), 2)
    if suggestion:
        cv2.putText(frame, f"Oneri: {suggestion}?", (30, 465), 1, 1.1, (0, 255, 255), 2)

    cv2.imshow('Sessiz Kelimeler - Turizm Asistani', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): 
        break
    elif key == ord('a'):
        if label == 'space': 
            accumulated_text += " "
        elif label == 'del': 
            accumulated_text = accumulated_text[:-1]
        elif label != 'nothing': 
            accumulated_text += label
            print(f"Güncel Metin: {accumulated_text}")

    elif key == ord('s'):
        if suggestion:
            words = accumulated_text.split()
            if words:
                words[-1] = suggestion
                accumulated_text = " ".join(words) + " "
            else:
                accumulated_text = suggestion + " "
            
            # Onaylanan kelimeyi seslendir
            seslendir(suggestion)
        else:
            print("Seçilecek bir öneri yok.")

    elif key == ord('v'):
        if accumulated_text.strip():
            seslendir(accumulated_text)
        else:
            print("Seslendirilecek metin boş.")

cap.release()
cv2.destroyAllWindows()