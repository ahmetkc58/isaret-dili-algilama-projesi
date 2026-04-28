# Gerekli kütüphanelerin içe aktarılması
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, backend as K
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold, ParameterGrid
import numpy as np
import os

# GPU kullanımı için (varsa)
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Squash fonksiyonu
def squash(vectors, axis=-1):
    """
    Squash fonksiyonu, kapsül vektörlerini normalleştirir.
    """
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm)
    return scale * vectors / K.sqrt(s_squared_norm + K.epsilon())

# Primary Capsule katmanı
class PrimaryCapsule(layers.Layer):
    def __init__(self, dim_capsule, n_channels, kernel_size, strides, padding, **kwargs):
        super(PrimaryCapsule, self).__init__(**kwargs)
        self.dim_capsule = dim_capsule
        self.n_channels = n_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding

    def build(self, input_shape):
        self.conv = layers.Conv2D(
            filters=self.dim_capsule * self.n_channels,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding
        )

    def call(self, inputs):
        outputs = self.conv(inputs)
        outputs = layers.Reshape(target_shape=[-1, self.dim_capsule])(outputs)
        return squash(outputs)

# Kapsül katmanı
class CapsuleLayer(layers.Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule      # Kapsül sayısı (sınıf sayısı)
        self.dim_capsule = dim_capsule      # Kapsül boyutu
        self.routings = routings            # Dinamik yönlendirme tekrar sayısı

    def build(self, input_shape):
        # Giriş kapsül sayısı ve boyutu
        self.input_num_capsule = int(input_shape[1])
        self.input_dim_capsule = int(input_shape[2])

        # Ağırlık matrisi
        self.W = self.add_weight(
            shape=[self.num_capsule, self.input_num_capsule, self.dim_capsule, self.input_dim_capsule],
            initializer='glorot_uniform',
            trainable=True,
            name='W'
        )

    def call(self, inputs):
        # Girişleri genişletme ve matris çarpımı
        inputs_expand = K.expand_dims(inputs, axis=1)
        inputs_tile = K.tile(inputs_expand, [1, self.num_capsule, 1, 1])
        inputs_hat = K.map_fn(lambda x: K.batch_dot(self.W, x, [3, 2]), elems=inputs_tile)

        # Başlatma
        b = K.zeros(shape=[K.shape(inputs_hat)[0], self.num_capsule, self.input_num_capsule])

        # Dinamik yönlendirme
        for i in range(self.routings):
            c = tf.nn.softmax(b, axis=1)
            outputs = squash(K.batch_dot(c, inputs_hat, [2, 2]))
            if i < self.routings - 1:
                b += K.batch_dot(outputs, inputs_hat, [2, 3])

        return outputs

# Modelin oluşturulması
def create_model(learning_rate=0.001, num_classes=10):
    # MobileNetV2 önceden eğitilmiş modeli
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Bazı katmanları eğitilebilir hale getirin
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    
    # Özellik çıkarımı
    x = base_model.output
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # Primary Capsule katmanı
    primary_caps = PrimaryCapsule(
        dim_capsule=8,
        n_channels=16,
        kernel_size=3,
        strides=2,
        padding='valid'
    )(x)
    
    # Digit Capsule katmanı
    digit_caps = CapsuleLayer(
        num_capsule=num_classes,  # Sınıf sayısı
        dim_capsule=16,
        routings=3
    )(primary_caps)
    
    # Kapsüllerin büyüklüklerini alarak çıktı katmanı
    out_caps = layers.Lambda(lambda z: K.sqrt(K.sum(K.square(z), axis=-1)))(digit_caps)
    
    # Modelin oluşturulması
    model = models.Model(inputs=base_model.input, outputs=out_caps)
    
    # Modelin derlenmesi
    model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# Veri setinin hazırlanması
def prepare_data(data_dir, batch_size=32):
    # Veri artırma ve ön işleme
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.1)

    # Eğitim verisi
    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    # Doğrulama verisi
    validation_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=True
    )

    # Sınıf sayısı
    num_classes = train_generator.num_classes

    return train_generator, validation_generator, num_classes

# Hiperparametre optimizasyonu
def hyperparameter_optimization(train_generator, validation_generator, num_classes):
    # Hiperparametre ızgarası
    param_grid = {
        'learning_rate': [0.001, 0.0001],
        'batch_size': [16, 32],
        'epochs': [5, 10]
    }

    best_accuracy = 0
    best_params = {}
    best_model = None

    for params in ParameterGrid(param_grid):
        print(f"Parametreler: {params}")
        
        # Modeli oluştur
        model = create_model(learning_rate=params['learning_rate'], num_classes=num_classes)
        
        # Erken durdurma callback'i
        early_stopping = EarlyStopping(monitor='val_loss', patience=3)
        
        # Modelin eğitilmesi
        history = model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=params['epochs'],
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Doğrulama doğruluğunu al
        val_accuracy = history.history['val_accuracy'][-1]
        print(f"Doğrulama Doğruluğu: {val_accuracy}")
        
        # En iyi modeli kaydet
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_params = params
            best_model = model
            model.save('best_model.h5')

    print(f"En iyi doğrulama doğruluğu: {best_accuracy} ile Parametreler: {best_params}")
    return best_model, best_params

# K-Fold çapraz doğrulama
def k_fold_cross_validation(data_dir, num_classes, k=5, epochs=5, learning_rate=0.001):
    # Veri setini Numpy arraylerine dönüştürme
    datagen = ImageDataGenerator(rescale=1./255)
    generator = datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )

    # Verileri ve etiketleri alma
    X = []
    y = []

    for _ in range(len(generator)):
        X_batch, y_batch = next(generator)
        X.append(X_batch)
        y.append(y_batch)

    X = np.vstack(X)
    y = np.vstack(y)

    # K-Fold çapraz doğrulama
    kf = KFold(n_splits=k, shuffle=True)

    fold = 1
    for train_index, val_index in kf.split(X):
        print(f"KFold {fold}/{k}")
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        # Modeli oluştur
        model = create_model(learning_rate=learning_rate, num_classes=num_classes)
        
        # Erken durdurma callback'i
        early_stopping = EarlyStopping(monitor='val_loss', patience=3)
        
        # Modelin eğitilmesi
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Doğrulama doğruluğunu al
        val_accuracy = history.history['val_accuracy'][-1]
        print(f"KFold {fold} Doğrulama Doğruluğu: {val_accuracy}")
        fold += 1

# Ana fonksiyon
def main():
    data_dir = 'data/'  # Veri setinizin klasör yolu
    batch_size = 32

    # Veri setini hazırla
    train_generator, validation_generator, num_classes = prepare_data(data_dir, batch_size)

    # Hiperparametre optimizasyonu
    best_model, best_params = hyperparameter_optimization(train_generator, validation_generator, num_classes)

    # K-Fold çapraz doğrulama (isteğe bağlı)
    # k_fold_cross_validation(data_dir, num_classes, k=5, epochs=5, learning_rate=best_params['learning_rate'])

    # En iyi model ile tahmin yapma veya kaydetme
    # best_model.save('mobilenet_capsule_best_model.h5')

if __name__ == "__main__":
    main()
