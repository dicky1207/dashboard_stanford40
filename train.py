# 1. Import Library dan Konfigurasi Awal
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import joblib
from sklearn.utils.class_weight import compute_class_weight


IMG_SIZE = (224, 224)          # Ukuran input MobileNetV2
BATCH_SIZE = 32
EPOCHS_STAGE1 = 30             # Epoch untuk tahap pertama (base model beku)
EPOCHS_STAGE2 = 15             # Epoch untuk fine-tuning (learning rate kecil)

CLASS_NAMES = joblib.load('class_names.joblib')     # Muat nama kelas dari hasil prepare_data.py
NUM_CLASSES = len(CLASS_NAMES)
print(f'Jumlah kelas: {NUM_CLASSES}')

# ============================================================
# 2. Data Augmentation dan Pembuatan Generator
# ============================================================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2          # 20% data training untuk validasi
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    subset='validation'
)

test_generator = test_datagen.flow_from_directory(
    'data/test',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    shuffle=False
)

print(f'Training samples: {train_generator.samples}')
print(f'Validation samples: {val_generator.samples}')
print(f'Test samples: {test_generator.samples}')

# ============================================================
# 3. CLASS WEIGHT (UNTUK MENGATASI IMBALANCED DATA)
# ============================================================
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weight_dict = dict(enumerate(class_weights))
print(f'Class weights: {class_weight_dict}')

# ============================================================
# 4. BANGUN MODEL TRANSFER LEARNING (MobileNetV2)
# ============================================================
# Muat MobileNetV2 tanpa lapisan klasifikasi
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False   # Bekukan semua layer di awal

# Tambahkan lapisan klasifikasi di atasnya
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

# Kompilasi dengan learning rate default untuk tahap 1
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ============================================================
# 5. Callback Stage 1
# ============================================================
early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=8,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    'best_har_model_stage1.keras',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=1e-6
)

# ============================================================
# 6. Stage 1: Pelatihan dengan Base Model Beku
# ============================================================
print("\n========== TAHAP 1: Pelatihan dengan Base Model Beku ==========")
history_stage1 = model.fit(
    train_generator,
    epochs=EPOCHS_STAGE1,
    validation_data=val_generator,
    callbacks=[early_stop, checkpoint, reduce_lr],
    class_weight=class_weight_dict,
    verbose=1
)

# Muat model terbaik dari stage 1
# Model yang dimuat ini akan digunakan untuk fine-tuning
model = tf.keras.models.load_model('best_har_model_stage1.keras')

# ============================================================
# 7. Stage 2: FINE-TUNING
# ============================================================
print("\n========== TAHAP 2: Fine-Tuning dengan Learning Rate Rendah ==========")

# AMBIL LAGI BASE_MODEL DARI MODEL YANG DIMUAT (bukan dari model awal)
base_model = model.layers[0]

# Buka 50 lapisan teratas agar ikut dilatih
base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False   # Bekukan 100 lapisan pertama, sisanya (sekitar 54) dapat dilatih

# Kompilasi ulang dengan learning rate sangat kecil
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Callback untuk fine-tuning (patience lebih kecil karena proses cepat)
checkpoint_fine = ModelCheckpoint(
    'best_har_model_final.keras',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)
early_stop_fine = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True
)

# Lanjutkan training
history_stage2 = model.fit(
    train_generator,
    epochs=EPOCHS_STAGE2,
    validation_data=val_generator,
    callbacks=[early_stop_fine, checkpoint_fine],
    class_weight=class_weight_dict,
    verbose=1
)

# Muat model final terbaik
model = tf.keras.models.load_model('best_har_model_final.keras')

# ============================================================
# 8. EVALUASI PADA DATA UJI
# ============================================================
test_loss, test_acc = model.evaluate(test_generator, verbose=0)
print(f'\nAkurasi akhir pada data uji: {test_acc:.4f}')

# Simpan model final dengan nama umum
model.save('har_40_model.keras')
print('Model final disimpan sebagai har_40_model.keras')

# ============================================================
# 9. PLOT KURVA PELATIHAN (GABUNGAN DUA TAHAP)
# ============================================================
# Gabungkan history dari dua tahap
acc = history_stage1.history['accuracy'] + history_stage2.history['accuracy']
val_acc = history_stage1.history['val_accuracy'] + history_stage2.history['val_accuracy']
loss = history_stage1.history['loss'] + history_stage2.history['loss']
val_loss = history_stage1.history['val_loss'] + history_stage2.history['val_loss']
epochs = range(1, len(acc)+1)

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(epochs, acc, label='Training')
plt.plot(epochs, val_acc, label='Validation')
plt.axvline(x=len(history_stage1.history['accuracy']), color='red', linestyle='--', label='Mulai Fine-Tuning')
plt.title('Akurasi per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Akurasi')
plt.legend()

plt.subplot(1,2,2)
plt.plot(epochs, loss, label='Training')
plt.plot(epochs, val_loss, label='Validation')
plt.axvline(x=len(history_stage1.history['loss']), color='red', linestyle='--', label='Mulai Fine-Tuning')
plt.title('Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig('training_curves.png')
plt.show()

# ============================================================
# 10. CONFUSION MATRIX DAN CLASSIFICATION REPORT
# ============================================================
test_generator.reset()
y_true = test_generator.classes
y_pred = model.predict(test_generator)
y_pred_class = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_true, y_pred_class)
plt.figure(figsize=(18, 16))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            annot_kws={'size': 8})
plt.title('Confusion Matrix - Data Uji')
plt.xlabel('Prediksi')
plt.ylabel('Aktual')
plt.xticks(rotation=90, fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

print('\n========== CLASSIFICATION REPORT ==========')
print(classification_report(y_true, y_pred_class, target_names=CLASS_NAMES))

print('\nPelatihan dan evaluasi selesai.')