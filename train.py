import os
import sys
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

# ============================================================
# 1. KONFIGURASI AWAL
# ============================================================
IMG_SIZE = (224, 224)          # Ukuran input MobileNetV2
BATCH_SIZE = 32
EPOCHS_STAGE1 = 30             # Epoch untuk tahap pertama
EPOCHS_STAGE2 = 10             # Epoch untuk fine-tuning (lebih sedikit)
LEARNING_RATE_STAGE1 = 1e-3
LEARNING_RATE_STAGE2 = 1e-5

# ============================================================
# 2. VALIDASI FILE YANG DIPERLUKAN
# ============================================================
if not os.path.exists('class_names.joblib'):
    print("❌ ERROR: File 'class_names.joblib' tidak ditemukan!")
    print("Jalankan 'prepare_data.py' terlebih dahulu.")
    sys.exit(1)

if not os.path.exists('data/train'):
    print("❌ ERROR: Folder 'data/train' tidak ditemukan!")
    print("Jalankan 'prepare_data.py' terlebih dahulu.")
    sys.exit(1)

# ============================================================
# 3. MUAT NAMA KELAS
# ============================================================
CLASS_NAMES = joblib.load('class_names.joblib')
NUM_CLASSES = len(CLASS_NAMES)
print(f'🏷️  Jumlah kelas: {NUM_CLASSES}')
print(f'📋 Nama kelas: {CLASS_NAMES[:5]}... ({NUM_CLASSES} total)')

# ============================================================
# 4. DATA AUGMENTATION DAN GENERATOR
# ============================================================
print('\n📊 Menyiapkan generator data...')

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
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

print(f'✅ Training samples: {train_generator.samples}')
print(f'✅ Validation samples: {val_generator.samples}')
print(f'✅ Test samples: {test_generator.samples}')

# Validasi ukuran dataset
if train_generator.samples < 100:
    print("⚠️  PERINGATAN: Dataset training sangat kecil (<100). Akurasi mungkin rendah.")
if val_generator.samples < 50:
    print("⚠️  PERINGATAN: Dataset validasi sangat kecil (<50). Evaluasi mungkin tidak stabil.")

# ============================================================
# 5. CLASS WEIGHT (UNTUK MENGATASI IMBALANCED DATA)
# ============================================================
print('\n⚖️  Menghitung class weights...')
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weight_dict = dict(enumerate(class_weights))
print(f'📊 Class weights: {class_weight_dict}')

# ============================================================
# 6. BANGUN MODEL TRANSFER LEARNING
# ============================================================
print('\n🏗️  Membangun model MobileNetV2...')

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE_STAGE1),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ============================================================
# 7. CALLBACK UNTUK TAHAP 1
# ============================================================
print('\n📋 Menyiapkan callback...')

early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=8,
    restore_best_weights=True,
    verbose=1
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
    min_lr=1e-6,
    verbose=1
)

# ============================================================
# 8. TAHAP 1: PELATIHAN DENGAN BASE MODEL BEKU
# ============================================================
print('\n' + '='*60)
print('🚀 TAHAP 1: Pelatihan dengan Base Model Beku')
print('='*60)

history_stage1 = model.fit(
    train_generator,
    epochs=EPOCHS_STAGE1,
    validation_data=val_generator,
    callbacks=[early_stop, checkpoint, reduce_lr],
    class_weight=class_weight_dict,
    verbose=1
)

# Muat model terbaik dari stage 1
model = tf.keras.models.load_model('best_har_model_stage1.keras')
print('✅ Model terbaik stage 1 dimuat.')

# ============================================================
# 9. TAHAP 2: FINE-TUNING (LEBIH HATI-HATI)
# ============================================================
print('\n' + '='*60)
print('🚀 TAHAP 2: Fine-Tuning dengan Learning Rate Rendah')
print('='*60)

# Ambil base_model dari model yang dimuat
base_model = model.layers[0]

# Buka hanya 20 lapisan terakhir (lebih aman)
# Total lapisan MobileNetV2 ~154, kita buka 20 lapisan teratas
base_model.trainable = True
for layer in base_model.layers[:-20]:  # Bekukan semua kecuali 20 lapisan terakhir
    layer.trainable = False

print(f'🔓 {sum(1 for l in base_model.layers if l.trainable)} lapisan dilatih (dari {len(base_model.layers)} total)')

# Kompilasi ulang dengan learning rate sangat kecil
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE_STAGE2),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Callback untuk fine-tuning
checkpoint_fine = ModelCheckpoint(
    'best_har_model_final.keras',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

early_stop_fine = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True,
    verbose=1
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
print('✅ Model final dimuat.')

# ============================================================
# 10. EVALUASI PADA DATA UJI
# ============================================================
print('\n' + '='*60)
print('📊 EVALUASI PADA DATA UJI')
print('='*60)

test_loss, test_acc = model.evaluate(test_generator, verbose=0)
print(f'\n🎯 Akurasi akhir pada data uji: {test_acc:.4f} ({test_acc*100:.2f}%)')

# Simpan model final
model.save('har_40_model.keras')
print('💾 Model final disimpan sebagai har_40_model.keras')

# ============================================================
# 11. PLOT KURVA PELATIHAN
# ============================================================
print('\n📈 Membuat plot kurva pelatihan...')

acc = history_stage1.history['accuracy'] + history_stage2.history['accuracy']
val_acc = history_stage1.history['val_accuracy'] + history_stage2.history['val_accuracy']
loss = history_stage1.history['loss'] + history_stage2.history['loss']
val_loss = history_stage1.history['val_loss'] + history_stage2.history['val_loss']
epochs = range(1, len(acc)+1)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(epochs, acc, 'b-', label='Training')
plt.plot(epochs, val_acc, 'r-', label='Validation')
plt.axvline(x=len(history_stage1.history['accuracy']), color='g', linestyle='--', 
            label='Mulai Fine-Tuning')
plt.title('Akurasi per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Akurasi')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'b-', label='Training')
plt.plot(epochs, val_loss, 'r-', label='Validation')
plt.axvline(x=len(history_stage1.history['loss']), color='g', linestyle='--', 
            label='Mulai Fine-Tuning')
plt.title('Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150)
plt.show()
print('💾 Kurva pelatihan disimpan ke training_curves.png')

# ============================================================
# 12. CONFUSION MATRIX DAN CLASSIFICATION REPORT
# ============================================================
print('\n📊 Membuat confusion matrix...')

test_generator.reset()
y_true = test_generator.classes
y_pred = model.predict(test_generator)
y_pred_class = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_true, y_pred_class)

# Plot confusion matrix
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
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()
print('💾 Confusion matrix disimpan ke confusion_matrix.png')

print('\n' + '='*60)
print('📋 CLASSIFICATION REPORT')
print('='*60)
print(classification_report(y_true, y_pred_class, target_names=CLASS_NAMES))

print('\n✅ Pelatihan dan evaluasi selesai!')