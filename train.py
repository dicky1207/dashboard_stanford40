import os
import sys
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, Dropout, GlobalAveragePooling2D, Input, Lambda, Concatenate
)
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import joblib
from sklearn.utils.class_weight import compute_class_weight
import random

# ============================================================
# 1. KONFIGURASI AWAL
# ============================================================
IMG_SIZE = (224, 224)          # ResNet50V2 juga pakai 224x224
BATCH_SIZE = 32
EPOCHS_STAGE1 = 30
EPOCHS_STAGE2 = 15
LEARNING_RATE_STAGE1 = 1e-3
LEARNING_RATE_STAGE2 = 1e-5
ALPHA_MIXUP = 0.2              # Parameter untuk Mixup

# ============================================================
# 2. VALIDASI FILE
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

# ============================================================
# 4. DATA GENERATOR DASAR (TANPA AUGMENTASI BERAT)
# ============================================================
# Kita akan menerapkan augmentasi manual di dalam generator custom
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)

# ============================================================
# 5. FUNGSI AUGMENTASI CUTOUT (RANDOM ERASING)
# ============================================================
def cutout(image, mask_size=(16, 16), num_masks=1):
    """Menerapkan Cutout (Random Erasing) pada gambar."""
    h, w, _ = image.shape
    mask_h, mask_w = mask_size
    for _ in range(num_masks):
        # Pilih posisi acak untuk mask
        y = np.random.randint(0, h - mask_h)
        x = np.random.randint(0, w - mask_w)
        # Set pixel di area mask menjadi 0 (hitam)
        image[y:y+mask_h, x:x+mask_w, :] = 0
    return image

# ============================================================
# 6. FUNGSI MIXUP (UNTUK BATCH)
# ============================================================
def mixup_batch(images, labels, alpha=0.2):
    """Menerapkan Mixup pada satu batch."""
    batch_size = tf.shape(images)[0]
    # Sample lambda dari distribusi Beta
    lam = tf.random.uniform((batch_size, 1, 1, 1), 0, 1, dtype=tf.float32)
    lam = tf.where(lam < alpha, tf.zeros_like(lam), lam)  # stabilisasi
    
    # Indeks untuk shuffle
    indices = tf.random.shuffle(tf.range(batch_size))
    
    # Gambar campuran
    mixed_images = lam * images + (1 - lam) * tf.gather(images, indices)
    
    # Label campuran (soft label)
    mixed_labels = lam * labels + (1 - lam) * tf.gather(labels, indices)
    
    return mixed_images, mixed_labels

# ============================================================
# 7. CUSTOM GENERATOR DENGAN CUTOUT DAN MIXUP
# ============================================================
class CustomDataGenerator(tf.keras.utils.Sequence):
    """Generator dengan Cutout dan Mixup."""
    def __init__(self, generator, use_cutout=True, use_mixup=True, alpha=0.2):
        self.generator = generator
        self.use_cutout = use_cutout
        self.use_mixup = use_mixup
        self.alpha = alpha
        self.batch_size = generator.batch_size
        self.samples = generator.samples
        self.classes = generator.classes
        
    def __len__(self):
        return len(self.generator)
    
    def __getitem__(self, index):
        # Ambil batch dari generator dasar
        images, labels = self.generator[index]
        
        # Terapkan Cutout (jika diaktifkan)
        if self.use_cutout:
            for i in range(images.shape[0]):
                images[i] = cutout(images[i], mask_size=(24, 24), num_masks=2)
        
        # Konversi label ke one-hot untuk Mixup
        labels_onehot = tf.keras.utils.to_categorical(labels, NUM_CLASSES)
        
        # Terapkan Mixup (jika diaktifkan)
        if self.use_mixup:
            images, labels_onehot = mixup_batch(images, labels_onehot, self.alpha)
        
        return images, labels_onehot

# ============================================================
# 8. MEMBUAT GENERATOR DASAR
# ============================================================
print('\n📊 Menyiapkan generator data...')

train_base = train_datagen.flow_from_directory(
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
    subset='validation',
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    'data/test',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    shuffle=False
)

print(f'✅ Training samples: {train_base.samples}')
print(f'✅ Validation samples: {val_generator.samples}')
print(f'✅ Test samples: {test_generator.samples}')

# ============================================================
# 9. BUNGKUS GENERATOR DENGAN CUSTOM (CUTOUT + MIXUP)
# ============================================================
train_generator = CustomDataGenerator(
    train_base,
    use_cutout=True,
    use_mixup=True,
    alpha=ALPHA_MIXUP
)

# ============================================================
# 10. CLASS WEIGHT (TETAP DIPAKAI UNTUK IMBALANCE)
# ============================================================
print('\n⚖️  Menghitung class weights...')
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_base.classes),
    y=train_base.classes
)
class_weight_dict = dict(enumerate(class_weights))
print(f'📊 Class weights: {class_weight_dict}')

# ============================================================
# 11. BANGUN MODEL DENGAN RESNET50V2
# ============================================================
print('\n🏗️  Membangun model ResNet50V2...')

# Input layer
inputs = Input(shape=(224, 224, 3))

# Base model ResNet50V2 (pre-trained ImageNet)
base_model = ResNet50V2(
    include_top=False,
    weights='imagenet',
    input_tensor=inputs
)
base_model.trainable = False

# Tambahkan lapisan klasifikasi
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE_STAGE1),
    loss='categorical_crossentropy',  # Karena Mixup menghasilkan soft label
    metrics=['accuracy']
)

model.summary()

# ============================================================
# 12. CALLBACK
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
# 13. TAHAP 1: PELATIHAN DENGAN BASE MODEL BEKU
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

# Muat model terbaik
model = tf.keras.models.load_model('best_har_model_stage1.keras')
print('✅ Model terbaik stage 1 dimuat.')

# ============================================================
# 14. TAHAP 2: FINE-TUNING (BUKA 50 LAPISAN TERAKHIR)
# ============================================================
print('\n' + '='*60)
print('🚀 TAHAP 2: Fine-Tuning dengan Learning Rate Rendah')
print('='*60)

# Ambil base model dari model yang dimuat
base_model = model.layers[1]  # ResNet50V2 adalah layer ke-1 (input layer ke-0)
base_model.trainable = True

# Bekukan semua lapisan kecuali 50 lapisan terakhir
for layer in base_model.layers[:-50]:
    layer.trainable = False

print(f'🔓 {sum(1 for l in base_model.layers if l.trainable)} lapisan dilatih (dari {len(base_model.layers)} total)')

# Kompilasi ulang
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE_STAGE2),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callback fine-tuning
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

history_stage2 = model.fit(
    train_generator,
    epochs=EPOCHS_STAGE2,
    validation_data=val_generator,
    callbacks=[early_stop_fine, checkpoint_fine],
    class_weight=class_weight_dict,
    verbose=1
)

# Muat model final
model = tf.keras.models.load_model('best_har_model_final.keras')
print('✅ Model final dimuat.')

# ============================================================
# 15. EVALUASI PADA DATA UJI
# ============================================================
print('\n' + '='*60)
print('📊 EVALUASI PADA DATA UJI')
print('='*60)

# Untuk evaluasi, kita tetap pakai sparse (karena model output softmax)
# Tapi karena training pakai categorical, evaluasi tetap bisa.
test_loss, test_acc = model.evaluate(test_generator, verbose=0)
print(f'\n🎯 Akurasi akhir pada data uji: {test_acc:.4f} ({test_acc*100:.2f}%)')

model.save('har_40_model.keras')
print('💾 Model final disimpan sebagai har_40_model.keras')

# ============================================================
# 16. PLOT KURVA
# ============================================================
print('\n📈 Membuat plot kurva pelatihan...')

acc = history_stage1.history['accuracy'] + history_stage2.history['accuracy']
val_acc = history_stage1.history['val_accuracy'] + history_stage2.history['val_accuracy']
loss = history_stage1.history['loss'] + history_stage2.history['loss']
val_loss = history_stage1.history['val_loss'] + history_stage2.history['val_loss']
epochs = range(1, len(acc)+1)

plt.figure(figsize=(12, 4))
plt.subplot(1,2,1)
plt.plot(epochs, acc, 'b-', label='Training')
plt.plot(epochs, val_acc, 'r-', label='Validation')
plt.axvline(x=len(history_stage1.history['accuracy']), color='g', linestyle='--', label='Mulai Fine-Tuning')
plt.title('Akurasi per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Akurasi')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1,2,2)
plt.plot(epochs, loss, 'b-', label='Training')
plt.plot(epochs, val_loss, 'r-', label='Validation')
plt.axvline(x=len(history_stage1.history['loss']), color='g', linestyle='--', label='Mulai Fine-Tuning')
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
# 17. CONFUSION MATRIX
# ============================================================
print('\n📊 Membuat confusion matrix...')
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
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()
print('💾 Confusion matrix disimpan ke confusion_matrix.png')

print('\n' + '='*60)
print('📋 CLASSIFICATION REPORT')
print('='*60)
print(classification_report(y_true, y_pred_class, target_names=CLASS_NAMES))

print('\n✅ Pelatihan dan evaluasi selesai!')