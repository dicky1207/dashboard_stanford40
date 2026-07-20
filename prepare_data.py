import os
import shutil
import glob
import sys
from sklearn.model_selection import train_test_split
import joblib
from tqdm import tqdm

# ============================================================
# 1. KONFIGURASI FOLDER
# ============================================================
SOURCE_DIR = 'stanford40'          # Folder hasil ekstrak dataset
TARGET_DIR = 'data'                # Folder tujuan terstruktur
TEST_SIZE = 0.2                    # Proporsi data uji
RANDOM_STATE = 42                  # Untuk reproduksibilitas

# ============================================================
# 2. VALIDASI FOLDER SUMBER
# ============================================================
if not os.path.exists(SOURCE_DIR):
    print(f"❌ ERROR: Folder '{SOURCE_DIR}' tidak ditemukan!")
    print(f"Pastikan Anda telah mengekstrak dataset Stanford 40 Actions ke folder '{SOURCE_DIR}'.")
    sys.exit(1)

# ============================================================
# 3. MENCARI SEMUA FILE GAMBAR
# ============================================================
images = glob.glob(os.path.join(SOURCE_DIR, '*.jpg'))
print(f'📂 Total gambar ditemukan: {len(images)}')

if len(images) == 0:
    print(f"❌ ERROR: Tidak ada file .jpg di folder '{SOURCE_DIR}'!")
    print("Pastikan folder berisi file dengan format: aktivitas_nomor.jpg")
    sys.exit(1)

# ============================================================
# 4. EKSTRAKSI NAMA KELAS DARI NAMA FILE
# ============================================================
class_names = set()
for f in images:
    basename = os.path.basename(f)          # misal: brushing_teeth_001.jpg
    parts = basename.split('_')
    # Gabungkan semua bagian kecuali bagian terakhir (nomor urut + ekstensi)
    class_name = '_'.join(parts[:-1])       # menghasilkan brushing_teeth
    class_names.add(class_name)

class_names = sorted(list(class_names))
print(f'🏷️  Jumlah kelas: {len(class_names)}')

if len(class_names) == 0:
    print("❌ ERROR: Tidak ada nama kelas yang diekstrak!")
    print("Pastikan format file adalah: aktivitas_nomor.jpg")
    sys.exit(1)

print(f'📋 Nama kelas (10 pertama): {class_names[:10]}...')

# ============================================================
# 5. PEMBAGIAN DATA LATIH DAN UJI (80:20)
# ============================================================
train_imgs, test_imgs = train_test_split(
    images, 
    test_size=TEST_SIZE, 
    random_state=RANDOM_STATE
)
print(f'📊 Data latih: {len(train_imgs)} gambar')
print(f'📊 Data uji: {len(test_imgs)} gambar')

# ============================================================
# 6. MEMBUAT DIREKTORI TARGET
# ============================================================
print('📁 Membuat struktur folder...')
for split in ['train', 'test']:
    for cls in class_names:
        os.makedirs(os.path.join(TARGET_DIR, split, cls), exist_ok=True)

# ============================================================
# 7. MENYALIN FILE KE DIREKTORI YANG SESUAI
# ============================================================
print('📋 Menyalin file...')

for img in tqdm(train_imgs, desc='Menyalin data latih'):
    basename = os.path.basename(img)
    parts = basename.split('_')
    cls = '_'.join(parts[:-1])
    dst = os.path.join(TARGET_DIR, 'train', cls, basename)
    shutil.copy2(img, dst)  # copy2 mempertahankan metadata

for img in tqdm(test_imgs, desc='Menyalin data uji'):
    basename = os.path.basename(img)
    parts = basename.split('_')
    cls = '_'.join(parts[:-1])
    dst = os.path.join(TARGET_DIR, 'test', cls, basename)
    shutil.copy2(img, dst)

# ============================================================
# 8. MENYIMPAN DAFTAR NAMA KELAS
# ============================================================
joblib.dump(class_names, 'class_names.joblib')
print('💾 Daftar kelas disimpan ke class_names.joblib')

print('\n✅ Persiapan data selesai!')
print(f'📁 Folder "{TARGET_DIR}" telah dibuat dengan {len(class_names)} kelas.')
print('🚀 Lanjutkan ke tahap pelatihan: python train.py')