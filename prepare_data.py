# 1. Impor Pustaka dan Penentuan Folder Sumber
import os
import shutil
import glob
from sklearn.model_selection import train_test_split
import joblib
from tqdm import tqdm

source_dir = 'stanford40'
target_dir = 'data'

# 2. Cari semua file gambar dan mengekstrak nama kelas
images = glob.glob(os.path.join(source_dir, '*.jpg'))
print(f'Total gambar ditemukan: {len(images)}')

class_names = set()
for f in images:
    basename = os.path.basename(f)
    # Pisahkan berdasarkan underscore, lalu gabungkan semua kecuali bagian terakhir (nomor)
    parts = basename.split('_')
    # Bagian terakhir adalah nomor urut + ekstensi, kelas adalah gabungan parts[:-1] dengan '_'
    class_name = '_'.join(parts[:-1])
    class_names.add(class_name)

class_names = sorted(list(class_names))
print(f'Jumlah kelas: {len(class_names)}')
print(f'Nama kelas: {class_names}')

# 3. Pembagian data (80:20) dan pembuatan direktori target
train_imgs, test_imgs = train_test_split(images, test_size=0.2, random_state=42)

for split in ['train', 'test']:
    for cls in class_names:
        os.makedirs(os.path.join(target_dir, split, cls), exist_ok=True)

# 4. Menyalin file dan menyimpan daftar kelas
for img in tqdm(train_imgs, desc='Menyalin data latih'):
    basename = os.path.basename(img)
    parts = basename.split('_')
    cls = '_'.join(parts[:-1])
    shutil.copy(img, os.path.join(target_dir, 'train', cls, basename))

for img in tqdm(test_imgs, desc='Menyalin data uji'):
    basename = os.path.basename(img)
    parts = basename.split('_')
    cls = '_'.join(parts[:-1])
    shutil.copy(img, os.path.join(target_dir, 'test', cls, basename))

joblib.dump(class_names, 'class_names.joblib')
print('Persiapan data selesai. Folder "data" telah dibuat dengan 40 kelas.')