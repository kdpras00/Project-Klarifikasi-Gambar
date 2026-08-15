# Klasifikasi Gambar Buah & Sayuran dengan CNN (Fruits-360)

Proyek akhir **Klasifikasi Gambar** (Dicoding) — model *Convolutional Neural Network (CNN)* untuk mengklasifikasikan gambar buah & sayuran dari dataset **Fruits-360**, menggunakan TensorFlow Keras.

## Ringkasan

| Aspek | Keterangan |
|---|---|
| Dataset | [Fruits-360](https://www.kaggle.com/datasets/moltean/fruits) (Kaggle, oleh Mihai Oltean) |
| Jumlah gambar | ±10.000 gambar (subset 20 kelas dari ±100.000, memenuhi kriteria ≥ 1.000 & saran ≥ 10.000) |
| Jumlah kelas | 20 kelas buah & sayuran (memenuhi kriteria ≥ 3 kelas) |
| Resolusi | Beragam (gambar asli, tanpa preprocessing) |
| Split | Training (50%) / Validation (25%) / Test (25%) |
| Arsitektur | CNN `Sequential`: Conv2D, BatchNormalization, MaxPooling2D, Dropout, Dense |
| Target akurasi | ≥ 85% (diusahakan ≥ 95%) pada training & test set |

## Hasil Pelatihan

*(Diisi otomatis oleh notebook saat dijalankan di Colab — lihat output notebook.ipynb)*

| Metrik | Nilai |
|---|---|
| Akurasi Training | ... |
| Akurasi Validation | ... |
| Akurasi Test Set | ... |
| Loss Test Set | ... |

## Struktur Submission

```
submission
├─── tfjs_model/        (model.json, group1-shard1of1.bin)
├─── tflite/            (model.tflite, label.txt)
├─── saved_model/       (saved_model.pb, variables/)
├─── notebook.ipynb
├─── README.md
└─── requirements.txt
```

## Cara Menjalankan

1. Buka `notebook.ipynb` di **Google Colab** dengan runtime **GPU T4**.
2. Siapkan kredensial Kaggle: kaggle.com → Settings → API → **Create New Token** (`kaggle.json`).
3. Jalankan seluruh sel secara berurutan (download dataset dilakukan otomatis).
4. Pada bagian akhir, notebook akan menyusun folder `submission/` dan mengunduhnya sebagai `.zip`.

## Fitur yang Diimplementasikan

- Optimasi kecepatan: subset dataset (20 kelas), `.cache()` + `.prefetch()`, `mixed_float16` (fp16), `BATCH_SIZE` lebih besar, `IMG_SIZE` sesuai resolusi asli
- Callback: `EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint`
- Data augmentation (hanya pada data training)
- Plot akurasi & loss
- Evaluasi pada test set
- Model disimpan dalam format **SavedModel**, **TF-Lite**, dan **TFJS**
- Inference menggunakan model **TF-Lite** (dengan bukti visual)

## Lisensi Dataset

Fruits-360 © Mihai Oltean — lisensi [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
# Project-Klarifikasi-Gambar
