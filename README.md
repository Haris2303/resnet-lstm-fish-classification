# 🎣 Klasifikasi Jenis Ikan Laut Kota Sorong Menggunakan Deep Learning (CNN (ResNet50) + LSTM)

Proyek ini bertujuan untuk mengembangkan model deep learning berbasis ResNet50 dan LSTM untuk klasifikasi **jenis ikan hasil laut di Kota Sorong**. Dataset berupa **urutan gambar (frame)** yang diambil dari video ikan.

---

## 📁 Dataset & Notebook

| Resource                           | Link                                                                                                            |
| ---------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| 📁 Dataset (6 Kelas, Terstruktur)  | [🔗 Google Drive Dataset](https://drive.google.com/drive/folders/172vAdvDHdbj8sijmLlElixssL5HDF1FF?usp=sharing) |
| 📓 Notebook Colab (cnn_lstm.ipynb) | [🔗 Google Colab Code](https://colab.research.google.com/drive/YOUR-NOTEBOOK-LINK)                              |

> 💡 _Silakan salin ke Google Drive Anda sendiri sebelum digunakan._

---

## 📌 Tujuan Riset

- Menggunakan **CNN (ResNet50)** untuk ekstraksi fitur gambar
- Menggabungkan **LSTM** untuk menangkap informasi berurutan antar frame
- Meningkatkan akurasi klasifikasi jenis ikan menggunakan pendekatan temporal
- Membangun pipeline **end-to-end**: ekstraksi data, preprocessing, training, evaluasi

---

## 🧠 Arsitektur Model

- Input: Urutan 10 frame gambar berukuran 224x224 piksel
- Feature extractor: `ResNet50` (pretrained ImageNet, frozen)
- Temporal model: `LSTM` (2 lapis, 128 unit) + `Dropout`
- Output: `Dense` dengan `Softmax` untuk klasifikasi 6 jenis ikan

---

## Metodologi Penelitian

<p align="center">
  <img src="https://drive.google.com/thumbnail?id=1NVaFObbuG4MpLJ1v_1fYJ2ZNV0RGA_zk" alt="Diagram Kerangka Penelitian" width="700"/>
  <br>
  <em>Gambar 1. Diagram Kerangka Penelitian Menggunakan <a href="https://drive.google.com/file/d/1NVaFObbuG4MpLJ1v_1fYJ2ZNV0RGA_zk/view?usp=drive_link">Draw Io</a></em>
</p>

## 🗂️ Struktur Folder

```bash
/content/
├── dataset/
│   ├── Kakap Merah/
│   │   ├── ikan001/
│   │   ├── ikan002/
│   ├── Bubara/
│   └── ...
├── cnn_lstm.ipynb   👈 notebook utama
├── saved_model/     👈 folder hasil model
```

## Referensi Jurnal

| Judul                                                                                           | Link                                                                                          |
| ----------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| Klasifikasi Data Penderita Skizofrenia Menggunakan CNN-LSTM dan CNN-GRU pada Data Sinyal EEG 2D | [🔗 Link PDF](https://pdfs.semanticscholar.org/2c86/e587981a81914d3febc824b9adb14c1cd96f.pdf) |
