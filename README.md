# 🔥 Fire & Smoke Detection System

Sistem deteksi api dan asap secara real-time menggunakan **YOLOv8** deep learning model, dengan backend **FastAPI** dan frontend web berbasis kamera.

---

## 📂 Struktur Proyek

```
project3/
├── backend/
│   ├── main.py              # FastAPI server & YOLOv8 inference
│   └── requirements.txt     # Dependensi backend
├── frontend/
│   ├── index.html           # Halaman utama
│   ├── style.css            # Styling
│   ├── script.js            # Logika kamera & deteksi
│   ├── server.js            # Static file server (Node.js)
│   └── package.json
├── runs/detect/fire_smoke_detector/
│   └── weights/best.pt      # Model terlatih (YOLOv8)
├── train.py                 # Script training model
├── prepare_data.py          # Script persiapan dataset
├── data.yaml                # Konfigurasi dataset
└── requirements.txt         # Dependensi training
```

---

## 📊 Dataset

Dataset **tidak disertakan** dalam repository karena ukurannya yang besar.

> [!IMPORTANT]
> Download dataset dari Kaggle:
> **https://www.kaggle.com/datasets/sekaabdul/fire-smoke-9-and-10**

Setelah diunduh, letakkan dataset sesuai struktur berikut:

```
project3/
├── dataset/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   └── val/
│       ├── images/
│       └── labels/
```

Atau jalankan `python prepare_data.py` untuk mempersiapkan dataset secara otomatis.

---

## 📋 Prerequisites

| Requirement       | Keterangan                                          |
|--------------------|-----------------------------------------------------|
| **Python 3.8+**    | Untuk menjalankan backend & training model           |
| **Node.js**        | Untuk menjalankan frontend                           |
| **Model Terlatih** | File `best.pt` sudah tersedia di repository          |

---

## � Tentang Proyek

Proyek ini merupakan sistem deteksi api dan asap berbasis kecerdasan buatan (AI) yang memanfaatkan model **YOLOv8** untuk mengenali tanda-tanda kebakaran secara real-time melalui feed kamera. Sistem terdiri dari backend **FastAPI** yang menjalankan inferensi model deep learning, serta frontend web yang menampilkan hasil deteksi langsung di browser pengguna dengan bounding box pada objek yang terdeteksi.

Dengan kemampuan mendeteksi **api (fire)** dan **asap (smoke)** dalam hitungan milidetik, sistem ini dirancang sebagai solusi **deteksi dini kebakaran** yang dapat diintegrasikan ke infrastruktur kota-kota besar.

## 🏙️ Peluang Pengembangan di Kota Besar

Kebakaran di wilayah perkotaan merupakan ancaman serius yang seringkali terlambat ditangani karena keterbatasan pengawasan manusia. Sistem ini memiliki potensi besar untuk dikembangkan dan diaplikasikan di kota-kota besar sebagai **layer pertama deteksi kebakaran** yang bekerja 24/7 secara otomatis.

**Skenario penerapan:**

- **Integrasi dengan jaringan CCTV kota** — Model deteksi ini dapat dihubungkan ke seluruh kamera CCTV yang sudah terpasang di berbagai titik kota, seperti persimpangan jalan, area permukiman padat, kawasan industri, pasar tradisional, dan gedung-gedung bertingkat. Setiap kamera menjadi "mata AI" yang memantau potensi kebakaran tanpa henti.

- **Sistem peringatan dini otomatis** — Ketika api atau asap terdeteksi, sistem dapat langsung mengirimkan notifikasi ke pemadam kebakaran, dinas terkait, atau pusat komando kota dalam hitungan detik, jauh sebelum laporan dari warga masuk. Respons yang lebih cepat berarti kerusakan yang lebih kecil dan nyawa yang lebih terselamatkan.

- **Dashboard monitoring terpusat** — Seluruh feed CCTV dapat dipantau melalui satu dashboard terpusat di pusat komando, dilengkapi dengan peta lokasi kejadian, tingkat keparahan, dan histori deteksi untuk analisis pola kebakaran.

- **Skalabilitas dan efisiensi biaya** — Karena memanfaatkan infrastruktur CCTV yang sudah ada, biaya implementasi jauh lebih rendah dibandingkan memasang sensor kebakaran fisik di setiap lokasi. Model YOLOv8 Nano yang ringan juga memungkinkan inferensi pada perangkat edge computing tanpa memerlukan server GPU mahal.

- **Smart City dan keselamatan publik** — Sistem ini sejalan dengan visi **Smart City** yang mengedepankan teknologi untuk meningkatkan keselamatan dan kualitas hidup warga kota. Dengan cakupan deteksi di seluruh titik strategis, kota dapat secara proaktif mencegah kebakaran besar sebelum terjadi.

---

## 🔗 API Endpoints

| Method | Endpoint      | Keterangan                       |
|--------|---------------|----------------------------------|
| `GET`  | `/health`     | Cek status server & model        |
| `POST` | `/detect`     | Upload gambar untuk deteksi      |
| `GET`  | `/model-info` | Info model yang sedang digunakan |

---

## 🧠 Training Model (Opsional)

Jika ingin melatih model sendiri:

1. Download dan siapkan dataset (lihat bagian [Dataset](#-dataset))
2. Jalankan training:

```bash
python train.py
```

Konfigurasi training:
- **Base Model:** YOLOv8n (Nano)
- **Image Size:** 640×640
- **Classes:** `fire`, `smoke`

Hasil training akan tersimpan di `runs/detect/fire_smoke_detector/weights/best.pt`.

---

## ⚠️ Troubleshooting

| Masalah                            | Solusi                                                        |
|------------------------------------|---------------------------------------------------------------|
| `localhost:3000` tidak bisa dibuka | Pastikan `node server.js` berjalan di terminal terpisah       |
| `localhost:8000` tidak bisa dibuka | Pastikan `python main.py` berjalan di terminal pertama        |
| Model tidak ditemukan              | Pastikan `runs/detect/fire_smoke_detector/weights/best.pt` ada |
| Port sudah dipakai                 | Tutup aplikasi lain yang memakai port 3000 atau 8000          |
| Module not found                   | Pastikan virtual environment aktif & dependencies terinstall  |
| Kamera tidak bisa diakses         | Berikan izin akses kamera di browser                          |
