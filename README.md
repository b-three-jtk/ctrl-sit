# 🧍‍♂️ Ctrl+Sit: Sistem Deteksi Postur dengan FastAPI & Jupyter Notebook

Sistem ini menggunakan **MediaPipe** dan **OpenCV** untuk mendeteksi postur tubuh manusia dari gambar. FastAPI digunakan sebagai backend REST API dan notebook disiapkan untuk eksplorasi serta pengujian model.

---

## 🗂️ Struktur File & Cara Menjalankan Notebook

```
.
├── assets/                      
│   ├── bad_posture.mp3              
│   └── good_posture.mp3                          
├── app.py                    # Entry point FastAPI
├── main.py                   # File utama aplikasi
├── posture_alert.py
├── requirements.txt          # File dependensi     
└── README.md
```

### 📒 Menjalankan Notebook

1. **Aktifkan virtual environment** (opsional tetapi disarankan):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate.bat # Windows
   ```

2. **Instal dependensi**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Jalankan Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

4. Buka `notebooks/eksplorasi.ipynb` dan jalankan sel satu per satu.

---

## ⚙️ Instruksi Setup FastAPI & Dependensi

1. **Clone repositori**:
   ```bash
   git clone https://github.com/namapengguna/ctrl-sit.git
   cd ctrl-sit
   ```

2. **Aktifkan virtual environment & instalasi dependensi**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # atau venv\Scripts\activate.bat (Windows)
   pip install -r requirements.txt
   ```

3. **Jalankan server FastAPI**:
   ```bash
   uvicorn app.main:app --reload
   ```

4. Akses dokumentasi API:
   ```
   http://localhost:8000/docs
   ```

---

## 📡 Dokumentasi Endpoint + Contoh Penggunaan API

### ▶️ `POST /detect-posture/`

#### Deskripsi:
Mendeteksi apakah postur tubuh pada gambar termasuk **baik** atau **buruk**, berdasarkan sudut leher dan torso.

#### Request:
- **Method**: `POST`
- **Form field**: `file` (gambar JPG/PNG)

#### Contoh cURL:
```bash
curl -X POST "http://localhost:8000/detect-posture/" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/image.jpg"
```

#### Contoh Respons:
```json
{
  "status": "success",
  "good_frames": 5,
  "bad_frames": 10
}
```

> Postur **baik** jika `good_frames > bad_frames`, **buruk** jika sebaliknya.

---

## 📎 Catatan Teknis

- Sudut leher dianggap **buruk** jika > 20°
- Sudut torso dianggap **buruk** jika > 10°
- API ini **hanya untuk gambar**, bukan video/live feed
- Eksperimen lengkap tersedia di notebook `eksplorasi.ipynb`

---

## 🛠 Teknologi yang Digunakan

- Python 3.10+
- FastAPI
- MediaPipe
- OpenCV
- Jupyter Notebook

---
