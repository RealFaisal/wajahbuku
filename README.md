# WajahBuku

Rekomendasi buku berdasarkan ekspresi wajah.

Project ini terdiri dari:

- **Deteksi & crop wajah** (MTCNN dari `facenet-pytorch`, fallback OpenCV Haar Cascade)
- **Prediksi emosi** (PyTorch model `.pt`)
- **Rekomendasi buku** (TF-IDF + cosine similarity)
- **UI** menggunakan Streamlit

## Struktur Folder (ringkas)

- `app.py` — aplikasi Streamlit
- `src/` — modul utama (preprocessing, inference, recommender)
- `models/` — model PyTorch `.pt`
- `data/books/` — dataset buku (`books.csv` (original), `books_processed.csv`, `genre_mapping.json`)
- `notebooks/` — notebook eksplorasi

## Prasyarat

- **Git** (untuk clone)
- **Python 3.10–3.12** (disarankan)
- (Opsional) **CUDA + GPU** jika ingin inference di GPU

> Catatan: Urutan instalasi penting karena **PyTorch sebaiknya di-install terlebih dahulu**.

## Cara Clone

```bash
git clone <URL_REPO_ANDA>
cd WajahBuku
```

## Setup Environment

### Windows (PowerShell / CMD)

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## Install Dependencies

### 1) Install PyTorch dulu (WAJIB)

Pilih salah satu sesuai perangkat Anda (lihat opsi paling baru di https://pytorch.org/get-started/locally/).

**GPU (contoh CUDA 13.0):**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

**CPU saja:**

```bash
pip install torch torchvision
```

### 2) Install package lainnya

```bash
pip install -r requirements.txt
```

## Menjalankan Notebook

Jika Anda ingin membuka notebook di `notebooks/`:

```bash
jupyter notebook
```

Lalu buka `notebooks/exploration.ipynb`.

## Menjalankan Aplikasi (Streamlit)

Dari root project:

```bash
streamlit run app.py
```

Jika berhasil, Streamlit akan menampilkan URL lokal (biasanya `http://localhost:8501`).

## Catatan Data & Model

- Dataset buku yang dipakai aplikasi default ada di: `data/books/books_processed.csv`
- Model emosi default ada di: `models/resnet50_fer2013_best.pt`

Jika Anda memindahkan file/folder, sesuaikan path di `app.py`.
