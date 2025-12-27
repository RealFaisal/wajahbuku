import streamlit as st
from PIL import Image
import pandas as pd

from src.preprocessing.face_detection import detect_and_crop_face
from src.inference.emotion_predict import EmotionPredictor, EMOTION_LABELS
from src.recommender.book_recommender import BookRecommender

# --------------------
MODEL_PATH_DEFAULT = "models/resnet50_fer2013_best.pt"
BOOKS_CSV = "data/books/books_processed.csv"
EMOTION_MAP_JSON = "data/books/genre_mapping.json"
# --------------------

st.set_page_config(page_title="Book Recommender by Face Emotion", layout="wide")

st.title("Rekomendasi Buku Berdasarkan Ekspresi Wajah")
st.markdown("Upload foto wajah Anda dan dapatkan rekomendasi buku yang sesuai dengan mood Anda!")
st.markdown("---")

# Sidebar: config
with st.sidebar:
    st.header("⚙️ Pengaturan")
    top_n = st.slider("Jumlah rekomendasi", 5, 50, 10)
    
    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.markdown("- Gunakan foto dengan wajah yang jelas")
    st.markdown("- Pastikan pencahayaan cukup")
    st.markdown("- Satu wajah per foto")
    

# Use default paths
model_path = MODEL_PATH_DEFAULT
books_csv = BOOKS_CSV
emotion_map = EMOTION_MAP_JSON

# Lazy load model & recommender with caching
@st.cache_resource
def load_predictor(p_path):
    return EmotionPredictor(p_path)

@st.cache_resource
def load_recommender(csv_path, map_path):
    return BookRecommender(books_csv_path=csv_path, emotion_map_path=map_path)

# Load resources
try:
    predictor = load_predictor(model_path)
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")
    predictor = None

try:
    recommender = load_recommender(books_csv, emotion_map)
except Exception as e:
    st.sidebar.error(f"Error loading recommender: {e}")
    recommender = None

# Pilihan: Upload atau Capture
st.subheader("Pilih Cara Input Foto")
input_mode = st.radio(
    "Pilih metode:",
    ["Upload Foto", "Capture dari Kamera"],
    horizontal=True
)

uploaded = None
captured = None

if input_mode == "Upload Foto":
    uploaded = st.file_uploader(
        "Upload gambar wajah (jpg, jpeg, png)",
        type=["jpg", "jpeg", "png"]
    )
else:
    captured = st.camera_input("Ambil foto dari kamera")

# Process image
image = None
if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
elif captured is not None:
    image = Image.open(captured).convert("RGB")

if image:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Foto Asli")
        st.image(image, width="stretch")
    
    # Detect face
    with st.spinner("🔍 Mendeteksi wajah..."):
        face_img = detect_and_crop_face(image)
    
    if face_img is None:
        st.error("❌ Tidak ada wajah terdeteksi. Coba foto lain dengan wajah yang lebih jelas.")
        st.stop()
    
    with col2:
        st.subheader("Wajah Terdeteksi")
        st.image(face_img, width="stretch")
    
    # Predict emotion
    if predictor is None:
        st.error("Model prediksi tidak tersedia.")
        st.stop()
    
    with st.spinner("🧠 Menganalisis emosi..."):
        pred_result = predictor.predict(face_img)
        emotion = pred_result["label"]
        probs = pred_result["probs"]
    
    st.markdown("---")
    st.subheader("🎭 Hasil Analisis Emosi")
    
    col_e1, col_e2 = st.columns([1, 2])
    with col_e1:
        emotion_emoji = {
            "angry": "😠",
            "disgust": "🤢",
            "fear": "😨",
            "happy": "😊",
            "neutral": "😐",
            "sad": "😢",
            "surprise": "😲"
        }
        st.markdown(f"### {emotion_emoji.get(emotion, '🤔')} **{emotion.upper()}**")
    
    with col_e2:
        confidence = probs[EMOTION_LABELS.index(emotion)]
        st.markdown(f"**Confidence Score:** {confidence:.1%}")
        st.progress(confidence)
    
    # Recommend books
    if recommender is None:
        st.error("Sistem rekomendasi tidak tersedia.")
        st.stop()
    
    st.markdown("---")
    st.subheader(f"📚 Rekomendasi Buku untukmu:")
    
    with st.spinner("🔍 Mencari buku yang cocok..."):
        recommendations = recommender.recommend(emotion, top_n=top_n)
    
    if not recommendations:
        st.warning("Tidak ada rekomendasi ditemukan.")
        st.stop()
    
    # Display recommendations in 3 columns
    for idx in range(0, len(recommendations), 3):
        cols = st.columns(3)
        
        for col_idx, col in enumerate(cols):
            book_idx = idx + col_idx
            if book_idx >= len(recommendations):
                break
            
            r = recommendations[book_idx]
            
            with col:
                # Thumbnail dengan ukuran tetap
                thumbnail_url = r.get("thumbnail", "")
                if thumbnail_url and thumbnail_url.strip():
                    try:
                        st.markdown(
                            f'<img src="{thumbnail_url}" style="width:100%; height:280px; object-fit:contain; background-color:#f0f0f0; border-radius:8px;">',
                            unsafe_allow_html=True
                        )
                    except:
                        st.markdown("📖", unsafe_allow_html=True)
                else:
                    st.markdown(
                        '<div style="width:100%; height:280px; display:flex; align-items:center; justify-content:center; background-color:#f0f0f0; border-radius:8px; font-size:80px;">📖</div>',
                        unsafe_allow_html=True
                    )
                
                # Info utama
                title = r.get("title", "Unknown Title")
                authors = r.get("authors", "Unknown Author")
                categories = r.get("categories", "Unknown Category")
                
                st.markdown(f"**{title}**")
                st.caption(f"👤 {authors}")
                st.caption(f"📂 {categories}")
                
                # Rating
                avg_rating = r.get("average_rating", "")
                if avg_rating and avg_rating != "":
                    try:
                        rating_val = float(avg_rating)
                        st.caption(f"⭐ {rating_val:.2f}")
                    except:
                        pass
                
                # Detail lengkap di popover
                with st.popover("ℹ️ Detail Lain"):
                    def safe_get(d, key, default="-"):
                        val = d.get(key, default)
                        if pd.isna(val) or val == "" or val == " ":
                            return default
                        return val
                    
                    subtitle = safe_get(r, "subtitle")
                    isbn13 = safe_get(r, "isbn13")
                    isbn10 = safe_get(r, "isbn10")
                    pub_year = safe_get(r, "published_year")
                    num_pages = safe_get(r, "num_pages")
                    ratings_count = safe_get(r, "ratings_count")
                    description = safe_get(r, "description")
                    
                    if subtitle != "-":
                        st.write(f"**Subtitle:** {subtitle}")
                    if isbn13 != "-":
                        st.write(f"**ISBN-13:** {isbn13}")
                    if isbn10 != "-":
                        st.write(f"**ISBN-10:** {isbn10}")
                    if pub_year != "-":
                        try:
                            st.write(f"**Tahun Terbit:** {int(float(pub_year))}")
                        except:
                            st.write(f"**Tahun Terbit:** {pub_year}")
                    if num_pages != "-":
                        try:
                            st.write(f"**Jumlah Halaman:** {int(float(num_pages))}")
                        except:
                            st.write(f"**Jumlah Halaman:** {num_pages}")
                    if ratings_count != "-":
                        try:
                            st.write(f"**Jumlah Rating:** {int(float(ratings_count))}")
                        except:
                            st.write(f"**Jumlah Rating:** {ratings_count}")
                    
                    if description != "-":
                        st.write("**Deskripsi:**")
                        st.write(description)
        
        if idx + 3 < len(recommendations):
            st.markdown("---")

else:
    # Landing page
    st.markdown("### 🌟 Cara Kerja:")
    st.markdown("""
    1. **Upload atau capture** foto wajah Anda
    2. **AI akan mendeteksi** emosi dari ekspresi wajah
    3. **Sistem akan merekomendasikan** buku yang sesuai dengan mood Anda
    4. **Sesuaikan bobot fitur** di sidebar untuk hasil yang lebih personal
    """)
    
    st.markdown("### 🎯 Metode Content-Based Multi-Feature:")
    st.markdown("""
    - **Genre Matching ()**: Mencocokkan kategori buku dengan emosi
    - **Content Similarity ()**: Menganalisis kesamaan deskripsi buku menggunakan TF-IDF
    - **Rating Quality ()**: Mempertimbangkan rating dan popularitas buku (IMDB formula)
    """)