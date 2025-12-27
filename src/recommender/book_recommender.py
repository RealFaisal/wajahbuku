from typing import List, Optional, Dict
import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_EMOTION_TO_GENRES = {
    "angry": [
        "Crime, Mystery & Thriller",
        "Psychology & Self-Help",
        "Drama & Performing Arts"
    ],
    "disgust": [
        "Crime, Mystery & Thriller",
        "Psychology & Self-Help",
        "Self-Help & Personal Development"
    ],
    "fear": [
        "Religion & Spirituality",
        "Psychology & Self-Help",
        "Health & Medicine",
        "Biography & Autobiography"
    ],
    "happy": [
        "Fiction & Literature",
        "Children & Young Adult",
        "Humor",
        "Comics & Graphic Novels"
    ],
    "neutral": [],
    "sad": [
        "Biography & Autobiography",
        "Self-Help & Personal Development",
        "Psychology & Self-Help",
        "Religion & Spirituality"
    ],
    "surprise": [
        "Crime, Mystery & Thriller",
        "Fantasy & Science Fiction",
        "Drama & Performing Arts"
    ]
}

# Keywords untuk content-based filtering (berdasarkan psychological response)
EMOTION_KEYWORDS = {
    "angry": [
        "fast-paced", "suspense", "thriller", "action", "intense",
        "gripping", "cathartic", "therapeutic", "powerful", "compelling",
        "dynamic", "energetic", "dramatic", "tension"
    ],
    "disgust": [
        "corruption", "injustice", "expose", "truth", "revelation",
        "uncover", "scandal", "morality", "ethics", "conscience",
        "redemption", "change", "reform", "awareness"
    ],
    "fear": [
        "overcoming", "hardship", "survival", "courage", "resilience",
        "reassuring", "hope", "strength", "triumph", "perseverance",
        "comfort", "consoling", "inspiring", "brave", "endurance"
    ],
    "happy": [
        "happy", "joy", "light-hearted", "funny", "humor", "comedy",
        "cheerful", "uplifting", "feel-good", "delightful", "entertaining",
        "amusing", "playful", "whimsical", "heartwarming"
    ],
    "neutral": [
        "diverse", "variety", "popular", "bestseller", "acclaimed",
        "award-winning", "classic", "recommended", "well-written", "engaging",
        "story", "narrative", "literature", "novel", "book"
    ],
    "sad": [
        "uplifting", "consoling", "inspirational", "hope", "comfort",
        "healing", "redemption", "encouraging", "moving", "touching",
        "emotional", "poignant", "meaningful", "reflective"
    ],
    "surprise": [
        "twist", "mystery", "suspense", "unexpected", "surprise",
        "revelation", "plot-twist", "shocking", "unpredictable", "intrigue",
        "enigma", "puzzle", "detective", "secret", "discover"
    ]
}


class BookRecommender:
    def __init__(
        self,
        books_csv_path: str = "data/books/books_processed.csv",
        emotion_map_path: Optional[str] = "data/books/genre_mapping.json",
        emotion_to_genres: Optional[Dict[str, List[str]]] = None,
        text_field: str = "description"
    ):
        self.df = self._load_books(books_csv_path)
        self.text_field = text_field
        
        if emotion_to_genres:
            self.emotion_to_genres = emotion_to_genres
        elif emotion_map_path:
            try:
                with open(emotion_map_path, "r") as f:
                    self.emotion_to_genres = json.load(f)
            except Exception:
                self.emotion_to_genres = DEFAULT_EMOTION_TO_GENRES.copy()
        else:
            self.emotion_to_genres = DEFAULT_EMOTION_TO_GENRES.copy()
        
        self.emotion_keywords = EMOTION_KEYWORDS.copy()
        self._tfidf = None
        self._doc_vectors = None

    def reload_emotion_map(self, emotion_map_path: str):
        with open(emotion_map_path, "r") as f:
            self.emotion_to_genres = json.load(f)

    def _load_books(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        
        expected = [
            "isbn13", "isbn10", "title", "subtitle", "authors", "categories",
            "thumbnail", "description", "published_year",
            "average_rating", "num_pages", "ratings_count"
        ]
        
        for col in expected:
            if col not in df.columns:
                df[col] = ""
        
        df = df.fillna("")
        return df

    def _ensure_tfidf(self):
        if self._tfidf is None or self._doc_vectors is None:
            docs = self.df[self.text_field].fillna("").astype(str)
            self._tfidf = TfidfVectorizer(
                max_features=500,
                stop_words='english',
                ngram_range=(1, 2)
            )
            self._doc_vectors = self._tfidf.fit_transform(docs)

    def _genre_match_score(self, genres: List[str]) -> pd.Series:
        if not genres:
            return pd.Series([0] * len(self.df), index=self.df.index)
        
        pattern = "|".join([g.lower() for g in genres])
        return self.df["categories"].str.lower().str.contains(pattern, na=False, case=False, regex=True)

    def _calculate_weighted_rating(self, df: pd.DataFrame, percentile: int = 80) -> pd.Series:
        """
        IMDB Weighted Rating Formula:
        WR = (v/(v+m)) * R + (m/(v+m)) * C
        - R = average rating
        - v = number of votes (ratings_count)
        - m = minimum votes required (percentile)
        - C = mean rating across all books
        """
        # Hitung rata-rata rating global dan threshold minimum votes
        df = df.copy()
        df['average_rating'] = pd.to_numeric(df['average_rating'], errors='coerce').fillna(0)
        df['ratings_count'] = pd.to_numeric(df['ratings_count'], errors='coerce').fillna(0)
        
        C = df['average_rating'].mean()
        m = df['ratings_count'].quantile(percentile / 100)
        
        def weighted_rating(row):
            try:
                v = row['ratings_count']
                R = row['average_rating']
                if v == 0:
                    return 0
                return (v / (v + m)) * R + (m / (v + m)) * C
            except:
                return 0
        
        return df.apply(weighted_rating, axis=1)

    def recommend_content_based(
        self, 
        emotion: str, 
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Content-Based Filtering dengan Multi-Feature Scoring:
        - Genre Matching (30%): Cocokkan kategori dengan emosi
        - Content Similarity (40%): TF-IDF similarity pada deskripsi
        - Weighted Rating (30%): IMDB formula untuk kredibilitas
        
        Parameters:
            emotion: Emosi yang terdeteksi
            top_n: Jumlah rekomendasi
        """
        # Bobot optimal (ubah di sini jika ingin mengatur bobot)
        w_genre = 0.3      # 30% - Genre matching
        w_content = 0.5    # 40% - Content similarity
        w_rating = 0.2     # 30% - Rating quality
        
        emotion = emotion.lower()
        genres = self.emotion_to_genres.get(emotion, [])
        keywords = self.emotion_keywords.get(emotion, ["story"])
        
        df_scores = self.df.copy()
        
        # 1. Genre Score (0 atau 1)
        genre_mask = self._genre_match_score(genres).astype(int)
        df_scores["genre_score"] = genre_mask
        
        # 2. Content Similarity Score (0-1)
        self._ensure_tfidf()
        seed_text = " ".join(keywords)
        seed_vec = self._tfidf.transform([seed_text])
        sims = cosine_similarity(seed_vec, self._doc_vectors).flatten()
        
        # Normalize similarity scores
        if sims.max() > 0:
            sims_norm = (sims - sims.min()) / (sims.max() - sims.min())
        else:
            sims_norm = sims
        
        df_scores["content_score"] = sims_norm
        
        # 3. Weighted Rating Score (0-5, normalized to 0-1)
        weighted_ratings = self._calculate_weighted_rating(df_scores)
        if weighted_ratings.max() > 0:
            weighted_ratings_norm = weighted_ratings / 5.0  # Normalize to 0-1
        else:
            weighted_ratings_norm = weighted_ratings
        
        df_scores["rating_score"] = weighted_ratings_norm
        
        # 4. Combined Score
        df_scores["final_score"] = (
            w_genre * df_scores["genre_score"] +
            w_content * df_scores["content_score"] +
            w_rating * df_scores["rating_score"]
        )
        
        # Sort by final score
        df_scores = df_scores.sort_values(
            by=["final_score", "average_rating", "ratings_count"], 
            ascending=[False, False, False]
        )
        
        return df_scores.head(top_n).reset_index(drop=True)

    def recommend(
        self, 
        emotion: str, 
        top_n: int = 10
    ) -> List[Dict]:
        """
        Main recommendation method menggunakan Content-Based Multi-Feature
        
        Returns list of recommended books dengan fields:
        - isbn13, title, authors, categories, thumbnail, description
        - published_year, average_rating, num_pages, ratings_count
        - isbn10, subtitle
        """
        df_rec = self.recommend_content_based(emotion, top_n=top_n)
        
        fields = [
            "isbn13", "isbn10", "title", "subtitle", "authors", "categories",
            "thumbnail", "description", "published_year",
            "average_rating", "num_pages", "ratings_count"
        ]
        
        results = []
        for _, row in df_rec.iterrows():
            book = {}
            for f in fields:
                if f in df_rec.columns:
                    book[f] = row[f]
            results.append(book)
        
        return results