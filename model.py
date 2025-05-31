import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import pickle

def load_data():
    df_lagu = pd.read_csv('data_lagu_bersih.csv')

    # Daftar fitur numerik yang akan digunakan
    fitur_lagu = [
        'danceability', 'energy', 'loudness', 'speechiness',
        'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
    ]

    # Konversi nilai koma ke titik untuk fitur numerik
    for fitur in fitur_lagu:
        df_lagu[fitur] = pd.to_numeric(
            df_lagu[fitur].astype(str).str.replace(',', '.').replace('', np.nan),
            errors='coerce'
        )

    # Hapus baris dengan NaN di fitur penting
    df_lagu = df_lagu.dropna(subset=fitur_lagu)

    # Siapkan dataframe fitur dengan index spotify_id
    df_fitur = df_lagu[['spotify_id'] + fitur_lagu].copy()
    df_fitur.set_index('spotify_id', inplace=True)

    return df_lagu, df_fitur

def train_model():
    df_rating = pd.read_csv('data_rating.csv')

    # Konversi rating ke float
    df_rating['rating'] = df_rating['rating'].astype(float)

    reader = Reader(rating_scale=(1,5))
    data = Dataset.load_from_df(df_rating[['user_id', 'spotify_id', 'rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    model_svd = SVD()
    model_svd.fit(trainset)

    with open('model_svd.pkl', 'wb') as f:
        pickle.dump(model_svd, f)

    print("Model trained and saved.")
    return model_svd

def load_model():
    with open('model_svd.pkl', 'rb') as f:
        model_svd = pickle.load(f)
    return model_svd

def rekomendasi_lagu_hybrid_4_1(user_id, judul_input, artis_input, genre_input, top_n=10, df_lagu=None, df_fitur=None, model_svd=None):
    if df_lagu is None or df_fitur is None or model_svd is None:
        raise ValueError("Dataset dan model harus sudah dimuat")

    # Cari lagu input berdasarkan judul dan artis
    lagu_input = df_lagu[
        (df_lagu['name'].str.lower() == judul_input.lower()) &
        (df_lagu['artists'].str.lower() == artis_input.lower())
    ]

    if lagu_input.empty:
        return f"Lagu dengan judul '{judul_input}' oleh '{artis_input}' tidak ditemukan."

    lagu_input_id = lagu_input.iloc[0]['spotify_id']

    # PERBAIKAN: Ambil vektor fitur dari df_lagu sesuai kolom df_fitur
    v_input = lagu_input[df_fitur.columns].values.reshape(1, -1)
    sim_scores = cosine_similarity(v_input, df_fitur.values).flatten()
    df_cbf_score = pd.DataFrame({
        'spotify_id': df_fitur.index,
        'cbf_score': sim_scores
    })

    # Collaborative filtering score dari model SVD
    try:
        df_cf_score = pd.DataFrame([{
            'spotify_id': song_id,
            'cf_score': model_svd.predict(user_id, song_id).est
        } for song_id in df_lagu['spotify_id']])
    except:
        df_cf_score = pd.DataFrame([{
            'spotify_id': song_id,
            'cf_score': np.nan
        } for song_id in df_lagu['spotify_id']])

    # Gabungkan skor CBF dan CF
    df_score = df_cbf_score.merge(df_cf_score, on='spotify_id')
    df_score = df_score.merge(df_lagu[['spotify_id', 'name', 'artists', 'genre', 'spotify_url']], on='spotify_id')

    # Filter genre dan hapus lagu input dari hasil
    df_score = df_score[
        (df_score['spotify_id'] != lagu_input_id) &
        (df_score['genre'].str.lower() == genre_input.lower())
    ]

    # Hapus duplikat
    df_score = df_score.drop_duplicates(subset=['name', 'artists'])

    # Normalisasi skor CF jika tersedia
    if not df_score['cf_score'].isnull().all():
        min_cf = df_score['cf_score'].min()
        max_cf = df_score['cf_score'].max()
        df_score['cf_score_normalized'] = (df_score['cf_score'] - min_cf) / (max_cf - min_cf + 1e-9)
    else:
        df_score['cf_score_normalized'] = 0.0

    # Hitung skor akhir (hybrid 4:1)
    df_score['score_4_1'] = df_score['cbf_score'] * 0.8 + df_score['cf_score_normalized'].fillna(0.0) * 0.2

    # Ambil top N rekomendasi
    top_rekomendasi = df_score.sort_values('score_4_1', ascending=False).head(top_n)

    # Format hasil akhir
    hasil = top_rekomendasi[['name', 'artists', 'genre', 'score_4_1', 'spotify_url']].copy()
    hasil.columns = ['Judul Lagu', 'Artis', 'Genre', 'Skor Hybrid', 'spotify_url']

    return hasil.reset_index(drop=True)
