from flask import Flask, request, jsonify
from flask_cors import CORS
import model
import pandas as pd


app = Flask(__name__)
CORS(app)

# Load model dan data
print("Loading data dan model...")
df_lagu, df_fitur = model.load_data()
model_svd = model.load_model()
print("Selesai loading.")

@app.route('/')
def home():
    return "API Rekomendasi Lagu Hybrid CBF:CF 4:1 siap."

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({'error': 'Request body harus berformat JSON dan tidak kosong'}), 400

    user_id = data.get('user_id')
    judul = data.get('judul_lagu')
    artis = data.get('artis')
    genre = data.get('genre')
    top_n = data.get('top_n', 10)

    if not user_id or not judul or not artis or not genre:
        return jsonify({'error': 'Parameter user_id, judul_lagu, artis, dan genre harus diisi'}), 400

    hasil = model.rekomendasi_lagu_hybrid_4_1(
        user_id=user_id,
        judul_input=judul,
        artis_input=artis,
        genre_input=genre,
        top_n=top_n,
        df_lagu=df_lagu,
        df_fitur=df_fitur,
        model_svd=model_svd
    )

    if isinstance(hasil, str):
        return jsonify({'message': hasil}), 404

    hasil_json = hasil.to_dict(orient='records')
    return jsonify({'recommendations': hasil_json})

@app.route('/lagu', methods=['GET'])
def daftar_lagu():
    df = pd.read_csv("data_lagu_bersih.csv")  # pastikan file ini ada
    lagu_list = [{"title": row["name"], "artist": row["artists"]} for _, row in df.iterrows()]
    return jsonify(lagu_list)




if __name__ == '__main__':
    app.run(debug=True)
