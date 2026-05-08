import streamlit as st
import pandas as pd
import re
import numpy as np
import pickle
import gensim
import gdown
import os
from gensim.models import KeyedVectors, Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# 1. SETUP HALAMAN
st.set_page_config(
    page_title="Skincare Recommender",
    page_icon="💄",
    layout="wide"
)

st.title("Sistem Rekomendasi Produk Kecantikan Lokal 💄")
st.markdown("Berbasis Content-Based Filtering menggunakan Model Natural Language Processing (NLP)")
st.write("---")


# 2. LOAD DATA & MODEL
@st.cache_resource
def load_model():
    return SentenceTransformer('indobenchmark/indobert-base-p1')  # embedding user input menggunakan indobert

@st.cache_resource
def load_w2v_model():
    from gensim.models import Word2Vec
    if not os.path.exists('idwiki_word2vec_300/idwiki_word2vec_300.model'):
        os.makedirs('idwiki_word2vec_300', exist_ok=True)
        gdown.download('https://drive.google.com/uc?id=1ljUZgFke8gyx_7nRi6qhiOzg0BaQIZRR', 'idwiki_word2vec_300/idwiki_word2vec_300.model', quiet=False)
        gdown.download('https://drive.google.com/uc?id=1b19YFHKCMp9B7hrXRFDWAIkjao4bO7n7', 'idwiki_word2vec_300/idwiki_word2vec_300.model.trainables.syn1neg.npy', quiet=False)
        gdown.download('https://drive.google.com/uc?id=1JLzYISZ2F8_Y6mjKFjkEbg0xNGhOPwqt', 'idwiki_word2vec_300/idwiki_word2vec_300.model.wv.vectors.npy', quiet=False)
    return Word2Vec.load('idwiki_word2vec_300/idwiki_word2vec_300.model')

model_w2v = load_w2v_model()

@st.cache_resource
def load_ft_model():
    from gensim.models import KeyedVectors
    if not os.path.exists('cc.id.300.vec/cc.id.300.vec'):
        os.makedirs('cc.id.300.vec', exist_ok=True)
        gdown.download('https://drive.google.com/uc?id=1xTIZpqgBxX_XoNccJZWB4mHdWmQZHRWq', 'cc.id.300.vec/cc.id.300.vec', quiet=False)
    return KeyedVectors.load_word2vec_format('cc.id.300.vec/cc.id.300.vec', binary=False, limit=100000)
    
model_ft = load_ft_model()

@st.cache_data
def load_data():
    df_model = pd.read_csv('data_final.csv')
    df_ui = pd.read_csv('data_ui.csv')
    embeddings = np.load('embeddings.npy')
    embeddings_w2v = np.load('embeddings_w2v_pretrained.npy')
    sim_weighted = np.load('similarity_matrix_alpha_0_5.npy')
    sim_tfidf = np.load('similarity_matrix_tfidf.npy')
    sim_w2v_pretrained = np.load('similarity_matrix_w2v_pretrained.npy')
    sim_ft_pretrained = np.load('similarity_matrix_ft_pretrained.npy')
    embeddings_ft = np.load('embeddings_ft_pretrained.npy')
    tfidf_matrix = np.load('tfidf_matrix.npy')
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    return df_model, df_ui, embeddings, embeddings_w2v, sim_weighted, sim_tfidf, sim_w2v_pretrained, sim_ft_pretrained, embeddings_ft, tfidf_matrix, tfidf

model = load_model()
df_model, df_ui, embeddings, embeddings_w2v, sim_weighted, sim_tfidf, sim_w2v_pretrained, sim_ft_pretrained, embeddings_ft, tfidf_matrix, tfidf = load_data()

# 3. FUNGSI REKOMENDASI
factory = StemmerFactory()
stemmer = factory.create_stemmer()
stopword_factory = StopWordRemoverFactory()
stopword_remover = stopword_factory.create_stop_word_remover()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'expired date.*?(\n|$)', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'peringatan.*?(\n|$)', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def simplify_query(text):
    text = clean_text(text)
    words = text.split()
    keywords = [w for w in words if len(w) > 2]
    return ' '.join(keywords[:20])

def preprocess_query_w2v(text):
    text = clean_text(text)
    text = stopword_remover.remove(text)  # remove stopwords dari Sastrawi
    words = text.split()
    keywords = [w for w in words if len(w) > 2]
    return ' '.join(keywords[:20])

def preprocess_query_tfidf(text):
    text = clean_text(text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = stemmer.stem(text)  # stemming
    text = stopword_remover.remove(text)  # remove stopwords dari Sastrawi
    words = text.split()
    keywords = [w for w in words if len(w) > 2]
    return ' '.join(keywords)

def get_embedding(text):
    return model.encode(text, normalize_embeddings=True)

def choose_diverse_top_indices(similarities, top_k=15, result_n=5, max_pairwise=0.88):
    top_indices = similarities.argsort()[::-1][:top_k]
    chosen = []
    for idx in top_indices:
        idx = int(idx)
        if not chosen:
            chosen.append(idx)
            continue
        pair_sims = cosine_similarity([embeddings[idx]], embeddings[chosen])[0]
        if pair_sims.max() < max_pairwise:
            chosen.append(idx)
        if len(chosen) >= result_n:
            break
    if len(chosen) < result_n:
        for idx in top_indices:
            idx = int(idx)
            if idx not in chosen:
                chosen.append(idx)
            if len(chosen) >= result_n:
                break
    return chosen

# Modal untuk deskripsi produk
@st.dialog("Deskripsi Produk")
def show_description_modal(idx):
    st.write(f"**{df_ui.loc[idx, 'Product Name']}**")
    st.write(df_ui.loc[idx, 'short description'])
    if st.button("Tutup"):
        st.session_state.pop('selected_idx', None)
        st.rerun()

# # 4. INPUT USER
# st.subheader("Masukkan kebutuhan skincare kamu (Contoh: ):")

# DROPDOWN PILIH MODEL
metode = st.selectbox(
    "Pilih sistem rekomendasi:",
    ("Sistem A", "Sistem B", "Sistem C", "Sistem D", "Sistem E", "Sistem F", "Sistem G")
)

user_input = st.text_area(
    "Masukkan produk acuan dan permasalahan kulit Anda:",
    placeholder="Contoh: 'Wardah Lightening Serum, kulit berjerawat..."
)

# =========================
# 5. TOMBOL
# =========================
if metode != st.session_state.get('last_metode'):
    st.session_state.pop('selected_idx', None)
    st.session_state['last_metode'] = metode

if st.button("Rekomendasikan Produk"):
    st.session_state.pop('selected_idx', None)  # reset produk yang dipilih sebelumnya
    st.session_state.pop('top_indices', None)  # reset hasil rekomendasi sebelumnya

    if not user_input.strip():
        st.warning("Harap isi input dahulu!")
    else:
        if metode == "Sistem A": #indobert
            query_clean = simplify_query(user_input)  # tanpa stemming
            query_embedding = get_embedding(query_clean)
            similarities = cosine_similarity([query_embedding], embeddings)[0]
            top_indices = choose_diverse_top_indices(similarities, top_k=20, result_n=5, max_pairwise=0.88)
            top_indices = [i for i in top_indices if similarities[i] >= 0.2]
            if not top_indices:
                top_indices = similarities.argsort()[::-1][:5].tolist()
            st.session_state['top_indices'] = top_indices

        elif metode == "Sistem B": #tf-idf
            query_clean = preprocess_query_tfidf(user_input)  # dengan stemming dan stopword removal
            query_vec = tfidf.transform([query_clean])
            similarities = cosine_similarity(query_vec, np.load('tfidf_matrix.npy'))[0]
            top_indices = similarities.argsort()[::-1][:5].tolist()
            st.session_state['top_indices'] = top_indices

        elif metode == "Sistem C": #word2vec pretrained
            query_clean = preprocess_query_w2v(user_input)  # clean text + stopword removal
            words = query_clean.split()
            # Hitung embedding untuk query menggunakan word2vec
            word_vectors = []
            for word in words:
                if word in model_w2v.wv:
                    word_vectors.append(model_w2v.wv[word])
            if word_vectors:
                query_embedding = np.mean(word_vectors, axis=0)
                similarities = cosine_similarity([query_embedding], embeddings_w2v)[0]
                top_indices = similarities.argsort()[::-1][:5].tolist()
                st.session_state['top_indices'] = top_indices
            else:
                st.warning("Kata tidak ditemukan dalam model Word2Vec")
                top_indices = []

        elif metode == "Sistem D": #fasttext pretrained
            query_clean = preprocess_query_w2v(user_input)  # tanpa stemming
            words = query_clean.split()
            word_vectors = []
            for word in words:
                if word in model_ft:  # pakai model_ft, bukan model_w2v
                    word_vectors.append(model_ft[word])
            if word_vectors:
                query_embedding_ft = np.mean(word_vectors, axis=0)
                similarities = cosine_similarity([query_embedding_ft], embeddings_ft)[0]  # embeddings_ft bukan sim_ft
                top_indices = similarities.argsort()[::-1][:5].tolist()
                st.session_state['top_indices'] = top_indices
            else:
                st.warning("Kata tidak ditemukan dalam model FastText")
                top_indices = []

        elif metode == "Sistem E": #word2vec + tf-idf (weighted)
            query_clean_w2v = preprocess_query_w2v(user_input)  # clean text + stopword removal untuk word2vec
            query_clean_tfidf = preprocess_query_tfidf(user_input)  # dengan stemming untuk tfidf
            
            # Word2Vec similarity
            words = query_clean_w2v.split()
            word_vectors = []
            for word in words:
                if word in model_w2v.wv:
                    word_vectors.append(model_w2v.wv[word])
            if word_vectors:
                query_embedding_w2v = np.mean(word_vectors, axis=0)
                sim_w2v_query = cosine_similarity([query_embedding_w2v], embeddings_w2v)[0]
            else:
                sim_w2v_query = np.zeros(len(embeddings_w2v))
            
            # TF-IDF similarity
            query_vec = tfidf.transform([query_clean_tfidf])
            sim_tfidf_query = cosine_similarity(query_vec, np.load('tfidf_matrix.npy'))[0]
            
            # Normalisasi
            sim_w2v_norm = (sim_w2v_query - sim_w2v_query.min()) / (sim_w2v_query.max() - sim_w2v_query.min() + 1e-9)
            sim_tfidf_norm = (sim_tfidf_query - sim_tfidf_query.min()) / (sim_tfidf_query.max() - sim_tfidf_query.min() + 1e-9)
            similarities = 0.5 * sim_w2v_norm + 0.5 * sim_tfidf_norm
            top_indices = similarities.argsort()[::-1][:5].tolist()
            st.session_state['top_indices'] = top_indices

        elif metode == "Sistem F": #fasttext + tf-idf (weighted)
            query_clean_ft = preprocess_query_w2v(user_input)   # tanpa stemming
            query_clean_tfidf = preprocess_query_tfidf(user_input)  # dengan stemming

            # FastText similarity
            words = query_clean_ft.split()
            word_vectors = []
            for word in words:
                if word in model_ft:
                    word_vectors.append(model_ft[word])
            if word_vectors:
                query_embedding_ft = np.mean(word_vectors, axis=0)
                sim_ft_query = cosine_similarity([query_embedding_ft], embeddings_ft)[0]
            else:
                sim_ft_query = np.zeros(len(embeddings_ft))

            # TF-IDF similarity
            query_vec = tfidf.transform([query_clean_tfidf])
            sim_tfidf_query = cosine_similarity(query_vec, np.load('tfidf_matrix.npy'))[0]

            # Normalisasi
            sim_ft_norm = (sim_ft_query - sim_ft_query.min()) / (sim_ft_query.max() - sim_ft_query.min() + 1e-9)
            sim_tfidf_norm = (sim_tfidf_query - sim_tfidf_query.min()) / (sim_tfidf_query.max() - sim_tfidf_query.min() + 1e-9)
            similarities = 0.7 * sim_ft_norm + 0.3 * sim_tfidf_norm
            top_indices = similarities.argsort()[::-1][:5].tolist()
            st.session_state['top_indices'] = top_indices

        else:  # IndoBERT + TF-IDF (Weighted) (Sistem G)
            query_bert = simplify_query(user_input)  # tanpa stemming untuk BERT
            query_tfidf_clean = preprocess_query_tfidf(user_input)  # dengan stemming untuk TF-IDF
            
            query_embedding = get_embedding(query_bert)
            sim_bert = cosine_similarity([query_embedding], embeddings)[0]
            
            query_vec = tfidf.transform([query_tfidf_clean])
            sim_tfidf_query = cosine_similarity(query_vec, np.load('tfidf_matrix.npy'))[0]

            # Normalisasi
            sim_bert_norm = (sim_bert - sim_bert.min()) / (sim_bert.max() - sim_bert.min() + 1e-9)
            sim_tfidf_norm = (sim_tfidf_query - sim_tfidf_query.min()) / (sim_tfidf_query.max() - sim_tfidf_query.min() + 1e-9)
            similarities = 0.5 * sim_bert_norm + 0.5 * sim_tfidf_norm # semua combined dijadiin 50:50 karena hasilnya kalau lbh banyak selain tf-idf hasilnya jelek; nnt stlh ground truth mgkn akan kelihatan di nilai eval
            top_indices = similarities.argsort()[::-1][:5].tolist()
            st.session_state['top_indices'] = top_indices

# =========================
# 6. DISPLAY HASIL (OUTSIDE if st.button)
# =========================
if 'top_indices' in st.session_state:
    top_indices = st.session_state['top_indices']
    
    st.write("---")
    st.subheader("Produk Lokal Rekomendasi untuk kamu:")
    
    num_cols = min(len(top_indices), 5)
    cols = st.columns(num_cols)
    for i, idx in enumerate(top_indices):
        with cols[i]:
            st.image(df_ui.loc[idx, 'Product Image 1'], caption="Produk")
            st.write(f"**{df_ui.loc[idx, 'Product Name']}**")
            st.caption(df_ui.loc[idx, 'short description'][:80] + "...")
            
            if st.button("Lihat Deskripsi", key=f"desc_{idx}"):
                st.session_state['selected_idx'] = idx

    if 'selected_idx' in st.session_state:
        show_description_modal(st.session_state['selected_idx'])

# =========================
# FOOTER
# =========================
st.write("---")
st.caption("Prototype ini dibuat dengan Streamlit")

# a bit inconsist: Jadi embeddings_w2v memang harus tetap di-load. Yang tidak terpakai dan bisa dihapus hanya sim_weighted, sim_tfidf, sim_w2v_pretrained. ✅