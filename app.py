import streamlit as st
import pandas as pd
import numpy as np
import math
import re
import altair as alt
import plotly.express as px
import nltk
from nltk.corpus import stopwords
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

nltk.download('stopwords')

# --- Stopwords setup ---
stop_factory = StopWordRemoverFactory()
stopword_sastrawi = set(stop_factory.get_stop_words())
stopwords_nltk = set(stopwords.words('english'))
more_stopword = {
    'gw','gue','gua','lu','lo','elo','elu','ya','ye','lah','mah','nih',
    'kok','dong','bgt','banget','kyk','kek','aja','udah','yg',
    'gak','gk','ga','nggak','gausah','kayak','cuma',
    'doang','sih','lagi','lg','jd','jdi','trs','trus','sy','tetep',
    'klo','gitu','terus','makanya','begitu','jg','skrng','pdhl',
    'tpi','ny','udh','knp','gt','deh','lho','lhoh','nya','donk','bener','koq','sich'
}
stopword_combined = stopword_sastrawi.union(stopwords_nltk).union(more_stopword)

# --- Fungsi Preprocessing ---
def remove_short_reviews(df):
    return df[df['content'].apply(lambda x: len(x.split()) >= 10)]

def case_folding(text):
    return text.lower() if isinstance(text, str) else ""

def normalize_app_name(text):
    if pd.isnull(text): return ""
    text = text.lower()
    text = re.sub(r'\bby\.?u\b', 'byu', text)
    text = re.sub(r'\bby u\b', 'byu', text)
    return text

def filtering_combined(text):
    if pd.isnull(text): return ""
    text = re.sub(r"([.,;:!?()\[\]{}\"'/])(\w)", r"\1 \2", text)
    text = re.sub(r"(\w)([.,;:!?()\[\]{}\"'/])", r"\1 \2", text)
    text = re.sub(r",(?=\S)", ", ", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def remove_stopwords(text):
    if isinstance(text, str):
        words = text.lower().split()
        filtered = [w for w in words if w not in stopword_combined]
        return " ".join(filtered)
    return ""

# --- Setup ---
st.set_page_config(
    page_title="Analisis Topik by.U",
    layout="wide"
)

st.title("Visualisasi Pemodelan Topik Ulasan Aplikasi by.U dengan BERTopic")

# --- Sidebar Navigasi Utama yang Lebih Menarik ---
with st.sidebar:
    # Judul Sidebar
    st.title("Navigasi Utama")
    # Pemisah garis horizontal
    st.markdown("---")
    # Selectbox Navigasi
    menu = st.selectbox(
        "Pilih Halaman:",
        ["Dataset Awal", "Preprocessing", "Visualisasi"]
    )

# --- Load Dataset dan Preprocessing ---
try:
    df = pd.read_csv("Testbyu_reviews.csv")

    if 'content' not in df.columns:
        st.error("Kolom 'content' tidak ditemukan.")
    else:
        # Preprocessing
        df = remove_short_reviews(df)
        df['case_folding'] = df['content'].apply(case_folding)
        df['normalized_text'] = df['case_folding'].apply(normalize_app_name)
        df['filtered_text'] = df['normalized_text'].apply(filtering_combined)
        df['final_text'] = df['filtered_text'].apply(remove_stopwords)

        # ========== Halaman 1: Dataset Awal ==========
        if menu == "Dataset Awal":
            st.subheader("Dataset Awal")
        # Informasi tambahan
            st.markdown("""
            **Informasi Dataset:**
            - Sumber data: Google Play Store (Aplikasi by.U)
            - Jumlah ulasan: 10.000 ulasan pengguna
            - Rentang waktu ulasan: 24 Oktober 2019 - 21 Mei 2025
            - Bahasa dominan: Bahasa Indonesia, dengan istilah serapan Bahasa Inggris
            """)

            st.info("Data ini digunakan sebagai dasar proses preprocessing dan pemodelan topik menggunakan metode BERTopic.")
            items_per_page = 20
            total_rows = len(df)
            total_pages = math.ceil(total_rows / items_per_page)
            page_number = st.number_input("Halaman:", 1, total_pages, 1, key="page_ds")
            start_idx = (page_number - 1) * items_per_page
            end_idx = start_idx + items_per_page
            st.dataframe(df.iloc[start_idx:end_idx][['userName', 'score', 'content', 'at']])

        # ========== Halaman 2: Preprocessing ==========
        elif menu == "Preprocessing":
            st.subheader("Tahapan Preprocessing")
            tab1, tab2, tab3, tab4 = st.tabs([
                "Case Folding", "Normalisasi 'byu'", "Filtering Simbol/Angka/URL", "Stopword Removal"
            ])
            items_per_page = 20
            total_rows = len(df)
            total_pages = math.ceil(total_rows / items_per_page)
            page_number = st.number_input("Halaman:", 1, total_pages, 1, key="page_pre")
            start_idx = (page_number - 1) * items_per_page
            end_idx = start_idx + items_per_page
            df_page = df.iloc[start_idx:end_idx]

            with tab1:
                st.info("Mengubah teks menjadi huruf kecil.")
                st.dataframe(df_page[['content', 'case_folding']].rename(columns={'content': 'Sebelum', 'case_folding': 'Sesudah'}))

            with tab2:
                st.info("Normalisasi variasi kata 'by u', 'by.u' menjadi 'byu'.")
                st.dataframe(df_page[['case_folding', 'normalized_text']].rename(columns={'case_folding': 'Sebelum', 'normalized_text': 'Sesudah'}))

            with tab3:
                st.info("Membersihkan simbol, angka, dan URL.")
                st.dataframe(df_page[['normalized_text', 'filtered_text']].rename(columns={'normalized_text': 'Sebelum', 'filtered_text': 'Sesudah'}))

            with tab4:
                st.info("Menghapus stopword (kata tidak penting).")
                st.dataframe(df_page[['filtered_text', 'final_text']].rename(columns={'filtered_text': 'Sebelum', 'final_text': 'Sesudah'}))

        # ========== Halaman 3: Visualisasi ==========
        elif menu == "Visualisasi":
            st.subheader("Visualisasi Antar Topik")
            tab1, tab2, tab3 = st.tabs([
                "Ringkasan Topik", "Distribusi Topik", "Topik per Dokumen"
            ])

            with tab1:
                st.info("Proporsi dokumen per topik dari hasil BERTopic.")

                try:
                    df_summary = pd.read_csv("topik_summary.csv")
                    df_summary = df_summary[df_summary["Topic"] != -1]

                    # Tampilkan pie chart interaktif
                    fig = px.pie(df_summary, values='Count', names='Name', title='Proporsi Dokumen per Topik')
                    st.plotly_chart(fig)

                    # Fitur pencarian kata kunci
                    search_query = st.text_input("Cari topik berdasarkan kata kunci:")
                    filtered = df_summary[df_summary['Name'].str.contains(search_query, case=False)] if search_query else df_summary

                    # Tabel ringkasan
                    st.dataframe(filtered[['Topic', 'Name', 'Count']])

                except FileNotFoundError:
                    st.error("File topik_summary.csv tidak ditemukan.")

            with tab2:
                st.subheader("Distribusi Topik")
                st.image("newplot.png", caption="Visualisasi Intertopic Distance Map (BERTopic)", width=600)


            with tab3:
                st.info("Tabel distribusi dokumen dan topik hasil klasifikasi BERTopic.")
                try:
                    df_doc = pd.read_csv("topik_per_dokumen.csv")
                    st.subheader("Topik per Dokumen")
                    st.dataframe(df_doc[['text', 'topic', 'probability', 'topic_label']])
                except FileNotFoundError:
                    st.error("File topik_per_dokumen.csv tidak ditemukan.")


except FileNotFoundError:
    st.error("File 'Testbyu_reviews.csv' tidak ditemukan.")
except Exception as e:
    st.error(f"Terjadi error: {e}")
