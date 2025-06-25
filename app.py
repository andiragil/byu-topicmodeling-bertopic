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
    df = pd.read_csv("byu_reviewsNew.csv")

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
            st.info("Data ini digunakan sebagai dasar proses preprocessing dan pemodelan topik menggunakan metode BERTopic.")

            # Informasi Dataset
            st.markdown("""
            ### 📌 Informasi Dataset:
            - **Sumber data:** Google Play Store (Aplikasi by.U)
            - **Jumlah ulasan:** 10.000 ulasan pengguna
            - **Rentang waktu ulasan:** 24 Oktober 2019 - 21 Mei 2025
            - **Bahasa dominan:** Bahasa Indonesia, dengan istilah serapan Bahasa Inggris
            """)

            # Penjelasan kolom
            st.markdown("""
            ### 🧾 Penjelasan Kolom:
            - **`content`**: Berisi isi ulasan yang ditulis oleh pengguna aplikasi. Kolom ini menjadi fokus utama dalam pemodelan topik.
            - **`at`**: Tanggal ketika ulasan diberikan. Digunakan untuk mengetahui rentang waktu pengumpulan data.
            """)

            # Pagination setup
            items_per_page = 10
            total_rows = len(df)
            total_pages = math.ceil(total_rows / items_per_page)

            col1, col2 = st.columns([1, 5])
            with col1:
                page_number = st.number_input("Pilih halaman:", 1, total_pages, 1, key="page_ds")
            with col2:
                st.markdown(f"**Halaman {page_number} dari {total_pages}**")

            start_idx = (page_number - 1) * items_per_page
            end_idx = start_idx + items_per_page
            st.dataframe(df.iloc[start_idx:end_idx][['content', 'at']])

        # ========== Halaman 2: Preprocessing ==========
        elif menu == "Preprocessing":
            st.subheader("Tahapan Preprocessing")
            st.markdown("""
            Proses preprocessing bertujuan untuk membersihkan dan menstandarkan data teks sebelum dilakukan pemodelan topik. 
            Berikut adalah empat tahapan utama dalam proses ini, masing-masing ditampilkan dalam tab berbeda.
            """)

            tab1, tab2, tab3, tab4 = st.tabs([
                "Case Folding", "Normalisasi 'byu'", "Filtering Simbol/Angka/URL", "Stopword Removal"
            ])

            # Pagination setup
            items_per_page = 10
            total_rows = len(df)
            total_pages = math.ceil(total_rows / items_per_page)

            col1, col2 = st.columns([1, 5])
            with col1:
                page_number = st.number_input("Pilih halaman:", 1, total_pages, 1, key="page_pre")
            with col2:
                st.markdown(f"**Halaman {page_number} dari {total_pages}**")

            start_idx = (page_number - 1) * items_per_page
            end_idx = start_idx + items_per_page
            df_page = df.iloc[start_idx:end_idx]

            # --- TAB 1: Case Folding ---
            with tab1:
                st.info("""
                Tahapan awal untuk mengubah seluruh huruf menjadi huruf kecil (lowercase). Hal ini bertujuan untuk menyamakan representasi kata, 
                contohnya 'Internet' dan 'internet' menjadi 'internet'.
                """)
                st.dataframe(df_page[['content', 'case_folding']].rename(columns={
                    'content': 'Sebelum',
                    'case_folding': 'Sesudah'
                }))

            # --- TAB 2: Normalisasi ---
            with tab2:
                st.info("""
                Proses ini mengganti variasi penulisan kata 'by u', 'by.u', dll menjadi bentuk konsisten 'byu'.
                Normalisasi ini penting untuk menghindari fragmentasi makna dalam analisis topik.
                """)
                st.dataframe(df_page[['case_folding', 'normalized_text']].rename(columns={
                    'case_folding': 'Sebelum',
                    'normalized_text': 'Sesudah'
                }))

            # --- TAB 3: Filtering ---
            with tab3:
                st.info("""
                Pada tahap ini dilakukan pembersihan terhadap:
                - Simbol atau tanda baca
                - Angka
                - URL atau tautan
                Proses ini membuat data lebih bersih dan bebas dari noise.
                """)
                st.dataframe(df_page[['normalized_text', 'filtered_text']].rename(columns={
                    'normalized_text': 'Sebelum',
                    'filtered_text': 'Sesudah'
                }))

            # --- TAB 4: Stopword Removal ---
            with tab4:
                st.info("""
                Stopword adalah kata-kata yang sangat umum dan tidak memberikan makna signifikan, seperti 'yang', 'dan', 'atau', dsb.
                Stopword dihapus agar topik yang dihasilkan lebih tajam dan relevan.
                
                Stopword yang digunakan: gabungan dari **Sastrawi**, **NLTK Bahasa Inggris**, dan kosakata informal Indonesia (seperti 'gak', 'aja', dll).
                """)
                st.dataframe(df_page[['filtered_text', 'final_text']].rename(columns={
                    'filtered_text': 'Sebelum',
                    'final_text': 'Sesudah'
                }))


        # ========== Halaman 3: Visualisasi ==========
        elif menu == "Visualisasi":
            st.subheader("Visualisasi Antar Topik")
            tab1, tab2, tab3 = st.tabs([
                "Ringkasan Topik", "Distribusi Topik", "Topik per Dokumen"
            ])

            with tab1:
                st.subheader("Ringkasan Topik")
                st.info("""
                    Ringkasan ini menunjukkan jumlah dokumen yang diklasifikasikan ke dalam masing-masing topik yang berhasil diidentifikasi dari data ulasan pengguna aplikasi by.U.

                    Masing-masing topik diberi nama berdasarkan 3-5 kata kunci paling representatif yang muncul dari algoritma BERTopic.
                    """)


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

                # Penjelasan dari Intertopic Distance Map
                st.markdown("""
                **Interpretasi Peta Jarak Topik (Intertopic Distance Map):**

                Peta di atas menggambarkan hubungan antar topik hasil BERTopic.  
                Semakin **jauh posisi antar lingkaran**, maka semakin **berbeda isi topik** tersebut.  
                Sebaliknya, lingkaran yang saling **berdekatan** cenderung memiliki **kemiripan kata kunci**.

                Berdasarkan visualisasi:
                - Sebagian besar topik terlihat **cukup tersebar**, menandakan **keragaman isi topik** yang baik.
                - Tidak ada tumpukan berlebihan antar topik, yang berarti model berhasil **memisahkan tema utama** dari data ulasan.
                """)

                # Tampilkan tab per topik dan gambarnya
                topic_tabs = st.tabs([f"Topik {i}" for i in range(8)])
                for i in range(8):
                    with topic_tabs[i]:
                        st.image(f"topik_{i}.png", caption=f"Topik {i}", width=400)


            with tab3:
                st.subheader("Kesimpulan Hasil Pemodelan Topik")
                st.markdown("""
                Hasil analisis topik menggunakan **BERTopic** menghasilkan sejumlah tema utama yang teridentifikasi dari pola kata dalam **10.000 ulasan pengguna aplikasi by.U**.  
                Berikut adalah kesimpulan hasil eksplorasi:
                """)

                st.markdown("""
                ### Ringkasan Umum:
                - **Jumlah topik terbentuk:** 9 topik utama (tanpa menghitung outlier).
                - **Topik paling dominan:** Topik 0, mencakup ~38% dokumen.

                ### Penjelasan Tiap Topik:
                """)

                st.markdown("""
                #### 🟦 Topik 0: *Aplikasi, login, buka, aplikasinya*
                - Topik ini paling besar dengan 1.963 dokumen (~38%).
                - Ulasan berisi keluhan tentang aplikasi tidak bisa dibuka, login gagal, atau aplikasi lambat.

                #### 🟥 Topik 1: *Game, ping, main, youtube*
                - Mewakili pengguna yang mengeluhkan performa saat bermain game atau streaming.
                - Istilah seperti "ping tinggi", "lag", atau "game tidak lancar" sering muncul.

                #### 🟪 Topik 2: *Mode, sinyal, pesawat, sinyalnya*
                - Fokus pada isu sinyal hilang, terutama setelah mode pesawat atau restart jaringan.
                - Banyak yang menyebut “sinyalnya hilang terus”, “mode pesawat tidak membantu”.

                #### 🟩 Topik 3: *Kartu, pembayaran, sim, transaksi*
                - Pengguna mengalami masalah saat membeli kartu SIM, proses pembayaran gagal, atau tidak diproses.
                - Ada juga yang menyebut kesulitan registrasi kartu.

                #### 🟦 Topik 4: *Mbps, unlimited, gb, fup*
                - Berisi keluhan tentang kecepatan internet yang tidak sesuai ekspektasi.
                - Istilah "unlimited tapi lambat", "FUP turun drastis" banyak ditemukan.

                #### 🟧 Topik 5: *Murah, harga, promo, mudah*
                - Topik dengan sentimen lebih positif.
                - Banyak ulasan yang menyebutkan harga terjangkau, promo menarik, dan mudah digunakan.

                #### 🟨 Topik 6: *CS, email, chat, respon*
                - Berisi kritik terhadap customer service.
                - Keluhan umum seperti "respon lambat", "tidak ada jawaban dari CS", atau "email dibalas lama".

                #### 🟫 Topik 7: *Pulsa, pulsanya, konter, kesedot*
                - Masalah terkait pulsa terpotong sendiri atau pembelian pulsa tidak masuk.
                - Beberapa menyebut “pulsa hilang padahal tidak digunakan”.

                #### 🟪 Topik 8: *APK, susah, ngelag, buka*
                - Topik minor dengan 95 dokumen, berisi ulasan tentang file APK, aplikasi yang berat atau tidak kompatibel.
                - Beberapa menyarankan untuk memperbaiki versi terbaru aplikasi.
                """)


except FileNotFoundError:
    st.error("File 'byu_reviewsNew.csv' tidak ditemukan.")
except Exception as e:
    st.error(f"Terjadi error: {e}")
