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
import streamlit.components.v1 as components

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
    df = pd.read_csv("byu_streamlit.csv")

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
            st.markdown("Data ini digunakan sebagai dasar proses preprocessing dan pemodelan topik menggunakan metode BERTopic.")

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

            # Setup pagination (hitung dulu)
            items_per_page = 10
            total_rows = len(df)
            total_pages = math.ceil(total_rows / items_per_page)

            # Ambil nilai default halaman pertama
            page_number = 1  # Default

            # Cek jika 'page_ds' sudah ada di session_state, gunakan nilainya
            if 'page_ds' in st.session_state:
                page_number = st.session_state['page_ds']

            start_idx = (page_number - 1) * items_per_page
            end_idx = start_idx + items_per_page

            # Tampilkan tabel dulu
            st.dataframe(df.iloc[start_idx:end_idx][['content', 'at']])

            # Pagination tampil di bawah tabel
            col1, col2 = st.columns([1, 5])
            with col1:
                page_number = st.number_input("Pilih halaman:", 1, total_pages, page_number, key="page_ds")
            with col2:
                st.markdown(f"**Halaman {page_number} dari {total_pages}**")

        # ========== Halaman 2: Preprocessing ==========
        elif menu == "Preprocessing":
            st.subheader("Tahapan Preprocessing")
            st.markdown("""
            Proses preprocessing bertujuan untuk membersihkan dan menstandarkan data teks sebelum dilakukan pemodelan topik. 
            Berikut adalah empat tahapan utama dalam proses ini, masing-masing ditampilkan dalam tab berbeda.
            """)

            tab1, tab2, tab3, tab4 = st.tabs([
                "Case Folding", "Normalisasi 'byu'", "Penghapusan Simbol/Angka/URL", "Stopword Removal"
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
                
                Stopword yang digunakan: gabungan dari **Sastrawi**, **NLTK Bahasa Inggris**, dan kosakata informal Indonesia.
                """)
                st.dataframe(df_page[['filtered_text', 'final_text']].rename(columns={
                    'filtered_text': 'Sebelum',
                    'final_text': 'Sesudah'
                }))


        # ========== Halaman 3: Visualisasi ==========
        elif menu == "Visualisasi":
            st.subheader("Visualisasi Antar Topik")
            st.markdown("""
                Tahap visualisasi ini bertujuan untuk memberikan gambaran umum mengenai distribusi dokumen ulasan berdasarkan hasil pemodelan topik menggunakan BERTopic.  
                Visualisasi ini membantu dalam mengidentifikasi topik-topik dominan yang paling banyak dibahas oleh pengguna.
                """)
            tab1, tab2, tab3 = st.tabs([
                "Ringkasan Topik", "Pemetaan Topik", "Kesimpulan Hasil"
            ])

            with tab1:
                try:
                    df_summary = pd.read_csv("topik_summary.csv")
                    df_summary = df_summary[df_summary["Topic"] != -1]

                    # Tampilkan pie chart interaktif
                    st.subheader("Proporsi Dokumen per Topik")
                    fig = px.pie(df_summary, values='Count', names='Name')
                    st.plotly_chart(fig)

                    # Tabel ringkasan
                    st.subheader("Tabel Ringkasan Topik")
                    st.markdown("""
                    Berikut ini adalah tabel ringkasan topik yang dihasilkan dari proses pemodelan BERTopic:

                    - **`Topic`**: ID atau nomor urut dari masing-masing topik yang terdeteksi.
                    - **`Name`**: Representasi topik berdasarkan kumpulan kata kunci paling dominan dari setiap topik.
                    - **`Count`**: Jumlah dokumen (ulasan) yang termasuk dalam masing-masing topik.

                    Tabel ini berguna untuk melihat topik mana yang paling sering dibahas oleh pengguna aplikasi by.U.
                    """)
                    st.dataframe(df_summary[['Topic', 'Name', 'Count']])

                    st.subheader("Visualisasi Topik (HTML)")
                    with open("barchart_topics.html", "r", encoding="utf-8") as f:
                        html_data = f.read()
                    components.html(html_data, height=600, scrolling=True)


                except FileNotFoundError:
                    st.error("File topik_summary.csv tidak ditemukan.")

            with tab2:
                st.subheader("Pemetaan Topik")
                st.image("newplot.png", caption="Visualisasi Intertopic Distance Map (BERTopic)", width=600)
                with open("intertopic_topics.html", "r", encoding="utf-8") as f:
                    inter_html = f.read()
                components.html(inter_html, height=650, scrolling=True)

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
                Hasil analisis topik menggunakan **BERTopic** berhasil mengelompokkan **10.000 ulasan pengguna aplikasi by.U** ke dalam sejumlah tema utama.  
                Setiap topik merepresentasikan pola keluhan, komentar, atau pujian yang muncul secara berulang.
                
                Berikut adalah ringkasan hasil eksplorasi:
                """)

                # Baca file hasil topik
                try:
                    df_summary = pd.read_csv("topik_summary.csv")
                    df_summary = df_summary[df_summary["Topic"] != -1]  # Exclude outliers
                    df_summary = df_summary.sort_values("Count", ascending=False)

                    # Ringkasan umum
                    st.markdown(f"""
                    ### Ringkasan Umum:
                    - **Jumlah topik terbentuk:** {len(df_summary)} topik utama.
                    - **Topik paling dominan:** Topik {df_summary.iloc[0]['Topic']}, mencakup ~{round(100 * df_summary.iloc[0]['Count'] / df_summary['Count'].sum(), 1)}% dokumen.
                    """)

                    # Penjelasan per topik
                    st.markdown("### Penjelasan Tiap Topik:")

                    for idx, row in df_summary.iterrows():
                        st.markdown(f"""
                        #### 🔹 Topik {row['Topic']}: *{row['Representation']}*
                        - Jumlah dokumen: {row['Count']}
                        - Contoh ulasan representatif:
                        > _"{row['Representative_Docs']}"_
                        """)
                except FileNotFoundError:
                    st.error("File topik_summary.csv tidak ditemukan.")



except FileNotFoundError:
    st.error("File 'byu_reviewsNew.csv' tidak ditemukan.")
except Exception as e:
    st.error(f"Terjadi error: {e}")
