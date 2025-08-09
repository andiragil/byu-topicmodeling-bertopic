import streamlit as st
import pandas as pd
import numpy as np
import math
import re
import altair as alt
import plotly.express as px

st.set_page_config(
    page_title="Analisis Topik by.U",
    layout="wide"
)

st.title("Visualisasi Pemodelan Topik Ulasan Aplikasi by.U dengan BERTopic")

with st.sidebar:
    st.title("Navigasi Utama")
    st.markdown("---")
    menu = st.selectbox(
        "Pilih Halaman:",
        ["Dataset Awal", "Preprocessing", "Visualisasi"]
    )

# --- Load Dataset ---
try:
    df = pd.read_csv("reviews_preprocessed.csv")

    if 'content' not in df.columns:
        st.error("Kolom 'content' tidak ditemukan.")
    else:
        # ========== Halaman 1: Dataset Awal ==========
        if menu == "Dataset Awal":
            st.subheader("Dataset Awal")
            st.markdown("Data ini digunakan sebagai dasar proses preprocessing dan pemodelan topik menggunakan metode BERTopic.")
            st.markdown("""
            ### 📌 Informasi Dataset:
            - **Sumber data:** Google Play Store (Aplikasi by.U)
            - **Jumlah ulasan:** 10.000 ulasan pengguna
            - **Rentang waktu ulasan:** 24 Oktober 2019 - 21 Mei 2025
            - **Bahasa dominan:** Bahasa Indonesia, dengan istilah serapan Bahasa Inggris
            """)
            st.markdown("""
            ### 🧾 Penjelasan Kolom:
            - **`content`**: Berisi isi ulasan yang ditulis oleh pengguna aplikasi. Kolom ini menjadi fokus utama dalam pemodelan topik.
            - **`at`**: Tanggal ketika ulasan diberikan. Digunakan untuk mengetahui rentang waktu pengumpulan data.
            """)

            items_per_page = 10
            total_rows = len(df)
            total_pages = math.ceil(total_rows / items_per_page)
            page_number = 1 
            if 'page_ds' in st.session_state:
                page_number = st.session_state['page_ds']
            start_idx = (page_number - 1) * items_per_page
            end_idx = start_idx + items_per_page
            st.dataframe(df.iloc[start_idx:end_idx][['content', 'at']])
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
                }), use_container_width=True)

             
        
            # --- TAB 2: Normalisasi ---
            with tab2:
                st.info("""
                Proses ini mengganti variasi penulisan kata 'by u', 'by.u', dll menjadi bentuk konsisten 'byu'.
                Normalisasi ini penting untuk menghindari fragmentasi makna dalam analisis topik.
                """)
                st.dataframe(df_page[['case_folding', 'normalized']].rename(columns={
                    'case_folding': 'Sebelum',
                    'normalized': 'Sesudah'
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
                st.dataframe(df_page[['normalized', 'filtered_text']].rename(columns={
                    'normalized': 'Sebelum',
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

                    st.subheader("Visualisasi dalam Bentuk Bar Chart")
                    st.markdown("""
                    Bar chart berikut menunjukkan **lima kata kunci dominan pada setiap topik** 
                    beserta skor bobotnya (C-TF-IDF). Skor ini menunjukkan tingkat kekhasan kata 
                    terhadap topik tersebut dibandingkan topik lain.
                    """)
                    st.image("barchart.png", caption="Top Words Scores", width=800)
                    
                    st.subheader("Visualisasi dalam Bentuk Pie Chart")
                    fig = px.pie(df_summary, values='Count', names='Name')
                    st.plotly_chart(fig)

                    with st.expander("Lihat Tabel dalam bentuk Tabel Topik"):
                        st.subheader("Tabel Ringkasan Topik")
                        st.markdown("""
                        Berikut ini adalah tabel ringkasan topik yang dihasilkan dari proses pemodelan BERTopic:

                        - **`Topic`**: ID atau nomor urut dari masing-masing topik yang terdeteksi.
                        - **`Name`**: Representasi topik berdasarkan kumpulan kata kunci paling dominan dari setiap topik.
                        - **`Count`**: Jumlah dokumen (ulasan) yang termasuk dalam masing-masing topik.

                        Tabel ini berguna untuk melihat topik mana yang paling sering dibahas oleh pengguna aplikasi by.U.
                        """)
                        st.dataframe(df_summary[['Topic', 'Name', 'Count']])


                except FileNotFoundError:
                    st.error("File topik_summary.csv tidak ditemukan.")

            with tab2:
                st.subheader("Pemetaan Topik")
                st.image("newplot.png", caption="Visualisasi Intertopic Distance Map (BERTopic)", width=600)
                st.markdown("""
                **Interpretasi Peta Jarak Topik (Intertopic Distance Map):**

                Peta di atas menggambarkan hubungan antar topik hasil BERTopic.  
                Semakin **jauh posisi antar lingkaran**, maka semakin **berbeda isi topik** tersebut.  
                Sebaliknya, lingkaran yang saling **berdekatan** cenderung memiliki **kemiripan kata kunci**.

                Berdasarkan visualisasi:
                - Sebagian besar topik terlihat **cukup tersebar**, menandakan **keragaman isi topik** yang baik.
                - Tidak ada tumpukan berlebihan antar topik, yang berarti model berhasil **memisahkan tema utama** dari data ulasan.
                """)


            with tab3:
                st.subheader("Kesimpulan Hasil Pemodelan Topik")
                st.markdown("""
                Hasil analisis topik menggunakan **BERTopic** berhasil mengelompokkan **10.000 ulasan pengguna aplikasi by.U** ke dalam sejumlah tema utama.  
                Setiap topik merepresentasikan pola keluhan, komentar, atau pujian yang muncul secara berulang.
                
                Berikut adalah ringkasan hasil eksplorasi:
                """)

                try:
                    df_summary = pd.read_csv("topik_summary.csv")
                    df_summary = df_summary[df_summary["Topic"] != -1]
                    df_summary = df_summary.sort_values("Count", ascending=False)
                    st.markdown(f"""
                    ### Ringkasan Umum:
                    - **Jumlah topik terbentuk:** {len(df_summary)} topik utama.
                    - **Topik paling dominan:** Topik {df_summary.iloc[0]['Topic']}, mencakup ~{round(100 * df_summary.iloc[0]['Count'] / df_summary['Count'].sum(), 1)}% dokumen.
                    """)

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
    st.error("File 'reviews_preprocessed.csv' tidak ada.")
except Exception as e:
    st.error(f"Terjadi error: {e}")