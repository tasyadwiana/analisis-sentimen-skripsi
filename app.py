import streamlit as st
import pandas as pd
import sqlite3
import hashlib
import pickle
import re
import nltk
import html
import os
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
from streamlit_option_menu import option_menu
from wordcloud import WordCloud
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

st.markdown("""
    <style>
        div[data-baseweb="base-input"] {
            background-color: #FFFFFF !important;
        }
         
         [data-baseweb="input"]{
            background-color: #FFFFFF !important;
         }

    </style>
""", unsafe_allow_html=True)

def format_tanggal_indo():
   hari_dict = {0: 'Senin', 1: 'Selasa', 2: 'Rabu', 3: 'Kamis', 4: 'Jumat', 5: 'Sabtu', 6: 'Minggu'}
   bulan_dict = {1: 'Januari', 2: 'Februari', 3: 'Maret', 4: 'April', 5: 'Mei', 6: 'Juni', 
                  7: 'Juli', 8: 'Agustus', 9: 'September', 10: 'Oktober', 11: 'November', 12: 'Desember'}

   tz_wib = datetime.timezone(datetime.timedelta(hours=7))
   now = datetime.datetime.now(tz_wib)

   hari = hari_dict[now.weekday()]
   bulan = bulan_dict[now.month]

   return f"{hari}, {now.day} {bulan} {now.year}"

def cetak_pdf_terpadu(pilihan_laporan, df_data=None, metrik=None):
   pdf = FPDF()
   pdf.add_page()


   pdf.image("logosentiment.png", x=10, y=8, w=20)
   pdf.set_font("Arial", "B", 12)
   pdf.multi_cell(0, 6, "LAPORAN ANALISIS SENTIMEN KOMENTAR YOUTUBE\nPODCAST KELUARGA ARTIS", align="C")
    
   pdf.line(10, 25, 200, 25)
   pdf.line(10, 26, 200, 26)
   pdf.ln(10)

   pdf.set_font("Arial", "B", 11)
   pdf.cell(0, 8, f"{pilihan_laporan}", ln=True, align="C")
   pdf.ln(5)

   pdf.set_font("Arial", "", 11)
   total_data = len(df_data)
   jml_positif = len(df_data[df_data['label'].astype(str).str.lower() == 'positif'])
   jml_negatif = len(df_data[df_data['label'].astype(str).str.lower() == 'negatif'])
   jml_netral = len(df_data[df_data['label'].astype(str).str.lower() == 'netral'])

   acc_nb = metrik['nb']['accuracy'] * 100 if metrik else 0
   acc_svm = metrik['svm']['accuracy'] * 100 if metrik else 0

   if pilihan_laporan == "Laporan Ringkasan Data Hasil Penelitian":
      pct_pos = (jml_positif / total_data) * 100
      pct_neg = (jml_negatif / total_data) * 100
      pct_netral = (jml_netral / total_data) * 100
      
      label_width = 70

      pdf.multi_cell(0, 8, "Berikut adalah ringkasan keseluruhan dari data dan pengujian yang dilakukan dalam penelitian ini:")
      pdf.set_x(pdf.l_margin) 
      pdf.cell(label_width, 8, "- Total Data Keseluruhan")
      pdf.cell(0, 8, f": {total_data} komentar", ln=True)

      pdf.cell(label_width, 8, "- Sentimen Positif")
      pdf.cell(0, 8, f": {pct_pos:.1f}%", ln=True)
      
      pdf.cell(label_width, 8, "- Sentimen Negatif")
      pdf.cell(0, 8, f": {pct_neg:.1f}%", ln=True)

      pdf.cell(label_width, 8, "- Sentimen Netral")
      pdf.cell(0, 8, f": {pct_netral:.1f}%", ln=True)

      pdf.ln(5)
      pdf.multi_cell(0, 8, "Hasil Kinerja Algoritma Klasifikasi (Akurasi):")
      pdf.set_x(pdf.l_margin) 
      pdf.cell(label_width, 8, "- Naive Bayes")
      pdf.cell(0, 8, f": {acc_nb:.2f}%", ln=True)
      pdf.cell(label_width, 8, "- Support Vector Machine")
      pdf.cell(0, 8, f": {acc_svm:.2f}%", ln=True)

   elif pilihan_laporan == "Laporan Evaluasi Performa Model":
      label_width = 80
      pdf.multi_cell(0, 8, "Perbandingan metrik performa pengujian algoritma (dalam persentase) setelah tahap SMOTE:")
      pdf.set_x(pdf.l_margin) 
      pdf.cell(label_width, 8, "- Tingkat Akurasi Naive Bayes")
      pdf.cell(0, 8, f": {acc_nb:.2f}%", ln=True)
      pdf.cell(label_width, 8, "- Tingkat Akurasi Support Vector Machine")
      pdf.cell(0, 8, f": {acc_svm:.2f}%", ln=True)
      pdf.ln(5)

      pdf.set_font("Arial", "B", 11)
      pdf.cell(0, 8, "Kesimpulan:", ln=True)
      pdf.set_font("Arial", "I", 11)
      if acc_svm > acc_nb:
         pdf.multi_cell(0, 8, f"Berdasarkan hasil pengujian di atas, algoritma Support Vector Machine (SVM) menunjukkan performa klasifikasi sentimen yang lebih unggul dibandingkan Naive Bayes dengan selisih tingkat akurasi {(acc_svm - acc_nb):.2f}%.")
      elif acc_nb > acc_svm:
         pdf.multi_cell(0, 8, f"Berdasarkan hasil pengujian di atas, algoritma Naive Bayes menunjukkan performa klasifikasi sentimen yang lebih unggul dibandingkan SVM dengan selisih tingkat akurasi {(acc_nb - acc_svm):.2f}%.")
      else:
         pdf.multi_cell(0, 8, "Berdasarkan hasil pengujian di atas, algoritma Naive Bayes dan Support Vector Machine (SVM) menunjukkan performa klasifikasi sentimen yang seimbang.")

   elif pilihan_laporan == "Laporan Kata Sentimen Positif Terbanyak":
      pdf.multi_cell(0, 8, "Visualisasi kata kunci yang paling dominan muncul pada komentar bersentimen Positif:")

      teks_pos = ' '.join(df_data[df_data['label'].astype(str).str.lower() == 'positif']['komentar_bersih'].dropna().astype(str))
      wordcloud = WordCloud(width=600, height=300, background_color='white', colormap='Greens').generate(teks_pos)
      wordcloud.to_file("temp_wc.png")

      pdf.image("temp_wc.png", x=15, w=180)
      pdf.ln(5)

      top_pos = Counter(teks_pos.split()).most_common(20)
      pdf.set_font("Arial", "B", 10)
      pdf.cell(0, 8, "Rincian Top 20 Kata Positif:", ln=True)
      pdf.set_font("Arial", "", 10)

      baris_kata = [f"{kata} ({jumlah})" for kata, jumlah in top_pos]
      pdf.multi_cell(0, 6, " | ".join(baris_kata))

      if os.path.exists("temp_wc.png"):
         os.remove("temp_wc.png")

   elif pilihan_laporan == "Laporan Distribusi Sentimen":
      pdf.multi_cell(0, 8, "Berikut adalah visualisasi bar chart yang merepresentasikan perbandingan jumlah kelas sentimen:")
        
      fig, ax = plt.subplots(figsize=(6, 4))
      sns.barplot(x=['Positif', 'Negatif', 'Netral'], y=[jml_positif, jml_negatif, jml_netral], palette='viridis', hue=['Positif', 'Negatif', 'Netral'], ax=ax)
      ax.set_ylabel("Jumlah Komentar")
      plt.tight_layout()
      plt.savefig("temp_bar.png")
      plt.close(fig)
   
      pdf.image("temp_bar.png", x=30, w=150)
      if os.path.exists("temp_bar.png"):
         os.remove("temp_bar.png")
      
      pdf.ln(5)

      label_width = 50

      pdf.cell(label_width, 8, "- Sentimen Positif")
      pdf.cell(0, 8, f": {jml_positif} komentar", ln=True)

      pdf.cell(label_width, 8, "- Sentimen Negatif")
      pdf.cell(0, 8, f": {jml_negatif} komentar", ln=True)

      pdf.cell(label_width, 8, "- Sentimen Netral")
      pdf.cell(0, 8, f": {jml_netral} komentar", ln=True)

   pdf.ln(15)
   pdf.set_font("Arial", "", 11)
   pdf.cell(120) 
   pdf.cell(60, 6, f"Jakarta, {format_tanggal_indo()}", ln=True, align="C")
   pdf.cell(120)
    
   return bytes(pdf.output())

def make_hashes(password):
   return hashlib.sha256(str.encode(password)).hexdigest()
def check_hashes(password, hashed_text):
   if make_hashes(password) == hashed_text:
      return True
   return False

def create_usertable():
   conn = sqlite3.connect('users_data.db')
   c = conn.cursor()
   c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT UNIQUE, password TEXT)')
   c.execute('''CREATE TABLE IF NOT EXISTS historytable(
             id INTEGER PRIMARY KEY AUTOINCREMENT,
             username TEXT,
             teks_asli TEXT,
             teks_bersih TEXT,
             pred_nb TEXT,
             pred_svm TEXT,
             waktu TIMESTAMP)''')
   conn.commit()
   conn.close()

def add_userdata(username, password):
    conn = sqlite3.connect('users_data.db')
    c = conn.cursor()
    c.execute('INSERT INTO userstable(username, password) VALUES (?,?)', (username, password))
    conn.commit()
    conn.close()

def login_user(username, password):
   conn = sqlite3.connect('users_data.db')
   c = conn.cursor()
   c.execute('SELECT password FROM userstable WHERE username = ?', (username,))
   data = c.fetchone()
   conn.close()
   return data
create_usertable()

def add_history(username, teks_asli, teks_bersih, pred_nb, pred_svm):
   conn = sqlite3.connect('users_data.db')
   c = conn.cursor()
   waktu = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
   c.execute('INSERT INTO historytable(username, teks_asli, teks_bersih, pred_nb, pred_svm, waktu) VALUES (?,?,?,?,?,?)',
             (username, teks_asli, teks_bersih, str(pred_nb), str(pred_svm), waktu))
   conn.commit()
   conn.close()

def view_all_history(username):
   conn = sqlite3.connect('users_data.db')
   df = pd.read_sql_query(f"SELECT teks_asli AS 'Teks Asli', teks_bersih AS 'Teks Bersih', pred_nb AS 'Naive Bayes', pred_svm AS 'SVM', waktu AS Waktu FROM historytable WHERE username='{username}' ORDER BY waktu DESC", conn)
   conn.close()
   return df
st.set_page_config(page_title="Dashboard Analisis Sentimen", layout="wide")

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')

@st.cache_resource
def load_models_and_metrics():
    with open ('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open ('model_nb.pkl', 'rb') as f:
        nb_model = pickle.load(f)
    with open ('model_svm.pkl', 'rb') as f:
        svm_model = pickle.load(f)
    with open ('eval_metrics.pkl', 'rb') as f:
       metrics = pickle.load(f)
    return vectorizer, nb_model, svm_model, metrics

vectorizer, model_nb, model_svm, evaluation_metrics= load_models_and_metrics()

@st.cache_data
def load_data():
   return pd.read_csv('dataset_dashboard.csv')

df = load_data()

list_stopwords = set(stopwords.words('indonesian'))
factory = StemmerFactory()
stemmer = factory.create_stemmer()

kamus_slang = {'gw': 'saya',
    'bgt': 'banget', 'yg': 'yang', 'gpp': 'gapapa', 'ky': 'kayak', 'jg': 'juga',
    'kejawa': 'ketawa', 'ak': 'saya', 'pdl': 'padahal', 'gk': 'tidak', 'dr': 'dari',
    'dgn': 'dengan', 'sm': 'sama', 'trs': 'terus', 'skrg': 'sekarang', 'bs': 'bisa',
    'jd': 'jadi', 'klo': 'kalau', 'knp': 'kenapa', 'tp': 'tapi', 'msh': 'masih', 'gbs': 'tidak bisa',
    'emg': 'memang', 'th': 'tahun', 'podkes': 'podcast', 'tgl': 'tanggal', 'jgn': 'jangan',
    'tdi': 'tadi', 'udh': 'sudah'}

def is_spam(text):
  if not isinstance(text, str):
    return False

  keywords = ['slot', 'gacor', 'alexis']
  for kw in keywords:
    if kw in text:
      return True
  if re.search(r'\b[a-z]{4,}\d{2,4}\b', text):
    return True
  return False

def cleaning_process(text):
  if not isinstance(text, str):
    return ""

  text = re.sub(r'<a href[^>]+>.*?</a>', ' ', text)
  text = re.sub(r'https?://\S+|www\.\S+', '', text)

  text = html.unescape(html.unescape(text))
  text = re.sub(r'<.*?>', ' ', text)

  text = re.sub(r'\b\d{1,2}:\d{2}(?::\d{2})?\b', ' ', text)

  text = re.sub(r'@[A-Za-z0-9_]+', '', text)
  text = re.sub(r'#\w+', '', text)

  text = re.sub(r'\b\d{3,}\b', ' ', text)

  text = re.sub(r'[^A-Za-z0-9 ]', '', text)
  text = re.sub(r'\s+', ' ', text).strip()

  words = text.split()
  normalized_words = [kamus_slang.get(word.lower(), word) for word in words]
  text = ' '.join(normalized_words)
  text = text.lower()
  tokens = word_tokenize(text)
  tokens = [word for word in tokens if word not in list_stopwords]
  text_joined = ' '.join(tokens)
  stemmed_text = stemmer.stem(text_joined)
  return stemmed_text

#inisialisasi login
if 'logged_in' not in st.session_state:
   st.session_state['logged_in'] = False

#login
def login_page():
   col1, col2, col3 = st.columns([1, 1.5, 1])

   with col2:
      st.title("Login")
      st.markdown("Silahkan masuk untuk mengakses web Analisis Sentimen")

      menu_login = st.selectbox("Pilih", ["Login", "Register"])
      if menu_login == "Login":
         username = st.text_input("Username")
         password = st.text_input("Password", type="password")

         if st.button("Masuk", type="primary", width="stretch"):
            result = login_user(username, password)

            if result:
               hashed_pswd = result[0]
               if check_hashes(password, hashed_pswd):
                  st.session_state['logged_in'] = True
                  st.session_state['username'] = username
                  st.success(f"Selamat datang, {username}!")
                  st.rerun()
               else:
                  st.error("Password salah!")
            else:
               st.warning("Username tidak ditemukan!")

      elif menu_login == "Register":
         st.subheader("Buat Akun Baru")
         new_user = st.text_input("Username")
         new_password = st.text_input("Password Baru", type="password")

         if st.button("Daftar Akun", type="primary", width="stretch"):
            if new_user and new_password:
               try: 
                  add_userdata(new_user, make_hashes(new_password))
                  st.success("Akun berhasil dibuat! Silakan pilih menu Login untuk masuk.")
               except sqlite3.IntegrityError:
                  st.error("Username tersebut sudah digunakan. Silakan pilih username lain.")
            else:
               st.warning("Username dan password tidak boleh kosong!")
         
# sidebar
def main_page():
   with st.sidebar:
      menu = option_menu(
         menu_title="Pilih Halaman:",
         options=["Dashboard", "Analisis Sentimen", "Riwayat Analisis", "Evaluasi Model", "Visualisasi", "Cetak Laporan"]
      )
   # dashboard
   if menu == "Dashboard":
      st.title("Dashboard Dataset")
      st.markdown("Ringkasan informasi dari dataset penelitian yang digunakan.")

      col1, col2, col3, col4 = st.columns(4)
      
      total_data = len(df)
      jml_positif = len(df[df['label'].astype(str).str.lower() == 'positif'])
      jml_negatif = len(df[df['label'].astype(str).str.lower() == 'negatif'])
      jml_netral = len(df[df['label'].astype(str).str.lower() == 'netral'])

      col1.metric("Total Data", total_data)
      col2.metric("Sentimen Positif", jml_positif)
      col3.metric("Sentimen Negatif", jml_negatif)
      col4.metric("Sentimen Netral", jml_netral)

      st.markdown("---")
      st.subheader("Contoh Dataset Bersih")
      df_tampil=df[['komentar_bersih','label']].iloc[1:21].copy().rename(columns={'komentar_bersih': 'Komentar Bersih','label': 'Label' })
      df_tampil.index = range(1, len(df_tampil) + 1)
      df_tampil.index.name = "No"

      styled_df = (
         df_tampil.style
         .set_table_styles([
            {
                  "selector": "th",
                  "props": [
                     ("background-color", "#FFFFFF"),
                     ("color", "#2B2B2B"),
                     ("font-weight", "bold"),
                     ("text-align", "center")
                  ]
            },
            {
                  "selector": "td",
                  "props": [
                     ("text-align", "left")
                  ]
            }
         ])
         .hide(axis="index")   # jika ingin menghilangkan index
      )

      st.markdown(styled_df.to_html(), unsafe_allow_html=True)

   # analisis sentimen
   elif menu == "Analisis Sentimen":
      st.title("Analisis Sentimen Real-Time")
      st.write("Masukkan komentar baru untuk diprediksi menggunakan model **Naive Bayes** dan **SVM**.")

      user_input = st.text_area("Masukkan teks komentar yang ingin dianalisis:", height=100)

      if st.button("Analisis Komentar", type="primary"):
         if user_input.strip() == "":
               st.warning("Silakan masukan teks komentar terlebih dahulu")
         else:
            with st.spinner('Memproses teks dan melakukan klasifikasi'):
               text_processed = cleaning_process(user_input)
               text_vectorized = vectorizer.transform([text_processed])

               pred_nb = model_nb.predict(text_vectorized)[0]
               pred_svm = model_svm.predict(text_vectorized)[0]

               #baru
               add_history(st.session_state['username'], user_input, text_processed, pred_nb, pred_svm)
               
         st.info(f"**Hasil Preprocessing Teks:** {text_processed}")
         st.markdown("<br>", unsafe_allow_html=True)

         col1, col2 = st.columns(2)
         with col1:
            st.subheader("Naive Bayes")
            if str(pred_nb).lower() == 'positif':
               st.success(f"**{pred_nb}**")
            elif str(pred_nb).lower() == 'negatif':
               st.error(f"**{pred_nb}**")
            else:
               st.warning(f"**{pred_nb}**")
         
         with col2:
            st.subheader("Support Vector Machine")
            if str(pred_svm).lower() == 'positif':
               st.success(f"**{pred_svm}**") 
            elif str(pred_svm).lower() == 'negatif':
               st.error(f"**{pred_svm}**")
            else:
               st.warning(f"**{pred_svm}**")

         st.markdown("---")
         if pred_nb != pred_svm:
            st.info("**Kesimpulan:** Terjadi perbedan prediksi antara kedua model. Algoritma SVM memisahkan kelas berdasarkan margin *hyperplane*, sedangkan Naive Bayes berdasarkan probabilitas kemunculan data.")
         else:
            st.success("**Kesimpulan:** Kedua algoritma sepakat memberikan klasifikasi sentimen yang sama untuk komentar ini.")
   
   #riwayat analisis
   elif menu == "Riwayat Analisis":
      st.title("Riwayat Analisis Sentimen")
      st.markdown(f"Menampilkan seluruh teks yang pernah dianalisis oleh pengguna: **{st.session_state['username']}**")

      history_df = view_all_history(st.session_state['username'])

      if not history_df.empty:
         styled_history = (
            history_df.style
            .set_table_styles([
               {
                     "selector": "th",
                     "props": [
                        ("background-color", "#FFFFFF"),
                        ("color", "#2B2B2B"),
                        ("font-weight", "bold"),
                        ("text-align", "center")
                     ]
               },
               {
                     "selector": "td",
                     "props": [
                        ("text-align", "left")
                     ]
               }
            ])
            .hide(axis="index")   # jika ingin menghilangkan index
         )

         st.markdown(styled_history.to_html(), unsafe_allow_html=True)

         csv = history_df.to_csv(index=False).encode('utf-8')
         st.download_button(
            label="Unduh Riwayat (CSV)",
            data=csv,
            file_name=f'riwayat_sentimen_{st.session_state["username"]}.csv',
            mime='text/csv',
            type="primary"
         )
      else:
         st.info("Belum ada riwayat analisis. Silakan lakukan prediksi di menu 'Analisis Sentimen' terlebih dahulu.")

   #evaluasi model
   elif menu == "Evaluasi Model":
      st.title("Evaluasi Performa Model")
      st.markdown("Perbandingan metrik evaluasi terhadap komentar YouTube Podcast Keluarga Artis.")

      labels = evaluation_metrics['labels']

      col1, col2 = st.columns(2)
      with col1:
         st.subheader("Naive Bayes")
         st.text(f"Accuracy    : {evaluation_metrics['nb']['accuracy'] * 100:.2f}%")
         st.text(f"Precision   : {evaluation_metrics['nb']['precision'] * 100:.2f}%")
         st.text(f"Recall      : {evaluation_metrics['nb']['recall'] * 100:.2f}%")
         st.text(f"F1-Score    : {evaluation_metrics['nb']['f1'] * 100:.2f}%")

         fig, ax = plt.subplots()
         sns.heatmap(evaluation_metrics['nb']['confusion_matrix'], annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
         plt.title("Confusion Matrix - Naive Bayes")
         st.pyplot(fig)
      with col2:
         st.subheader("Support Vector Machine (SVM)")
         st.text(f"Accuracy    : {evaluation_metrics['svm']['accuracy'] * 100:.2f}%")
         st.text(f"Precision   : {evaluation_metrics['svm']['precision'] * 100:.2f}%")
         st.text(f"Recall      : {evaluation_metrics['svm']['recall'] * 100:.2f}%")
         st.text(f"F1-Score    : {evaluation_metrics['svm']['f1'] * 100:.2f}%")

         fig, ax = plt.subplots()
         sns.heatmap(evaluation_metrics['svm']['confusion_matrix'], annot=True, fmt='d', cmap='Greens', xticklabels=labels, yticklabels=labels)
         plt.title("Confusion Matrix - SVM")
         st.pyplot(fig)

      st.markdown("---")
      st.subheader("Kesimpulan Performa")

      acc_nb = evaluation_metrics['nb']['accuracy']
      acc_svm = evaluation_metrics['svm']['accuracy']

      if acc_svm > acc_nb:
        st.success(f"Berdasarkan metrik pengujian, **Support Vector Machine (SVM)** menunjukkan performa yang lebih unggul dengan tingkat akurasi **{acc_svm * 100:.2f}%** dibandingkan Naïve Bayes ({acc_nb * 100:.2f}%).")
      elif acc_nb > acc_svm:
        st.success(f"Berdasarkan metrik pengujian, **Naïve Bayes** menunjukkan performa yang lebih unggul dengan tingkat akurasi **{acc_nb * 100:.2f}%** dibandingkan SVM ({acc_svm * 100:.2f}%).")
      else:
        st.info(f"Kedua algoritma menunjukkan performa yang seimbang dengan tingkat akurasi **{acc_nb * 100:.2f}%**.")
   #visualisasi
   elif menu == "Visualisasi":
      st.title("Visualisasi Data Sentimen")

      sentimen_counts = df['label'].value_counts()

      col1, col2 = st.columns(2)
      with col1:
         st.subheader("Pie Chart Distribusi Sentimen")
         fig1, ax1 = plt.subplots()
         ax1.pie(sentimen_counts, labels=sentimen_counts.index, autopct='%1.1f%%', startangle=90, colors=['#ff9999', '#66b3ff', '#99ff99'])
         ax1.axis('equal')
         st.pyplot(fig1)
      with col2:
         st.subheader("Bar Chart Distribusi Sentimen")
         fig2, ax2 = plt.subplots()
         sns.barplot(x=sentimen_counts.index, y=sentimen_counts.values, palette='viridis', ax=ax2)
         ax2.set_ylabel("Jumlah Komentar")
         st.pyplot(fig2)
         
      st.markdown("---")
      st.subheader("Word Cloud (Kata yang sering muncul)")
      all_text = ' '.join(df['komentar_bersih'].dropna().astype(str).tolist())
      wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='tab10').generate(all_text)
         
      fig3, ax3, = plt.subplots(figsize=(10, 5))
      ax3.imshow(wordcloud, interpolation='bilinear')
      ax3.axis('off')
      st.pyplot(fig3)
#cetak laporan
   elif menu == "Cetak Laporan":
      st.title("Cetak Dokumen Laporan")
      st.write("Unduh hasil analisis sentimen dalam format dokumen PDF.")
      st.markdown("---")

      daftar_laporan = [
            "Laporan Ringkasan Data Hasil Penelitian",
            "Laporan Evaluasi Performa Model",
            "Laporan Kata Sentimen Positif Terbanyak",
            "Laporan Distribusi Sentimen"
      ]
      
      pilihan = st.selectbox("Pilih Kategori Laporan:", daftar_laporan)

      if st.button("Siapkan Dokumen", type="primary"):
         with st.spinner("Menyusun dokumen PDF beserta visualisasi grafik..."):
            
            pdf_bytes = cetak_pdf_terpadu(pilihan_laporan=pilihan, df_data=df, metrik=evaluation_metrics)
                  
            st.success(f"Dokumen **{pilihan}** berhasil disiapkan!")
                  
            st.download_button(
               label="Unduh File PDF",
               data=pdf_bytes,
               file_name=f"{pilihan.replace(' ', '_')}.pdf",
               mime="application/pdf"
                  )
         

if not st.session_state['logged_in']:
   login_page()
else:
   main_page()