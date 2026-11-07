import os
import re
import json
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

# ---------------------------------------------------------
# KONFIGURASI
# ---------------------------------------------------------
st.set_page_config(
    page_title="Analisis Sentimen Kesehatan",
    page_icon="💬",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "sentiment_mlp_pipeline.joblib")
DATA_FILE = os.path.join(BASE_DIR, "youtube_kesehatan_labeled.csv")
METRICS_FILE = os.path.join(BASE_DIR, "model_metrics.json")
CONFUSION_FILE = os.path.join(BASE_DIR, "confusion_matrix.png")

SENTIMENT_ORDER = ["positif", "negatif", "netral"]
EMOJI = {"positif": "😊", "negatif": "😟", "netral": "😐"}

# ---------------------------------------------------------
# FUNGSI PEMUATAN FILE
# ---------------------------------------------------------
@st.cache_resource
def load_model(path):
    return joblib.load(path)

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

@st.cache_data
def load_metrics(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    return None

# ---------------------------------------------------------
# PREPROCESSING PREDIKSI
# Harus konsisten dengan preprocessing pada notebook.
# ---------------------------------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def normalize_words(text):
    # Mengurangi huruf berulang: "sukaaa" menjadi "suka".
    text = re.sub(r"(.)\1{2,}", r"\1", str(text).lower())

    replacements = {
        "aq": "aku",
        "gw": "saya",
        "gue": "saya",
        "gak": "tidak",
        "ga": "tidak",
        "nggak": "tidak",
        "enggak": "tidak",
        "makasih": "terima kasih"
    }
    return " ".join(replacements.get(word, word) for word in text.split())

# ---------------------------------------------------------
# VALIDASI FILE
# ---------------------------------------------------------
st.title("💬 Dashboard Analisis Sentimen YouTube")
st.caption(
    "Topik: Kesehatan | Pelabelan: Rule-Based | "
    "Klasifikasi: TF-IDF dan Neural Network MLP"
)

missing_files = [
    name for name, path in [
        ("Model MLP", MODEL_FILE),
        ("Dataset berlabel", DATA_FILE)
    ]
    if not os.path.exists(path)
]

if missing_files:
    st.error(
        "File berikut belum ditemukan: "
        + ", ".join(missing_files)
        + ". Simpan file model dan CSV dalam folder yang sama dengan app_sentimen_nn.py."
    )
    st.stop()

try:
    model = load_model(MODEL_FILE)
    df = load_data(DATA_FILE)
except Exception as error:
    st.error(f"Gagal memuat model atau dataset: {error}")
    st.stop()

required_columns = {"sentimen", "clean_text"}
if not required_columns.issubset(df.columns):
    st.error("Dataset harus memiliki kolom 'sentimen' dan 'clean_text'.")
    st.stop()

df["sentimen"] = df["sentimen"].astype(str).str.lower().str.strip()
df["clean_text"] = df["clean_text"].fillna("").astype(str)
df = df[df["sentimen"].isin(SENTIMENT_ORDER)].copy()

if df.empty:
    st.error("Dataset tidak memiliki baris dengan label sentimen yang valid.")
    st.stop()

counts = (
    df["sentimen"]
    .value_counts()
    .reindex(SENTIMENT_ORDER, fill_value=0)
)
percentages = (counts / counts.sum() * 100).round(2)

# ---------------------------------------------------------
# RINGKASAN DATASET
# ---------------------------------------------------------
st.header("Ringkasan Dataset")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Komentar", f"{len(df):,}".replace(",", "."))
c2.metric("Positif", f"{counts['positif']:,}".replace(",", "."))
c3.metric("Negatif", f"{counts['negatif']:,}".replace(",", "."))
c4.metric("Netral", f"{counts['netral']:,}".replace(",", "."))

st.caption(
    "Distribusi berikut berasal dari dataset berlabel yang digunakan dalam praktikum. "
    "Label dibuat menggunakan pendekatan Rule-Based."
)

# ---------------------------------------------------------
# DISTRIBUSI SNETIMEN
# ---------------------------------------------------------
st.header("Distribusi Sentimen")
left, right = st.columns(2)

with left:
    fig, ax = plt.subplots(figsize=(7, 4.8))
    bars = ax.bar(
        counts.index,
        counts.values,
        color=["#2E9D70", "#D95D5D", "#6486B8"],
        width=0.62
    )
    ax.set_title("Jumlah Komentar per Sentimen", fontsize=13, pad=14)
    ax.set_xlabel("Kategori Sentimen")
    ax.set_ylabel("Jumlah Komentar")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.bar_label(bars, padding=4, fontsize=10)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with right:
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.pie(
        counts.values,
        labels=[label.capitalize() for label in counts.index],
        autopct="%1.1f%%",
        startangle=90,
        colors=["#2E9D70", "#D95D5D", "#6486B8"],
        textprops={"fontsize": 10},
        wedgeprops={"edgecolor": "white", "linewidth": 1}
    )
    ax.set_title("Persentase Sentimen", fontsize=13, pad=14)
    ax.axis("equal")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

distribution_table = pd.DataFrame({
    "Kategori Sentimen": [label.capitalize() for label in counts.index],
    "Jumlah Komentar": counts.values,
    "Persentase (%)": percentages.values
})
with st.expander("Lihat tabel distribusi sentimen"):
    st.dataframe(distribution_table, hide_index=True, use_container_width=True)

# ---------------------------------------------------------
# WORDCLOUD
# ---------------------------------------------------------
st.header("WordCloud per Kategori Sentimen")
st.caption(
    "Kata umum yang kurang informatif disaring. Istilah kesehatan tetap dipertahankan "
    "agar topik pembicaraan tidak hilang."
)

tabs = st.tabs(["😊 Positif", "😟 Negatif", "😐 Netral"])

# Stopwords tambahan untuk mengurangi kata percakapan yang terlalu umum.
# Kata negasi seperti 'tidak', 'belum', dan 'bukan' sengaja tidak dihapus.
extra_stopwords = {
    "yang", "dan", "di", "ke", "dari", "ini", "itu", "saya", "aku",
    "kamu", "dia", "kami", "kita", "nya", "ya", "yg", "untuk",
    "dengan", "karena", "jadi", "sudah", "saja", "aja", "dok",
    "dokter", "pak", "bu", "bang", "dong", "deh", "sih", "kok",
    "lah", "nih", "tuh", "kan", "lebih", "sangat", "banget",
    "bisa", "ada", "apa", "kalau", "kalo", "atau", "juga",
    "saat", "sudah", "masih", "nya", "video", "youtube"
}
wordcloud_stopwords = set(STOPWORDS).union(extra_stopwords)

for tab, category in zip(tabs, SENTIMENT_ORDER):
    with tab:
        text_data = " ".join(
            df.loc[df["sentimen"] == category, "clean_text"]
            .dropna()
            .astype(str)
        )

        if text_data.strip():
            try:
                wc = WordCloud(
                    width=1200,
                    height=520,
                    background_color="white",
                    stopwords=wordcloud_stopwords,
                    collocations=False,
                    max_words=100,
                    min_font_size=8,
                    contour_width=0
                ).generate(text_data)

                fig, ax = plt.subplots(figsize=(13, 5.2))
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                fig.tight_layout(pad=0)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            except ValueError:
                st.info("Kata yang tersisa belum cukup untuk membentuk WordCloud.")
        else:
            st.info(f"Tidak ada komentar pada kategori {category}.")

# ---------------------------------------------------------
# EVALUASI MODEL
# Angka diambil dari file metrik hasil notebook.
# Confusion matrix ditampilkan dari hasil evaluasi yang disimpan.
# ---------------------------------------------------------
st.header("Evaluasi Neural Network MLP")
metrics = load_metrics(METRICS_FILE)

if metrics:
    metric_columns = st.columns(4)
    metric_columns[0].metric("Accuracy", f"{metrics.get('accuracy', 0) * 100:.2f}%")
    metric_columns[1].metric("Precision Macro", f"{metrics.get('precision_macro', 0) * 100:.2f}%")
    metric_columns[2].metric("Recall Macro", f"{metrics.get('recall_macro', 0) * 100:.2f}%")
    metric_columns[3].metric("F1-score Macro", f"{metrics.get('f1_macro', 0) * 100:.2f}%")

    st.caption(
        f"Evaluasi menggunakan {metrics.get('total_training', 0):,} data training "
        f"dan {metrics.get('total_testing', 0):,} data testing. "
        "Metrik ini mengukur prediksi terhadap label Rule-Based pada data testing."
    )
else:
    st.warning(
        "File model_metrics.json belum ditemukan. Jalankan notebook sampai bagian evaluasi "
        "dan penyimpanan metrik."
    )

if os.path.exists(CONFUSION_FILE):
    st.subheader("Confusion Matrix")
    st.image(
        CONFUSION_FILE,
        caption="Confusion Matrix hasil pengujian Neural Network MLP",
        use_container_width=True
    )
else:
    st.warning(
        "File confusion_matrix.png belum ditemukan. Jalankan cell evaluasi confusion matrix "
        "pada notebook."
    )

# ---------------------------------------------------------
# PREDIKSI KOMENTAR BARU
# ---------------------------------------------------------
st.header("Prediksi Sentimen Komentar Baru")
st.write(
    "Masukkan komentar berbahasa Indonesia untuk memperkirakan kategori sentimennya."
)

with st.form("prediction_form"):
    comment = st.text_area(
        "Masukkan komentar",
        placeholder="Contoh: Terima kasih dokter, penjelasannya sangat bermanfaat.",
        height=120
    )
    submitted = st.form_submit_button("Prediksi Sentimen", type="primary")

if submitted:
    if not comment.strip():
        st.warning("Silakan masukkan komentar terlebih dahulu.")
    else:
        cleaned = normalize_words(clean_text(comment))

        if not cleaned:
            st.warning("Teks tidak memiliki kata yang dapat diproses.")
        else:
            try:
                prediction = model.predict([cleaned])[0]
                probabilities = model.predict_proba([cleaned])[0]
                classes = model.named_steps["mlp"].classes_

                st.success(
                    f"Hasil prediksi: {EMOJI.get(prediction, '')} "
                    f"{str(prediction).capitalize()}"
                )

                probability_df = pd.DataFrame({
                    "Sentimen": [str(label).capitalize() for label in classes],
                    "Probabilitas (%)": probabilities * 100
                }).sort_values("Probabilitas (%)", ascending=False)

                probability_df["Probabilitas (%)"] = (
                    probability_df["Probabilitas (%)"].round(2)
                )

                st.subheader("Probabilitas setiap kategori")
                st.dataframe(
                    probability_df,
                    hide_index=True,
                    use_container_width=True
                )

                st.bar_chart(
                    probability_df.set_index("Sentimen"),
                    y="Probabilitas (%)",
                    height=220
                )

                with st.expander("Lihat teks setelah preprocessing"):
                    st.write(cleaned)

                st.info(
                    "Probabilitas merupakan keluaran model berdasarkan pola data latih, "
                    "bukan jaminan bahwa sentimen telah dipahami dengan benar. "
                    "Model ini digunakan untuk praktikum analisis komentar, bukan diagnosis medis."
                )

            except Exception as error:
                st.error(f"Prediksi gagal dilakukan: {error}")

# ---------------------------------------------------------
# CATATAN METODOLOGI
# ---------------------------------------------------------
with st.expander("Catatan metodologi dan keterbatasan"):
    st.markdown(
        """
        - Dataset terdiri atas komentar YouTube bertopik kesehatan.
        - Label sentimen pada dataset dibuat menggunakan metode Rule-Based berbasis kamus.
        - Model klasifikasi menggunakan TF-IDF dan Multilayer Perceptron (MLP).
        - Distribusi kelas tidak seimbang, sehingga nilai accuracy perlu dibaca bersama
          precision, recall, F1-score, dan confusion matrix.
        - Model mempelajari pola label otomatis. Oleh karena itu, hasil prediksi tidak
          dapat dianggap sebagai penilaian sentimen manusia yang sempurna.
        - Hasil prediksi bukan nasihat atau diagnosis medis.
        """
    )

st.divider()
st.caption("Praktikum Analisis Sentimen | Topik Kesehatan | TF-IDF + Neural Network MLP")