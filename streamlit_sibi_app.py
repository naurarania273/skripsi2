import streamlit as st
from streamlit_webrtc import (
    webrtc_streamer,
    VideoTransformerBase,
    WebRtcMode,
)
import av
import numpy as np
import cv2
import joblib
import gdown
import mediapipe as mp
from collections import deque, Counter

import requests
from dotenv import load_dotenv
import os
import time
load_dotenv()

# === Fungsi untuk memperbaiki kalimat menggunakan OpenRouter/Gemini ===
def clearer(raw_text):
    """
    Memperbaiki teks bahasa Indonesia menggunakan OpenRouter (preferred).
    Jika gagal, mengembalikan raw_text sebagai fallback.
    Hasil yang dikembalikan adalah string final (bukan JSON).
    """
    if not raw_text:
        return ""

    system_prompt = """
Tolong ubah teks Bahasa Indonesia di bawah ini menjadi kalimat lengkap dan mudah dibaca. Teks ini ditulis tanpa spasi dan mungkin mengandung salah ketik (typo). Koreksi kata-kata yang salah, tambahkan spasi, dan susun menjadi kalimat yang benar sesuai tata bahasa Indonesia.

Berikan hasil akhirnya hanya dalam tanda petik ganda seperti "..." tanpa penjelasan tambahan.
Contoh:
Input: sayamaukepasarmembelibuahtapipadinyagakadaorangterusterpaksaakubaliklagikekeretadanmenungguteman
Output: "Saya mau ke pasar membeli buah, tapi padinya nggak ada orang. Terus, terpaksa aku balik lagi ke kereta dan menunggu teman."
"""

    # 1) Coba pakai OpenRouter (jika API key ada)
    or_key = os.getenv("OR_APIKEY")
    if or_key:
        try:
            payload = {
                "model": "google/gemma-3-12b-it:free",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": raw_text}
                ]
            }
            headers = {
                "Authorization": f"Bearer {or_key}",
                "Content-Type": "application/json"
            }
            resp = requests.post("https://openrouter.ai/api/v1/chat/completions", json=payload, headers=headers, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                # sesuaikan path response jika beda
                return data["choices"][0]["message"]["content"].strip()
            else:
                # lanjut ke fallback
                print("OpenRouter status:", resp.status_code, resp.text)
        except Exception as e:
            print("OpenRouter error:", e)

    # 2) Fallback: jika punya Google Generative API key
    ggl_key = os.getenv("GGLAI_H")
    if ggl_key:
        try:
            payload = {
                "contents": [
                    {
                        "role": "model",
                        "parts": [{"text": system_prompt}]
                    },
                    {
                        "role": "user",
                        "parts": [{"text": raw_text}]
                    }
                ]
            }
            headers = {
                "X-goog-api-key": ggl_key,
                "Content-Type": "application/json"
            }
            resp = requests.post(
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
                json=payload, headers=headers, timeout=15
            )
            if resp.status_code == 200:
                data = resp.json()
                return data["candidates"][0]["content"]["parts"][0]["text"].strip()
            else:
                print("Gemini API status:", resp.status_code, resp.text)
        except Exception as e:
            print("Gemini API error:", e)

    # 3) Jika semua gagal, kembalikan raw_text sebagai fallback (atau kosong)
    return raw_text

# === Konfigurasi halaman utama ===
st.set_page_config(page_title="Deteksi SIBI", layout="wide")

# === Sidebar Interaktif ===
with st.sidebar:
    st.image("assets/img/alfabet.jpg", width=180)
    st.markdown("## 🧭 Menu Utama")
    halaman = st.radio("📂 Pilih Halaman:", ["🏠 Beranda", "📷 Deteksi SIBI"])

# === Halaman: Beranda ===
if halaman.startswith("🏠"):
    st.title("📘 Abjad Bahasa Isyarat SIBI")
    st.markdown("""
    Selamat datang di aplikasi deteksi huruf **SIBI (Sistem Isyarat Bahasa Indonesia)** satu tangan secara real-time.
    Aplikasi ini memanfaatkan **MediaPipe** untuk mendeteksi koordinat landmark tangan, kemudian diklasifikasikan menggunakan **Random Forest**. 
    Hasil deteksi huruf disusun menjadi kata dan kalimat, serta diperbaiki secara otomatis menggunakan **LLM Gemma 3 melalui OpenRouter** agar sesuai dengan kaidah bahasa Indonesia.
    """)
    st.image("assets/img/alfabet.jpg", caption="Abjad dalam SIBI", use_container_width=True)
    st.info("👉 Pindah ke halaman **Deteksi SIBI** melalui sidebar untuk memulai pengenalan huruf.")

# === Halaman: Deteksi SIBI ===
elif halaman.startswith("📷"):

    st.title("🤖 Deteksi Huruf SIBI Real-Time")

    # Pastikan ada file log untuk menyimpan kata sementara
    if not os.path.exists("./log_raw.txt"):
        with open("./log_raw.txt", "w") as f:
            f.write("")

    # --- Download model jika belum ada ---
    model_filename = "sibi_rf_model.pkl"
    if not os.path.exists(model_filename):
        # Terima link Google Drive share atau link raw; ubah sesuai link kamu.
        # Contoh share link: https://drive.google.com/file/d/1U2Me1TWPst6OFHwkuJISQmrbYhz-VJay/view?usp=sharing
        drive_link = "https://drive.google.com/file/d/1U2Me1TWPst6OFHwkuJISQmrbYhz-VJay/view?usp=sharing"  # ganti dengan link kamu

        # Jika link berformat /file/d/ID/..., ekstrak ID
        file_id = None
        if "drive.google.com" in drive_link:
            import re
            m = re.search(r"/d/([a-zA-Z0-9_-]+)", drive_link)
            if m:
                file_id = m.group(1)
                download_url = f"https://drive.google.com/uc?id={file_id}"
            else:
                # mungkin sudah dalam format uc?id=...
                download_url = drive_link
        else:
            download_url = drive_link

        try:
            st.info("Mengunduh model, tunggu sebentar...")
            gdown.download(download_url, model_filename, quiet=False)
        except Exception as e:
            st.error(f"Gagal mendownload model: {e}")

    # Setelah ada file, load model (cek lagi kalau gagal)
    try:
        model = joblib.load(model_filename)
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        st.stop()

    # Inisialisasi MediaPipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )

    # === Video Processor ===
    class SignPredictor(VideoTransformerBase):
        def __init__(self):
            self.prediction_history = deque(maxlen=9)
            self.current_prediction = "..."
            self.last_appended = None
            self.kata = ""

        def extract_raw_landmarks(self, image):
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                if coords.shape == (21, 3):
                    return coords.flatten()
            return None

        def transform(self, frame: av.VideoFrame) -> np.ndarray:
            img = frame.to_ndarray(format="bgr24")
            img = cv2.flip(img, 1)

            features = self.extract_raw_landmarks(img)
            if features is not None:
                try:
                    pred = model.predict([features])[0]
                    self.prediction_history.append(pred)

                    if len(self.prediction_history) >= 6:
                        most_common, count = Counter(self.prediction_history).most_common(1)[0]
                        if count >= 6:
                            self.current_prediction = most_common
                            if self.current_prediction != self.last_appended:
                                self.kata += self.current_prediction
                                self.last_appended = self.current_prediction
                                # tulis ke file log agar nanti bisa diproses saat STOP
                                with open("./log_raw.txt", "w") as f:
                                    f.write(self.kata)
                        else:
                            self.current_prediction = "Menstabilkan..."
                    else:
                        self.current_prediction = "Menstabilkan..."
                except Exception as e:
                    self.current_prediction = f"Error: {str(e)}"
            else:
                self.prediction_history.clear()
                self.current_prediction = "Tangan tidak terdeteksi"

            # === Tampilkan huruf prediksi (kiri atas) ===
            cv2.rectangle(img, (0, 0), (400, 40), (0, 255, 0), 1)
            cv2.putText(img, f"{self.current_prediction}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            # === Tampilkan kata (kiri bawah) ===
            raw = self.kata if self.kata else ''
            kata_text = f"KATA = {raw}"

            (text_w, _), _ = cv2.getTextSize(kata_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            y1 = img.shape[0] - 50
            y2 = img.shape[0]
            cv2.rectangle(img, (0, y1), (text_w + 20, y2), (0, 0, 0), -1)
            cv2.putText(img, kata_text, (10, y2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            return img

    # Layout dan webrtc
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # Initialize post-processing flag
        if "processed_on_stop" not in st.session_state:
            st.session_state.processed_on_stop = False

        webrtc_ctx = webrtc_streamer(
            key="sibi-app",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=SignPredictor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    # Detect STOP: PLAYING → READY
    if not webrtc_ctx.state.playing and not st.session_state.processed_on_stop:
        # Baca file log jika ada
        if os.path.exists("./log_raw.txt"):
            with open("./log_raw.txt", "r") as f:
                raw_text = f.read().strip()
        else:
            raw_text = ""

        st.write(f"raw text: {raw_text}")

        # RESET file log
        with open("./log_raw.txt", "w") as f:
            f.write("")

        # Proses perbaikan kalimat
        with st.spinner("🛑 Stream stopped. Running post-processing..."):
            if raw_text:
                startt = time.time()
                result = clearer(raw_text)
                endt = time.time() - startt

                st.write(f"result: {result}")
                st.write(f"{round(endt, 2)} seconds")
            else:
                st.write("No Words")

            st.session_state.processing_done = True
            st.session_state.processed_on_stop = True

    # Reset ketika stream berjalan lagi
    if webrtc_ctx.state.playing:
        st.session_state.processed_on_stop = False

    st.markdown("---")
    st.info("💡 Tips: Letakkan tangan kanan di tengah kamera dan tahan selama 1–2 detik. Untuk menghapus kata yang sudah terbentuk, silakan hentikan dan jalankan kembali kamera.")
    st.caption("Deteksi dan Perbaikan Kalimat Bahasa Isyarat")
