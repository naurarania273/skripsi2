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
import mediapipe as mp
from collections import deque, Counter
import requests  # Diperlukan untuk download dari Hugging Face
from dotenv import load_dotenv
import os
import time

# Muat environment variables jika ada (untuk API keys)
load_dotenv()

# === Fungsi untuk memperbaiki kalimat menggunakan LLM (tidak diubah) ===
def clearer(raw_text):
    if not raw_text:
        return ""
    system_prompt = """
Tolong ubah teks Bahasa Indonesia di bawah ini menjadi kalimat lengkap dan mudah dibaca. Teks ini ditulis tanpa spasi dan mungkin mengandung salah ketik (typo). Koreksi kata-kata yang salah, tambahkan spasi, dan susun menjadi kalimat yang benar sesuai tata bahasa Indonesia. Berikan hasil akhirnya hanya dalam tanda petik ganda seperti "..." tanpa penjelasan tambahan. Contoh: Input: sayamaukepasarmembelibuahtapipadinyagakadaorangterusterpaksaakubaliklagikekeretadanmenungguteman Output: "Saya mau ke pasar membeli buah, tapi padinya nggak ada orang. Terus, terpaksa aku balik lagi ke kereta dan menunggu teman."
"""
    # 1) Coba pakai OpenRouter
    or_key = os.getenv("OR_APIKEY")
    if or_key:
        try:
            payload = {"model": "google/gemma-2-9b-it:free", "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": raw_text}]}
            headers = {"Authorization": f"Bearer {or_key}", "Content-Type": "application/json"}
            resp = requests.post("https://openrouter.ai/api/v1/chat/completions", json=payload, headers=headers, timeout=20)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"].strip('"')
            else:
                print("OpenRouter status:", resp.status_code, resp.text)
        except Exception as e:
            print("OpenRouter error:", e)
    # 2) Fallback: Google Generative AI
    ggl_key = os.getenv("GGLAI_H")
    if ggl_key:
        try:
            payload = {"contents": [{"role": "model", "parts": [{"text": system_prompt}]}, {"role": "user", "parts": [{"text": raw_text}]}]}
            headers = {"X-goog-api-key": ggl_key, "Content-Type": "application/json"}
            resp = requests.post("https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent", json=payload, headers=headers, timeout=20)
            if resp.status_code == 200:
                return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip('"')
            else:
                print("Gemini API status:", resp.status_code, resp.text)
        except Exception as e:
            print("Gemini API error:", e)
    return f'"{raw_text}" (gagal diproses LLM)'

# === Konfigurasi halaman utama ===
st.set_page_config(page_title="Deteksi SIBI", layout="wide")

# === Sidebar Interaktif ===
with st.sidebar:
    st.image("https://raw.githubusercontent.com/rizal-muhamad/sibi-app/main/assets/img/alfabet.jpg", width=180)
    st.markdown("## 🧭 Menu Utama")
    halaman = st.radio("📂 Pilih Halaman:", ["🏠 Beranda", "📷 Deteksi SIBI"])

# === Halaman: Beranda (tidak diubah) ===
if halaman.startswith("🏠"):
    st.title("📘 Abjad Bahasa Isyarat SIBI")
    st.markdown("""
    Selamat datang di aplikasi deteksi huruf **SIBI (Sistem Isyarat Bahasa Indonesia)** satu tangan secara real-time.
    Aplikasi ini memanfaatkan **MediaPipe** untuk mendeteksi koordinat landmark tangan, kemudian diklasifikasikan menggunakan **Random Forest**. 
    Hasil deteksi huruf disusun menjadi kata dan kalimat, serta diperbaiki secara otomatis menggunakan **model bahasa (LLM)** agar sesuai dengan kaidah bahasa Indonesia.
    """)
    st.image("https://raw.githubusercontent.com/rizal-muhamad/sibi-app/main/assets/img/alfabet.jpg", caption="Abjad dalam SIBI", use_container_width=True)
    st.info("👉 Pindah ke halaman **Deteksi SIBI** melalui sidebar untuk memulai pengenalan huruf.")

# === Halaman: Deteksi SIBI ===
elif halaman.startswith("📷"):

    st.title("🤖 Deteksi Huruf SIBI Real-Time")
    
    # Pastikan ada file log untuk menyimpan kata sementara
    log_file = "./log_raw.txt"
    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            f.write("")

    # --- IMPLEMENTASI BARU: Download dan Load Model dari Hugging Face ---
    @st.cache_resource
    def load_model_from_hf():
        """
        Mengunduh model dari Hugging Face Hub jika belum ada, lalu memuatnya.
        Fungsi ini di-cache agar tidak dijalankan berulang kali.
        """
        model_filename = "sibi_rf_model.pkl"
        
        if not os.path.exists(model_filename):
            # ⬇️ GANTI URL DI BAWAH INI DENGAN LINK DARI REPO HUGGING FACE ANDA ⬇️
            # Contoh: "https://huggingface.co/nama-user/nama-repo/resolve/main/sibi_rf_model.pkl"
            hf_url = "https://huggingface.co/nept28/sibi-random-forest-classifier/resolve/main/sibi_rf_model.pkl" # <-- GANTI INI
            
            try:
                with st.spinner(f"Mengunduh model '{model_filename}' dari Hugging Face..."):
                    response = requests.get(hf_url)
                    response.raise_for_status()  # Cek jika download gagal
                    with open(model_filename, "wb") as f:
                        f.write(response.content)
            except Exception as e:
                st.error(f"Gagal mengunduh model: {e}")
                return None
                
        # Muat model dari file lokal yang sudah diunduh
        try:
            model = joblib.load(model_filename)
            return model
        except Exception as e:
            st.error(f"Gagal memuat model dari file: {e}")
            return None

    # Panggil fungsi baru untuk memuat model
    model = load_model_from_hf()

    # Hentikan aplikasi jika model gagal dimuat
    if model is None:
        st.stop()
    # --------------------------------------------------------------------

    # Inisialisasi MediaPipe (tidak diubah)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )

    # === Video Processor (tidak diubah) ===
    class SignPredictor(VideoTransformerBase):
        def __init__(self):
            self.prediction_history = deque(maxlen=15)
            self.current_prediction = "..."
            self.last_appended_char = None
            self.word_string = ""

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

                    if len(self.prediction_history) >= 10:
                        most_common, count = Counter(self.prediction_history).most_common(1)[0]
                        if count >= 8:
                            self.current_prediction = most_common
                            if self.current_prediction != self.last_appended_char:
                                self.word_string += self.current_prediction
                                self.last_appended_char = self.current_prediction
                                with open(log_file, "w") as f:
                                    f.write(self.word_string)
                        else:
                            self.current_prediction = "Menstabilkan..."
                    else:
                        self.current_prediction = "Menstabilkan..."
                except Exception:
                    self.current_prediction = "Error prediksi"
            else:
                self.prediction_history.clear()
                self.current_prediction = "Tangan tidak terdeteksi"
                self.last_appended_char = None

            cv2.rectangle(img, (0, 0), (400, 40), (21, 113, 23, 255), -1)
            cv2.putText(img, f"Prediksi: {self.current_prediction}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            kata_text = f"KATA: {self.word_string}"
            (text_w, _), _ = cv2.getTextSize(kata_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            y1, y2 = img.shape[0] - 50, img.shape[0]
            cv2.rectangle(img, (0, y1), (text_w + 20, y2), (0, 0, 0, 128), -1)
            cv2.putText(img, kata_text, (10, y2 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            return img

    # Layout dan webrtc (tidak diubah)
    col1, col2 = st.columns([1, 4])
    with col1:
        st.info("Hasil Perbaikan")
        result_placeholder = st.empty()
        result_placeholder.markdown("`Belum ada kalimat yang terbentuk.`")

    with col2:
        webrtc_ctx = webrtc_streamer(
            key="sibi-app",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=SignPredictor,
            media_stream_constraints={"video": {"height": 720}, "audio": False},
            async_processing=True,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )

    # Proses perbaikan teks setelah stream berhenti (tidak diubah)
    if not webrtc_ctx.state.playing and os.path.exists(log_file):
        with open(log_file, "r") as f:
            raw_text = f.read().strip()
        if raw_text:
            with st.spinner("Memperbaiki kalimat yang terbentuk..."):
                final_result = clearer(raw_text)
                result_placeholder.success(f"**Hasil:** {final_result}")
            with open(log_file, "w") as f:
                f.write("")
                
    st.markdown("---")
    st.info("💡 **Tips:** Posisikan tangan kanan di tengah kamera dan tahan setiap huruf selama 1-2 detik. Untuk menghapus kalimat, hentikan dan jalankan ulang kamera.")
