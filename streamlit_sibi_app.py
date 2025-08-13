import streamlit as st
from streamlit_webrtc import (
    webrtc_streamer, 
    VideoTransformerBase, 
    WebRtcMode,
    RTCConfiguration
)
import av
import numpy as np
import cv2
import joblib
import mediapipe as mp
from collections import deque, Counter

import requests
from dotenv import load_dotenv
import os
import time
import uuid
import requests
import nest_asyncio
nest_asyncio.apply()

import asyncio

try:
    loop = asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)


load_dotenv()


if "session_key" not in st.session_state:
    st.session_state["session_key"] = f"user-{uuid.uuid4()}"

sess_k = st.session_state["session_key"]

def clearer(raw_text):
    system_prompt="""
Tolong ubah teks Bahasa Indonesia di bawah ini menjadi kalimat lengkap dan mudah dibaca. Teks ini ditulis tanpa spasi dan mungkin mengandung salah ketik (typo). Koreksi kata-kata yang salah, tambahkan spasi, dan susun menjadi kalimat yang benar sesuai tata bahasa Indonesia.

Berikan hasil akhirnya **hanya** dalam tanda petik ganda seperti "..." tanpa penjelasan tambahan.
Contoh:
Input: sayamaukepasarmembelibuahtapipadinyagakadaorangterusterpaksaakubaliklagikekeretadanmenungguteman  
Output: "Saya mau ke pasar membeli buah, tapi padinya nggak ada orang. Terus, terpaksa aku balik lagi ke kereta dan menunggu teman."

"""

    payload = {
        "model": "google/gemma-3-12b-it:free",
        "messages": [
            {
                "role":"system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": raw_text
            }
        ]
    }
    headers = {
        "Authorization": f"Bearer {os.getenv('OR_APIKEY')}",
        "Content-Type": "application/json"
    }
def clearer(raw_text):
    system_prompt = """
Tolong ubah teks Bahasa Indonesia di bawah ini menjadi kalimat lengkap dan mudah dibaca. Teks ini ditulis tanpa spasi dan mungkin mengandung salah ketik (typo). Koreksi kata-kata yang salah, tambahkan spasi, dan susun menjadi kalimat yang benar sesuai tata bahasa Indonesia.

Berikan hasil akhirnya **hanya** dalam tanda petik ganda seperti "..." tanpa penjelasan tambahan.
Contoh:
Input: sayamaukepasarmembelibuahtapipadinyagakadaorangterusterpaksaakubaliklagikekeretadanmenungguteman  
Output: "Saya mau ke pasar membeli buah, tapi padinya nggak ada orang. Terus, terpaksa aku balik lagi ke kereta dan menunggu teman."
"""

    payload = {
        "model": "google/gemma-3-12b-it:free",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": raw_text}
        ]
    }
    headers = {
        "Authorization": f"Bearer {os.getenv('OR_APIKEY')}",
        "Content-Type": "application/json"
    }

    
    response = requests.post("https://openrouter.ai/api/v1/chat/completions", json=payload, headers=headers)
    stat = response.status_code
    print("::::::::::::",stat)
    
    if str(stat) == "200":
        return response.json()["choices"][0]["message"]["content"]
    else:
        print("move to gemini AI API")
        payload = {
            "contents": [
                {
                    "role": "model",
                    "parts": [
                        {
                            "text": system_prompt
                        }
                    ]
                },
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": raw_text
                        }
                    ]
                }
            ]
        }
    
        headers = {
            "X-goog-api-key": os.getenv('GGLAI_H'),
            "Content-Type": "application/json"
        }
        
        response = requests.post("https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent" ,json=payload, headers=headers)
        
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]

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


    > Gunakan **tangan kanan**, posisikan di depan kamera untuk mengenali huruf A–Y.  
    Huruf **J** dan **Z** tidak didukung karena melibatkan gerakan dinamis.
    """)
    st.image("assets/img/alfabet.jpg", caption="Abjad dalam SIBI", use_container_width=True)
    st.info("👉 Pindah ke halaman **Deteksi SIBI** melalui sidebar untuk memulai pengenalan huruf.")

# === Halaman: Deteksi SIBI ===
elif halaman.startswith("📷"):

    st.title("🤖 Deteksi Huruf SIBI Real-Time")
    raw_letters = ""
    # Load model
    model = joblib.load("sibi_rf_model.pkl")

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
                                self.raw_letters = self.kata
                                print(f"DEBUG::: {self.kata}")
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

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # Initialize post-processing flag
        if "processed_on_stop" not in st.session_state:
            st.session_state.processed_on_stop = False
        
        # SETUP WEBRTC
        RTC_CONFIGURATION = RTCConfiguration({
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        })

        print("log sess_k:", sess_k)
        
        webrtc_ctx = webrtc_streamer(
            key=sess_k,
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=SignPredictor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
            # rtc_configuration=RTC_CONFIGURATION
        )

    # Detect STOP: PLAYING → READY
    if not webrtc_ctx.state.playing and not st.session_state.processed_on_stop:
        with open("./log_raw.txt", "r") as f:
            raw_text = f.read().strip()
        st.write(f"raw text: {raw_text}")

        # RESET
        with open("./log_raw.txt", "w") as f:
            f.write("")

        # Your custom logic here
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


    # Reset when stream starts again
    if webrtc_ctx.state.playing:
        st.session_state.processed_on_stop = False

    st.markdown("---")
    st.info("💡 Tips: Letakkan tangan kanan di tengah kamera dan tahan selama 1–2 detik. Untuk menghapus kata yang sudah terbentuk, silakan hentikan dan jalankan kembali kamera.")
    st.caption("Deteksi dan Perbaikan Kalimat Bahasa Isyarat")
