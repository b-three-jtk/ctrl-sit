import streamlit as st
import base64
import time

def play_audio(file_path: str):
    with open(file_path, 'rb') as f:
        audio_bytes = f.read()
        encoded = base64.b64encode(audio_bytes).decode()
        audio_html = f"""
            <audio autoplay>
                <source src="data:audio/mp3;base64,{encoded}" type="audio/mp3">
            </audio>
        """
        st.markdown(audio_html, unsafe_allow_html=True)

def handle_posture_audio(is_bad_posture: bool):
    if 'last_posture' not in st.session_state:
        st.session_state.last_posture = None

    if is_bad_posture and st.session_state.last_posture != 'bad':
        play_audio('assets/bad_posture.mp3')
        st.session_state.last_posture = 'bad'
        st.warning("⚠️ Postur kamu sedang tidak baik, perbaiki posisi dudukmu.")
        
    elif not is_bad_posture and st.session_state.last_posture != 'good':
        play_audio('assets/good_posture.mp3')
        st.session_state.last_posture = 'good'
        st.success("✅ Postur kamu sudah baik. Pertahankan ya!")