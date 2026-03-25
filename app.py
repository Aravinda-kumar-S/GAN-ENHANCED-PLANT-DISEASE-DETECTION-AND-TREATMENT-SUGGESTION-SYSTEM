import streamlit as st
import os
import time
import numpy as np
import tensorflow as tf
import pandas as pd
import plotly.express as px
from PIL import Image
import json
import base64
from utils import (get_ai_explanation, get_ai_chat_response, get_weather_data, 
                   interpret_weather_for_spraying, identify_with_gemini, 
                   get_recommendations, get_nearby_vendors, get_user_location,
                   get_gov_links, get_state_from_coords, get_climate_zone, speak_text)
from explainability import make_gradcam_heatmap, save_and_display_gradcam, get_last_conv_layer_name
from translations import TRANSLATIONS
from streamlit_js_eval import get_geolocation
from streamlit_mic_recorder import mic_recorder
import google.generativeai as genai

# --- GLOBAL SETTINGS ---
MODEL_PATH = 'best_plant_disease_model.keras'
CLASS_NAMES_PATH = 'class_names.json'

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="PlantVision AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SESSION STATE INIT ---
if 'lang' not in st.session_state:
    st.session_state.lang = "English"
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'user_loc' not in st.session_state:
    st.session_state.user_loc = None

T = TRANSLATIONS[st.session_state.lang]

# --- VISUAL STYLING (CSS) ---
def load_css():
    st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
        
        html, body, [class*="css"] {{
            font-family: 'Outfit', sans-serif;
            background-color: #F8FAF8;
        }}

        .main-header {{
            background: linear-gradient(90deg, #1B5E20 0%, #43A047 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            font-size: 3.5rem;
            margin-bottom: 10px;
        }}

        .stCard {{
            background-color: #FFFFFF;
            padding: 24px;
            border-radius: 16px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.05);
            border: 1px solid #E8F5E9;
            margin-bottom: 20px;
            transition: transform 0.3s ease;
        }}
        .stCard:hover {{
            transform: translateY(-5px);
        }}
        
        div[data-testid="stMetricValue"] {{
            font-size: 28px;
            color: #2E7D32;
            font-weight: 700;
        }}

        section[data-testid="stSidebar"] {{
            background-color: #FFFFFF;
            border-right: 1px solid #E8F5E9;
        }}
        
        .stButton > button {{
            background: linear-gradient(90deg, #2E7D32 0%, #43A047 100%);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 14px 28px;
            font-weight: 600;
            width: 100%;
            transition: all 0.3s ease;
        }}
        .stButton > button:hover {{
            box-shadow: 0 8px 20px rgba(46, 125, 50, 0.3);
            opacity: 0.9;
        }}

        .lang-selector {{
            position: absolute;
            top: 10px;
            right: 10px;
        }}
        </style>
    """, unsafe_allow_html=True)

load_css()

# --- GEOLOCATION ---
def fetch_precise_location():
    if not st.session_state.user_loc:
        with st.sidebar:
            st.write("📍 Detecting Location...")
            loc = get_geolocation()
            if loc:
                st.session_state.user_loc = {
                    "lat": loc['coords']['latitude'],
                    "lon": loc['coords']['longitude'],
                    "state": get_state_from_coords(loc['coords']['latitude'], loc['coords']['longitude']),
                    "zone": get_climate_zone(loc['coords']['latitude'], loc['coords']['longitude'])
                }
                st.success(f"Location Verified: {st.session_state.user_loc['state']}")

fetch_precise_location()

# --- MODEL LOADING ---
@st.cache_resource
def load_model_resources():
    model = None
    class_names = []
    if os.path.exists(MODEL_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
        except: pass
    if os.path.exists(CLASS_NAMES_PATH):
        with open(CLASS_NAMES_PATH, 'r') as f:
            class_names = json.load(f)
    else:
        class_names = ["Healthy", "Early Blight", "Late Blight", "Rust", "Powdery Mildew"]
    return model, class_names

model, class_names = load_model_resources()

# --- SIDEBAR NAVIGATION ---
st.sidebar.markdown(f"## 🌱 {T['title']}")
selected_lang = st.sidebar.selectbox(T['select_lang'], list(TRANSLATIONS.keys()), index=list(TRANSLATIONS.keys()).index(st.session_state.lang))
if selected_lang != st.session_state.lang:
    st.session_state.lang = selected_lang
    st.rerun()

st.sidebar.markdown("---")
nav_options = [T['home'], T['disease_diagnosis'], T['ai_chat_adviser'], T['gov_schemes'], T['nearby_vendors'], T['weather_forecast'], T['market_prices']]
selected_page = st.sidebar.radio("Navigation", nav_options)

# --- PAGE FUNCTIONS ---

def home_page():
    st.markdown(f"<h1 class='main-header'>{T['welcome']}</h1>", unsafe_allow_html=True)
    st.markdown(f"### {T['tagline']}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"<div class='stCard'><h3>🌡️ {T['weather_forecast']}</h3><p>Get real-time updates and spray windows.</p></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='stCard'><h3>🌾 {T['market_prices']}</h3><p>Track latest crop prices in your region.</p></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='stCard'><h3>📜 {T['gov_schemes']}</h3><p>Access location-filtered subsidies.</p></div>", unsafe_allow_html=True)

    st.image("https://images.unsplash.com/photo-1592982537447-7440770cbfc9?auto=format&fit=crop&q=80&w=1000", use_column_width=True)

def disease_diagnosis_page():
    st.markdown(f"<h1 class='main-header'>{T['disease_diagnosis']}</h1>", unsafe_allow_html=True)
    st.write(T['upload'])
    
    col1, col2 = st.columns([1, 1])
    with col1:
        input_method = st.radio(T['input_method'], [T['upload_img'], T['live_cam']], horizontal=True)
        img_file = st.file_uploader("Choose Image", type=['jpg', 'png']) if input_method == T['upload_img'] else st.camera_input("Take Photo")
        
        if img_file:
            image = Image.open(img_file)
            st.image(image, use_column_width=True)
            if st.button(T['analyze']):
                with st.spinner("🤖 AI Expert is analyzing symptoms..."):
                    plant, disease, conf, _ = identify_with_gemini(image)
                    
                    # Get environmental context for the AI
                    lat, lon, _ = get_user_location()
                    weather = get_weather_data(lat, lon)
                    
                    # Generate the Structured Agricultural Diagnosis (as requested)
                    structured_report = get_ai_explanation(image, disease, conf, weather)
                    
                    st.session_state.last_result = {
                        "disease": disease, 
                        "conf": conf, 
                        "report": structured_report, 
                        "image": image
                    }
                    st.rerun()

    with col2:
        if 'last_result' in st.session_state:
            res = st.session_state.last_result
            st.markdown(f"<div class='stCard'><h3>Results: {res['disease']}</h3><p>Confidence: {res['conf']*100:.1f}%</p></div>", unsafe_allow_html=True)
            st.progress(res['conf'])
            
            st.markdown("### 📋 AI Agricultural Diagnosis")
            st.info(res['report'])
            
            recs = get_recommendations(res['disease'])
            with st.expander("🛒 Marketplace - Direct Purchase Links", expanded=False):
                # Organic Section
                st.markdown("#### 🌿 Organic Solutions")
                for o in recs.get('Organic', []):
                    st.write(f"**{o['name']}** ({o['price']}) - {o['dosage']}")
                    cols = st.columns(len(o['marketplaces']))
                    for idx, mk in enumerate(o['marketplaces']):
                        cols[idx].link_button(f"🛒 Buy on {mk['site']}", mk['link'])
                
                # Chemical Section
                st.markdown("#### 🧪 Chemical Solutions")
                for c in recs.get('Chemical', []):
                    st.write(f"**{c['name']}** ({c['price']}) - {c['dosage']}")
                    cols = st.columns(len(c['marketplaces']))
                    for idx, mk in enumerate(c['marketplaces']):
                        cols[idx].link_button(f"🛒 Buy on {mk['site']}", mk['link'])

                # Seeds Section
                if 'Seeds' in recs:
                    st.markdown("#### 🌾 Resistant Seeds")
                    for s in recs['Seeds']:
                        st.write(f"**{s['name']}** ({s['price']})")
                        cols = st.columns(len(s['marketplaces']))
                        for idx, mk in enumerate(s['marketplaces']):
                            cols[idx].link_button(f"🛒 Buy on {mk['site']}", mk['link'])
                
                st.markdown("---")
                st.info(f"**Safety**: {recs['Safety']}")
                
                # Geofenced Local Vendor Link
                st.markdown("#### 📍 Local Availability")
                st.write("Prefer to buy locally?")
                if st.button("🏪 Find at Nearest Agri-Store"):
                    top_v = get_nearby_vendors(st.session_state.user_loc.get('lat', 20.5), st.session_state.user_loc.get('lon', 78.5))[0]
                    st.success(f"Recommended Store: **{top_v['name']}** ({top_v['dist']} km away)")
                    st.link_button(f"Visit Website", top_v['link'])

def chat_adviser_page():
    st.markdown(f"<h1 class='main-header'>{T['ai_chat_adviser']}</h1>", unsafe_allow_html=True)
    
    # Voice Input Section
    st.markdown("### 🎙️ Voice Interaction")
    col_v1, col_v2 = st.columns([1, 3])
    with col_v1:
        audio = mic_recorder(start_prompt="🎙️ Start Recording", stop_prompt="🛑 Stop Speaking")
    
    if audio:
        with st.spinner("Processing Voice..."):
            # Display Audio
            st.audio(audio['bytes'])
            # Send to Gemini
            response = get_ai_chat_response(st.session_state.messages + [{"role": "user", "content": "Voice Input"}], 
                                            audio_bytes=audio['bytes'])
            
            st.session_state.messages.append({"role": "user", "content": "🎤 *Voice Message Submitted*"})
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Auto-speak the response
            audio_b64 = speak_text(response[:300], lang='en')
            if audio_b64:
                st.markdown(f'<audio autoplay src="data:audio/mp3;base64,{audio_b64}">', unsafe_allow_html=True)
            st.rerun()

    st.markdown("---")
    # Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about your crops..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        
        with st.chat_message("assistant"):
            response = get_ai_chat_response(st.session_state.messages)
            st.markdown(response)
            
            # TTS
            audio_b64 = speak_text(response[:300], lang='en')
            if audio_b64:
                st.markdown(f'<audio autoplay src="data:audio/mp3;base64,{audio_b64}">', unsafe_allow_html=True)
            
        st.session_state.messages.append({"role": "assistant", "content": response})

def gov_schemes_page():
    st.markdown(f"<h1 class='main-header'>{T['gov_schemes']}</h1>", unsafe_allow_html=True)
    state = st.session_state.user_loc['state'] if st.session_state.user_loc else "Default"
    st.info(f"Showing schemes for: **{state}**")
    
    links = get_gov_links(state)
    for link in links:
        st.markdown(f"""
        <div class='stCard'>
            <h4>{link['name']}</h4>
            <a href="{link['link']}" target="_blank">View Details →</a>
        </div>
        """, unsafe_allow_html=True)

def nearby_vendors_page():
    st.markdown(f"<h1 class='main-header'>{T['nearby_vendors']}</h1>", unsafe_allow_html=True)
    lat = st.session_state.user_loc['lat'] if st.session_state.user_loc else 20.59
    lon = st.session_state.user_loc['lon'] if st.session_state.user_loc else 78.96
    
    vendors = get_nearby_vendors(lat, lon)
    v_df = pd.DataFrame(vendors)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        for v in vendors:
            with st.container():
                st.markdown(f"""
                <div class='stCard' style='padding: 15px; margin-bottom: 10px;'>
                    <h4 style='margin:0;'>{v['name']}</h4>
                    <p style='margin:5px 0;'>📍 {v['dist']} km away</p>
                    <p style='margin:5px 0;'>📞 {v['phone']}</p>
                </div>
                """, unsafe_allow_html=True)
                if v['link'] != "#":
                    st.link_button("Visit Website", v['link'])
    with col2:
        fig = px.scatter_mapbox(v_df, lat="lat", lon="lon", hover_name="name", zoom=12, height=400)
        fig.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig, use_column_width=True)

def weather_forecast_page():
    st.markdown(f"<h1 class='main-header'>{T['weather_forecast']}</h1>", unsafe_allow_html=True)
    lat = st.session_state.user_loc['lat'] if st.session_state.user_loc else 20.59
    lon = st.session_state.user_loc['lon'] if st.session_state.user_loc else 78.96
    
    weather = get_weather_data(lat, lon)
    if "error" not in weather:
        st.metric("Temperature", f"{weather['temp']}°C")
        st.metric("Humidity", f"{weather['humidity']}%")
        st.info(f"Condition: {weather['description'].title()}")
        advice = interpret_weather_for_spraying(weather)
        st.warning(advice)
    else:
        st.error("Weather API key not configured.")

def market_prices_page():
    st.markdown(f"<h1 class='main-header'>{T['market_prices']}</h1>", unsafe_allow_html=True)
    st.write("Tracking latest Mandi prices...")
    # Add a mock table for prices
    data = {
        "Crop": ["Tomato", "Potato", "Wheat", "Rice", "Onion"],
        "Price (per Quintal)": ["₹2,500", "₹1,800", "₹2,200", "₹3,100", "₹1,200"],
        "Trend": ["↑ 5%", "↓ 2%", "→ 0%", "↑ 8%", "↓ 10%"]
    }
    st.table(pd.DataFrame(data))

# --- ROUTING ---
if selected_page == T['home']: home_page()
elif selected_page == T['disease_diagnosis']: disease_diagnosis_page()
elif selected_page == T['ai_chat_adviser']: chat_adviser_page()
elif selected_page == T['gov_schemes']: gov_schemes_page()
elif selected_page == T['nearby_vendors']: nearby_vendors_page()
elif selected_page == T['weather_forecast']: weather_forecast_page()
elif selected_page == T['market_prices']: market_prices_page()
