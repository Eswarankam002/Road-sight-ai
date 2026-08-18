import os
import tempfile
import math
from pathlib import Path

import base64
import io
import cv2
import numpy as np
import requests
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from ultralytics import YOLO
from streamlit_javascript import st_javascript


MODEL_PATH = "best1.pt"
UPLOADS_DIR = Path("uploads")
UPLOADS_DIR.mkdir(exist_ok=True)
FRAME_STRIDE = 4
VIDEO_IMGSZ = 320


st.set_page_config(
    page_title="ROAD SIGHT AI",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# Initialize dark mode state
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

dark_mode = st.session_state.dark_mode

# Light mode styles
if not dark_mode:
    base_style = """
    <style>
        .stApp {
            background:
                radial-gradient(circle at top left, #dbeafe 0%, transparent 35%),
                linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%);
            color: #0f172a;
        }

        .block-container {
            max-width: 1200px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .hero-section {
            text-align: center;
            padding: 32px 20px 20px;
            background: linear-gradient(180deg, rgba(0,100,255,0.04) 0%, transparent 100%);
            border-radius: 20px;
            margin-bottom: 10px;
        }

        .hero-icon {
            margin-bottom: 12px;
        }

        .upload-heading {
            font-size: 22px;
            font-weight: 700;
            color: #172554;
            margin-bottom: 6px;
        }

        .upload-description {
            color: #64748b;
            font-size: 15px;
            margin-bottom: 16px;
        }

        .info-card {
            background: rgba(255, 255, 255, 0.75);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            padding: 18px;
            border-radius: 16px;
            min-height: 150px;
            border: 1px solid rgba(0, 212, 255, 0.15);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.06);
            margin-bottom: 10px;
        }

        .info-icon {
            font-size: 28px;
            margin-bottom: 6px;
        }

        .info-title {
            font-size: 18px;
            font-weight: 700;
            color: #0066cc;
            margin-bottom: 6px;
        }

        .info-text {
            color: #64748b;
            font-size: 14px;
            line-height: 1.6;
        }

        .section-title {
            font-size: 24px;
            font-weight: 750;
            color: #0f172a;
            margin-top: 20px;
            margin-bottom: 10px;
        }

        .result-card {
            background: rgba(255, 255, 255, 0.8);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            padding: 16px 18px;
            border-radius: 14px;
            border-left: 5px solid #00d4ff;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.06);
            margin-bottom: 12px;
            color: #0f172a;
            font-weight: 600;
            line-height: 1.4;
        }

        .result-card h3 {
            font-size: 18px;
            margin: 0 0 8px 0;
            color: #0f172a;
            font-weight: 800;
        }

        .result-card p,
        .result-card strong {
            color: #0f172a;
            font-weight: 700;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 999px;
            padding: 0.6rem 1rem;
            background: rgba(255, 255, 255, 0.85);
        }

        [data-testid="stFileUploader"] {
            background: rgba(255, 255, 255, 0.8);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            padding: 22px;
            border: 2px dashed #00b4d8;
            border-radius: 18px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.06);
        }

        .stButton > button {
            width: 100%;
            height: 46px;
            border: none;
            border-radius: 12px;
            background: linear-gradient(135deg, #0077cc, #00b4d8);
            color: white;
            font-weight: 700;
            box-shadow: 0 4px 15px rgba(0, 180, 216, 0.3);
        }

        .stButton > button:hover {
            box-shadow: 0 6px 20px rgba(0, 180, 216, 0.5);
        }

        div[data-testid="stCheckbox"] {
            display: none !important;
        }

        .theme-toggle {
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 999;
            background: rgba(255, 255, 255, 0.85);
            backdrop-filter: blur(10px);
            border-radius: 50%;
            width: 50px;
            height: 50px;
            border: 1px solid rgba(0, 212, 255, 0.2);
            cursor: pointer;
            font-size: 24px;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
            transition: all 0.3s ease;
        }

        .theme-toggle:hover {
            transform: scale(1.1);
            box-shadow: 0 15px 40px rgba(0, 180, 216, 0.2);
        }

        #MainMenu, footer, header {
            visibility: hidden;
        }
    </style>
    """
else:
    # Dark mode styles
    base_style = """
    <style>
        .stApp {
            background: #060b18;
            color: #e2e8f0;
        }

        .block-container {
            max-width: 1200px;
            padding-top: 2rem;
            padding-bottom: 3rem;
            background: #0a1020 !important;
            color: #e2e8f0 !important;
        }

        .hero-section {
            text-align: center;
            padding: 32px 20px 20px;
            background: linear-gradient(180deg, rgba(0,100,255,0.08) 0%, transparent 100%);
            border-radius: 20px;
            border: 1px solid rgba(0, 212, 255, 0.08);
        }

        .hero-icon {
            margin-bottom: 12px;
        }

        .upload-heading {
            font-size: 22px;
            font-weight: 700;
            color: #e2e8f0;
            margin-bottom: 6px;
        }

        .upload-description {
            color: #94a3b8;
            font-size: 15px;
            margin-bottom: 16px;
        }

        .info-card {
            background: rgba(10, 20, 40, 0.8);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            padding: 18px;
            border-radius: 16px;
            min-height: 150px;
            border: 1px solid rgba(0, 212, 255, 0.12);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.4);
            margin-bottom: 10px;
            color: #e2e8f0;
        }

        .info-icon {
            font-size: 28px;
            margin-bottom: 6px;
        }

        .info-title {
            font-size: 18px;
            font-weight: 700;
            color: #00d4ff;
            margin-bottom: 6px;
        }

        .info-text {
            color: #94a3b8;
            font-size: 14px;
            line-height: 1.6;
        }

        .section-title {
            font-size: 24px;
            font-weight: 750;
            color: #e2e8f0;
            margin-top: 20px;
            margin-bottom: 10px;
        }

        .result-card {
            background: rgba(10, 20, 40, 0.8);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            padding: 16px 18px;
            border-radius: 14px;
            border-left: 5px solid #00d4ff;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.4);
            margin-bottom: 12px;
            color: #e2e8f0;
            font-weight: 600;
            line-height: 1.4;
        }

        .result-card h3 {
            font-size: 18px;
            margin: 0 0 8px 0;
            color: #e2e8f0;
            font-weight: 800;
        }

        .result-card p,
        .result-card strong {
            color: #e2e8f0;
            font-weight: 700;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 999px;
            padding: 0.6rem 1rem;
            background: rgba(10, 20, 40, 0.8) !important;
            border-color: rgba(0, 212, 255, 0.15) !important;
            color: #e2e8f0 !important;
        }

        [data-testid="stFileUploader"] {
            background: rgba(10, 20, 40, 0.8) !important;
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            padding: 22px;
            border: 2px dashed rgba(0, 212, 255, 0.4);
            border-radius: 18px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
            color: #e2e8f0;
        }

        .stButton > button {
            width: 100%;
            height: 46px;
            border: none;
            border-radius: 12px;
            background: linear-gradient(135deg, #0066cc, #00b4d8);
            color: white;
            font-weight: 700;
            box-shadow: 0 4px 15px rgba(0, 180, 216, 0.25);
        }

        .stButton > button:hover {
            box-shadow: 0 6px 20px rgba(0, 180, 216, 0.4);
        }

        div[data-testid="stCheckbox"] {
            display: none !important;
        }

        .theme-toggle {
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 999;
            background: rgba(10, 20, 40, 0.85);
            backdrop-filter: blur(12px);
            border-radius: 50%;
            width: 50px;
            height: 50px;
            border: 1px solid rgba(0, 212, 255, 0.2);
            cursor: pointer;
            font-size: 24px;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
            transition: all 0.3s ease;
            color: #00d4ff;
        }

        .theme-toggle:hover {
            transform: scale(1.1);
            box-shadow: 0 15px 40px rgba(0, 180, 216, 0.2);
            background: rgba(15, 30, 60, 0.95) !important;
        }

        #MainMenu, footer, header {
            visibility: hidden;
        }
    </style>
    """

st.markdown(base_style, unsafe_allow_html=True)

# Add animated icons CSS and SVG styles
animated_icons_style = """
<style>
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-8px); }
    }
    
    .icon-road {
        display: inline-block;
        width: 56px;
        height: 56px;
        animation: float 3s ease-in-out infinite;
    }
    
    .icon-camera {
        display: inline-block;
        width: 28px;
        height: 28px;
        animation: pulse 2s ease-in-out infinite;
        color: #3b82f6;
    }
    
    .icon-video {
        display: inline-block;
        width: 28px;
        height: 28px;
        animation: pulse 2s ease-in-out infinite 0.3s;
        color: #8b5cf6;
    }
    
    .icon-check {
        display: inline-block;
        width: 28px;
        height: 28px;
        animation: bounce 2s ease-in-out infinite;
        color: #10b981;
    }
    
    .icon-location {
        display: inline-block;
        width: 20px;
        height: 20px;
        animation: float 2.5s ease-in-out infinite;
        color: #ef4444;
    }
    
    .icon-moon {
        display: inline-block;
        animation: float 2s ease-in-out infinite;
    }
    
    .icon-sun {
        display: inline-block;
        animation: spin 20s linear infinite;
    }
    
    .theme-toggle {
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 999;
        background: rgba(255, 255, 255, 0.95);
        border-radius: 50%;
        width: 50px;
        height: 50px;
        border: none;
        cursor: pointer;
        font-size: 24px;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
        transition: all 0.3s ease;
    }

    .theme-toggle:hover {
        transform: scale(1.1);
        box-shadow: 0 15px 40px rgba(15, 23, 42, 0.15);
    }
    
    /* Dark mode icon colors */
    @media (prefers-color-scheme: dark) {
        .icon-camera {
            color: #60a5fa;
        }
        .icon-video {
            color: #a78bfa;
        }
        .icon-check {
            color: #34d399;
        }
        .icon-location {
            color: #f87171;
        }
    }
</style>
"""
st.markdown(animated_icons_style, unsafe_allow_html=True)

# Add custom positioning for theme toggle
custom_button_style = """
<style>
    [data-testid="stHorizontalBlock"] {
        position: relative;
    }
</style>
"""
st.markdown(custom_button_style, unsafe_allow_html=True)

# Create theme toggle at top right - using columns for positioning
_, _, theme_col = st.columns([0.7, 0.2, 0.1])

# Moon icon SVG with unique creative design
moon_icon_svg = """
<svg width="28" height="28" viewBox="0 0 28 28" xmlns="http://www.w3.org/2000/svg" style="display: inline-block;">
    <defs>
        <style>
            .moon-body { animation: moonFloat 3s ease-in-out infinite; }
            .moon-glow { animation: moonGlow 2s ease-in-out infinite; }
            @keyframes moonFloat {
                0%, 100% { transform: translateY(0px); }
                50% { transform: translateY(-3px); }
            }
            @keyframes moonGlow {
                0%, 100% { opacity: 0.6; }
                50% { opacity: 1; }
            }
        </style>
    </defs>
    <!-- Glow effect -->
    <circle class="moon-glow" cx="14" cy="14" r="11" fill="none" stroke="#FBBF24" stroke-width="1.5" opacity="0.6"/>
    
    <!-- Moon body -->
    <circle class="moon-body" cx="14" cy="14" r="9" fill="#FCD34D"/>
    
    <!-- Moon craters for realism -->
    <circle cx="10" cy="11" r="1.5" fill="#F59E0B" opacity="0.7"/>
    <circle cx="17" cy="16" r="1" fill="#F59E0B" opacity="0.7"/>
    <circle cx="12" cy="18" r="1.2" fill="#F59E0B" opacity="0.7"/>
    
    <!-- Shadow effect -->
    <ellipse cx="11" cy="12" rx="3" ry="4" fill="#1F2937" opacity="0.3"/>
</svg>
"""

# Sun icon SVG with unique creative design
sun_icon_svg = """
<svg width="28" height="28" viewBox="0 0 28 28" xmlns="http://www.w3.org/2000/svg" style="display: inline-block;">
    <defs>
        <style>
            .sun-core { animation: sunPulse 2s ease-in-out infinite; }
            .sun-rays { animation: sunSpin 8s linear infinite; }
            @keyframes sunPulse {
                0%, 100% { r: 6; }
                50% { r: 7; }
            }
            @keyframes sunSpin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        </style>
    </defs>
    <!-- Sun rays -->
    <g class="sun-rays" transform-origin="14 14">
        <rect x="13" y="2" width="2" height="5" fill="#FCD34D"/>
        <rect x="13" y="21" width="2" height="5" fill="#FCD34D"/>
        <rect x="2" y="13" width="5" height="2" fill="#FCD34D"/>
        <rect x="21" y="13" width="5" height="2" fill="#FCD34D"/>
        
        <!-- Diagonal rays -->
        <line x1="20" y1="8" x2="23" y2="5" stroke="#FCD34D" stroke-width="2" stroke-linecap="round"/>
        <line x1="5" y1="23" x2="8" y2="20" stroke="#FCD34D" stroke-width="2" stroke-linecap="round"/>
        <line x1="8" y1="8" x2="5" y2="5" stroke="#FCD34D" stroke-width="2" stroke-linecap="round"/>
        <line x1="23" y1="23" x2="20" y2="20" stroke="#FCD34D" stroke-width="2" stroke-linecap="round"/>
    </g>
    
    <!-- Sun core -->
    <circle class="sun-core" cx="14" cy="14" r="6" fill="#FDE047"/>
    
    <!-- Inner bright spot -->
    <circle cx="12" cy="12" r="2" fill="#FEFCE8" opacity="0.8"/>
</svg>
"""

# Moon and Sun SVG icons
moon_icon = f"""<span class="icon-moon" style="display: inline-block; cursor: pointer;">{moon_icon_svg}</span>"""
sun_icon = f"""<span class="icon-sun" style="display: inline-block; cursor: pointer;">{sun_icon_svg}</span>"""

with theme_col:
    if st.button("🌙" if not dark_mode else "☀️", key="theme_toggle_btn", help="Toggle dark/light mode"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file '{MODEL_PATH}' was not found.")
    return YOLO(MODEL_PATH)


def run_detection(image: Image.Image):
    model = load_model()
    image_array = np.array(image)
    results = model(image_array, conf=0.25, stream=False)
    result_image = results[0].plot()
    result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
    total_detections = len(results[0].boxes)

    detections = []
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        damage_name = results[0].names[class_id]
        detections.append({
            "name": damage_name,
            "confidence": confidence,
        })

    return result_image, total_detections, detections


def get_route_distance(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def show_route_map(start_loc, end_loc, waypoints):
    if not start_loc or not end_loc:
        return

    slat, slon = start_loc["latitude"], start_loc["longitude"]
    elat, elon = end_loc["latitude"], end_loc["longitude"]

    all_pts = [f"{slat},{slon}"]
    for wp in waypoints:
        all_pts.append(f"{wp['latitude']},{wp['longitude']}")
    all_pts.append(f"{elat},{elon}")

    route_coords_js = ",".join(["[" + p + "]" for p in all_pts])
    min_lat = min(slat, elat, *[w["latitude"] for w in waypoints]) - 0.005
    max_lat = max(slat, elat, *[w["latitude"] for w in waypoints]) + 0.005
    min_lon = min(slon, elon, *[w["longitude"] for w in waypoints]) - 0.005
    max_lon = max(slon, elon, *[w["longitude"] for w in waypoints]) + 0.005

    start_name = get_place_name(slat, slon)
    end_name = get_place_name(elat, elon)

    route_distance = get_route_distance(slat, slon, elat, elon)

    st.markdown(f"""
    <div class="result-card">
        <h3>Route Summary</h3>
        <p><strong>Start:</strong> {start_name}<br>
        <strong>End:</strong> {end_name}<br>
        <strong>Straight-line distance:</strong> {route_distance:.0f} m ({route_distance/1000:.2f} km)<br>
        <strong>GPS waypoints recorded:</strong> {len(waypoints) + 2}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="result-card">
        <h3>Start Location</h3>
        <p><strong>Place:</strong> {start_name}<br>
        <strong>Coordinates:</strong> {slat:.6f}, {slon:.6f}</p>
    </div>
    <div class="result-card">
        <h3>End Location</h3>
        <p><strong>Place:</strong> {end_name}<br>
        <strong>Coordinates:</strong> {elat:.6f}, {elon:.6f}</p>
    </div>
    """, unsafe_allow_html=True)

    center_lat = (slat + elat) / 2
    center_lon = (slon + elon) / 2

    leaflet_html = """
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <div id="route-map" style="width:100%;height:100%;border-radius:12px;"></div>
    <script>
    var map = L.map('route-map').setView([CENTER_LAT, CENTER_LON], 14);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: 'Road Sight AI',
        maxZoom: 19
    }).addTo(map);
    var startIcon = L.divIcon({html:'<div style="background:#22c55e;width:14px;height:14px;border-radius:50%;border:2px solid white;box-shadow:0 0 6px rgba(34,197,94,0.6);"></div>', className:'', iconSize:[14,14], iconAnchor:[7,7]});
    var endIcon = L.divIcon({html:'<div style="background:#ef4444;width:14px;height:14px;border-radius:50%;border:2px solid white;box-shadow:0 0 6px rgba(239,68,68,0.6);"></div>', className:'', iconSize:[14,14], iconAnchor:[7,7]});
    L.marker([SLAT, SLON], {icon:startIcon}).addTo(map).bindPopup('<b>Start</b><br>START_NAME');
    L.marker([ELAT, ELON], {icon:endIcon}).addTo(map).bindPopup('<b>End</b><br>END_NAME');
    var routeCoords = [ROUTE_COORDS];
    L.polyline(routeCoords, {color:'#00d4ff', weight:4, opacity:0.85, dashArray:'8,6'}).addTo(map);
    routeCoords.forEach(function(pt, i){
        if(i>0 && i<routeCoords.length-1 && i%3===0){
            L.circleMarker(pt, {radius:3, fillColor:'#00d4ff', color:'white', weight:1, fillOpacity:0.7}).addTo(map);
        }
    });
    map.fitBounds([[MIN_LAT, MIN_LON],[MAX_LAT, MAX_LON]]);
    </script>
    """.replace("CENTER_LAT", str(center_lat)) \
      .replace("CENTER_LON", str(center_lon)) \
      .replace("SLAT", str(slat)) \
      .replace("SLON", str(slon)) \
      .replace("ELAT", str(elat)) \
      .replace("ELON", str(elon)) \
      .replace("START_NAME", start_name) \
      .replace("END_NAME", end_name) \
      .replace("ROUTE_COORDS", route_coords_js) \
      .replace("MIN_LAT", str(min_lat)) \
      .replace("MIN_LON", str(min_lon)) \
      .replace("MAX_LAT", str(max_lat)) \
      .replace("MAX_LON", str(max_lon))

    components.html(leaflet_html, height=420)


def get_place_name(latitude, longitude):
    try:
        response = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={
                "lat": latitude,
                "lon": longitude,
                "format": "json",
                "addressdetails": 1,
            },
            timeout=10,
            headers={"User-Agent": "RoadSightAI/1.0"},
        )
        response.raise_for_status()
        data = response.json()
        address = data.get("address", {})
        parts = []
        for key in ["village", "town", "city", "municipality", "county", "state", "country"]:
            value = address.get(key)
            if value:
                parts.append(str(value))
        if parts:
            return ", ".join(parts[:5])
        return data.get("display_name", "Unknown location")
    except Exception:
        return "Location name unavailable"


def show_location_details(location):
    if not location:
        return

    latitude = location.get("latitude")
    longitude = location.get("longitude")
    if latitude is None or longitude is None:
        return

    place_name = get_place_name(latitude, longitude)
    location["place_name"] = place_name
    st.markdown(
        f"<div class=\"result-card\">\n"
        f"    <h3>Detection Location</h3>\n"
        f"    <p><strong>Place:</strong> {place_name}<br>\n"
        f"    <strong>Latitude:</strong> {latitude:.4f}<br>\n"
        f"    <strong>Longitude:</strong> {longitude:.4f}</p>\n"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.caption(f"Coordinates: {latitude:.4f}, {longitude:.4f}")
    map_embed = (
        f"https://www.openstreetmap.org/export/embed.html?"
        f"bbox={longitude-0.01:.6f}%2C{latitude-0.01:.6f}%2C{longitude+0.01:.6f}%2C{latitude+0.01:.6f}"
        f"&layer=mapnik&marker={latitude:.6f},{longitude:.6f}"
    )
    components.iframe(map_embed, height=320)


def show_location_map(location):
    if not location:
        return

    latitude = location.get("latitude")
    longitude = location.get("longitude")
    if latitude is None or longitude is None:
        return

    place_name = get_place_name(latitude, longitude)
    location["place_name"] = place_name
    st.markdown(
        f"📍 Current location: {place_name}"
    )
    st.caption(f"Coordinates: {latitude:.4f}, {longitude:.4f}")
    map_embed = (
        f"https://www.openstreetmap.org/export/embed.html?"
        f"bbox={longitude-0.01:.6f}%2C{latitude-0.01:.6f}%2C{longitude+0.01:.6f}%2C{latitude+0.01:.6f}"
        f"&layer=mapnik&marker={latitude:.6f},{longitude:.6f}"
    )
    components.iframe(map_embed, height=320)


def summarize_detections(detections):
    total_detections = len(detections)
    if total_detections == 0:
        return total_detections, []

    grouped = {}
    for detection in detections:
        name = detection["name"]
        if name not in grouped:
            grouped[name] = {"count": 0, "confidence_sum": 0.0}
        grouped[name]["count"] += 1
        grouped[name]["confidence_sum"] += detection["confidence"]

    damage_summary = []
    for name, values in grouped.items():
        count = values["count"]
        average_confidence = values["confidence_sum"] / count
        percentage = (count / total_detections) * 100
        damage_summary.append({
            "name": name,
            "count": count,
            "percentage": percentage,
            "average_confidence": average_confidence,
        })

    damage_summary.sort(key=lambda item: item["count"], reverse=True)
    return total_detections, damage_summary


def show_detection_results(result_image, total_detections, detections, title="Detection Result", location=None):
    st.success("Road Sight AI analysis completed successfully!")
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)

    if location:
        show_location_details(location)

    result_column, details_column = st.columns([1.5, 1], gap="large")

    with result_column:
        st.image(result_image, caption="Road Sight AI - Detected Damage", use_container_width=True)

    with details_column:
        st.markdown(
            f"""
            <div class="result-card">
                <h3>Detection Summary</h3>
                <p><strong>Total detections:</strong> {total_detections}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if total_detections == 0:
            st.warning("No trained road-damage object was detected in this image.")
        else:
            _, damage_summary = summarize_detections(detections)
            for summary in damage_summary:
                st.markdown(
                    f"""
                    <div class="result-card">
                        <strong>{summary['name']}</strong><br>
                        Count: {summary['count']}<br>
                        Share: {summary['percentage']:.1f}% of detected damages<br>
                        Avg confidence: {summary['average_confidence'] * 100:.2f}%
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            for index, detection in enumerate(detections, start=1):
                st.markdown(
                    f"""
                    <div class="result-card">
                        <strong>Detection {index}</strong><br>
                        Damage type: {detection['name']}<br>
                        Confidence: {detection['confidence'] * 100:.2f}%
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def process_video(uploaded_video):
    model = load_model()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(uploaded_video.getbuffer())
        input_path = temp_file.name

    output_path = UPLOADS_DIR / f"detected_{uploaded_video.name}"
    capture = cv2.VideoCapture(input_path)

    if not capture.isOpened():
        raise ValueError("The uploaded video could not be read.")

    fps = capture.get(cv2.CAP_PROP_FPS) or 24
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width <= 0 or height <= 0:
        raise ValueError("The uploaded video does not contain valid frame dimensions.")

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise ValueError("The output video writer could not be initialized.")

    frame_index = 0
    all_detections = []
    while True:
        success, frame = capture.read()
        if not success:
            break

        frame_index += 1
        if frame_index % FRAME_STRIDE != 0:
            writer.write(frame)
            continue

        results = model(frame, conf=0.25, stream=False, imgsz=VIDEO_IMGSZ)
        annotated_frame = results[0].plot()
        writer.write(annotated_frame)

        for box in results[0].boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            damage_name = results[0].names[class_id]
            all_detections.append({
                "name": damage_name,
                "confidence": confidence,
            })

    capture.release()
    writer.release()
    os.remove(input_path)

    return output_path, all_detections


heading_color = "#e2e8f0" if dark_mode else "#0f172a"
description_color = "#cbd5e1" if dark_mode else "#64748b"

# RoadSight AI professional logo
roadsight_logo_svg = """
<svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg" style="width:80px;height:80px;">
    <defs>
        <linearGradient id="logoBg" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" style="stop-color:#0a0e27;stop-opacity:1" />
            <stop offset="100%" style="stop-color:#111b3a;stop-opacity:1" />
        </linearGradient>
        <linearGradient id="roadG" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" style="stop-color:#00d4ff;stop-opacity:0.15" />
            <stop offset="100%" style="stop-color:#00d4ff;stop-opacity:0.02" />
        </linearGradient>
        <linearGradient id="scanLine" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" style="stop-color:#00d4ff;stop-opacity:0.9" />
            <stop offset="100%" style="stop-color:#0066ff;stop-opacity:0.3" />
        </linearGradient>
        <filter id="glow">
            <feGaussianBlur stdDeviation="2" result="blur"/>
            <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
    </defs>
    <!-- Outer ring -->
    <circle cx="60" cy="60" r="56" fill="none" stroke="#00d4ff" stroke-width="1.5" opacity="0.3"/>
    <!-- Background -->
    <circle cx="60" cy="60" r="52" fill="url(#logoBg)" stroke="#00d4ff" stroke-width="0.5" opacity="0.9"/>
    <!-- Road perspective -->
    <polygon points="30,85 90,85 72,35 48,35" fill="url(#roadG)" stroke="#00d4ff" stroke-width="0.5" opacity="0.4"/>
    <!-- Road center line (dashed) -->
    <line x1="60" y1="38" x2="60" y2="82" stroke="#00d4ff" stroke-width="1.5" stroke-dasharray="4,4" filter="url(#glow)" opacity="0.7"/>
    <!-- Scanning frame corners -->
    <path d="M38,40 L38,32 L46,32" fill="none" stroke="#00d4ff" stroke-width="2" filter="url(#glow)"/>
    <path d="M82,40 L82,32 L74,32" fill="none" stroke="#00d4ff" stroke-width="2" filter="url(#glow)"/>
    <path d="M38,78 L38,86 L46,86" fill="none" stroke="#00d4ff" stroke-width="2" filter="url(#glow)"/>
    <path d="M82,78 L82,86 L74,86" fill="none" stroke="#00d4ff" stroke-width="2" filter="url(#glow)"/>
    <!-- Scan line -->
    <line x1="40" y1="58" x2="80" y2="58" stroke="url(#scanLine)" stroke-width="1" filter="url(#glow)" opacity="0.8">
        <animate attributeName="y1" values="35;80;35" dur="3s" repeatCount="indefinite"/>
        <animate attributeName="y2" values="35;80;35" dur="3s" repeatCount="indefinite"/>
    </line>
    <!-- Damage detection dots -->
    <circle cx="50" cy="65" r="2.5" fill="#ff4444" filter="url(#glow)" opacity="0.9">
        <animate attributeName="opacity" values="0.4;1;0.4" dur="2s" repeatCount="indefinite"/>
    </circle>
    <circle cx="70" cy="72" r="2" fill="#ff4444" filter="url(#glow)" opacity="0.7">
        <animate attributeName="opacity" values="0.4;1;0.4" dur="2s" repeatCount="indefinite" begin="0.5s"/>
    </circle>
    <!-- Neural network nodes -->
    <circle cx="28" cy="50" r="1.5" fill="#00d4ff" filter="url(#glow)" opacity="0.5"/>
    <circle cx="25" cy="60" r="1.5" fill="#00d4ff" filter="url(#glow)" opacity="0.5"/>
    <circle cx="28" cy="70" r="1.5" fill="#00d4ff" filter="url(#glow)" opacity="0.5"/>
    <circle cx="92" cy="50" r="1.5" fill="#00d4ff" filter="url(#glow)" opacity="0.5"/>
    <circle cx="95" cy="60" r="1.5" fill="#00d4ff" filter="url(#glow)" opacity="0.5"/>
    <circle cx="92" cy="70" r="1.5" fill="#00d4ff" filter="url(#glow)" opacity="0.5"/>
    <!-- Connecting lines -->
    <line x1="28" y1="50" x2="38" y2="58" stroke="#00d4ff" stroke-width="0.5" opacity="0.3"/>
    <line x1="25" y1="60" x2="38" y2="60" stroke="#00d4ff" stroke-width="0.5" opacity="0.3"/>
    <line x1="28" y1="70" x2="38" y2="62" stroke="#00d4ff" stroke-width="0.5" opacity="0.3"/>
    <line x1="92" y1="50" x2="82" y2="58" stroke="#00d4ff" stroke-width="0.5" opacity="0.3"/>
    <line x1="95" y1="60" x2="82" y2="60" stroke="#00d4ff" stroke-width="0.5" opacity="0.3"/>
    <line x1="92" y1="70" x2="82" y2="62" stroke="#00d4ff" stroke-width="0.5" opacity="0.3"/>
    <!-- Camera/vision icon at top -->
    <rect x="54" y="24" width="12" height="8" rx="2" fill="none" stroke="#00d4ff" stroke-width="1" filter="url(#glow)" opacity="0.6"/>
    <circle cx="60" cy="28" r="2" fill="#00d4ff" opacity="0.8" filter="url(#glow)"/>
</svg>
"""

st.markdown(
    f"""
    <div class="hero-section">
        <div class="hero-icon">{roadsight_logo_svg}</div>
        <h1 style="margin:0; font-size:2.4rem; color:{heading_color}; font-weight:800; letter-spacing:3px; text-transform:uppercase;">Road Sight AI</h1>
        <p style="margin:0.2rem auto 0; max-width:760px; color:#00d4ff; font-size:0.85rem; letter-spacing:2px; text-transform:uppercase; font-weight:600;">
            AI-Powered Road Damage Detection
        </p>
        <p style="margin:0.4rem auto 0; max-width:760px; color:{description_color}; font-size:1rem;">
            Choose how you want to inspect the road: upload a still image, upload a video, or use the live camera to detect damage in real time.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


st.write("")

st.markdown('<div class="upload-heading">Detection Location</div>', unsafe_allow_html=True)

location = st.session_state.get("location")

if location and location.get("latitude"):
    st.success(f"Location set: {location['place_name']} ({location['latitude']:.6f}, {location['longitude']:.6f})")
    map_url = f"https://www.openstreetmap.org/export/embed.html?bbox={location['longitude']-0.008:.6f}%2C{location['latitude']-0.008:.6f}%2C{location['longitude']+0.008:.6f}%2C{location['latitude']+0.008:.6f}&layer=mapnik&marker={location['latitude']:.6f},{location['longitude']:.6f}"
    components.iframe(map_url, height=200)
else:
    st.warning("Please set your location before uploading images or videos.")

col_detect, col_manual = st.columns([1, 1])

with col_detect:
    with st.expander("Auto-detect location", expanded=not location):
        st.caption("Click below. Your browser will ask permission - click Allow.")
        if st.button("Detect My Location", key="gps_detect_btn", use_container_width=True):
            with st.spinner("Getting your GPS location..."):
                js_code = """
                new Promise(function(resolve) {
                    if (!navigator.geolocation) {
                        resolve(JSON.stringify({error: 'Geolocation not supported'}));
                        return;
                    }
                    navigator.geolocation.getCurrentPosition(
                        function(pos) {
                            resolve(JSON.stringify({
                                lat: pos.coords.latitude,
                                lon: pos.coords.longitude
                            }));
                        },
                        function(err) {
                            resolve(JSON.stringify({error: err.message}));
                        },
                        {enableHighAccuracy: true, timeout: 15000, maximumAge: 0}
                    );
                });
                """
                result = st_javascript(js_code)
            if isinstance(result, str):
                import json
                try:
                    data = json.loads(result)
                    if "lat" in data and "error" not in data:
                        lat, lon = float(data["lat"]), float(data["lon"])
                        place = get_place_name(lat, lon)
                        st.session_state.location = {
                            "latitude": lat,
                            "longitude": lon,
                            "place_name": place,
                            "accuracy": "exact (GPS)",
                        }
                        st.success(f"Location found: {place}")
                        st.rerun()
                    else:
                        err_msg = data.get("error", "unknown error")
                        if "denied" in err_msg.lower() or "permission" in err_msg.lower():
                            st.error("Permission denied. Fix steps:")
                            st.markdown("1. Click the **lock icon** in the address bar\n2. Set **Location** to **Allow**\n3. **Reload** this page\n4. Click **Detect My Location** again")
                        else:
                            st.error(f"GPS error: {err_msg}")
                except Exception:
                    st.error("Could not read GPS result.")
            else:
                st.error("GPS did not respond. Try again or use manual entry.")

with col_manual:
    with st.expander("Enter coordinates manually", expanded=not location):
        st.caption("Google Maps -> long-press your spot -> copy lat/long numbers.")
        col_lat, col_lon = st.columns(2)
        with col_lat:
            default_lat = location["latitude"] if location else 17.0000
            manual_lat = st.number_input("Latitude", value=default_lat, format="%.6f", step=0.0001)
        with col_lon:
            default_lon = location["longitude"] if location else 82.2500
            manual_lon = st.number_input("Longitude", value=default_lon, format="%.6f", step=0.0001)
        if st.button("Save Location", key="save_manual_loc", use_container_width=True):
            place = get_place_name(manual_lat, manual_lon)
            st.session_state.location = {
                "latitude": manual_lat,
                "longitude": manual_lon,
                "place_name": place,
                "accuracy": "exact (manual)",
            }
            st.rerun()

image_tab, video_tab, live_tab = st.tabs([
    "Image Detection", 
    "Video Upload", 
    "Live Camera"
])

with image_tab:
    st.markdown('<div class="upload-heading">Upload a Road Image</div>', unsafe_allow_html=True)
    st.markdown('<div class="upload-description">Choose whether to upload a road image or use your camera.</div>', unsafe_allow_html=True)

    image_input_method = st.radio(
        "Select input method",
        ["Upload image", "Use camera"],
        index=0,
        horizontal=True,
        label_visibility="collapsed",
    )

    image = None
    if image_input_method == "Use camera":
        st.info("Click the camera widget below to open your camera. Allow camera access when prompted by your browser.")
        camera_input_data = st.camera_input("Capture a road photo for detection", key="image_camera")
        if camera_input_data is not None:
            try:
                image = Image.open(camera_input_data).convert("RGB")
                st.image(image, caption="Captured camera image", use_container_width=True)
            except Exception as error:
                st.error(f"Unable to load the captured image: {error}")
    else:
        uploaded_file = st.file_uploader("Upload road image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption="Uploaded road image", use_container_width=True)
            except Exception as error:
                st.error(f"Unable to load the uploaded image: {error}")

    if image is not None:
        try:
            with st.spinner("Analyzing the selected image..."):
                result_image, total_detections, detections = run_detection(image)
            show_detection_results(result_image, total_detections, detections, location=location)
        except Exception as error:
            st.error(f"Unable to process the image: {error}")

with video_tab:
    st.markdown('<div class="upload-heading">Process a Road Video</div>', unsafe_allow_html=True)
    st.markdown('<div class="upload-description">Upload a video file or capture a live camera frame for Road Sight AI detection.</div>', unsafe_allow_html=True)

    video_input_method = st.radio(
        "Select video source",
        ["Upload video", "Use camera"],
        index=0,
        horizontal=True,
        label_visibility="collapsed",
    )

    if video_input_method == "Use camera":
        st.info("Click the camera widget below to open your camera. Allow camera access when prompted by your browser.")
        camera_frame = st.camera_input("Capture a road frame for detection", key="video_camera")
        if camera_frame is not None:
            try:
                image = Image.open(camera_frame).convert("RGB")
                st.image(image, caption="Captured live frame", use_container_width=True)
                with st.spinner("Analyzing the captured camera frame..."):
                    result_image, total_detections, detections = run_detection(image)
                show_detection_results(result_image, total_detections, detections, title="Live Camera Frame Detection", location=location)
            except Exception as error:
                st.error(f"Unable to process the captured camera frame: {error}")
    else:
        video_file = st.file_uploader("Upload road video", type=["mp4", "mov", "avi"], label_visibility="collapsed")

        if video_file is not None:
            st.video(video_file)
            st.info(f"Fast mode is enabled: the app analyzes every {FRAME_STRIDE}th frame and uses a smaller inference size to speed up video processing.")
            with st.spinner("Processing the uploaded video in fast mode..."):
                try:
                    output_path, all_detections = process_video(video_file)
                    total_detections, damage_summary = summarize_detections(all_detections)
                    st.success("Video detection completed successfully.")
                    if location:
                        show_location_details(location)
                    st.video(str(output_path))
                    st.markdown(
                        f"""
                        <div class="result-card">
                            <h3>Video Detection Summary</h3>
                            <p><strong>Total detections:</strong> {total_detections}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    if total_detections == 0:
                        st.warning("No trained road-damage object was detected in the sampled video frames.")
                    else:
                        for summary in damage_summary:
                            st.markdown(
                                f"""
                                <div class="result-card">
                                    <strong>{summary['name']}</strong><br>
                                    Count: {summary['count']}<br>
                                    Share: {summary['percentage']:.1f}% of detected damages<br>
                                    Avg confidence: {summary['average_confidence'] * 100:.2f}%
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                    st.download_button(
                        label="Download processed video",
                        data=open(output_path, "rb").read(),
                        file_name=output_path.name,
                        mime="video/mp4",
                    )
                except Exception as error:
                    st.error(f"Unable to process the video: {error}")

with live_tab:
    st.markdown('<div class="upload-heading">Use the Live Camera</div>', unsafe_allow_html=True)
    st.markdown('<div class="upload-description">Capture a live frame from your webcam and run Road Sight AI detection with automatic GPS route tracking.</div>', unsafe_allow_html=True)

    col_start, col_stop = st.columns(2)
    with col_start:
        start_tracking = st.button("Start Live Session", key="start_live_tracking")
    with col_stop:
        stop_tracking = st.button("Stop Live Session", key="stop_live_tracking")

    if start_tracking:
        loc = st.session_state.get("location")
        if loc:
            st.session_state.live_tracking_active = True
            st.session_state.route_start = loc
            st.session_state.route_end = None
            st.session_state.route_waypoints = [loc]
            st.success(f"Live session started. Start point: {loc['place_name']}")
        else:
            st.warning("Please set your location first in the Detection Location section above.")

    if st.session_state.get("live_tracking_active"):
        loc = st.session_state.get("location")
        if loc:
            waypoints = st.session_state.get("route_waypoints", [])
            last = waypoints[-1] if waypoints else None
            if not last or get_route_distance(
                last["latitude"], last["longitude"],
                loc["latitude"], loc["longitude"]
            ) > 10:
                waypoints.append(loc)
                st.session_state.route_waypoints = waypoints
            st.info(f"Tracking active. Waypoints: {len(waypoints)} | Location: {loc['latitude']:.5f}, {loc['longitude']:.5f}")

    if stop_tracking and st.session_state.get("live_tracking_active"):
        st.session_state.live_tracking_active = False
        loc = st.session_state.get("location")
        if loc:
            st.session_state.route_end = loc

    live_image_data = st.camera_input("Capture a live frame for analysis", key="live_camera")
    if live_image_data is not None:
        try:
            live_image = Image.open(live_image_data).convert("RGB")
            st.image(live_image, caption="Captured live frame", use_container_width=True)
            with st.spinner("Analyzing the live frame..."):
                result_image, total_detections, detections = run_detection(live_image)
            show_detection_results(result_image, total_detections, detections, title="Live Camera Detection Result", location=location)
        except Exception as error:
            st.error(f"Unable to process the live camera frame: {error}")

    if not st.session_state.get("live_tracking_active") and st.session_state.get("route_end"):
        start_loc = st.session_state.get("route_start")
        end_loc = st.session_state.route_end
        waypoints = st.session_state.get("route_waypoints", [])
        if start_loc and end_loc:
            show_route_map(start_loc, end_loc, waypoints)
        st.session_state.route_start = None
        st.session_state.route_end = None
        st.session_state.route_waypoints = []


st.write("")

# Define animated SVG icons
camera_icon_svg = """
<svg class="icon-camera" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
    <defs>
        <style>
            .camera-lens { animation: lens-focus 2s ease-in-out infinite; }
            @keyframes lens-focus {
                0%, 100% { r: 4; }
                50% { r: 6; }
            }
        </style>
    </defs>
    <rect x="2" y="5" width="20" height="14" rx="2" fill="none" stroke="currentColor" stroke-width="2"/>
    <circle class="camera-lens" cx="12" cy="12" r="4" fill="none" stroke="currentColor" stroke-width="2"/>
    <circle cx="18" cy="7" r="1.5" fill="currentColor"/>
</svg>
"""

video_icon_svg = """
<svg class="icon-video" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
    <defs>
        <style>
            .play-button { animation: play-pulse 2s ease-in-out infinite 0.3s; }
            @keyframes play-pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.6; }
            }
        </style>
    </defs>
    <rect x="2" y="4" width="20" height="16" rx="2" fill="none" stroke="currentColor" stroke-width="2"/>
    <polygon class="play-button" points="9,8 9,16 16,12" fill="currentColor"/>
</svg>
"""

check_icon_svg = """
<svg class="icon-check" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
    <defs>
        <style>
            .checkmark { stroke-dasharray: 24; stroke-dashoffset: 24; animation: check-draw 1s ease-in-out infinite; }
            @keyframes check-draw {
                0% { stroke-dashoffset: 24; }
                50% { stroke-dashoffset: 0; }
                100% { stroke-dashoffset: 0; }
            }
        </style>
    </defs>
    <circle cx="12" cy="12" r="10" fill="none" stroke="currentColor" stroke-width="2"/>
    <path class="checkmark" d="M7 12l3 3 7-7" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
"""

step1, step2, step3 = st.columns(3)
with step1:
    st.markdown(
        f"""
        <div class="info-card">
            <div class="info-icon">{camera_icon_svg}</div>
            <div class="info-title">1. Upload Image</div>
            <div class="info-text">Choose a clear road image for instant AI damage detection.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with step2:
    st.markdown(
        f"""
        <div class="info-card">
            <div class="info-icon">{video_icon_svg}</div>
            <div class="info-title">2. Process Video</div>
            <div class="info-text">Upload a moving scene and detect damage frame by frame with Road Sight AI.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with step3:
    st.markdown(
        f"""
        <div class="info-card">
            <div class="info-icon">{check_icon_svg}</div>
            <div class="info-title">3. Review Results</div>
            <div class="info-text">Inspect the annotated output and confidence scores for each detected defect.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )