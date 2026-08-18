# ROAD SIGHT AI

AI-powered road damage detection system using YOLOv8 and Streamlit.

## Features

- **Image Detection** - Upload or capture road photos for instant damage analysis
- **Video Detection** - Upload road videos with frame-by-frame analysis
- **Live Camera** - Real-time detection using webcam
- **GPS Location** - Detect and map damage locations
- **Route Tracking** - Track monitoring routes with interactive maps
- **Dark/Light Mode** - Futuristic dark and clean light themes

## Damage Types Detected

- Alligator Cracks (AC)
- Longitudinal Cracks (L)
- Transverse Cracks (T)
- Potholes (P)
- Surface Erosion (SE)
- Utility Hole Damage (UH)

## Requirements

- Python 3.10+
- Ultralytics YOLOv8
- OpenCV
- Streamlit
- streamlit-javascript

## Installation

```bash
# Clone the repository
git clone https://github.com/Eswarankam002/Road-sight-ai.git
cd Road-sight-ai

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Train model (optional - model file needed)
python main.py

# Run the app
streamlit run app.py
```

The app opens at `http://localhost:8501`

## Model

- Uses YOLOv8 trained on road damage dataset
- Model file: `best1.pt`
- Training config: `data.yaml`

## Project Structure

```
Road-sight-ai/
├── app.py              # Main Streamlit application
├── main.py             # Model training script
├── requirements.txt    # Python dependencies
├── data.yaml           # Dataset configuration
├── best1.pt            # Trained YOLOv8 model
├── static/
│   └── style.css       # Custom styles
└── templates/
    └── index.html      # HTML template
```

## License

MIT
