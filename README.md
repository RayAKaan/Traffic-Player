# 🚦 SmarTSignalAI – Intelligent Traffic Detection & Signal Management System

SmarTSignalAI is a full-stack, AI-powered Traffic Management System designed to detect, track, and classify vehicles in real-time from traffic camera feeds or uploaded videos, and dynamically control traffic signals. It leverages **YOLOv8**, **Deep SORT**, and **FastAPI**, and is built with **MLOps integration** for seamless deployment, monitoring, and version control.

---

## 📌 Project Status

**🔧 Actively in Development**

| Module                         | Status     |
|-------------------------------|------------|
| 🚘 Vehicle Detection (YOLOv8) | ✅ Completed |
| 🎯 Vehicle Tracking (Deep SORT) | ✅ Completed |
| 📊 Dashboard UI (React)       | ✅ Completed |
| 📦 Backend API (FastAPI)      | ✅ Completed |
| 🧠 Scene Analysis & Insights  | ✅ Completed |
| 🔄 Signal Override & Routing  | ⚙️ In Progress |
| 📈 ML Monitoring (MLflow)     | ⚙️ Partially integrated |
| 🚨 Accident Risk Analysis     | ✅ Basic version implemented |
| 🗺️ Real-time Route Planning   | 🚧 Coming soon |
| 🚦 Live Feed Support          | 🧪 Prototype phase |
| 🐳 Dockerized Deployment      | ✅ In place |
| 🧠 YOLOv10 / Advanced Models  | 🚧 Future plan |

---

## 💡 Key Features

- ✅ **YOLOv8-based Vehicle Detection**
- ✅ **Real-time Tracking** via Deep SORT  
- ✅ **Enhanced Scene Understanding** (speed, density, congestion, accident risk)  
- ✅ **Modular FastAPI Backend**
- ✅ **Professional React Frontend Dashboard**
- ✅ **MLflow Tracking** for model versioning and performance
- ✅ **FFmpeg-based Video Processing** for browser compatibility
- ✅ **Smart Traffic Light Logic** (manual + automated override)
- ✅ **MLOps Ready**: Docker, CI/CD (GitHub Actions planned)

---

## 📂 Project Structure

<details>
<summary><b>📦 Click to view project tree</b></summary>

SmarTSignalAI/
│
├── app/
│ ├── main.py
│ ├── core/
│ │ ├── config.py
│ │ ├── logger.py
│ │ └── utils.py
│ │
│ ├── routes/
│ │ ├── video.py
│ │ ├── inference.py
│ │ ├── dashboard.py
│ │ └── monitor.py
│ │
│ ├── services/
│ │ ├── detection_service.py
│ │ ├── tracking_service.py
│ │ ├── analytics_service.py
│ │ └── mlflow_service.py
│ │
│ ├── models/
│ │ ├── yolo_inference.py
│ │ ├── deep_sort_tracker.py
│ │ └── postprocessing.py
│ │
│ └── data/
│ ├── uploads/
│ └── processed/
│
├── frontend/
│ ├── src/
│ │ ├── pages/
│ │ │ ├── Dashboard.tsx
│ │ │ ├── VideoProcessing.tsx
│ │ │ ├── MLMonitoring.tsx
│ │ │ ├── LiveFeed.tsx
│ │ │ └── RoutingConsole.tsx
│ │ │
│ │ ├── components/
│ │ │ ├── Navbar.tsx
│ │ │ ├── Sidebar.tsx
│ │ │ ├── VideoPlayer.tsx
│ │ │ └── InsightsCard.tsx
│ │ │
│ │ └── api/
│ │ ├── videoAPI.tsx
│ │ ├── inferenceAPI.tsx
│ │ └── dashboardAPI.tsx
│ │
│ ├── vite.config.ts
│ └── package.json
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
└── .gitignore

Copy code

</details>

---

## 🧠 How It Works

### 1. 🔍 **Detection & Tracking**
- **YOLOv8** detects vehicles (car, bus, truck, motorbike, bicycle).  
- **Deep SORT** assigns consistent IDs to track motion across frames.

### 2. 📊 **Scene Analysis**
- Computes:
  - Vehicle counts (class-wise)
  - Average speed (in km/h)
  - Road density
  - Idle vs moving vehicles
  - Congestion level
  - Accident risk estimation

### 3. 🎞️ **Video Processing**
- Users upload traffic video via the frontend.  
- Backend processes it frame-by-frame.  
- Processed video is saved using FFmpeg for compatibility.

### 4. 📈 **ML Monitoring**
- All inference logs, stats, and model configurations are tracked via **MLflow**.

### 5. 🧠 **Dashboard Insights**
- Displays congestion, risk alerts, scene descriptions, and traffic conditions.

---

## 🚀 Getting Started

### ✅ Prerequisites

- **Python ≥ 3.10**
- **Node.js ≥ 18.x**
- **FFmpeg** installed system-wide
- **Docker** (optional but recommended)
- **CUDA GPU** (for faster inference)

---

### ⚙️ 1. Clone the Repository

```bash
git clone https://github.com/RayAKaan/SmarTSignalAI.git
cd SmarTSignalAI
🧱 2. Backend Setup
bash
Copy code
cd app
python -m venv venv
source venv/bin/activate       # (Linux/macOS)
venv\Scripts\activate          # (Windows)
pip install -r requirements.txt
Run the backend server:

bash
Copy code
uvicorn app.main:app --reload
💻 3. Frontend Setup
bash
Copy code
cd frontend
npm install
npm run dev
Access the app at:

arduino
Copy code
http://localhost:5173
🐳 4. (Optional) Docker Setup
bash
Copy code
docker-compose up --build
🧩 5. Run MLflow (Optional)
bash
Copy code
mlflow ui --port 5000
Access at http://localhost:5000

🤝 Contribution Guide
Fork and clone the repo.

Create a new branch for your feature.

Follow modular structure & PEP8.

Submit a PR with a clear description.
