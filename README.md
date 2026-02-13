# 🛰️ SAR Multi-Agent System

A LangGraph-based multi-agent system for SAR data search, download, and InSAR processing with intelligent AI assistance

> Intelligent SAR Data Processing with LangGraph, SNAP, and FastAPI

## 🌟 Key Features

### 1. Retrieval Agent
- Web search and information extraction
- Location name → Coordinates conversion
- Date/Event information extraction
- RAG-based Q&A

### 2. SAR Processing Agent
- **SAR Data Search**: Location/date-based Sentinel-1 data search
- **Automatic Download**: ASF API integration
- **InSAR Processing**: Ground deformation analysis using SNAP
  - Automatic Master/Slave selection
  - Background processing (20-30 minutes)
  - Phase/Coherence map generation

### 3. Vision Agent
- Image Segmentation
- Object Detection
- Image Classification

## 🏗️ Project Structure

```
sar-multi-agent/
├── server.py                   # Main agent server (port 8000)
├── web_ui.py                   # Streamlit UI
├── graph.py                    # LangGraph workflow
├── state.py                    # GraphState definition
│
├── nodes/                      # Agent nodes
│   ├── retrieval/              # Search/Download nodes
│   │   └── prompts/            # LLM prompts
│   ├── sar/                    # SAR/InSAR nodes
│   │   └── prompts/
│   └── vision/                 # Vision nodes
│       └── prompts/
│
├── services/                   # External API services
│   ├── sar_download/           # SAR Download API (port 8001)
│   ├── insar_processing/       # InSAR Processing API (port 8002)
│   └── cv_vision/              # Computer Vision API (planned)
│
└── scripts/                    # Utility scripts
    ├── start_all.sh            # Start all services
    ├── stop_all.sh             # Stop all services
    └── check_services.sh       # Check service status
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create Conda environment
conda create -n rag python=3.11
conda activate rag

# Install Python packages
pip install -r requirements.txt

# Configure environment variables (.env file)
# Copy .env.example if available, or create .env with:
# - TAVILY_API_KEY: Web search API key
# - SAR_DATA_PATHS: SAR data storage paths (comma-separated)
#   Example: /mnt/sar,/home/user/sar_data,/data/sar

# Install SNAP Python API (for InSAR)
cd services/insar_processing
bash INSTALL_ESA_SNAPPY.sh
cd ../..
```

### 2. Start Services

**Option A: Start all services at once**
```bash
bash scripts/start_all.sh
```

**Option B: Start services individually**
```bash
# SAR Download API (port 8001)
cd services/sar_download
bash start_sar_api.sh

# InSAR Processing API (port 8002)
cd services/insar_processing
bash start_insar_api.sh

# Agent Server (port 8000)
python server.py
```

### 3. Access UI

**Streamlit UI:**
```bash
streamlit run web_ui.py
```

**LangServe Playground:**
```
http://localhost:8000/chat/playground
```

## 📡 API Ports

| Service | Port | Purpose |
|---------|------|---------|
| Agent Server | 8000 | LangGraph main agent |
| SAR Download | 8001 | Sentinel-1 data search/download |
| InSAR Processing | 8002 | SNAP InSAR processing |
| CV Vision | 8003 | Computer Vision processing (planned) |

## 🧪 Usage Examples

### InSAR Processing
```
User: "Process InSAR with Turkey earthquake data from 2023"
Agent: → Location search → Data download → Master/Slave selection → InSAR processing
```

### Direct File Specification
```
User: "Process InSAR with these files: /mnt/sar/S1A_...zip /mnt/sar/S1A_...zip"
Agent: → Master/Slave selection → Parameter input → Start InSAR processing
```

### SAR Data Search
```
User: "Get me SAR data for Japan Noto Peninsula earthquake 2024"
Agent: → Location search → Coordinate conversion → SAR data search → Download
```

## 🛠️ Development

### Modifying Prompts
Prompts are managed in separate files:
- `nodes/retrieval/prompts/` - Search/Classification prompts
- `nodes/sar/prompts/` - SAR/InSAR prompts

### Code Structure
- **LangGraph**: Workflow graph definition (`graph.py`)
- **State Management**: TypedDict-based (`state.py`)
- **Nodes**: Functions for each processing step (`nodes/`)
- **Routing**: Conditional edges (`graph.py`)

## 📊 Architecture

```
User → Streamlit UI → Agent Server (LangGraph)
                         ↓
                   [Main Router]
                    /    |    \
                   /     |     \
            Retrieval  SAR    Vision
               ↓       ↓         ↓
          Web Search  SAR API  CV API
               ↓       ↓
          RAG/QA   InSAR API
```

## 🔧 Troubleshooting

### When services don't start
```bash
# Check service status
bash scripts/check_services.sh

# Stop existing processes
bash scripts/stop_all.sh

# Restart
bash scripts/start_all.sh
```

### InSAR Processing Errors
- Check SNAP installation: `/home/mjh/esa-snap`
- Verify esa_snappy setup: `services/insar_processing/INSTALL_ESA_SNAPPY.sh`
- Ensure sufficient disk space (10GB+)

## 📦 Dependencies

```bash
# Core
langgraph==0.2.45
langchain==0.3.7
langchain-community==0.3.5
langchain-openai==0.2.5

# API
fastapi==0.115.5
uvicorn==0.32.1

# SAR Processing
asf_search==9.0.3
esa_snappy (SNAP Python API)

# Utilities
requests==2.32.3
pydantic==2.10.2
```

## 📝 License

MIT

## 👥 Author

- Jeongho Min
