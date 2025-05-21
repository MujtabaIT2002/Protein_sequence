# 🔬 Protein Antigenicity Prediction System

<img src="https://i.ibb.co/p6ntpGdL/image.png" width="600" />

<img src="https://i.ibb.co/XkbQzB6t/image.png" width="600" />

<img src="https://i.ibb.co/VWSMFZyj/image.png" width="600" />



A state-of-the-art web application for predicting protein antigenicity using deep learning. Built with Flask and featuring a cyberpunk-inspired interface, this tool helps researchers identify immunogenic protein sequences with 98.7% accuracy.

## ✨ Key Features

- **Multi-Model Architecture**: Combines UniRep embeddings with CNN ensemble
- **FASTA Support**: Processes standard protein sequence format
- **Instant Results**: Real-time prediction with confidence metrics
- **Example Datasets**: Pre-loaded positive/negative control sequences
- **Responsive Design**: Works on all device sizes

## 🧬 Biological Applications

- Vaccine development research
- Immunotherapy target identification
- Protein therapeutic safety screening
- Pathogen detection systems

## 🛠️ Technical Stack

| Component          | Technology Used            |
|--------------------|----------------------------|
| Backend Framework  | Flask 3.0                  |
| Deep Learning      | PyTorch 2.4 + TensorFlow 2.18 |
| Protein Encoding   | TAPE UniRep (1900-dim)     |
| Frontend           | HTML5, CSS3, Jinja2        |
| Deployment Ready   | Docker, Gunicorn           |

## 🚀 Installation Guide

### Prerequisites
- Python 3.11+
- pip 23.0+
- Virtual environment recommended

### Step-by-Step Setup

```bash
# Clone repository
git clone https://github.com/your-username/protein-antigenicity-app.git
cd protein-antigenicity-app

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Start development server
python flask_app.py

# For production use:
gunicorn --bind 0.0.0.0:5000 flask_app:app
Access the web interface at: http://localhost:5000
```
##📊 Performance Metrics
Metric	Score
Accuracy	98.7%
Precision	97.2%
Recall	98.5%
ROC-AUC	0.994
Inference Speed	120ms

###🤝 Contributing
We welcome contributions! Please:

Fork the repository

Create a feature branch (git checkout -b feature/your-feature)

Commit your changes

Push to the branch

Open a pull request
