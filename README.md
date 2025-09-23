# Project AI – E-Commerce ML System

## 📌 Overview
This project implements an **end-to-end machine learning system** for e-commerce, covering the full lifecycle of model development:  
- Data preparation & training  
- Model deployment (basic & enhanced APIs)  
- Real-time inference  
- Model monitoring  
- Documentation & reports  

The system is designed to be modular, scalable, and production-ready, with emphasis on both **baseline ML deployment** and **enhanced features** for monitoring and robustness.

---

## 🚀 Features
- **Model Training** – CatBoost-based training pipeline (`ecommerce_ml_training.py`, `run_mm_train.py`).  
- **Deployment** – Scripts for exporting & serving models (`deploy_model.py`, `deployed_model.pkl`).  
- **APIs** – Multiple REST API implementations (`api_example.py`, `enhanced_api.py`, `enhanced_production_api.py`).  
- **Model Monitoring** – Automated monitoring tools (`model_monitoring.py`, logs in `prep_log.txt`, `test_performance_log.json`).  
- **Testing Suite** – Unit tests and integration tests (`test_deployed_model.py`, `test_enhanced_system.py`, `test_generalization.py`).  
- **Documentation** – Detailed reports and flowcharts describing workflows and system design.  

---

## 📂 Project Structure
```
project_ai/
│── ecommerce_ml_training.py     # Core training pipeline
│── deploy_model.py              # Model deployment script
│── run_mm_prep.py / run_mm_train.py   # Data prep & training automation
│── model_monitoring.py          # Tools for monitoring deployed model
│── api_example.py               # Basic API for serving predictions
│── enhanced_api.py              # Enhanced version of API
│── enhanced_production_api.py   # Production-grade API
│── enhanced_ml_system.py        # Extended ML pipeline
│── enhanced_requirements.txt    # Dependencies for enhanced system
│── test_*.py                    # Test scripts for models and APIs
│── *.pkl                        # Trained model artifacts
│── catboost_info/               # Training logs and metrics
│── docs & reports (.md/.html)   # Project reports and system guides
```

---

## ⚙️ Setup & Installation
1. **Clone the repository**
   ```bash
   git clone https://github.com/Krish8732/project_ai.git
   cd project_ai
   ```

2. **Install dependencies**  
   Basic setup:
   ```bash
   pip install -r requirements.txt
   ```
   Enhanced system setup:
   ```bash
   pip install -r enhanced_requirements.txt
   ```

3. **Verify installation**
   ```bash
   python -m unittest discover
   ```

---

## Model Files

You can access the model files needed for this project by clicking the button below:

[📁 Access Model Files](https://drive.google.com/drive/folders/1rHuIGTnD2JCR4D3zsbYXqT4vXF6ZCcUu?usp=sharing)


## 🏋️ Training the Model
To train from scratch:
```bash
python ecommerce_ml_training.py
```

Or using the automated runner:
```bash
python run_mm_prep.py
python run_mm_train.py
```

Artifacts will be saved as `.pkl` files.

---

## 🌐 Deploying the Model
Start a simple API server:
```bash
python api_example.py
```

For the enhanced/production API:
```bash
python enhanced_production_api.py
```

Test with:
```bash
python test_realistic_api.py
```

---

## 📊 Monitoring
Run monitoring tools:
```bash
python model_monitoring.py
```
Logs are stored in `prep_log.txt` and `test_performance_log.json`.

---

## 📑 Documentation & Reports
- **Deployment Guide** → `DEPLOYMENT_GUIDE.md`  
- **Enhanced System Guide** → `ENHANCED_SYSTEM_GUIDE.md`  
- **System Flowcharts** → `SYSTEM_FLOWCHARTS.md`  
- **Project Reports** → `E_COMMERCE_ML_PROJECT_REPORT.md`, `FINAL_PROJECT_REPORT.md/html`  
- **Summaries** → `PROJECT_SUMMARY_DOCUMENT.md`, `ENHANCEMENT_SUMMARY.md`  

---

## 🔮 Future Enhancements
- Advanced monitoring dashboards  
- Support for model versioning  
- CI/CD pipeline integration  
- API authentication & rate limiting  

---

## 🤝 Contributing
Feel free to fork this repo and submit pull requests. For major changes, please open an issue first to discuss what you’d like to change.

---

## 📜 License
This project is licensed under the MIT License.
