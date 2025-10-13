# System Flowcharts and Architecture Diagrams (Updated)

## 1. Overall System Architecture

```
┌──────────────────────────────┐      ┌───────────────────────────┐
│                              │      │                           │
│   Browser Extension          │      │   Local FastAPI Server    │
│   (content_predict.js)       │──────▶│   (extension_api.py)      │
│                              │      │                           │
└──────────────────────────────┘      └───────────────────────────┘
           │                                ▲
           ▼                                │
┌──────────────────────────────┐      ┌───────────────────────────┐
│                              │      │                           │
│   User Behavior Tracking     │      │   Prediction Endpoint     │
│   (clicks, scrolls, etc.)    │      │   (/predict_event)        │
│                              │      │                           │
└──────────────────────────────┘      └───────────────────────────┘
```

## 2. Data Processing Pipeline

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Raw Data      │    │   Data          │    │   Feature       │    │   Time-based    │
│   (CSVs)        │───▶│   Preprocessing │───▶│   Engineering   │───▶│   Split         │
│                 │    │   & Cleaning    │    │                 │    │   (Train/Val/Test)│
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
      │                                                                    │
      ▼                                                                    ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   SMOTE         │    │   Model         │    │   Model         │    │   Model         │
│   (on Train)    │───▶│   Training      │───▶│   Evaluation    │───▶│   Deployment    │
│                 │    │   (CatBoost)    │    │   (F1, AUC)     │    │   (Save .pkl)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 3. Machine Learning Model Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                 DEPLOYED MODEL                                  │
└─────────────────────────────────────────────────────────────────────────────────┘

                    ┌─────────────────────────────────────────────────┐
                    │                INPUT FEATURES                   │
                    │                                                 │
                    │ • User Behavior (clicks, scrolls, time)        │
                    │ • Product Features (price, brand, category)    │
                    │ • Temporal Features (hour, day, weekend)       │
                    └─────────────────┬───────────────────────────────┘
                                      │
                    ┌─────────────────┴───────────────────────────────┐
                    │                 CatBoost Model                  │
                    │                                                 │
                    │  • Pre-trained on historical data               │
                    │  • Loaded from 'deployed_model.pkl'             │
                    └─────────────────┬───────────────────────────────┘
                                      │
                    ┌─────────────────┴───────────────────────────────┐
                    │              FINAL PREDICTION                   │
                    │                                                 │
                    │ • Purchase Probability (0-1)                   │
                    │ • Recommendation (High, Medium, Low)           │
                    └─────────────────────────────────────────────────┘
```

## 4. Real-time Prediction Flow

```
┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
│   User Action    │   │ content_predict.js │   │ FastAPI Server   │   │   CatBoost       │
│   (e.g., scroll) │──▶│   (Tracks user)    │──▶│ (extension_api.py)│──▶│   Model          │
└──────────────────┘   └──────────────────┘   └──────────────────┘   └──────────────────┘
      │                                                                     │
      │                                                                     ▼
      │                                                            ┌──────────────────┐
      │                                                            │   Prediction     │
      │                                                            │   (Probability)  │
      │                                                            └──────────────────┘
      │                                                                     │
      ▼                                                                     ▼
┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
│  Update Storage  │◀──│  Send Response   │◀──│  Format Response │◀──│   Apply Logic    │
│ (chrome.storage) │   │   to content.js  │   │   (JSON)         │   │ (Recommendation) │
└──────────────────┘   └──────────────────┘   └──────────────────┘   └──────────────────┘
      │
      ▼
┌──────────────────┐
│   Popup UI       │
│   (popup.js)     │
│   (Displays      │
│    prediction)   │
└──────────────────┘
```

## 5. Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                LOCAL DEPLOYMENT                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────┐
│           Local Machine            │
│                                  │
│   ┌──────────────────────────┐   │
│   │                          │   │
│   │   Browser Extension      │   │
│   │                          │   │
│   └────────────┬─────────────┘   │
│                │                 │
│                │ HTTP Request    │
│                │ (localhost:8001)│
│                │                 │
│   ┌────────────▼─────────────┐   │
│   │                          │   │
│   │   FastAPI Server         │   │
│   │   (extension_api.py)     │   │
│   │                          │   │
│   │   - Loads deployed_model.pkl│   │
│   │   - /predict_event endpoint│   │
│   │                          │   │
│   └──────────────────────────┘   │
│                                  │
└──────────────────────────────────┘
```