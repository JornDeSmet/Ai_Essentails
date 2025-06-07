# 🍄 Mushroom Classification API

This repository contains a machine learning-powered web API to classify mushrooms as **edible** or **poisonous**, based on physical characteristics like cap color, gill shape, stem size, and more.

## 🚀 Live Demo

You can try out the deployed version here:  
📍 **[Live API on Render](https://ai-essentails.onrender.com/predict)**

---

## 🔍 Project Overview

This project leverages a **Random Forest** classifier trained on a labeled mushroom dataset. It exposes a prediction endpoint using **FastAPI**, making it easy to integrate into applications or test with tools like Postman.

### ✅ Features

- Trained ML model (`mushroom_model.pkl`)
- Categorical encoders (`label_encoders.pkl`)
- API built using **FastAPI**
- JSON-based prediction interface
- Public deployment on Render.com

---

## 📦 How to Use

### 1. Clone the repository

```bash
git clone https://github.com/JornDeSmet/Ai_Essentails.git
cd Ai_Essentails
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the server locally

```bash
uvicorn app:app --reload
```

### 4. Send a prediction request

POST to `/predict` with JSON like this:

```json
{
  "features": {
    "cap-diameter": 14.17,
    "cap-shape": "f",
    "cap-surface": "h",
    "cap-color": "e",
    "does-bruise-or-bleed": "f",
    "gill-attachment": "e",
    "gill-spacing": "c",
    "gill-color": "w",
    "stem-height": 17.86,
    "stem-width": 18.02,
    "stem-surface": "y",
    "stem-color": "w",
    "has-ring": "t",
    "ring-type": "p",
    "habitat": "d",
    "season": "a"
  }
}
```

The response will be:

```json
{
  "prediction": "edible" or "poisonous"
}
```

---

## 🧠 Tech Stack

- Python
- scikit-learn
- FastAPI
- Pandas
- Joblib
- Render (deployment)

---

## 📁 Files

- `app.py` – FastAPI application
- `mushroom_model.pkl` – Trained Random Forest model
- `label_encoders.pkl` – Encoders for categorical features
- `feature_order.pkl` – Ensures correct input order for predictions

---

## 🙋‍♂️ Author

Created by **Jorn De Smet**  
📍 GitHub: [JornDeSmet](https://github.com/JornDeSmet)

---

## 📄 License

This project is licensed under the MIT License.
