# Email Spam Detection Classifier

A machine learning-based email spam detection application built with Streamlit using Naive Bayes classification.

## Features

- **TF-IDF Vectorization**: Converts email text to numerical features
- **Multinomial Naive Bayes**: Classifies emails as spam or ham
- **Real-time Predictions**: Get instant spam probability scores
- **User-Friendly Interface**: Simple Streamlit web interface

## Setup Instructions

### 1. Create Virtual Environment
```bash
python -m venv .venv
.\.venv\Scripts\activate  # On Windows
source .venv/bin/activate  # On Mac/Linux
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the Model (If needed)
```bash
python train_model.py
```

This will:
- Download the public UCI SMS Spam Collection dataset (5574 messages)
- Train a TF-IDF Vectorizer
- Train a Multinomial Naive Bayes classifier (97.67% accuracy)
- Save `vectorizer.pkl` and `model.pkl`

### 4. Run the App
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## How It Works

1. **Input**: User enters an email message
2. **Preprocessing**: Text is cleaned, tokenized, and stemmed
3. **Vectorization**: TF-IDF converts text to numerical vectors
4. **Prediction**: Naive Bayes classifies the message
5. **Output**: Shows classification (Spam/Ham) with confidence scores

## Files

- `app.py` - Main Streamlit application
- `train_model.py` - Model training script
- `vectorizer.pkl` - Trained TF-IDF vectorizer
- `model.pkl` - Trained Naive Bayes classifier
- `practice.ipynb` - Jupyter notebook with full analysis
- `requirements.txt` - Python dependencies

## Dataset

The model is trained on the **UCI SMS Spam Collection** dataset:
- 5574 total messages
- 747 spam messages
- 4827 legitimate messages
- 97.67% test accuracy

To use your own dataset, place a CSV file named `spam.csv` with columns `v1` (label) and `v2` (text) in the project directory, then run `train_model.py`.

## Deployment

To deploy on Streamlit Cloud:
1. Push to GitHub
2. Connect your repo to Streamlit Cloud
3. Ensure all model files are committed
4. Access via your Streamlit Cloud URL

## Troubleshooting

**Error: "This MultinomialNB instance is not fitted yet"**
- Run `python train_model.py` to retrain the model

**Error: Model files not found**
- Ensure `vectorizer.pkl` and `model.pkl` exist in the project directory

**Encoding issues**
- The app handles both ASCII and extended characters

## Author
Adeel Munir - 2026
