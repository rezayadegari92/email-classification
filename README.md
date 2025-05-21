# Email Classification System

This project implements an AI-based email classification system that automatically categorizes incoming emails into four categories:
- Customer Support / Complaints
- Sales Inquiry / Pricing Request
- Partnership / Collaboration Proposal
- Spam / Advertisements

## Features

- Machine learning model for email classification
- REST API for real-time predictions
- Pre-trained model included
- Easy setup and deployment

## Setup Instructions

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download NLTK data:
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

4. Train the model:
```bash
python train_model.py
```

5. Start the API server:
```bash
uvicorn api:app --reload
```

The API will be available at `http://localhost:8000`

## API Usage

### Predict Email Category

**Endpoint:** `POST /predict`

**Request Body:**
```json
{
    "subject": "Your email subject",
    "body": "Your email body"
}
```

**Response:**
```json
{
    "category": "predicted_category",
    "confidence": 0.95
}
```

## Project Structure

- `train_model.py`: Script to train and save the classification model
- `api.py`: FastAPI implementation for the prediction endpoint
- `model.py`: Core model implementation and text preprocessing
- `email_dataset.csv`: Training dataset
- `model.joblib`: Saved model file (generated after training)

## Model Details

The system uses a combination of TF-IDF vectorization and a Support Vector Machine (SVM) classifier. The text preprocessing includes:
- Lowercase conversion
- Stopword removal
- Punctuation removal
- Tokenization

## License

MIT License