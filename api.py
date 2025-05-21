from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model import EmailClassifier
import uvicorn

app = FastAPI(
    title="Email Classification API",
    description="API for classifying emails into predefined categories",
    version="1.0.0"
)

# Load the trained model
classifier = EmailClassifier()
try:
    classifier.load_model()
except:
    raise RuntimeError("Model file not found. Please train the model first using train_model.py")

class EmailRequest(BaseModel):
    subject: str
    body: str

class EmailResponse(BaseModel):
    category: str
    confidence: float

@app.post("/predict", response_model=EmailResponse)
async def predict_email_category(request: EmailRequest):
    """
    Predict the category of an email based on its subject and body.
    
    - **subject**: The email subject line
    - **body**: The email body text
    
    Returns the predicted category and confidence score.
    """
    try:
        result = classifier.predict(request.subject, request.body)
        return EmailResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Welcome message and API information."""
    return {
        "message": "Welcome to the Email Classification API",
        "endpoints": {
            "/predict": "POST - Predict email category",
            "/docs": "GET - API documentation"
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 