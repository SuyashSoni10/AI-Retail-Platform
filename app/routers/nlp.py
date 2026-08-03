from fastapi import APIRouter
from app.schemas import SentimentRequest, SentimentResponse
from app.services.nlp_service import nlp_service

router = APIRouter(
    prefix="/nlp",
    tags=["nlp"]
)

@router.post("/analyze-sentiment", response_model=SentimentResponse)
async def analyze_sentiment(payload: SentimentRequest):
    result = nlp_service.predict_sentiment(payload.text)
    return result
