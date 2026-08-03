from pydantic import BaseModel

class SentimentRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float

class ChatbotRequest(BaseModel):
    message: str

class ChatbotResponse(BaseModel):
    reply: str
    intent: str

class ProductClassResponse(BaseModel):
    category: str
    confidence: float

class FaceRecognitionResponse(BaseModel):
    name: str
    matched: bool
    logged: bool
    engine: str
