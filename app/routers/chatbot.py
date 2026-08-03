from fastapi import APIRouter
from app.schemas import ChatbotRequest, ChatbotResponse
from app.services.chatbot_service import chatbot_service

router = APIRouter(
    prefix="/chatbot",
    tags=["chatbot"]
)

@router.post("", response_model=ChatbotResponse)
async def get_response(payload: ChatbotRequest):
    result = chatbot_service.get_chatbot_reply(payload.message)
    return result
