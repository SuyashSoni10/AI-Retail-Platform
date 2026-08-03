from fastapi import APIRouter, File, UploadFile
from app.schemas import ProductClassResponse, FaceRecognitionResponse
from app.services.cv_service import cv_service

router = APIRouter(
    prefix="/vision",
    tags=["vision"]
)

@router.post("/classify-product", response_model=ProductClassResponse)
async def classify_product(file: UploadFile = File(...)):
    image_bytes = await file.read()
    result = cv_service.classify_product(image_bytes)
    return result

@router.post("/recognize-face", response_model=FaceRecognitionResponse)
async def recognize_face(file: UploadFile = File(...)):
    image_bytes = await file.read()
    result = cv_service.recognize_face(image_bytes)
    return result
