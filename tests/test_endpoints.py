import io
from PIL import Image
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "docs_url" in data

def test_dashboard_stats():
    response = client.get("/dashboard/stats")
    assert response.status_code == 200
    data = response.json()
    assert "telemetry" in data
    assert "sentiment_breakdown" in data
    assert "product_sales_distribution" in data
    assert "chatbot_analytics" in data

def test_sentiment_analysis():
    response = client.post(
        "/nlp/analyze-sentiment",
        json={"text": "The delivery was delayed but the product itself is amazing and works great!"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] in ["positive", "neutral", "negative"]
    assert "confidence" in data
    assert isinstance(data["confidence"], float)

def test_chatbot_response():
    response = client.post(
        "/chatbot",
        json={"message": "What are your store hours?"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "reply" in data
    assert "intent" in data
    assert isinstance(data["reply"], str)

def test_classify_product_endpoint():
    # Create a mock image file
    file_stream = io.BytesIO()
    img = Image.new('RGB', (128, 128), color='green')
    img.save(file_stream, format='JPEG')
    file_stream.seek(0)
    
    response = client.post(
        "/vision/classify-product",
        files={"file": ("product.jpg", file_stream, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "category" in data
    assert "confidence" in data
    assert data["category"] in ['bags', 'clothing', 'electronics', 'groceries', 'shoes']

def test_recognize_face_endpoint():
    # Create a mock image file
    file_stream = io.BytesIO()
    img = Image.new('RGB', (128, 128), color='blue')
    img.save(file_stream, format='JPEG')
    file_stream.seek(0)
    
    response = client.post(
        "/vision/recognize-face",
        files={"file": ("face.jpg", file_stream, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "matched" in data
    assert "logged" in data
    assert "engine" in data
