from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import vision, nlp, chatbot

app = FastAPI(
    title="Smart Retail & Customer Intelligence Platform API",
    description="Backend API serving NLP Sentiment analysis, Chatbot intents, Product Category classifier, and Face recognition.",
    version="1.0.0"
)

# Enable CORS for frontend and dashboard integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Routers
app.include_router(vision.router)
app.include_router(nlp.router)
app.include_router(chatbot.router)

@app.get("/", tags=["root"])
async def root():
    return {
        "message": "Welcome to the Smart Retail AI Platform API!",
        "docs_url": "/docs",
        "status": "healthy"
    }

@app.get("/dashboard/stats", tags=["dashboard"])
async def get_dashboard_stats():
    # Return telemetry for the dashboard
    return {
        "telemetry": {
            "total_visits": 1250,
            "returning_customers": 420,
            "average_dwell_time_minutes": 18.5,
        },
        "sentiment_breakdown": {
            "positive": 72.4,
            "neutral": 18.1,
            "negative": 9.5
        },
        "product_sales_distribution": {
            "shoes": 35.2,
            "bags": 24.8,
            "clothing": 18.0,
            "groceries": 12.5,
            "electronics": 9.5
        },
        "chatbot_analytics": {
            "total_queries": 845,
            "resolved_automatically": 92.5
        }
    }
