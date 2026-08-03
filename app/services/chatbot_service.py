import os
import re
import pickle
import random

class ChatbotService:
    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, 'models', 'chatbot_model.pkl')
        
        print(f"Loading Chatbot model from {model_path}...")
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
            
        self.model = data['model']
        self.vectorizer = data['vectorizer']
        self.response_map = data['response_map']

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text.strip()

    def get_chatbot_reply(self, message: str) -> dict:
        cleaned = self.clean_text(message)
        if not cleaned:
            return {"reply": "I'm sorry, I didn't quite catch that. Could you please rephrase?", "intent": "unknown"}

        # Vectorize and predict intent tag
        vectorized = self.vectorizer.transform([cleaned])
        tag = self.model.predict(vectorized)[0]

        # Get response candidates
        responses = self.response_map.get(tag, ["I'm sorry, I don't have an answer for that right now."])
        reply = random.choice(responses)

        return {
            "reply": reply,
            "intent": tag
        }

# Instantiate singleton service instance
chatbot_service = ChatbotService()
