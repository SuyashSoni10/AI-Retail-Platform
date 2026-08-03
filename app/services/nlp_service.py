import os
import re
import pickle

class NLPService:
    def __init__(self):
        # Resolve path to the models directory
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, 'models', 'sentiment_model.pkl')
        vectorizer_path = os.path.join(base_dir, 'models', 'vectorizer.pkl')

        print(f"Loading NLP models from {model_path} and {vectorizer_path}...")
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
            
        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)

        self.stopwords = {"is", "an", "the", "a", "and", "to", "in", "of", "for", "on", "with", "at", "by", "this", "it", "that"}

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        words = text.split()
        cleaned_words = [w for w in words if w not in self.stopwords]
        return " ".join(cleaned_words)

    def predict_sentiment(self, text: str) -> dict:
        cleaned = self.clean_text(text)
        if not cleaned:
            return {"sentiment": "neutral", "confidence": 0.5}
            
        # Transform and predict
        vectorized = self.vectorizer.transform([cleaned])
        prediction = self.model.predict(vectorized)[0]
        
        # Calculate confidence score
        probabilities = self.model.predict_proba(vectorized)[0]
        class_idx = list(self.model.classes_).index(prediction)
        confidence = float(probabilities[class_idx])

        return {
            "sentiment": prediction,
            "confidence": confidence
        }

# Instantiate singleton service instance
nlp_service = NLPService()
