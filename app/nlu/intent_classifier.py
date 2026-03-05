import torch
import os
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from app.config import IDX_TO_INTENT, INTENT_LABELS
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class IntentClassifier:
    """Intent classifier using fine-tuned DistilBERT."""

    def __init__(self, model_path: str = "models/intent_model"):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_labels = len(INTENT_LABELS)
        self._load_model()

    def _load_model(self):
        """Load fine-tuned model or initialize pretrained."""
        model_name = "distilbert-base-uncased"
        if os.path.exists(self.model_path):
            logger.info(f"Loading fine-tuned model from {self.model_path}")
            self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_path)
            self.model = DistilBertForSequenceClassification.from_pretrained(self.model_path)
        else:
            logger.warning(f"Fine-tuned model not found. Loading base {model_name}")
            self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
            self.model = DistilBertForSequenceClassification.from_pretrained(
                model_name, num_labels=self.num_labels
            )
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Model loaded on {self.device}")

    def predict(self, text: str) -> dict:
        """
        Predict intent from text.

        Args:
            text: Input text string

        Returns:
            dict with 'intent', 'confidence', and 'all_scores'
        """
        if not text or not text.strip():
            logger.warning("Empty text input - returning fallback intent")
            return {
                "intent": "general_complaint",
                "confidence": 0.0,
                "all_scores": {}
            }

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        predicted_idx = int(probs.argmax())
        confidence = float(probs[predicted_idx])
        intent = IDX_TO_INTENT.get(predicted_idx, "general_complaint")

        all_scores = {
            INTENT_LABELS[i]: float(probs[i])
            for i in range(len(INTENT_LABELS))
        }

        logger.info(f"Predicted intent: {intent} (confidence: {confidence:.3f})")
        return {
            "intent": intent,
            "confidence": confidence,
            "all_scores": all_scores
        }

    def predict_batch(self, texts: list) -> list:
        """Predict intents for a batch of texts."""
        return [self.predict(text) for text in texts]
