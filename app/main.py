from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import tempfile
import os
import logging
from app.asr.whisper_asr import WhisperASR
from app.nlu.intent_classifier import IntentClassifier
from app.response.response_generator import ResponseGenerator
from app.tts.tts_engine import TTSEngine
from app.utils.logger import setup_logger
from app.config import settings

app = FastAPI(
    title="AI Voice Bot - Customer Support",
    description="Production-ready AI Voice Bot for Customer Support Automation",
    version="1.0.0"
)

logger = setup_logger(__name__)

# Initialize modules
asr = WhisperASR(model_size=settings.WHISPER_MODEL_SIZE)
intent_clf = IntentClassifier(model_path=settings.INTENT_MODEL_PATH)
response_gen = ResponseGenerator()
tts = TTSEngine()


class TextInput(BaseModel):
    text: str


class IntentInput(BaseModel):
    intent: str
    context: dict = {}


class SynthesizeInput(BaseModel):
    text: str
    rate: float = 1.0


@app.get("/")
async def root():
    return {"message": "AI Voice Bot API is running", "version": "1.0.0"}


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    """Convert speech audio to text using Whisper ASR."""
    try:
        logger.info(f"Transcribing audio file: {audio.filename}")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            content = await audio.read()
            tmp.write(content)
            tmp_path = tmp.name
        result = asr.transcribe(tmp_path)
        os.unlink(tmp_path)
        logger.info(f"Transcription result: {result['text']}")
        return {"text": result["text"], "language": result.get("language", "en")}
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-intent")
async def predict_intent(body: TextInput):
    """Classify intent from text input."""
    try:
        logger.info(f"Predicting intent for: {body.text}")
        result = intent_clf.predict(body.text)
        return {
            "intent": result["intent"],
            "confidence": result["confidence"],
            "all_scores": result["all_scores"]
        }
    except Exception as e:
        logger.error(f"Intent prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-response")
async def generate_response(body: IntentInput):
    """Generate contextual response based on predicted intent."""
    try:
        logger.info(f"Generating response for intent: {body.intent}")
        response = response_gen.generate(body.intent, body.context)
        return {"response": response, "intent": body.intent}
    except Exception as e:
        logger.error(f"Response generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/synthesize")
async def synthesize(body: SynthesizeInput):
    """Convert text to speech audio."""
    try:
        logger.info(f"Synthesizing speech for: {body.text[:50]}...")
        audio_path = tts.synthesize(body.text, rate=body.rate)
        return FileResponse(audio_path, media_type="audio/mpeg", filename="response.mp3")
    except Exception as e:
        logger.error(f"TTS error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/voicebot")
async def voicebot(audio: UploadFile = File(...)):
    """Unified endpoint: Audio input -> Audio output (full pipeline)."""
    try:
        logger.info("Processing voicebot request (audio -> audio)")
        # Step 1: ASR
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            content = await audio.read()
            tmp.write(content)
            tmp_path = tmp.name
        transcription = asr.transcribe(tmp_path)
        os.unlink(tmp_path)
        text = transcription["text"]
        logger.info(f"Transcribed: {text}")

        # Step 2: Intent Classification
        intent_result = intent_clf.predict(text)
        intent = intent_result["intent"]
        confidence = intent_result["confidence"]
        logger.info(f"Intent: {intent} (confidence: {confidence:.2f})")

        # Step 3: Response Generation
        response_text = response_gen.generate(intent)
        logger.info(f"Response: {response_text}")

        # Step 4: TTS
        audio_path = tts.synthesize(response_text)
        return FileResponse(audio_path, media_type="audio/mpeg", filename="bot_response.mp3")
    except Exception as e:
        logger.error(f"Voicebot pipeline error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
