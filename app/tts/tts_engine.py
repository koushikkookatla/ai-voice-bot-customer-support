import tempfile
import os
from gtts import gTTS
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class TTSEngine:
    """Text-to-Speech engine using gTTS (Google TTS)."""

    def __init__(self, language: str = "en", slow: bool = False):
        self.language = language
        self.slow = slow
        logger.info(f"TTSEngine initialized - language: {language}, slow: {slow}")

    def synthesize(self, text: str, rate: float = 1.0) -> str:
        """
        Convert text to speech and save as MP3.

        Args:
            text: Text to synthesize
            rate: Speaking rate (1.0 = normal, < 1.0 = slower, > 1.0 = faster)

        Returns:
            Path to generated audio file
        """
        if not text or not text.strip():
            logger.warning("Empty text provided to TTS - using default message")
            text = "I'm here to help you."

        # gTTS slow mode for rate < 0.8
        slow = rate < 0.8

        try:
            logger.info(f"Synthesizing: '{text[:50]}...' | rate: {rate}")
            tts = gTTS(text=text, lang=self.language, slow=slow)
            tmp_file = tempfile.NamedTemporaryFile(
                delete=False, suffix=".mp3", prefix="tts_"
            )
            tmp_file.close()
            tts.save(tmp_file.name)
            logger.info(f"Audio saved to: {tmp_file.name}")
            return tmp_file.name
        except Exception as e:
            logger.error(f"TTS synthesis failed: {str(e)}")
            raise

    def synthesize_to_file(self, text: str, output_path: str, rate: float = 1.0) -> str:
        """
        Synthesize text to a specific output file.

        Args:
            text: Text to synthesize
            output_path: Output file path
            rate: Speaking rate

        Returns:
            Output file path
        """
        slow = rate < 0.8
        tts = gTTS(text=text, lang=self.language, slow=slow)
        tts.save(output_path)
        logger.info(f"Audio saved to: {output_path}")
        return output_path

    def cleanup(self, file_path: str):
        """Remove temporary audio file."""
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
                logger.debug(f"Cleaned up: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to cleanup {file_path}: {str(e)}")
