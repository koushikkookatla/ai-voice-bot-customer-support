import whisper
import torch
import numpy as np
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class WhisperASR:
    """Automatic Speech Recognition using OpenAI Whisper."""

    def __init__(self, model_size: str = "base"):
        self.model_size = model_size
        logger.info(f"Loading Whisper model: {model_size}")
        self.model = whisper.load_model(model_size)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Whisper model loaded on {self.device}")

    def transcribe(self, audio_path: str) -> dict:
        """
        Transcribe audio file to text.

        Args:
            audio_path: Path to WAV audio file

        Returns:
            dict with 'text' and 'language' keys
        """
        try:
            logger.info(f"Transcribing: {audio_path}")
            result = self.model.transcribe(
                audio_path,
                language=None,  # Auto-detect language
                fp16=False       # Disable FP16 for CPU compatibility
            )
            transcribed_text = result["text"].strip()
            language = result.get("language", "en")

            if not transcribed_text:
                logger.warning("Empty transcription result - possibly silent or noisy audio")
                transcribed_text = "Sorry, I could not understand the audio."

            logger.info(f"Transcription: '{transcribed_text}' | Language: {language}")
            return {"text": transcribed_text, "language": language}

        except FileNotFoundError:
            logger.error(f"Audio file not found: {audio_path}")
            raise
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            raise

    def compute_wer(self, references: list, hypotheses: list) -> float:
        """
        Compute Word Error Rate on evaluation set.

        Args:
            references: List of ground truth transcripts
            hypotheses: List of model predicted transcripts

        Returns:
            WER as float (0.0 to 1.0)
        """
        total_words = 0
        total_errors = 0
        for ref, hyp in zip(references, hypotheses):
            ref_words = ref.lower().split()
            hyp_words = hyp.lower().split()
            total_words += len(ref_words)
            # Simple Levenshtein-based WER
            errors = self._levenshtein(ref_words, hyp_words)
            total_errors += errors

        wer = total_errors / max(total_words, 1)
        logger.info(f"WER: {wer:.4f} ({wer*100:.2f}%)")
        return wer

    def _levenshtein(self, ref: list, hyp: list) -> int:
        m, n = len(ref), len(hyp)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref[i-1] == hyp[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        return dp[m][n]
