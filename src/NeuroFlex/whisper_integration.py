import whisper
import numpy as np
from typing import Union, List, Dict
import logging

class WhisperIntegration:
    def __init__(self, model_name: str = "base"):
        self.model = whisper.load_model(model_name)
        logging.info(f"Whisper model '{model_name}' loaded successfully")

    def transcribe(self, audio: Union[str, np.ndarray]) -> Dict[str, Union[str, List[Dict[str, Union[float, str]]]]]:
        """
        Transcribe audio using the Whisper model.

        Args:
            audio (Union[str, np.ndarray]): Path to audio file or numpy array of audio data.

        Returns:
            Dict[str, Union[str, List[Dict[str, Union[float, str]]]]]: Transcription result containing text and segments.
        """
        try:
            result = self.model.transcribe(audio)
            logging.info("Audio transcription completed successfully")
            return result
        except Exception as e:
            logging.error(f"Error during transcription: {str(e)}")
            raise

    def detect_language(self, audio: Union[str, np.ndarray]) -> str:
        """
        Detect the language of the audio using the Whisper model.

        Args:
            audio (Union[str, np.ndarray]): Path to audio file or numpy array of audio data.

        Returns:
            str: Detected language code.
        """
        try:
            audio_data = self.model.pad_or_trim(audio)
            mel = self.model.log_mel_spectrogram(audio_data)
            _, probs = self.model.detect_language(mel)
            detected_language = max(probs, key=probs.get)
            logging.info(f"Detected language: {detected_language}")
            return detected_language
        except Exception as e:
            logging.error(f"Error during language detection: {str(e)}")
            raise

def create_whisper_integration(model_name: str = "base") -> WhisperIntegration:
    """
    Create an instance of WhisperIntegration.

    Args:
        model_name (str): Name of the Whisper model to use. Defaults to "base".

    Returns:
        WhisperIntegration: An instance of the WhisperIntegration class.
    """
    return WhisperIntegration(model_name)
