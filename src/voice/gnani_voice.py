# src/voice/gnani_voice.py
"""
Handles Gnani STT and TTS.
"""
import os
from pathlib import Path
from gnani.stt import GnaniSTTClient
from gnani.tts import GnaniTTSClient

from src.config import (
    GNANI_API_KEY,
    GNANI_STT_LANGUAGE,
    GNANI_TTS_VOICE,
    GNANI_TTS_MODEL,
    GNANI_TTS_LANGUAGE,
    PATHS
)
from src.agent.agent import SupportAgent

class VoiceAgent:
    def __init__(self):
        if not GNANI_API_KEY:
            raise ValueError("GNANI_API_KEY is not set in .env")
            
        # Initialize Gnani clients
        self.stt = GnaniSTTClient()
        self.tts = GnaniTTSClient()
        self.agent = SupportAgent()
        self.outputs_dir = PATHS.outputs
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

    def transcribe(self, audio_path: str) -> str:
        """Converts speech to text using Gnani STT."""
        try:
            result = self.stt.transcribe(audio_path, language_code=GNANI_STT_LANGUAGE)
            if isinstance(result, dict) and "transcript" in result:
                return result["transcript"].strip()
            return str(result).strip()
        except Exception as e:
            print(f"STT Error: {e}")
            return ""

    def synthesize(self, text: str, output_filename: str = "reply.wav", output_dir: Path = None) -> Path:
        """Converts text to speech using Gnani TTS."""
        if output_dir is None:
            output_dir = PATHS.outputs
        output_path = output_dir / output_filename
        
        try:
            self.tts.synthesize(
                text,
                voice=GNANI_TTS_VOICE,
                model=GNANI_TTS_MODEL,
                language=GNANI_TTS_LANGUAGE,
                output_file=str(output_path)
            )
            return output_path
        except Exception as e:
            print(f"TTS Error: {e}")
            return None

    def voice_turn(self, audio_path: str, output_filename: str = "reply.wav", output_dir: Path = None):
        """
        Process one full turn: Audio -> Text -> Agent -> Text -> Audio
        Returns a tuple: (transcribed_text, agent_reply_text, output_audio_path)
        """
        print(f"Listening to {audio_path}...")
        heard = self.transcribe(audio_path)
        
        if not heard:
            reply = "Sorry, I didn't catch that. Could you say it again?"
            out_path = self.synthesize(reply, output_filename, output_dir)
            return "", reply, out_path

        print(f"Customer said: {heard}")
        
        # Pass the text to our Step 3 Agent
        reply = self.agent.chat(heard)
        print(f"Agent replied: {reply}")
        
        # Speak the reply
        print("Generating voice output...")
        out_path = self.synthesize(reply, output_filename, output_dir)
        
        return heard, reply, out_path