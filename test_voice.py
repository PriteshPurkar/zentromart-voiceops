# test_voice.py
import sys
from pathlib import Path
from src.voice.gnani_voice import VoiceAgent

def main():
    # Point this to your actual .wav file
    audio_file = Path("sample_outputs/test_customer.wav")
    
    if not audio_file.exists():
        print(f"Error: Audio file not found at {audio_file.absolute()}")
        print("Please record a short message (e.g., 'Hi, my order is 69520867 and my pincode is 160017') and save it there.")
        sys.exit(1)

    print("Initializing Voice Agent (this may take a moment to load models)...")
    agent = VoiceAgent()

    # Run the voice turn
    heard_text, reply_text, output_audio = agent.voice_turn(
        audio_path=str(audio_file),
        output_filename="test_reply.wav"
    )

    print("\n" + "="*50)
    print("TRANSCRIPTION (What Gnani heard):")
    print(heard_text)
    print("\nAGENT REPLY (Text):")
    print(reply_text)
    print(f"\nAGENT AUDIO SAVED TO:")
    print(output_audio)
    print("="*50)

if __name__ == "__main__":
    main()