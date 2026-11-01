# run.py
import argparse
import json
from pathlib import Path
import shutil
from src.voice.gnani_voice import VoiceAgent
from src.config import PATHS

def main():
    parser = argparse.ArgumentParser(description="Batch process customer audio files.")
    parser.add_argument("audio_dir", type=str, help="Path to folder containing .wav files")
    args = parser.parse_args()

    input_dir = Path(args.audio_dir)
    if not input_dir.exists():
        print(f"Error: Directory {input_dir} not found.")
        return

    # Create a dedicated output folder for this batch
    batch_out_dir = PATHS.outputs / f"batch_{input_dir.name}"
    batch_out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading Voice Agent...")
    agent = VoiceAgent()
    
    # Find all .wav files
    audio_files = list(input_dir.glob("*.wav"))
    if not audio_files:
        print(f"No .wav files found in {input_dir}")
        return

    results_summary = []

    for audio_file in audio_files:
        print(f"\nProcessing: {audio_file.name}")
        
        # We need a unique output filename for this specific audio
        out_wav_name = f"{audio_file.stem}_reply.wav"
        
        # Run the voice turn
        heard, reply, out_audio_path = agent.voice_turn(
            audio_path=str(audio_file),
            output_filename=out_wav_name,
            output_dir=batch_out_dir
        )
        
        # Get the retrieved chunks for this specific turn
        # We do a quick retrieval just to log what the RAG system found
        chunks = agent.agent.retriever.search(heard, k=3)
        sources = [c.chunk.section_id for c in chunks]
        
        # Save the text transcript and metadata
        result_data = {
            "audio_file": audio_file.name,
            "transcription": heard,
            "agent_reply": reply,
            "retrieved_sources": sources,
            "reply_audio_file": out_wav_name
        }
        results_summary.append(result_data)
        
        # Save individual text file
        txt_path = batch_out_dir / f"{audio_file.stem}_transcript.json"
        with open(txt_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, indent=2)
            
        print(f"Saved reply to {out_audio_path}")

    # Save a combined summary
    summary_path = batch_out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)
        
    print(f"\nBatch processing complete! Check {batch_out_dir}")

if __name__ == "__main__":
    main()