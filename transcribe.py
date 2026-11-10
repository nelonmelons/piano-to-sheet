import sys, time, torch
import librosa
from piano_transcription_inference import PianoTranscription, sample_rate

audio_path = sys.argv[1] if len(sys.argv) > 1 else "audio.wav"
out_path = sys.argv[2] if len(sys.argv) > 2 else "transcription.mid"

audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
transcriptor = PianoTranscription(device=device, checkpoint_path=None)
t0 = time.time()
result = transcriptor.transcribe(audio, out_path)
print(f"done in {time.time()-t0:.1f}s on {device}; {len(result['est_note_events'])} notes, {len(result['est_pedal_events'])} pedal events -> {out_path}")
