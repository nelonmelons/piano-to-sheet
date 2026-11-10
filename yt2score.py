#!/usr/bin/env python3
"""yt2score: YouTube URL (or local audio file) -> MuseScore-ready MusicXML + MIDI.

Downloads/transcribes piano audio with ByteDance piano_transcription_inference,
detects tempo from onset spacing, caps pedal-sustained note lengths (keeping
the true sounding end for rest-filling), then hands off to video2midi.py's
`export` subcommand for quantization/hand-splitting/notation.

Run with the piano_transcription_inference venv python (needs torch + that
package); video2midi.py itself is invoked with system python3.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

import mido
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
VIDEO2MIDI = Path(__file__).resolve().parent / "video2midi.py"
MSCORE = next(
    (p for p in ["/Applications/MuseScore 4.app/Contents/MacOS/mscore", shutil.which("mscore4"), shutil.which("mscore"), shutil.which("musescore")] if p and Path(p).exists()),
    "mscore",
)

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def note_name(midi: int) -> str:
    """MIDI number -> note name, using the convention MIDI 60 = C4."""
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def slugify(text: str) -> str:
    """ASCII-only, dash-separated slug for a directory/file name."""
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    text = re.sub(r"[\s_-]+", "-", text)
    return text[:60].strip("-") or "track"


def is_url(source: str) -> bool:
    """Treat http(s):// sources as URLs to fetch; anything else is a local path."""
    return source.startswith("http://") or source.startswith("https://")


def _ytdlp() -> List[str]:
    """yt-dlp command prefix; use Node as the JS runtime only if it's installed."""
    return ["yt-dlp"] + (["--js-runtimes", "node"] if shutil.which("node") else [])


CHECKPOINT = Path.home() / "piano_transcription_inference_data" / "note_F1=0.9677_pedal_F1=0.9186.pth"
CHECKPOINT_URL = "https://zenodo.org/record/4034264/files/CRNN_note_F1%3D0.9677_pedal_F1%3D0.9186.pth?download=1"


def ensure_checkpoint() -> None:
    """Download the ~170 MB model once (the package's own downloader needs wget)."""
    if CHECKPOINT.exists() and CHECKPOINT.stat().st_size >= 1.6e8:
        return
    import urllib.request
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading the transcription model (~170 MB, one time) to {CHECKPOINT} ...")
    urllib.request.urlretrieve(CHECKPOINT_URL, CHECKPOINT)


def get_youtube_title(url: str) -> str:
    """yt-dlp --print title, without downloading anything."""
    out = subprocess.run(
        _ytdlp() + ["--print", "title", url],
        capture_output=True, text=True, check=True,
    )
    return out.stdout.strip().splitlines()[-1]


def download_audio(url: str, out_dir: Path) -> Path:
    """yt-dlp-extract the URL's audio as WAV into out_dir/audio.wav."""
    out_tmpl = str(out_dir / "audio.%(ext)s")
    subprocess.run(
        _ytdlp() + ["-x", "--audio-format", "wav", "-o", out_tmpl, url],
        check=True,
    )
    wav_path = out_dir / "audio.wav"
    if not wav_path.exists():
        sys.exit(f"ERROR: expected {wav_path} after yt-dlp download but it's missing")
    return wav_path


def transcribe(audio_path: Path, out_path: Path) -> None:
    """Run ByteDance piano_transcription_inference on CPU, mirroring transcribe.py."""
    import torch  # noqa: F401 -- import here so --help works without torch installed
    import librosa
    from piano_transcription_inference import PianoTranscription, sample_rate

    # librosa.load (not the package's own load_audio, which is broken on new librosa).
    audio, _ = librosa.load(str(audio_path), sr=sample_rate, mono=True)
    ensure_checkpoint()
    transcriptor = PianoTranscription(device="cpu", checkpoint_path=None)
    result = transcriptor.transcribe(audio, str(out_path))
    print(
        f"Transcribed {len(result['est_note_events'])} notes, "
        f"{len(result['est_pedal_events'])} pedal events -> {out_path}"
    )


def parse_transcription_midi(mid_path: Path) -> List[dict]:
    """mido-iterate a type-1 MIDI (note_on vel=0 doubles as note-off) into raw
    note events with absolute start/end seconds. Iterating a mido.MidiFile
    directly (rather than walking ticks by hand) converts ticks -> seconds
    using its own tempo map, so this is correct even with tempo changes.
    """
    mid = mido.MidiFile(str(mid_path))
    active: dict = {}  # pitch -> FIFO list of (start_s, velocity) for repeated same-pitch attacks
    events = []
    t = 0.0
    for msg in mid:
        t += msg.time
        if msg.type != "note_on":
            continue
        if msg.velocity > 0:
            active.setdefault(msg.note, []).append((t, msg.velocity))
        else:
            starts = active.get(msg.note)
            if not starts:
                continue
            start_s, velocity = starts.pop(0)
            events.append({"pitch": msg.note, "velocity": velocity, "start_s": start_s, "sound_end_s": t})
    events.sort(key=lambda e: e["start_s"])
    return events


def _phase_fraction(starts: np.ndarray, step_s: float, tol_s: float, n_phases: int) -> Tuple[float, float]:
    """Best phase (0..step_s) and the fraction of onsets landing within tol_s of it."""
    best_phase, best_frac = 0.0, -1.0
    for i in range(n_phases):
        phase = step_s * i / n_phases
        residual = (starts - phase) % step_s
        residual = np.minimum(residual, step_s - residual)  # circular distance
        frac = float(np.mean(residual <= tol_s))
        if frac > best_frac:
            best_frac, best_phase = frac, phase
    return best_phase, best_frac


def detect_bpm(starts: np.ndarray) -> float:
    """Scan 60..200 BPM (0.5 steps) for the 16th-note grid that best fits onset
    times (fraction within 25ms); among candidates within 0.01 of the best
    fit, pick the lowest BPM to avoid a double-time false positive.
    """
    candidates = []
    for i in range(int(round((200.0 - 60.0) / 0.5)) + 1):
        bpm = 60.0 + 0.5 * i
        step_s = (60.0 / bpm) / 4.0  # seconds per 16th note
        _, frac = _phase_fraction(starts, step_s, tol_s=0.025, n_phases=60)
        candidates.append((bpm, frac))

    candidates.sort(key=lambda c: -c[1])
    print("Top 5 BPM candidates (onset-grid fit):")
    for bpm_c, frac_c in candidates[:5]:
        print(f"  {bpm_c:6.1f} BPM  fit={frac_c * 100:.1f}%")

    best_frac = candidates[0][1]
    near_best = [bpm_c for bpm_c, frac_c in candidates if best_frac - frac_c <= 0.01]
    chosen = min(near_best)
    chosen_frac = next(frac_c for bpm_c, frac_c in candidates if bpm_c == chosen)
    print(f"Chosen BPM: {chosen:g} (fit={chosen_frac * 100:.1f}%)")
    if best_frac < 0.90:
        print(
            "WARNING: best onset-grid fit is below 90% -- this performance doesn't "
            "look metronomic, so the quantized score will be rough. Consider passing --bpm."
        )
    return chosen


def build_notes_json(events: List[dict], bpm: float, max_held_beats: float, out_path: Path) -> None:
    """Cap each note's duration at max_held_beats (undoes the pedal-sustain
    stretch), keeping the true sounding end as sound_end_s for video2midi's
    same-hand rest-filling to extend back into (never past it).
    """
    cap_s = max_held_beats * 60.0 / bpm
    notes = []
    for e in events:
        sound_end = e["sound_end_s"]
        duration = min(sound_end - e["start_s"], cap_s)
        notes.append({
            "pitch": e["pitch"], "name": note_name(e["pitch"]), "velocity": e["velocity"],
            "start_s": e["start_s"], "duration_s": duration, "sound_end_s": sound_end,
        })
    out_path.write_text(json.dumps({"notes": notes}, indent=2))
    print(f"Wrote {len(notes)} notes -> {out_path}")


def run_video2midi_export(notes_path: Path, out_mid: Path, bpm: float, title: str, fill_rest_max_beats: float) -> None:
    """Shell out to video2midi.py export with the same Python interpreter."""
    cmd = [
        sys.executable, str(VIDEO2MIDI), "export",
        "--notes", str(notes_path), "--out", str(out_mid),
        "--bpm", str(bpm), "--format", "both", "--title", title,
        "--fill-rest-max-beats", str(fill_rest_max_beats),
    ]
    print(f"\nRunning: {' '.join(cmd)}")
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        sys.exit(f"ERROR: video2midi.py export failed with exit code {proc.returncode}")


def render_pages(musicxml_path: Path, out_dir: Path) -> None:
    """Render PNG pages for the MusicXML via MuseScore 4's CLI."""
    png_out = out_dir / "page.png"
    cmd = [MSCORE, "-o", str(png_out), str(musicxml_path)]
    print(f"Running: {' '.join(cmd)}")
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        sys.exit(f"ERROR: MuseScore render failed with exit code {proc.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="yt2score: YouTube/audio -> MuseScore MusicXML + MIDI")
    parser.add_argument("source", help="YouTube URL or local audio file path")
    parser.add_argument("--out-dir", type=str, default=None, help="output directory (default ./out/<slug>)")
    parser.add_argument("--bpm", type=float, default=None, help="override the detected tempo")
    parser.add_argument("--title", type=str, default=None, help="score title (default: YouTube title, or file stem)")
    parser.add_argument("--max-held-beats", type=float, default=0.25, help="cap each note's duration at this many beats")
    parser.add_argument("--fill-rest-max-beats", type=float, default=4.0, help="passed through to video2midi.py's --fill-rest-max-beats")
    parser.add_argument("--render", action="store_true", help="render PNG pages with MuseScore 4")
    parser.add_argument("--skip-transcribe", action="store_true", help="reuse an existing transcription.mid in out-dir")
    args = parser.parse_args()

    url = is_url(args.source)
    src_path = None
    if url:
        title = args.title or get_youtube_title(args.source)
    else:
        src_path = Path(args.source).expanduser().resolve()
        if not src_path.exists():
            sys.exit(f"ERROR: {src_path} not found")
        title = args.title or src_path.stem

    slug = slugify(title)
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (SCRIPT_DIR / "out" / slug)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Title: {title}")
    print(f"Output directory: {out_dir}")

    transcription_path = out_dir / "transcription.mid"
    if args.skip_transcribe:
        if not transcription_path.exists():
            sys.exit(f"ERROR: --skip-transcribe given but {transcription_path} doesn't exist")
        print(f"Reusing existing transcription: {transcription_path}")
    else:
        if url:
            audio_path = download_audio(args.source, out_dir)
        else:
            audio_path = out_dir / f"audio{src_path.suffix or '.wav'}"
            if audio_path.resolve() != src_path:
                shutil.copy(src_path, audio_path)
        transcribe(audio_path, transcription_path)

    events = parse_transcription_midi(transcription_path)
    if not events:
        sys.exit(f"ERROR: no notes parsed from {transcription_path}")
    print(f"Parsed {len(events)} note events from {transcription_path}")

    starts = np.array([e["start_s"] for e in events])
    bpm = args.bpm if args.bpm is not None else detect_bpm(starts)
    print(f"Using BPM: {bpm:g}")

    notes_path = out_dir / "notes.json"
    build_notes_json(events, bpm, args.max_held_beats, notes_path)

    out_mid = out_dir / f"{slug}.mid"
    run_video2midi_export(notes_path, out_mid, bpm, title, args.fill_rest_max_beats)

    musicxml_path = out_mid.with_suffix(".musicxml")
    if args.render:
        render_pages(musicxml_path, out_dir)

    print()
    print(f"MusicXML: {musicxml_path}")
    print(f"MIDI:     {out_mid}")
    print("Open the .musicxml in MuseScore 4 (not the .mid).")


if __name__ == "__main__":
    main()
