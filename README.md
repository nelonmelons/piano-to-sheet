# piano-to-sheet

Turn a piano piece into sheet music you can open in MuseScore. There are two ways in:

1. **From a falling-notes video** (`video2midi.py`): reads the colored bars off a Synthesia-style piano video frame by frame.
2. **From audio** (`yt2score.py`): runs ByteDance's [piano transcription model](https://github.com/qiuqiangkong/piano_transcription_inference) on a YouTube link or an audio file, then uses the same cleanup and export.

I built this because I wanted to learn YOASOBI's *Halzion* and couldn't find sheet music for it anywhere. You can read more in [the write-up](https://nelsonsiu.fun/writing/halzion).

## Setup

```bash
pip install -r requirements.txt
```

MuseScore 4 is optional. You only need it for `--render`.

## From a video

Put the video next to the script as `video.mp4`, then run:

```bash
python3 video2midi.py calibrate   # find the 88 keys -> calibration.json / .png
python3 video2midi.py detect      # track the falling bars -> notes.json
python3 video2midi.py export --format musicxml --bpm 132
```

The constants at the top of `video2midi.py` (hit line, bar colors, scroll speed) are tuned for one video. Check `calibration.png` and adjust them for yours. `check_keylit.py` cross-checks the detected note onsets against the frames where the keys light up.

## From audio

```bash
python3 yt2score.py "https://www.youtube.com/watch?v=..." --render
```

This detects the tempo, caps the pedal-extended note lengths, splits the notes between the two hands, and writes MusicXML to `out/<title>/`.

## Known gaps

- The left/right hand split struggles in very dense passages.
- The left hand doesn't switch clefs, so its high passages show up as ledger lines in bass clef.
