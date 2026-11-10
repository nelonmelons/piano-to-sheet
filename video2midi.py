#!/usr/bin/env python3
"""video2midi: extract notes from a Synthesia-style falling-note piano video.

Stage 1 (working): keyboard calibration -- locate every white/black key's
pixel x-range and build a MIDI note lookup + annotated debug image.
Stage 2/3 (working): note-bar detection over time -> raw note events.
Stage 4 (stub): export detected events to a .mid file via mido.

Run `python3 video2midi.py calibrate` to produce calibration.json/.png.
Run `python3 video2midi.py detect` to produce notes.json.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict, field
from fractions import Fraction
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import mido
import numpy as np

# --------------------------------------------------------------------------
# CONFIG -- module-level constants. Tune these per-video; defaults below are
# measured values for the current video.mp4.
# --------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent

VIDEO_PATH = "video.mp4"  # resolved relative to SCRIPT_DIR
VIDEO_WIDTH, VIDEO_HEIGHT = 1920, 1080  # expected frame size (informational)
VIDEO_FPS = 60.0  # expected fps (informational; actual fps read from file)
VIDEO_FRAME_COUNT = 13287  # expected frame count (informational)
VIDEO_DURATION_S = 221.0  # expected duration in seconds (informational)

KEYBOARD_Y_TOP = 709  # top edge of the keyboard (black key tops)
KEYBOARD_Y_BOTTOM = 944  # bottom edge of the keyboard (white key bottoms)
BLACK_KEY_Y_BOTTOM = 865  # black keys end here; below this it's white-only

WHITE_KEY_SCAN_ROWS = (895, 915)  # rows over the white-key-only area
WHITE_GAP_GRAY_MAX = 120  # columns with mean gray below this are a gap

BLACK_KEY_SCAN_ROWS = (790, 810)  # rows over the black-key area
BLACK_KEY_GRAY_MAX = 60  # columns with mean gray below this are a black key
BLACK_KEY_MIN_RUN_WIDTH = 6  # minimum px width to count as a black-key run
BLACK_KEY_HALF_WIDTH = 10  # default half-width when a run isn't detected
BLACK_KEY_BOUNDARY_TOL = 12  # px tolerance to match a run to an expected key

WHITE_BOUNDARY_EDGE_TOL = 5  # px tolerance for adding x=0/width-1 boundaries

FIRST_MIDI_NOTE = 21  # A0, the lowest key on an 88-key piano
EXPECTED_WHITE_KEYS = 52
EXPECTED_BLACK_KEYS = 36

# y at which the key lights (bar bottom reaches the bottom of the dark-red
# band); measured from key-lit cross-check, 0.034 s x 438 px/s below the ROI
# edge. This is the TIMING reference for t_on/t_off; masking still stops at
# NOTE_ROI_Y_MAX.
HIT_LINE_Y = 704
NOTE_ROI_Y_MAX = 688  # bars are only searched for in rows 0..NOTE_ROI_Y_MAX

SAT_MIN = 150  # HSV saturation threshold for note-bar color detection
VAL_MIN = 150  # HSV value threshold for note-bar color detection
# All falling bars are the same instrument/colour -- one hue band, no track
# split by colour.
NOTE_HUE = (5, 25)  # red-orange .. yellow-orange
TRACK_NAME = "piano"

SCROLL_PX_PER_SEC = 438.0  # measured 7.3 px/frame @ 60fps, constant
SCROLL_AUTO = True  # let a later stage re-measure scroll speed automatically

MAX_SECONDS = 10.0  # Stage 2/3: caps how much video `detect` scans
FRAME_STEP = 2  # Stage 2/3: frame stride for `detect`
MIN_BAR_H = 6  # minimum lit-run height to count as a note-bar observation (px)
STRIP_HALF_W_WHITE = 6  # column-strip half-width for a white key's bar (~28px wide)
STRIP_HALF_W_BLACK = 3  # column-strip half-width for a black key's bar (~20px wide)
LIT_FRACTION_MIN = 0.6  # a row counts as "lit" if this fraction of its strip is masked
GLOW_MIN_H = 30  # below this, a bottom-clipped run is flame glow or an already-tracked tail
MATCH_TOL_PX = 12  # px tolerance between predicted and observed bar top/bottom
TRACK_MAX_MISSES = 3  # steps a track can go unmatched before it's closed
# A real bar's bottom/top move at a constant SCROLL_PX_PER_SEC, so its per-frame
# t_on/t_off estimates agree tightly; a fixed-in-screen-space artifact (title
# card hole, flame glow) drifts instead. MAD is the spread test.
T_ON_SPREAD_MAX_S = 0.03  # max allowed median-absolute-deviation of t_on/t_off estimates
# 17px at SCROLL_PX_PER_SEC; tuned to this video -- the shortest real note seen
# is A5 at 0.048s, and a 32nd note at 120bpm is 0.0625s, so this has headroom.
MIN_NOTE_DURATION_S = 0.02
DEBUG_EVERY_N_FRAMES = 30  # write an annotated debug frame this often

CALIB_FRAME_TIME_S = 60.0  # default single-frame timestamp (with --time)
CALIB_SAMPLE_TIMES_S = (0.5, 20, 45, 60, 90, 120, 150, 180, 210)  # multi-frame default
DEBUG_DIR = "debug"  # relative to SCRIPT_DIR

# Stage 4: export
BPM = 132.0  # user said ~130; onset grid fit gives 132.0 exactly (99.3% on 16th grid, 5.8 ms RMS) vs 52% at 130
QUANT_GRID = 16  # subdivisions per whole note (16 = 16th notes; 32 allowed)
TICKS_PER_BEAT = 480
LEAD_IN_BEATS = 0
MIDI_VELOCITY = 80
MIDI_PATH = "output.mid"

# Stage 4: hand separation + rest filling
HAND_MAX_SPAN = 14  # semitones a single hand can plausibly cover at once
HAND_INIT_LEFT = 48  # C3 -- starting left-hand centroid before any notes are seen
HAND_INIT_RIGHT = 72  # C5 -- starting right-hand centroid
HAND_EMA_ALPHA = 0.3  # how fast each hand's centroid follows where it just played
HAND_CROSS_PENALTY = 15.0  # beam: per new note that lands on the "wrong side" of a still-held note of the other hand
HAND_SPLIT_MODE = "beam"  # {beam, greedy}: how _split_hands picks each onset group's left/right split
HAND_SPLIT_BEAM_WIDTH = 48  # beam: partial assignments kept per onset group
HAND_SPAN_PENALTY = 60.0  # beam: flat cost when one hand's simultaneous notes span > HAND_MAX_SPAN
HAND_SPAN_EXCESS_W = 6.0  # beam: extra cost per semitone beyond HAND_MAX_SPAN
HAND_SPAN_SOFT_FROM = 12  # beam: spans above this (but <= HAND_MAX_SPAN) are mildly costly
HAND_SPAN_SOFT_W = 0.6  # beam: cost per semitone of span above HAND_SPAN_SOFT_FROM
HAND_MAX_NOTES = 5  # beam: a hand has five fingers
HAND_COUNT_PENALTY = 40.0  # beam: cost per simultaneous note beyond HAND_MAX_NOTES in one hand
HAND_REACH_FREE = 5  # beam: semitones a hand can move between groups without any cost
HAND_MOVE_W = 1.0  # beam: cost per semitone of hand jump beyond HAND_REACH_FREE (at zero time gap)
HAND_MOVE_TIME_REF = 1.0  # beam: jump cost is scaled by REF/(REF+gap_beats), so fast leaps cost most
HAND_LH_COMFORT_MAX = 72  # C5: LH notes above this are "better played with the right hand"
HAND_RH_COMFORT_MIN = 48  # C3: RH notes below this are "better played with the left hand"
HAND_REGISTER_W = 0.8  # beam: cost per semitone outside the comfortable register, per note
HAND_IDLE_SPAN = 9  # beam: one hand idle while the other stretches wider than this ...
HAND_IDLE_W = 0.5  # ... costs this per extra semitone
HAND_CENTER_W = 0.05  # beam: tiny pull of each note toward its hand's nominal centre (tie-breaker)
FILL_REST_MAX_BEATS = 0.5  # close an inter-note gap in the same hand up to this long

EXPORT_FORMAT = "both"  # {midi, musicxml, both}
AUTO_DOWNBEAT = True  # search for a metrically-sensible lead-in unless --lead-in-beats is explicit
KEY_CHANGE_PENALTY = 12  # Viterbi cost to change key signature between consecutive bars
SCORE_TITLE = "Halzion (ハルジオン) – YOASOBI, fruit Arr. (piano transcription)"
SCORE_COMPOSER = ""

OTTAVA_ENABLED = True
# Ledger-line-cost Viterbi (see _apply_ottava): pick, per 0.25-beat slot, the
# notation (plain vs. shifted-an-octave) that minimises total ledger-line
# load, subject to a fixed penalty for switching into a shifted run.
# A dead zone excuses the first few ledger lines for free (engravers tolerate
# 1-2 ledger lines without reaching for a bracket); only the excess beyond it
# is costed, squared.
OTTAVA_FREE_LEDGERS_ABOVE = 2.0  # ledger lines above a staff that cost nothing
OTTAVA_FREE_LEDGERS_BELOW = 1.0  # ledger lines below a staff that cost nothing
OTTAVA_SWITCH_PENALTY = 16.0  # ledger-line-equivalents charged for each bracket start
OTTAVA_MIN_GAP_BEATS = 2.0  # spans/regions closer than this are merged into one
OTTAVA_MIN_SPAN_BEATS = 2.0  # drop a region/span this short unless it merges with a neighbour
OTTAVA_SLOT_BEATS = 0.25
OTTAVA_15MA_EXTRA = 0.5  # a 15ma bracket costs OTTAVA_SWITCH_PENALTY * (1 + this): only worth it when 8va still leaves many ledger lines

WHITE_PITCH_CLASSES = {0, 2, 4, 5, 7, 9, 11}  # C D E F G A B
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


@dataclass
class KeyRegion:
    """One piano key's identity and pixel span."""

    midi: int
    name: str
    is_black: bool
    x0: int
    x1: int
    x_center: int


@dataclass
class Track:
    """An in-progress or closed note-bar track across frames."""

    pitch: int
    track_id: int
    last_y_bottom: float
    last_y_top: float
    last_frame_time: float
    last_frame_idx: int
    misses: int = 0
    frames_seen: int = 0
    t_on_estimates: List[float] = field(default_factory=list)
    t_off_estimates: List[float] = field(default_factory=list)
    heights: List[int] = field(default_factory=list)
    bottom_obs: List[Tuple[float, float]] = field(default_factory=list)  # (time, y_bottom), unclipped only


@dataclass
class Note:
    """A finished detected note event."""

    pitch: int
    name: str
    start_s: float
    duration_s: float
    track: str = TRACK_NAME
    truncated: bool = False  # True if the note's end lies beyond the scanned window


def note_name(midi: int) -> str:
    """MIDI number -> note name, using the convention MIDI 60 = C4."""
    pitch = NOTE_NAMES[midi % 12]
    octave = midi // 12 - 1
    return f"{pitch}{octave}"


def _find_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Group a boolean array into (start, end) index runs of consecutive True."""
    runs = []
    start = None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def _min_col_means(frames: List[np.ndarray], y0: int, y1: int) -> np.ndarray:
    """Per-column mean gray over rows y0..y1, minimum across all sample frames.

    A key that is dark (or lit up) in any sample frame is treated as dark --
    this catches black keys that happen to be pressed/lit in a single frame.
    """
    means = [cv2.cvtColor(f[y0 : y1 + 1], cv2.COLOR_BGR2GRAY).mean(axis=0) for f in frames]
    return np.min(np.stack(means, axis=0), axis=0)


def find_white_boundaries(frames: List[np.ndarray]) -> List[int]:
    """Find x positions of the dark gaps between white keys, plus frame edges."""
    y0, y1 = WHITE_KEY_SCAN_ROWS
    col_means = _min_col_means(frames, y0, y1)
    gap_mask = col_means < WHITE_GAP_GRAY_MAX
    boundaries = [round((s + e) / 2) for s, e in _find_runs(gap_mask)]

    width = frames[0].shape[1]
    if not boundaries or boundaries[0] > WHITE_BOUNDARY_EDGE_TOL:
        boundaries.insert(0, 0)
    if not boundaries or (width - 1 - boundaries[-1]) > WHITE_BOUNDARY_EDGE_TOL:
        boundaries.append(width - 1)
    return sorted(set(boundaries))


def find_black_runs(frames: List[np.ndarray]) -> List[Tuple[int, int]]:
    """Find x-ranges of black keys as dark, sufficiently-wide column runs."""
    y0, y1 = BLACK_KEY_SCAN_ROWS
    col_means = _min_col_means(frames, y0, y1)
    mask = col_means < BLACK_KEY_GRAY_MAX
    runs = _find_runs(mask)
    return [(s, e) for s, e in runs if (e - s + 1) >= BLACK_KEY_MIN_RUN_WIDTH]


def assign_white_midi(count: int) -> List[int]:
    """Walk MIDI numbers from FIRST_MIDI_NOTE, collecting the white pitch classes."""
    result = []
    m = FIRST_MIDI_NOTE
    while len(result) < count:
        if m % 12 in WHITE_PITCH_CLASSES:
            result.append(m)
        m += 1
    return result


def build_black_keys(
    boundaries: List[int], white_midis: List[int], black_runs: List[Tuple[int, int]]
) -> Tuple[List[Tuple[int, int, int]], int]:
    """Compute (midi, x0, x1) for each black key; refine x-range from detected runs.

    A black key sits on the shared boundary between two white keys whose MIDI
    numbers differ by 2. Returns the keys plus how many were refined by a
    detected run (vs. left at the default +/-BLACK_KEY_HALF_WIDTH range).
    """
    black_keys = []
    refined = 0
    for i in range(len(white_midis) - 1):
        if white_midis[i + 1] - white_midis[i] != 2:
            continue
        boundary = boundaries[i + 1]
        midi = white_midis[i] + 1
        x0, x1 = boundary - BLACK_KEY_HALF_WIDTH, boundary + BLACK_KEY_HALF_WIDTH
        for rs, re in black_runs:
            if abs((rs + re) / 2 - boundary) <= BLACK_KEY_BOUNDARY_TOL:
                x0, x1 = rs, re
                refined += 1
                break
        black_keys.append((midi, x0, x1))
    return black_keys, refined


def build_key_regions(
    boundaries: List[int],
    white_midis: List[int],
    black_keys: List[Tuple[int, int, int]],
    width: int,
) -> List[KeyRegion]:
    """Turn white boundaries + black key spans into the full KeyRegion list."""
    regions = []
    n_white = len(white_midis)
    for i, midi in enumerate(white_midis):
        x0 = boundaries[i]
        x1 = width - 1 if i == n_white - 1 else boundaries[i + 1] - 1
        regions.append(KeyRegion(midi, note_name(midi), False, x0, x1, (x0 + x1) // 2))
    for midi, x0, x1 in black_keys:
        regions.append(KeyRegion(midi, note_name(midi), True, x0, x1, (x0 + x1) // 2))
    return sorted(regions, key=lambda r: r.midi)


def build_x_to_note(regions: List[KeyRegion], width: int) -> np.ndarray:
    """Fill a width-long array mapping each x column to its MIDI note.

    White keys are filled first, then black keys overwrite their x-ranges --
    this matches how the keys visually overlap.
    """
    x_to_note = np.full(width, -1, dtype=int)
    for r in regions:
        if not r.is_black:
            x_to_note[max(r.x0, 0) : min(r.x1, width - 1) + 1] = r.midi
    for r in regions:
        if r.is_black:
            x_to_note[max(r.x0, 0) : min(r.x1, width - 1) + 1] = r.midi
    return x_to_note


def draw_calibration_image(
    frame: np.ndarray, regions: List[KeyRegion], boundaries: List[int]
) -> np.ndarray:
    """Draw calibration overlays (boundaries, black-key boxes, hit line, labels)."""
    img = frame.copy()
    width = img.shape[1]
    font = cv2.FONT_HERSHEY_SIMPLEX

    for x in boundaries:
        cv2.line(img, (x, KEYBOARD_Y_TOP), (x, KEYBOARD_Y_BOTTOM), (0, 255, 0), 1)

    for r in regions:
        if r.is_black:
            cv2.rectangle(
                img, (r.x0, KEYBOARD_Y_TOP), (r.x1, BLACK_KEY_Y_BOTTOM), (255, 0, 255), 1
            )
            cv2.putText(
                img, r.name, (r.x0, KEYBOARD_Y_TOP + 12), font, 0.35, (255, 0, 255), 1
            )
        else:
            cv2.putText(
                img, r.name, (r.x0 + 2, KEYBOARD_Y_BOTTOM - 8), font, 0.35, (0, 200, 0), 1
            )
        if r.name.startswith("C") and not r.is_black:
            cv2.putText(
                img, r.name, (r.x0 + 2, KEYBOARD_Y_TOP - 20), font, 0.55, (0, 165, 255), 2
            )

    cv2.line(img, (0, HIT_LINE_Y), (width - 1, HIT_LINE_Y), (255, 0, 0), 1)
    cv2.line(img, (0, NOTE_ROI_Y_MAX), (width - 1, NOTE_ROI_Y_MAX), (255, 255, 0), 1)

    n_white = sum(1 for r in regions if not r.is_black)
    n_black = sum(1 for r in regions if r.is_black)
    midi_lo, midi_hi = min(r.midi for r in regions), max(r.midi for r in regions)
    legend = [
        f"white keys: {n_white}  black keys: {n_black}",
        f"MIDI {midi_lo} ({note_name(midi_lo)}) .. {midi_hi} ({note_name(midi_hi)})",
        f"scroll: {SCROLL_PX_PER_SEC:.1f} px/s",
    ]
    for i, line in enumerate(legend):
        cv2.putText(img, line, (10, 20 + 18 * i), font, 0.5, (255, 255, 255), 1)

    return img


def save_calibration_json(
    path: Path,
    video_path: Path,
    width: int,
    height: int,
    fps: float,
    frame_count: int,
    boundaries: List[int],
    regions: List[KeyRegion],
    x_to_note: np.ndarray,
) -> None:
    """Write calibration.json with everything a later stage needs to reuse."""
    data = {
        "video_path": str(video_path),
        "frame_width": width,
        "frame_height": height,
        "fps": fps,
        "frame_count": frame_count,
        "white_boundaries": boundaries,
        "key_regions": [asdict(r) for r in regions],
        "x_to_note": x_to_note.tolist(),
        "config": {
            "keyboard_y_top": KEYBOARD_Y_TOP,
            "keyboard_y_bottom": KEYBOARD_Y_BOTTOM,
            "black_key_y_bottom": BLACK_KEY_Y_BOTTOM,
            "hit_line_y": HIT_LINE_Y,
            "note_roi_y_max": NOTE_ROI_Y_MAX,
            "note_hue": NOTE_HUE,
            "sat_min": SAT_MIN,
            "val_min": VAL_MIN,
            "track_name": TRACK_NAME,
            "scroll_px_per_sec": SCROLL_PX_PER_SEC,
            "scroll_auto": SCROLL_AUTO,
        },
    }
    path.write_text(json.dumps(data, indent=2))


def load_calibration(path: Path) -> Tuple[List[KeyRegion], np.ndarray]:
    """Load calibration.json back into (KeyRegion list, x_to_note array).

    For use by later stages (note detection / export) so they don't need to
    re-run keyboard detection.
    """
    data = json.loads(Path(path).read_text())
    regions = [KeyRegion(**r) for r in data["key_regions"]]
    x_to_note = np.array(data["x_to_note"], dtype=int)
    return regions, x_to_note


def _resolve(path_str: str) -> Path:
    """Resolve a CLI path relative to the script dir unless it's absolute."""
    p = Path(path_str)
    return p if p.is_absolute() else SCRIPT_DIR / p


def _read_frame_at(cap: cv2.VideoCapture, fps: float, t: float) -> Optional[np.ndarray]:
    """Seek to timestamp t (seconds) and read that single frame."""
    frame_idx = int(round(t * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    return frame if ok else None


def cmd_calibrate(args: argparse.Namespace) -> None:
    """Stage 1: locate the keyboard's keys across sample frames and save the mapping."""
    video_path = _resolve(VIDEO_PATH)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"ERROR: could not open video at {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or VIDEO_FPS
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # --time forces a single frame; otherwise sample several frames and take
    # the per-column minimum brightness, so a black key that's lit up (and
    # therefore bright) in one frame is still caught as black in another.
    sample_times = [args.time] if args.time is not None else list(CALIB_SAMPLE_TIMES_S)
    frames = [f for f in (_read_frame_at(cap, fps, t) for t in sample_times) if f is not None]
    cap.release()
    if not frames:
        sys.exit(f"ERROR: could not read any calibration frames from {video_path}")
    print(f"Sampled {len(frames)}/{len(sample_times)} frames for calibration: {sample_times}")

    boundaries = find_white_boundaries(frames)
    n_white = len(boundaries) - 1
    spacing = np.median(np.diff(boundaries))
    print(f"White keys detected: {n_white} (expected {EXPECTED_WHITE_KEYS})")
    print(f"Median white key spacing: {spacing:.1f} px")
    if n_white != EXPECTED_WHITE_KEYS:
        print(
            "WARNING: white key count mismatch -- tune WHITE_KEY_SCAN_ROWS "
            "and/or WHITE_GAP_GRAY_MAX and re-run calibrate."
        )

    white_midis = assign_white_midi(n_white)
    black_runs = find_black_runs(frames)
    black_keys, refined = build_black_keys(boundaries, white_midis, black_runs)
    print(f"Black keys detected from image runs: {refined} ({len(black_keys)} expected/placed)")
    if len(black_keys) != EXPECTED_BLACK_KEYS:
        print(
            "WARNING: black key count mismatch -- tune BLACK_KEY_SCAN_ROWS "
            "and/or BLACK_KEY_GRAY_MAX and re-run calibrate."
        )

    regions = build_key_regions(boundaries, white_midis, black_keys, width)
    x_to_note = build_x_to_note(regions, width)

    print(f"\n{'MIDI':>4}  {'name':<4}  {'black?':<6}  x0-x1")
    for r in regions:
        print(f"{r.midi:>4}  {r.name:<4}  {'yes' if r.is_black else 'no':<6}  {r.x0}-{r.x1}")
    midi_lo, midi_hi = regions[0].midi, regions[-1].midi
    print(
        f"\n{len(regions)} keys, MIDI {midi_lo} ({note_name(midi_lo)}) .. "
        f"{midi_hi} ({note_name(midi_hi)}), white width ~= {spacing:.1f} px"
    )

    json_path = _resolve(args.json)
    save_calibration_json(json_path, video_path, width, height, fps, frame_count, boundaries, regions, x_to_note)
    print(f"\nWrote {json_path}")

    annotated = draw_calibration_image(frames[0], regions, boundaries)
    out_path = _resolve(args.out)
    cv2.imwrite(str(out_path), annotated)
    print(f"Wrote {out_path}")


def _lit_runs(lit: np.ndarray) -> List[Tuple[int, int]]:
    """Runs of True in a 1-D boolean array, bridging isolated single-row gaps.

    A lone False between two True rows is filled in first (anti-aliasing);
    any gap of 2+ rows is left alone and correctly splits the run.
    """
    bridged = lit.copy()
    if len(bridged) > 2:
        bridged[1:-1] |= lit[:-2] & lit[2:]
    return _find_runs(bridged)


def _detect_runs(frame: np.ndarray, regions: List[KeyRegion]) -> Tuple[dict, np.ndarray]:
    """HSV-threshold the ROI once, then per key take a narrow column strip over
    x_center and find vertical lit-row runs -- no contours, no cross-key bleed.
    """
    roi = frame[: NOTE_ROI_Y_MAX + 1]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower = (NOTE_HUE[0], SAT_MIN, VAL_MIN)
    upper = (NOTE_HUE[1], 255, 255)
    mask = cv2.inRange(hsv, lower, upper) > 0

    width = mask.shape[1]
    runs_by_pitch: dict = {}
    for r in regions:
        half = STRIP_HALF_W_BLACK if r.is_black else STRIP_HALF_W_WHITE
        x0, x1 = max(r.x_center - half, 0), min(r.x_center + half, width - 1)
        lit = mask[:, x0 : x1 + 1].mean(axis=1) > LIT_FRACTION_MIN
        kept = [(y0, y1) for y0, y1 in _lit_runs(lit) if (y1 - y0 + 1) >= MIN_BAR_H]
        if kept:
            runs_by_pitch[r.midi] = kept
    return runs_by_pitch, mask


def _match_track(
    tracks: List[Track], pitch: int, y_top: float, y_bottom: float, frame_time: float, claimed: set
) -> Optional[Track]:
    """Find the best unclaimed active track of this pitch whose predicted top or bottom matches.

    The bottom prediction is clamped at NOTE_ROI_Y_MAX because that's where a bar
    visually clips as it nears the hit line; without the clamp a clipped track's
    predicted bottom keeps advancing past 688 while the observed one stays pinned,
    so it never matches and the tail re-spawns a new track every step. The top
    prediction is only trusted once the track has already shown an unclipped top
    (last_y_top > 1). On a tie between two candidate tracks, prefer the one that
    matched via its top (a real, currently-visible bar) over a bottom-only match
    (which could be lingering glow at the clamped edge).
    """
    best, best_diff, best_via_top = None, None, False
    for t in tracks:
        if t.pitch != pitch or t.track_id in claimed:
            continue
        dt = frame_time - t.last_frame_time
        pred_bottom = min(t.last_y_bottom + SCROLL_PX_PER_SEC * dt, NOTE_ROI_Y_MAX)
        bottom_diff = abs(pred_bottom - y_bottom)
        top_diff = abs(t.last_y_top + SCROLL_PX_PER_SEC * dt - y_top) if t.last_y_top > 1 else None
        if top_diff is not None and top_diff <= bottom_diff:
            diff, via_top = top_diff, True
        else:
            diff, via_top = bottom_diff, False
        if diff > MATCH_TOL_PX:
            continue
        if best is None or diff < best_diff or (diff == best_diff and via_top and not best_via_top):
            best, best_diff, best_via_top = t, diff, via_top
    return best


def _mad(values: List[float]) -> float:
    """Median absolute deviation -- small for a real, scroll-consistent bar; large for a drifting artifact."""
    arr = np.array(values)
    return float(np.median(np.abs(arr - np.median(arr))))


def _measure_scroll_speed(closed_tracks: List[Track]) -> Optional[float]:
    """Least-squares fit y_bottom vs time per track; return the median slope."""
    slopes = []
    for t in closed_tracks:
        if len(t.bottom_obs) < 4:
            continue
        times = np.array([o[0] for o in t.bottom_obs])
        ys = np.array([o[1] for o in t.bottom_obs])
        a = np.vstack([times, np.ones(len(times))]).T
        slope, _ = np.linalg.lstsq(a, ys, rcond=None)[0]
        slopes.append(slope)
    return float(np.median(slopes)) if slopes else None


def _count_same_pitch_overlaps(notes: List[Note]) -> int:
    """Count pairs of same-pitch notes whose [start, start+duration) ranges overlap."""
    by_pitch: dict = {}
    for n in notes:
        by_pitch.setdefault(n.pitch, []).append(n)
    overlaps = 0
    for group in by_pitch.values():
        group = sorted(group, key=lambda n: n.start_s)
        for a, b in zip(group, group[1:]):
            if b.start_s < a.start_s + a.duration_s:
                overlaps += 1
    return overlaps


def _resolve_same_pitch_overlaps(candidates: List[Tuple[Note, int]]) -> Tuple[List[Note], int]:
    """Drop the lower-confidence side of any same-pitch time overlap.

    A real bar is one column, so two same-pitch notes overlapping in time is
    physically impossible -- one is a fragment (e.g. a spark punching a
    transient hole in the mask). Keep whichever had more track observations;
    on a tie keep the earlier start. Only adjacent (by start) pairs need
    checking: intervals sorted by start can't skip-overlap a non-adjacent one.
    """
    by_pitch: dict = {}
    for note, frames_seen in candidates:
        by_pitch.setdefault(note.pitch, []).append([note, frames_seen])
    dropped = 0
    result: List[Note] = []
    for group in by_pitch.values():
        group.sort(key=lambda ng: ng[0].start_s)
        changed = True
        while changed:
            changed = False
            for i in range(len(group) - 1):
                a_note, a_frames = group[i]
                b_note, b_frames = group[i + 1]
                if b_note.start_s < a_note.start_s + a_note.duration_s:
                    drop_idx = i + 1 if a_frames >= b_frames else i  # tie -> drop the later one
                    del group[drop_idx]
                    dropped += 1
                    changed = True
                    break
        result.extend(ng[0] for ng in group)
    return result, dropped


def _draw_debug_frame(
    frame: np.ndarray,
    boxes: List[Tuple[Tuple[int, int, int, int], int, int]],
    frame_idx: int,
    frame_time: float,
) -> np.ndarray:
    """Annotate a frame with detected boxes (note name + track id), and the hit line."""
    img = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    for (x, y, w, h), pitch, track_id in boxes:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        label = f"{note_name(pitch)}#{track_id}" if pitch >= 0 else f"?#{track_id}"
        cv2.putText(img, label, (x, max(y - 4, 10)), font, 0.4, (0, 255, 0), 1)
    cv2.line(img, (0, HIT_LINE_Y), (img.shape[1] - 1, HIT_LINE_Y), (255, 0, 0), 1)
    cv2.putText(img, f"frame {frame_idx}  t={frame_time:.3f}s", (10, 20), font, 0.5, (255, 255, 255), 1)
    return img


def cmd_detect(args: argparse.Namespace) -> None:
    """Stage 2/3: scan frames over time and detect falling note-bar events."""
    wall_start = time.perf_counter()
    calib_path = _resolve("calibration.json")
    if not calib_path.exists():
        sys.exit(f"ERROR: {calib_path} not found -- run `python3 video2midi.py calibrate` first.")
    regions, x_to_note = load_calibration(calib_path)  # x_to_note kept for reference; detection no longer uses it
    regions_by_pitch = {r.midi: r for r in regions}

    video_path = _resolve(VIDEO_PATH)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"ERROR: could not open video at {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or VIDEO_FPS
    max_seconds = args.max_seconds if args.max_seconds > 0 else float("inf")  # 0 = whole video

    debug_dir = _resolve(DEBUG_DIR)
    if args.debug:
        debug_dir.mkdir(parents=True, exist_ok=True)

    tracks: List[Track] = []
    closed_tracks: List[Track] = []
    next_track_id = 0
    frames_processed = 0
    idx = 0
    last_processed_time = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_time = idx / fps
        if frame_time >= max_seconds:
            break
        if idx % args.step != 0:
            idx += 1
            continue
        last_processed_time = frame_time

        runs_by_pitch, mask = _detect_runs(frame, regions)

        # A track whose bar has fully passed (predicted top beyond the ROI +
        # tolerance) must not linger to absorb an unrelated later observation
        # at the same pitch (e.g. glow merging into the next bar's approach).
        still, expired = [], []
        for t in tracks:
            if t.last_y_top > 1 and t.last_y_top + SCROLL_PX_PER_SEC * (frame_time - t.last_frame_time) > NOTE_ROI_Y_MAX + MATCH_TOL_PX:
                expired.append(t)
            else:
                still.append(t)
        tracks = still
        closed_tracks.extend(expired)

        claimed = set()
        debug_boxes = []
        for pitch, runs in runs_by_pitch.items():
            for y0, y1 in runs:
                height = y1 - y0 + 1
                if y1 >= NOTE_ROI_Y_MAX - 2 and height < GLOW_MIN_H:
                    continue  # flame glow, or the last ~70ms of a bar already tracked
                y_top, y_bottom = float(y0), float(y1)

                track = _match_track(tracks, pitch, y_top, y_bottom, frame_time, claimed)
                if track is None:
                    track = Track(
                        pitch=pitch, track_id=next_track_id,
                        last_y_bottom=y_bottom, last_y_top=y_top,
                        last_frame_time=frame_time, last_frame_idx=idx,
                    )
                    next_track_id += 1
                    tracks.append(track)
                claimed.add(track.track_id)

                track.last_y_bottom = y_bottom
                track.last_y_top = y_top
                track.last_frame_time = frame_time
                track.last_frame_idx = idx
                track.frames_seen += 1
                track.misses = 0
                track.heights.append(height)
                if y_bottom < NOTE_ROI_Y_MAX - 2:
                    track.t_on_estimates.append(frame_time + (HIT_LINE_Y - y_bottom) / SCROLL_PX_PER_SEC)
                    track.bottom_obs.append((frame_time, y_bottom))
                if y_top > 1:
                    track.t_off_estimates.append(frame_time + (HIT_LINE_Y - y_top) / SCROLL_PX_PER_SEC)
                region = regions_by_pitch[pitch]
                debug_boxes.append(((region.x0, y0, region.x1 - region.x0, height), pitch, track.track_id))

        for t in tracks:
            if t.track_id not in claimed:
                t.misses += 1
        still_active, newly_closed = [], []
        for t in tracks:
            (newly_closed if t.misses > TRACK_MAX_MISSES else still_active).append(t)
        tracks = still_active
        closed_tracks.extend(newly_closed)

        if args.debug and (frames_processed == 0 or frames_processed % DEBUG_EVERY_N_FRAMES == 0):
            annotated = _draw_debug_frame(frame, debug_boxes, idx, frame_time)
            cv2.imwrite(str(debug_dir / f"frame_{idx:06d}.png"), annotated)
        if args.debug and frames_processed == 0:
            cv2.imwrite(str(debug_dir / f"mask_{idx:06d}.png"), (mask.astype(np.uint8) * 255))

        frames_processed += 1
        idx += 1

    cap.release()
    still_active_at_end = list(tracks)
    closed_tracks.extend(still_active_at_end)  # anything still active at the end also gets finalized
    truncated_ids = {t.track_id for t in still_active_at_end}

    measured_scroll = _measure_scroll_speed(closed_tracks)
    if measured_scroll is not None:
        print(f"Measured scroll speed: {measured_scroll:.1f} px/s (configured: {SCROLL_PX_PER_SEC:.1f} px/s)")
        if abs(measured_scroll - SCROLL_PX_PER_SEC) / SCROLL_PX_PER_SEC > 0.03:
            print("WARNING: measured scroll speed differs from configured by more than 3%.")

    # A real bar is visible for ~NOTE_ROI_Y_MAX/SCROLL_PX_PER_SEC seconds; below
    # 3 observations in that many processed frames is likely a stray blip, but
    # a large --step that would never see 3 real observations shouldn't reject
    # everything, hence the floor at 1.
    expected_observations = (NOTE_ROI_Y_MAX / SCROLL_PX_PER_SEC) * fps / args.step
    min_observations = max(1, min(3, int(expected_observations // 4)))

    dropped_no_ton = dropped_few_obs = dropped_drift_ton = dropped_drift_toff = 0
    dropped_no_toff = dropped_short = 0
    truncated_count = 0
    candidates: List[Tuple[Note, int]] = []  # (note, frames_seen) -- frames_seen decides overlap resolution
    for t in closed_tracks:
        if not t.t_on_estimates:
            dropped_no_ton += 1
            continue
        if t.frames_seen < min_observations:
            dropped_few_obs += 1
            continue
        if len(t.t_on_estimates) >= 3 and _mad(t.t_on_estimates) > T_ON_SPREAD_MAX_S:
            dropped_drift_ton += 1
            continue
        if len(t.t_off_estimates) >= 3 and _mad(t.t_off_estimates) > T_ON_SPREAD_MAX_S:
            dropped_drift_toff += 1
            continue
        start = float(np.median(t.t_on_estimates))
        truncated = False
        if t.t_off_estimates:
            end = float(np.median(t.t_off_estimates))
        elif t.track_id in truncated_ids:
            # Still falling when the scan window ended -- true end is unknown;
            # cap it at the last processed frame and flag it as truncated.
            end = last_processed_time
            truncated = True
            truncated_count += 1
        else:
            # Closed (miss-timeout) before the top ever became unclipped, with
            # no truncation excuse -- unexpected after the clip-aware match fix.
            dropped_no_toff += 1
            continue
        duration = end - start
        if duration < MIN_NOTE_DURATION_S:
            dropped_short += 1
            continue
        note = Note(pitch=t.pitch, name=note_name(t.pitch), start_s=start, duration_s=duration, truncated=truncated)
        candidates.append((note, t.frames_seen))

    notes, dropped_overlap = _resolve_same_pitch_overlaps(candidates)
    notes.sort(key=lambda n: n.start_s)
    overlap_count = _count_same_pitch_overlaps(notes)  # sanity check -- should be 0 after resolution

    print(f"\nFrames processed: {frames_processed}")
    print(f"Tracks created: {next_track_id}")
    print(
        f"Notes kept: {len(notes)}  (dropped: no-t_on={dropped_no_ton}, "
        f"few-observations={dropped_few_obs}, drift-t_on={dropped_drift_ton}, "
        f"drift-t_off={dropped_drift_toff}, too-short={dropped_short}, "
        f"no-t_off-unexpected={dropped_no_toff}, same-pitch-overlap={dropped_overlap})"
    )
    if truncated_count:
        print(f"Truncated notes (end beyond scan window): {truncated_count}")
    print(f"Same-pitch overlaps: {overlap_count}")

    print(f"\n{'start':>7}  {'dur':>6}  {'MIDI':>4}  name")
    for n in notes[:40]:
        print(f"{n.start_s:7.3f}  {n.duration_s:6.3f}  {n.pitch:>4}  {n.name}")

    out_path = _resolve(args.out)
    data = {
        "header": {
            "fps": fps,
            "scroll_px_per_sec_used": SCROLL_PX_PER_SEC,
            "scroll_px_per_sec_measured": measured_scroll,
            "frames_processed": frames_processed,
        },
        "notes": [asdict(n) for n in notes],
    }
    out_path.write_text(json.dumps(data, indent=2))
    print(f"\nWrote {out_path}")
    print(f"Elapsed wall time: {time.perf_counter() - wall_start:.1f}s")


def _find_grid_phase(starts: np.ndarray, step_s: float) -> Tuple[float, float, float]:
    """Scan candidate phases for the one minimizing mean squared circular onset distance.

    Returns (phase_s, rms_residual_s, fraction of onsets within 25% of a grid point).
    """
    n_candidates = 200
    best_phase, best_mse = 0.0, None
    for i in range(n_candidates):
        phase = step_s * i / n_candidates
        residual = (starts - phase) % step_s
        residual = np.minimum(residual, step_s - residual)  # circular distance
        mse = float(np.mean(residual ** 2))
        if best_mse is None or mse < best_mse:
            best_mse, best_phase = mse, phase
    residual = (starts - best_phase) % step_s
    residual = np.minimum(residual, step_s - residual)
    rms = float(np.sqrt(np.mean(residual ** 2)))
    on_grid_frac = float(np.mean(residual <= 0.25 * step_s))
    return best_phase, rms, on_grid_frac


def _resolve_collisions(quantized: List[dict], step_s: float, by_hand: bool = False) -> Tuple[int, int]:
    """Trim an earlier same-pitch note's end to the next one's start; drop it if that
    leaves it under one grid unit. With by_hand, groups by (hand, pitch) instead of
    pitch alone -- the same pitch overlapping across two different hands/tracks is
    legal MIDI (rendered on separate staves) and must be left alone.
    Returns (trimmed_count, dropped_count).
    """
    by_pitch: dict = {}
    for q in quantized:
        key = (q.get("hand"), q["pitch"]) if by_hand else q["pitch"]
        by_pitch.setdefault(key, []).append(q)
    trimmed = dropped = 0
    to_drop_ids = set()
    for group in by_pitch.values():
        group.sort(key=lambda q: q["start_s"])
        for a, b in zip(group, group[1:]):
            if a["end_s"] > b["start_s"] + 1e-9:  # epsilon: don't trim floating-point noise as a real overlap
                a["end_s"] = b["start_s"]
                trimmed += 1
                if a["end_s"] - a["start_s"] < step_s - 1e-9:
                    to_drop_ids.add(id(a))
                    dropped += 1
    quantized[:] = [q for q in quantized if id(q) not in to_drop_ids]
    return trimmed, dropped


def _pitch_span(pitches: List[int]) -> int:
    """Highest minus lowest pitch, or 0 for an empty list."""
    return max(pitches) - min(pitches) if pitches else 0


def _best_hand_split(
    pitches: List[int], left_center: float, right_center: float, held_left: List[int], held_right: List[int]
) -> Tuple[int, bool]:
    """Pick the 0..n split of ascending `pitches` into (left, right) closest to each
    hand's running centroid, subject to HAND_MAX_SPAN (a hand's still-sounding notes
    count toward its span). A lone note (n==1) is exempt from the span check and
    simply goes to whichever hand's centroid it's nearer to. Falls back to splitting
    at the single largest pitch gap if no split satisfies both spans.

    Returns (k, fell_back) where left = pitches[:k], right = pitches[k:].
    """
    n = len(pitches)
    best_k, best_cost = None, None
    for k in range(n + 1):
        left, right = pitches[:k], pitches[k:]
        if n > 1:
            if left and _pitch_span(held_left + left) > HAND_MAX_SPAN:
                continue
            if right and _pitch_span(held_right + right) > HAND_MAX_SPAN:
                continue
        cost = 0.0
        if left:
            cost += abs(float(np.mean(left)) - left_center)
        if right:
            cost += abs(float(np.mean(right)) - right_center)
        if best_cost is None or cost < best_cost:
            best_cost, best_k = cost, k
    if best_k is not None:
        return best_k, False
    gaps = [pitches[i + 1] - pitches[i] for i in range(n - 1)]
    return int(np.argmax(gaps)) + 1, True


def _split_hands_greedy(quantized: List[dict]) -> Tuple[int, List[int]]:
    """Greedily assign each quantized note to 'L' or 'R' (mutates in place).

    Notes are processed one onset-group (identical start_s) at a time, in time
    order, tracking each hand's EMA pitch centroid and its still-sounding notes
    (so a held chord can't be split out from under a hand by a later group).
    Returns (fallback_count, spans_seen) for reporting.
    """
    centers = {"L": float(HAND_INIT_LEFT), "R": float(HAND_INIT_RIGHT)}
    held: dict = {"L": [], "R": []}  # each: list of (end_s, pitch) still sounding
    fallback_count = 0
    spans_seen: List[int] = []

    groups: dict = {}
    for q in quantized:
        groups.setdefault(q["start_s"], []).append(q)

    for start in sorted(groups):
        group = sorted(groups[start], key=lambda q: q["pitch"])
        pitches = [q["pitch"] for q in group]

        for hand in ("L", "R"):
            held[hand] = [(e, p) for e, p in held[hand] if e > start]
        held_pitches = {h: [p for _, p in held[h]] for h in ("L", "R")}

        k, fell_back = _best_hand_split(pitches, centers["L"], centers["R"], held_pitches["L"], held_pitches["R"])
        fallback_count += fell_back
        left_group, right_group = group[:k], group[k:]
        for q in left_group:
            q["hand"] = "L"
        for q in right_group:
            q["hand"] = "R"

        for hand, new_group in (("L", left_group), ("R", right_group)):
            if not new_group:
                continue
            realized_span = _pitch_span(held_pitches[hand] + [q["pitch"] for q in new_group])
            spans_seen.append(realized_span)
            centers[hand] = HAND_EMA_ALPHA * float(np.mean([q["pitch"] for q in new_group])) + (1 - HAND_EMA_ALPHA) * centers[hand]
            held[hand].extend((q["end_s"], q["pitch"]) for q in new_group)

    return fallback_count, spans_seen


def _hand_group_cost(
    hand: str, new: List[int], held: List[int], other_held: List[int], last: Tuple[float, float, float], t: float
) -> float:
    """Playability cost of one hand playing `new` pitches (ascending) at beat `t`,
    given its own still-held pitches, the other hand's still-held pitches, and
    its previous position `last` = (lo, hi, onset_beat). Terms: span/finger
    count, movement from the previous position, register, crossing, centre pull.
    """
    if not new:
        return 0.0
    cost = 0.0
    allp = held + new
    if len(allp) >= 2:
        span = max(allp) - min(allp)
        if span > HAND_MAX_SPAN:
            cost += HAND_SPAN_PENALTY + HAND_SPAN_EXCESS_W * (span - HAND_MAX_SPAN)
        elif span > HAND_SPAN_SOFT_FROM:
            cost += HAND_SPAN_SOFT_W * (span - HAND_SPAN_SOFT_FROM)
        if len(allp) > HAND_MAX_NOTES:
            cost += HAND_COUNT_PENALTY * (len(allp) - HAND_MAX_NOTES)
    lo, hi = new[0], new[-1]
    gap = max(0.0, last[0] - hi, lo - last[1])
    excess = gap - HAND_REACH_FREE
    if excess > 0:
        dt = max(0.0, t - last[2])
        cost += HAND_MOVE_W * excess * HAND_MOVE_TIME_REF / (HAND_MOVE_TIME_REF + dt)
    if hand == "L":
        cost += HAND_REGISTER_W * sum(max(0, p - HAND_LH_COMFORT_MAX) for p in new)
        if other_held:
            floor_r = min(other_held)
            cost += HAND_CROSS_PENALTY * sum(1 for p in new if p > floor_r)
        cost += HAND_CENTER_W * sum(abs(p - HAND_INIT_LEFT) for p in new)
    else:
        cost += HAND_REGISTER_W * sum(max(0, HAND_RH_COMFORT_MIN - p) for p in new)
        if other_held:
            ceil_l = max(other_held)
            cost += HAND_CROSS_PENALTY * sum(1 for p in new if p < ceil_l)
        cost += HAND_CENTER_W * sum(abs(p - HAND_INIT_RIGHT) for p in new)
    return cost


def _split_hands_beam(quantized: List[dict], beat_s: float, width: int) -> None:
    """Beam search over onset groups for the left/right split index k of each group
    (left = lowest k notes), minimising the summed _hand_group_cost plus an
    idle/balance term. Each beam state carries both hands' last position and
    still-held notes so the next group can be costed. Mutates q["hand"].
    """
    groups: dict = {}
    for q in quantized:
        groups.setdefault(q["start_s"], []).append(q)
    starts = sorted(groups)
    far = -1e9
    # state: (cost, lastL, lastR, heldL, heldR); held = tuple of (end_s, pitch)
    beam = [(0.0, (float(HAND_INIT_LEFT),) * 2 + (far,), (float(HAND_INIT_RIGHT),) * 2 + (far,), (), ())]
    layers: List[List[Tuple[int, int]]] = []  # per group: (parent index in previous beam, k) per surviving state
    for start in starts:
        group = sorted(groups[start], key=lambda q: q["pitch"])
        pitches = [q["pitch"] for q in group]
        ends = [q["end_s"] for q in group]
        n = len(group)
        t = start / beat_s
        cands: dict = {}
        for si, (cost0, last_l, last_r, held_l, held_r) in enumerate(beam):
            hl = tuple(h for h in held_l if h[0] > start + 1e-9)
            hr = tuple(h for h in held_r if h[0] > start + 1e-9)
            hlp, hrp = [p for _, p in hl], [p for _, p in hr]
            for k in range(n + 1):
                left, right = pitches[:k], pitches[k:]
                c = cost0 + _hand_group_cost("L", left, hlp, hrp, last_l, t) + _hand_group_cost("R", right, hrp, hlp, last_r, t)
                if not left and not hl and right and len(hrp) + len(right) >= 2:
                    c += HAND_IDLE_W * max(0, max(hrp + right) - min(hrp + right) - HAND_IDLE_SPAN)
                if not right and not hr and left and len(hlp) + len(left) >= 2:
                    c += HAND_IDLE_W * max(0, max(hlp + left) - min(hlp + left) - HAND_IDLE_SPAN)
                nl = (left[0], left[-1], t) if left else last_l
                nr = (right[0], right[-1], t) if right else last_r
                nhl = hl + tuple(zip(ends[:k], left))
                nhr = hr + tuple(zip(ends[k:], right))
                key = (nhl, nhr, nl, nr)
                prev = cands.get(key)
                if prev is None or c < prev[0][0]:
                    cands[key] = ((c, nl, nr, nhl, nhr), si, k)
        best = sorted(cands.values(), key=lambda v: v[0][0])[:width]
        layers.append([(si, k) for _, si, k in best])
        beam = [st for st, _, _ in best]
    idx = min(range(len(beam)), key=lambda i: beam[i][0])
    for gi in range(len(starts) - 1, -1, -1):
        si, k = layers[gi][idx]
        group = sorted(groups[starts[gi]], key=lambda q: q["pitch"])
        for j, q in enumerate(group):
            q["hand"] = "L" if j < k else "R"
        idx = si


def _hand_split_metrics(quantized: List[dict]) -> dict:
    """Readability counters for the current q["hand"] assignment (using the
    pre-fill note ends): LH notes >= E5, RH notes <= E3, crossings (a new RH
    note below a still-held LH note, or a new LH note above a still-held RH
    note) and span>HAND_MAX_SPAN events (a hand's new+held notes, >=2 of them).
    """
    groups: dict = {}
    for q in quantized:
        groups.setdefault(q["start_s"], []).append(q)
    held: dict = {"L": [], "R": []}
    m = {"lh_ge_e5": 0, "rh_le_e3": 0, "crossings": 0, "span_gt_max": 0}
    for start in sorted(groups):
        for h in held:
            held[h] = [(e, p) for e, p in held[h] if e > start + 1e-9]
        new = {"L": [q["pitch"] for q in groups[start] if q["hand"] == "L"], "R": [q["pitch"] for q in groups[start] if q["hand"] == "R"]}
        hp = {h: [p for _, p in held[h]] for h in held}
        m["lh_ge_e5"] += sum(1 for p in new["L"] if p >= 76)
        m["rh_le_e3"] += sum(1 for p in new["R"] if p <= 52)
        if hp["L"]:
            m["crossings"] += sum(1 for p in new["R"] if p < max(hp["L"]))
        if hp["R"]:
            m["crossings"] += sum(1 for p in new["L"] if p > min(hp["R"]))
        for h in ("L", "R"):
            allp = hp[h] + new[h]
            if new[h] and len(allp) >= 2 and max(allp) - min(allp) > HAND_MAX_SPAN:
                m["span_gt_max"] += 1
        for h in ("L", "R"):
            held[h].extend((q["end_s"], q["pitch"]) for q in groups[start] if q["hand"] == h)
    return m


def _split_hands(quantized: List[dict], beat_s: float, mode: str = HAND_SPLIT_MODE) -> Tuple[int, List[int], dict]:
    """Assign every note to 'L'/'R' (mutates q["hand"]) with the greedy tracker or the
    beam search (`mode`). Both are always computed so their readability metrics can
    be logged side by side; the chosen one is left in place.
    Returns (fallback_count, spans_seen, info) where info holds both metric dicts
    and the greedy hand list (used to keep the lead-in search independent of the mode).
    """
    fallback_count, spans_seen = _split_hands_greedy(quantized)
    greedy_hands = [q["hand"] for q in quantized]
    info = {"greedy": _hand_split_metrics(quantized), "greedy_hands": greedy_hands, "mode": mode}
    if mode == "beam":
        _split_hands_beam(quantized, beat_s, HAND_SPLIT_BEAM_WIDTH)
        info["beam"] = _hand_split_metrics(quantized)
        # recompute fallback/span stats for the beam assignment
        fallback_count, spans_seen = 0, []
        groups: dict = {}
        for q in quantized:
            groups.setdefault(q["start_s"], []).append(q)
        held: dict = {"L": [], "R": []}
        for start in sorted(groups):
            for h in held:
                held[h] = [(e, p) for e, p in held[h] if e > start + 1e-9]
            for h in ("L", "R"):
                new = [q for q in groups[start] if q["hand"] == h]
                if not new:
                    continue
                span = _pitch_span([p for _, p in held[h]] + [q["pitch"] for q in new])
                spans_seen.append(span)
                if len(new) + len(held[h]) > 1 and span > HAND_MAX_SPAN:
                    fallback_count += 1
                held[h].extend((q["end_s"], q["pitch"]) for q in new)
    return fallback_count, spans_seen, info


def _make_single_voice(notes: List[dict], step_s: float) -> Tuple[int, int]:
    """Force one voice within a hand: every note starting at the same quantized
    time gets the same end (extended to the next onset, capped at the group's
    own latest original end), so MuseScore never needs a second voice.
    Returns (shortened_count, lengthened_count).
    """
    groups: dict = {}
    for q in notes:
        groups.setdefault(q["start_s"], []).append(q)
    starts = sorted(groups)
    shortened = lengthened = 0
    for i, start in enumerate(starts):
        group = groups[start]
        max_end = max(q["end_s"] for q in group)
        next_start = starts[i + 1] if i + 1 < len(starts) else None
        new_end = min(next_start, max_end) if next_start is not None else max_end
        new_end = max(new_end, start + step_s)  # minimum one grid unit
        for q in group:
            if new_end < q["end_s"]:
                shortened += 1
            elif new_end > q["end_s"]:
                lengthened += 1
            q["end_s"] = new_end
    return shortened, lengthened


def _assert_single_voice(notes: List[dict]) -> None:
    """Verify that within `notes` every pair is identical in (start,end) or non-overlapping."""
    groups: dict = {}
    for q in notes:
        groups.setdefault(q["start_s"], []).append(q)
    starts = sorted(groups)
    for start in starts:
        ends = {q["end_s"] for q in groups[start]}
        assert len(ends) == 1, f"single-voice violation: notes at start={start} have differing ends {ends}"
    for i in range(len(starts) - 1):
        end_i = groups[starts[i]][0]["end_s"]
        assert end_i <= starts[i + 1] + 1e-9, (
            f"single-voice violation: group at start={starts[i]} (end={end_i}) overlaps next group at {starts[i + 1]}"
        )


def _fill_rests(notes: List[dict], beat_s: float, max_beats: float = FILL_REST_MAX_BEATS) -> int:
    """Extend each note's end to the next onset in `notes` if the gap is short.

    `notes` should be one hand's (or the whole piece's, if hands aren't split)
    notes. If a note carries a quantized "sound_end_s" (pedal-extended sounding
    length), the extension is capped there instead of running all the way to
    the next onset -- this reproduces a held note's true sustain without ever
    reaching past it. Returns how many were extended.
    """
    max_gap_s = max_beats * beat_s
    ordered = sorted(notes, key=lambda q: q["start_s"])
    extended = 0
    for i, q in enumerate(ordered):
        next_on = next((o["start_s"] for o in ordered[i + 1 :] if o["start_s"] > q["start_s"]), None)
        if next_on is None:
            continue
        gap = next_on - q["end_s"]
        if not (0 < gap <= max_gap_s):
            continue
        sound_end = q.get("sound_end_s")
        new_end = min(next_on, sound_end) if sound_end is not None else next_on
        if new_end > q["end_s"]:
            q["end_s"] = new_end
            extended += 1
    return extended


def _choose_lead_in(quantized: List[dict], beat_s: float) -> Tuple[float, float]:
    """Search candidate lead-in shifts (0, 0.25, .., 3.75 beats) and pick the one
    that lands the most LEFT-hand note-duration on a downbeat (start%4==0, full
    weight) or beat 3 (start%4==2, half weight) -- sustained chords are usually
    placed on a strong beat, so this finds where the piece's actual bar lines are.
    Requires quantized notes to already carry a "hand" field. Returns (lead_in_beats, score).
    """
    left = [q for q in quantized if q.get("hand") == "L"]
    best_lead_in, best_score = 0.0, -1.0
    for i in range(16):
        lead_in = i * 0.25
        score = 0.0
        for q in left:
            phase = (q["start_s"] / beat_s + lead_in) % 4
            dur_beats = (q["end_s"] - q["start_s"]) / beat_s
            if abs(phase) < 1e-6:
                score += dur_beats
            elif abs(phase - 2) < 1e-6:
                score += 0.5 * dur_beats
        if score > best_score:
            best_score, best_lead_in = score, lead_in
    return best_lead_in, best_score


_KEY_NAMES = {  # fifths -> (major tonic, minor tonic), for display only
    -6: ("G♭", "e♭"), -5: ("D♭", "b♭"), -4: ("A♭", "f"),
    -3: ("E♭", "c"), -2: ("B♭", "g"), -1: ("F", "d"), 0: ("C", "a"),
    1: ("G", "e"), 2: ("D", "b"), 3: ("A", "f♯"), 4: ("E", "c♯"),
    5: ("B", "g♯"), 6: ("F♯", "d♯"),
}


def _key_name(fifths: int) -> str:
    """Display name for a key signature, both major and minor since we don't decide."""
    major, minor = _KEY_NAMES[fifths]
    return f"{major} major / {minor} minor"


def _fit_key_signature_bars(quantized: List[dict], beat_s: float, n_bars: int) -> List[int]:
    """Viterbi-fit one key signature (fifths, -6..6) per bar.

    Cost of a bar/fifths pair is the number of note *onsets* (both hands, before
    ties -- each quantized note counted once at its own start bar) whose pitch
    class falls outside that key's major scale; changing fifths between
    consecutive bars costs KEY_CHANGE_PENALTY. Returns one fifths value per bar.
    """
    fifths_list = list(range(-6, 7))
    scale_pcs = {f: {(7 * f + iv) % 12 for iv in (0, 2, 4, 5, 7, 9, 11)} for f in fifths_list}

    bar_pcs: List[List[int]] = [[] for _ in range(n_bars)]
    for q in quantized:
        b = int(q["start_s"] / beat_s // 4)
        if 0 <= b < n_bars:
            bar_pcs[b].append(q["pitch"] % 12)

    cost = [[sum(1 for pc in bar_pcs[b] if pc not in scale_pcs[f]) for f in fifths_list] for b in range(n_bars)]

    n_f = len(fifths_list)
    dp = [cost[0][:]] + [[0.0] * n_f for _ in range(n_bars - 1)]
    back = [[0] * n_f for _ in range(n_bars)]
    for b in range(1, n_bars):
        for fi in range(n_f):
            best_prev, best_cost = 0, None
            for fpi in range(n_f):
                c = dp[b - 1][fpi] + (0 if fpi == fi else KEY_CHANGE_PENALTY)
                if best_cost is None or c < best_cost:
                    best_cost, best_prev = c, fpi
            dp[b][fi] = best_cost + cost[b][fi]
            back[b][fi] = best_prev

    path = [0] * n_bars
    path[-1] = min(range(n_f), key=lambda fi: dp[-1][fi])
    for b in range(n_bars - 1, 0, -1):
        path[b - 1] = back[b][path[b]]
    return [fifths_list[i] for i in path]


_STEP_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def _bar_beat_str(x: Fraction) -> str:
    """Absolute beat position (0-based) -> "bar.beat" display string, both 1-indexed."""
    bar = int(x // 4) + 1
    beat = float(x - (bar - 1) * 4) + 1
    return f"{bar}.{beat:.2f}"


def _collect_note_records(root: ET.Element) -> List[dict]:
    """Walk every <part>/<measure> with the same cursor logic as the validator,
    recording each <note>'s absolute start/end (in beats, as Fraction), staff,
    and MIDI pitch (None for rests). Used by the ottava pass.
    """
    records: List[dict] = []
    for part in root.findall("part"):
        measures = part.findall("measure")
        divisions = None
        for mi, measure in enumerate(measures):
            for attrs in measure.findall("attributes"):
                div_el = attrs.find("divisions")
                if div_el is not None:
                    divisions = int(div_el.text)
            if divisions is None:
                continue
            cursor = 0
            chord_start = 0  # start of the current chord's base (non-chord) note
            for child in measure:
                if child.tag == "note":
                    staff_el, voice_el, dur_el = child.find("staff"), child.find("voice"), child.find("duration")
                    staff = staff_el.text if staff_el is not None else "1"
                    dur = int(dur_el.text) if dur_el is not None else 0
                    is_chord = child.find("chord") is not None
                    # A <chord/> note shares its base note's onset -- the cursor has
                    # already advanced past that onset by the time we reach it here.
                    if is_chord:
                        start_ticks = chord_start
                    else:
                        start_ticks = cursor
                        chord_start = cursor
                        cursor += dur
                    midi = None
                    pitch_el = child.find("pitch")
                    if pitch_el is not None:
                        step = pitch_el.find("step").text
                        alter_el = pitch_el.find("alter")
                        alter = int(alter_el.text) if alter_el is not None else 0
                        octave_el = pitch_el.find("octave")
                        octave = int(octave_el.text)
                        midi = (octave + 1) * 12 + _STEP_PC[step] + alter
                    records.append({
                        "el": child, "measure": measure, "staff": staff, "voice": voice_el.text if voice_el is not None else None,
                        "start": mi * 4 + Fraction(start_ticks, divisions),
                        "end": mi * 4 + Fraction(start_ticks + dur, divisions),
                        "midi": midi, "octave_el": octave_el if pitch_el is not None else None,
                        "order": len(records),
                    })
                elif child.tag == "backup":
                    dur_el = child.find("duration")
                    cursor -= int(dur_el.text) if dur_el is not None else 0
                elif child.tag == "forward":
                    dur_el = child.find("duration")
                    cursor += int(dur_el.text) if dur_el is not None else 0
    return records


def _make_octave_shift_direction(shift_type: str, placement: str, staff: str, number: str, size: str = "8") -> ET.Element:
    """Build one <direction><octave-shift> element. `number` distinguishes
    concurrent brackets: MuseScore 4 pairs octave-shift start/stop globally by
    this number (not scoped per staff), so a staff-1 8va and a staff-2 8vb
    that are simultaneously open must use different numbers -- verified
    empirically 2026-09-20: sharing "1" between them corrupts the pairing and
    the bracket that loses out plays back at the wrong octave.
    """
    direction = ET.Element("direction", {"placement": placement})
    dtype = ET.SubElement(direction, "direction-type")
    ET.SubElement(dtype, "octave-shift", {"type": shift_type, "size": size, "number": number})
    staff_el = ET.SubElement(direction, "staff")
    staff_el.text = staff
    return direction


def _insert_direction_relative_to(measure: ET.Element, target_el: ET.Element, direction_el: ET.Element, before: bool) -> None:
    """Insert direction_el as a sibling of target_el inside measure, immediately
    before or after it. Re-finds target_el's live index each call (by identity)
    so earlier insertions in the same measure don't invalidate the position.
    """
    children = list(measure)
    idx = next(i for i, c in enumerate(children) if c is target_el)
    measure.insert(idx if before else idx + 1, direction_el)


_SLOT_BEATS = Fraction(1, int(round(1 / OTTAVA_SLOT_BEATS)))  # OTTAVA_SLOT_BEATS as an exact Fraction


# Octave-bracket states per staff. `delta` = DISPLAYED pitch minus sounding pitch in
# semitones. <pitch> in the XML stays the SOUNDING pitch (MusicXML spec) and is
# never modified; MuseScore 4 draws the notes shifted itself. Verified by direct
# import tests: type="down" imports as 8va/15ma (drawn lower), type="up" as 8vb
# (drawn higher). `number` pairs start/stop globally across staves, so every
# (staff, direction) gets its own.
_OTTAVA_STATES = {
    "1": [
        {"kind": "plain", "delta": 0},
        {"kind": "8va", "delta": -12, "type": "down", "size": "8", "placement": "above", "number": "1"},
        {"kind": "15ma", "delta": -24, "type": "down", "size": "15", "placement": "above", "number": "1"},
        {"kind": "8vb", "delta": 12, "type": "up", "size": "8", "placement": "below", "number": "4"},
    ],
    "2": [
        {"kind": "plain", "delta": 0},
        {"kind": "8vb", "delta": 12, "type": "up", "size": "8", "placement": "below", "number": "2"},
        {"kind": "8va", "delta": -12, "type": "down", "size": "8", "placement": "above", "number": "3"},
    ],
}


def _state_enter_penalty(state: dict) -> float:
    """Cost of starting a bracket in `state` (from plain or from another bracket)."""
    if state["kind"] == "plain":
        return 0.0
    return OTTAVA_SWITCH_PENALTY * (1.0 + (OTTAVA_15MA_EXTRA if state["kind"] == "15ma" else 0.0))


def _ledger_cost(midi: int, staff: str, octave_shift: int) -> float:
    """Squared ledger-line cost of notating `midi` on `staff` with its written
    pitch moved `octave_shift` octaves (written = midi + 12*octave_shift; 0 =
    plain, -1 = 8va, -2 = 15ma, +1 = 8vb). One ledger line is approximated as
    3.5 semitones. The first OTTAVA_FREE_LEDGERS_ABOVE/BELOW ledger lines are
    free (engravers don't reach for a bracket over 1-2 ledger lines); only the
    excess is costed, squared so a handful of extreme ledger lines dominates a
    slot's total over several mildly-out-of-range notes.
    """
    p = midi + 12 * octave_shift
    if staff == "2":
        above, below = max(0.0, (p - 59) / 3.5), max(0.0, (41 - p) / 3.5)
    else:
        above, below = max(0.0, (p - 79) / 3.5), max(0.0, (62 - p) / 3.5)
    above = max(0.0, above - OTTAVA_FREE_LEDGERS_ABOVE)
    below = max(0.0, below - OTTAVA_FREE_LEDGERS_BELOW)
    return above ** 2 + below ** 2


def _ledger_lines(staff: str, step: str, octave: int) -> int:
    """Actual number of ledger lines a written note needs (diatonic count)."""
    d = octave * 7 + "CDEFGAB".index(step)
    if staff == "2":
        bottom, top = 18, 26  # G2, A3 (bass staff lines)
    else:
        bottom, top = 30, 38  # E4, F5 (treble staff lines)
    if d >= top + 2:
        return (d - top) // 2
    if d <= bottom - 2:
        return (bottom - d) // 2
    return 0


def _staff_slot_costs(pitched: List[dict], staff: str, states: List[dict]) -> List[List[float]]:
    """0.25-beat slot timeline -> per-state ledger-line cost per slot (index 0 = plain).
    Each note adds its cost to every slot it overlaps, so a long note weighs
    more than a short one; simultaneous notes on the staff add together.
    """
    n_slots = math.ceil(max(r["end"] for r in pitched) / _SLOT_BEATS)
    costs = [[0.0] * n_slots for _ in states]
    for r in pitched:
        i0 = math.floor(r["start"] / _SLOT_BEATS)
        i1 = math.ceil(r["end"] / _SLOT_BEATS)
        per_state = [_ledger_cost(r["midi"], staff, st["delta"] // 12) for st in states]
        for i in range(i0, min(i1, n_slots)):
            slot_start, slot_end = i * _SLOT_BEATS, (i + 1) * _SLOT_BEATS
            if r["start"] < slot_end and r["end"] > slot_start:
                for si, c in enumerate(per_state):
                    costs[si][i] += c
    return costs


def _viterbi_ottava_states(costs: List[List[float]], enter: List[float]) -> List[int]:
    """Multi-state Viterbi over per-slot ledger-line costs (state 0 = plain).
    Entering a bracketed state costs enter[state]. A bracket can only be left
    for plain, and then at least OTTAVA_MIN_GAP_BEATS + 1 beat of plain slots
    must follow before the next bracket starts (modelled with a chain of
    "cooldown" plain states), so two brackets on one staff -- same or different
    kind -- never touch (MuseScore would merge them). Returns the chosen state
    index (0 = plain, else a bracket state) per slot.
    """
    k, n = len(costs), len(costs[0]) if costs else 0
    if n == 0:
        return []
    gap = int(round((OTTAVA_MIN_GAP_BEATS + 1.0) / float(_SLOT_BEATS)))  # plain slots required between brackets
    assert gap >= 2
    cd = list(range(k, k + gap - 1))  # cooldown states C_1..C_{gap-1}; plain state 0 doubles as C_gap
    total = k + len(cd)
    inf = float("inf")
    dp = [[inf] * n for _ in range(total)]
    back = [[-1] * n for _ in range(total)]
    dp[0][0] = costs[0][0]
    for si in range(1, k):
        dp[si][0] = costs[si][0] + enter[si]
    for i in range(1, n):
        c0 = costs[0][i]
        # plain (eligible): stay plain, or finish the cooldown chain
        cands = [(dp[0][i - 1], 0), (dp[cd[-1]][i - 1], cd[-1])]
        v, pj = min(cands)
        dp[0][i], back[0][i] = v + c0, pj
        for si in range(1, k):
            stay, enter_v = dp[si][i - 1], dp[0][i - 1] + enter[si]
            if stay <= enter_v:
                dp[si][i], back[si][i] = stay + costs[si][i], si
            else:
                dp[si][i], back[si][i] = enter_v + costs[si][i], 0
        best = min((dp[b][i - 1], b) for b in range(1, k))
        dp[cd[0]][i], back[cd[0]][i] = best[0] + c0, best[1]
        for j in range(1, len(cd)):
            dp[cd[j]][i], back[cd[j]][i] = dp[cd[j - 1]][i - 1] + c0, cd[j - 1]
    cur = min(range(total), key=lambda si: dp[si][n - 1])
    path = [0] * n
    for i in range(n - 1, -1, -1):
        path[i] = cur if cur < k else 0
        if i > 0:
            cur = back[cur][i]
    return path


def _mask_to_regions(mask: List[bool]) -> List[List[int]]:
    """Contiguous True runs in a slot mask -> [start_idx, end_idx) pairs."""
    regions: List[List[int]] = []
    i, n = 0, len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        regions.append([i, j])
        i = j
    return regions


def _merge_and_filter_regions(
    regions: List[List[int]], cost_plain: List[float], cost_shifted: List[float], penalty: float = OTTAVA_SWITCH_PENALTY
) -> List[Tuple[int, int]]:
    """Merge slot regions (of ONE state) separated by <= OTTAVA_MIN_GAP_BEATS, then
    drop any (possibly newly-merged) region that's still shorter than
    OTTAVA_MIN_SPAN_BEATS, or whose net ledger-line saving over its full span
    doesn't cover `penalty` -- a safety net against the merge step
    stitching two barely-worthwhile regions into a net-negative one.
    Empirically (2026-09-20) MuseScore 4 itself visually merges two brackets
    exactly OTTAVA_MIN_GAP_BEATS apart into one, silently dropping a bracket --
    so the boundary must be inclusive here, not strict, to match.
    """
    min_gap_slots = OTTAVA_MIN_GAP_BEATS / float(_SLOT_BEATS)
    min_span_slots = OTTAVA_MIN_SPAN_BEATS / float(_SLOT_BEATS)
    merged: List[List[int]] = []
    for reg in regions:
        if merged and reg[0] - merged[-1][1] <= min_gap_slots:
            merged[-1][1] = reg[1]
        else:
            merged.append(list(reg))
    kept = []
    for r0, r1 in merged:
        if r1 - r0 < min_span_slots:
            continue
        saving = sum(cost_plain[i] - cost_shifted[i] for i in range(r0, r1))
        if saving >= penalty:
            kept.append((r0, r1))
    return kept


def _enforce_span_spacing(spans: List[dict], costs: Optional[List[List[float]]] = None) -> List[dict]:
    """Final safety pass on one staff's converted note spans (dicts with start,
    end, state index `si`): merge spans separated by <= OTTAVA_MIN_GAP_BEATS
    (tie-extension, run after regions are merged, can still leave two spans this
    close together), then drop anything left shorter than OTTAVA_MIN_SPAN_BEATS.
    Two spans of different states that close (or overlapping) can't coexist, so
    they merge into whichever of the two states has the lower ledger cost over
    the union (`costs[state][slot]`; first state wins ties). A merged
    cross-state span is dropped if it then saves nothing over plain. The gap
    boundary is inclusive to match MuseScore 4's own import behaviour, which
    visually merges (and drops a bracket for) two spans exactly
    OTTAVA_MIN_GAP_BEATS apart -- verified empirically 2026-09-20.
    """
    if not spans:
        return spans

    def _slots(a: Fraction, b: Fraction) -> range:
        return range(math.floor(a / _SLOT_BEATS), math.ceil(b / _SLOT_BEATS))

    ordered = sorted(spans, key=lambda s: (s["start"], s["end"]))
    merged = [dict(ordered[0])]
    for sp in ordered[1:]:
        prev = merged[-1]
        if float(sp["start"] - prev["end"]) <= OTTAVA_MIN_GAP_BEATS:
            prev["end"] = max(prev["end"], sp["end"])
            if sp["si"] != prev["si"]:
                prev["cross"] = True
                if costs is not None:
                    rng = _slots(prev["start"], prev["end"])
                    prev["si"] = min((prev["si"], sp["si"]), key=lambda si: sum(costs[si][i] for i in rng if i < len(costs[si])))
        else:
            merged.append(dict(sp))
    result = []
    for sp in merged:
        if float(sp["end"] - sp["start"]) < OTTAVA_MIN_SPAN_BEATS:
            continue
        if sp.get("cross") and costs is not None:
            rng = [i for i in _slots(sp["start"], sp["end"]) if i < len(costs[0])]
            if sum(costs[0][i] - costs[sp["si"]][i] for i in rng) < 0:
                continue
        result.append(sp)
    for a, b in zip(result, result[1:]):
        assert float(b["start"] - a["end"]) > OTTAVA_MIN_GAP_BEATS, "span spacing invariant violated"
    for sp in result:
        assert float(sp["end"] - sp["start"]) >= OTTAVA_MIN_SPAN_BEATS, "span length invariant violated"
    return result


def _apply_ottava(root: ET.Element) -> dict:
    """Insert automatic octave brackets (the notes' sounding <pitch> is untouched;
    MuseScore draws them shifted). Per staff, a multi-state ledger-line-cost Viterbi picks, per
    0.25-beat slot, the notation that minimises total ledger-line load:
      staff 1 (treble): plain / 8va / 15ma / 8vb
      staff 2 (bass):   plain / 8vb / 8va
    Entering any bracket (also from another bracket) costs OTTAVA_SWITCH_PENALTY
    (15ma: +OTTAVA_15MA_EXTRA of it). Regions then go through the merge/min-span
    filters per state, tie-safe span extension, and a cross-state spacing pass.
    Direction elements carry no duration, so the timing validator is unaffected.
    Returns stats for the caller to print.
    """
    records = _collect_note_records(root)
    stats = {
        "span_info": [], "span_lines": [], "counts": {}, "notes_shifted": 0,
        "rh_high_not_covered": 0, "rh_over_shifted": 0,
        "cost_plain_total": 0.0, "cost_chosen_total": 0.0,
        "ledger": {}, "n_8va": 0, "n_15ma": 0, "n_8vb": 0,
    }

    def _ledger_stats(staff_recs: List[dict], staff: str) -> Tuple[int, int]:
        total = ge4 = 0
        for r in staff_recs:
            pe = r["el"].find("pitch")
            n = _ledger_lines(staff, pe.find("step").text, int(pe.find("octave").text) + r["delta"] // 12)
            total += n
            ge4 += n >= 4
        return total, ge4

    for staff in ("1", "2"):
        states = _OTTAVA_STATES[staff]
        enter = [_state_enter_penalty(st) for st in states]
        pitched = [r for r in records if r["staff"] == staff and r["midi"] is not None]
        if not pitched:
            continue
        for r in pitched:
            r["delta"] = 0
        before = _ledger_stats(pitched, staff)

        costs = _staff_slot_costs(pitched, staff, states)
        path = _viterbi_ottava_states(costs, enter)
        n_slots = len(path)

        spans = []
        chosen_slot_state = [0] * n_slots
        for si in range(1, len(states)):
            regions = _merge_and_filter_regions(
                _mask_to_regions([p == si for p in path]), costs[0], costs[si], enter[si]
            )
            for r0, r1 in regions:
                region_start, region_end = r0 * _SLOT_BEATS, r1 * _SLOT_BEATS
                in_region = [r for r in pitched if region_start <= r["start"] < region_end]
                if not in_region:
                    continue
                spans.append({"start": min(r["start"] for r in in_region), "end": max(r["end"] for r in in_region), "si": si})

        # Extend spans so they never cut through a tied note: our earlier
        # barline pre-split ties every note that crosses a bar into several
        # same-pitch <note> fragments, each with its own start/end. If a span
        # boundary fell strictly inside one of these, only some fragments of
        # one logical note would get shifted -- a tie whose pitch jumps
        # midway. Walk outward from each span while its edge note has a tie
        # into a fragment currently outside the span.
        sorted_pitched = sorted(pitched, key=lambda r: r["start"])
        for sp in spans:
            changed = True
            while changed:
                changed = False
                in_span = [r for r in sorted_pitched if sp["start"] <= r["start"] < sp["end"]]
                if not in_span:
                    break
                first = min(in_span, key=lambda r: r["start"])
                if "stop" in {t.get("type") for t in first["el"].findall("tie")}:
                    pred = next((r for r in sorted_pitched if r["end"] == sp["start"] and r["midi"] == first["midi"]), None)
                    if pred is not None and pred["start"] < sp["start"]:
                        sp["start"] = pred["start"]
                        changed = True
                        continue
                last = max(in_span, key=lambda r: r["end"])
                if "start" in {t.get("type") for t in last["el"].findall("tie")}:
                    succ = next((r for r in sorted_pitched if r["start"] == last["end"] and r["midi"] == last["midi"]), None)
                    if succ is not None and succ["end"] > sp["end"]:
                        sp["end"] = succ["end"]
                        changed = True

        spans = _enforce_span_spacing(spans, costs)

        for sp in spans:
            st = states[sp["si"]]
            span_notes = [r for r in pitched if sp["start"] <= r["start"] < sp["end"]]
            if not span_notes:
                continue
            for r in span_notes:
                r["delta"] = st["delta"]  # displayed shift only; <pitch> is left untouched
                stats["notes_shifted"] += 1
            # Anchor on a primary-voice note when possible: MusicXML serializes
            # each voice as its own contiguous block per measure (not merged
            # chronologically), so a secondary-voice note's document position
            # can be earlier/later than its actual time relative to another
            # voice -- which would put two adjacent spans' start/stop markers
            # in the wrong relative order. The primary voice's block is always
            # in genuine chronological order, so anchoring there is safe.
            primary_voice = "1" if staff == "1" else "5"
            primary_notes = [r for r in span_notes if r["voice"] == primary_voice] or span_notes
            first_note = min(primary_notes, key=lambda r: (r["start"], r["order"]))
            last_note = max(primary_notes, key=lambda r: (r["end"], r["order"]))
            start_dir = _make_octave_shift_direction(st["type"], st["placement"], staff, st["number"], st["size"])
            _insert_direction_relative_to(first_note["measure"], first_note["el"], start_dir, before=True)
            stop_dir = _make_octave_shift_direction("stop", st["placement"], staff, st["number"], st["size"])
            _insert_direction_relative_to(last_note["measure"], last_note["el"], stop_dir, before=False)

            up = st["delta"] < 0  # 8va/15ma: the extreme that matters is the top note
            extreme = max(r["midi"] for r in span_notes) if up else min(r["midi"] for r in span_notes)
            stats["span_info"].append({"staff": staff, "kind": st["kind"], "start": sp["start"], "end": sp["end"], "n": len(span_notes), "extreme": note_name(extreme), "up": up})
            stats["span_lines"].append(
                f"staff {staff} {st['kind']:<4} bars {_bar_beat_str(sp['start'])}-{_bar_beat_str(sp['end'])} "
                f"({len(span_notes)} notes, {'top' if up else 'bottom'} {note_name(extreme)})"
            )
            stats["counts"][(staff, st["kind"])] = stats["counts"].get((staff, st["kind"]), 0) + 1
            stats["n_" + st["kind"]] += 1
            for i in range(math.floor(sp["start"] / _SLOT_BEATS), min(n_slots, math.ceil(sp["end"] / _SLOT_BEATS))):
                chosen_slot_state[i] = sp["si"]

        stats["cost_plain_total"] += sum(costs[0])
        stats["cost_chosen_total"] += sum(costs[chosen_slot_state[i]][i] for i in range(n_slots))
        stats["cost_chosen_total"] += sum(enter[next(si for si, s_ in enumerate(states) if s_["kind"] == sp_["kind"])] for sp_ in stats["span_info"] if sp_["staff"] == staff)
        after = _ledger_stats(pitched, staff)
        stats["ledger"][staff] = {"lines_before": before[0], "lines_after": after[0], "ge4_before": before[1], "ge4_after": after[1]}

        if staff == "1":
            # Coverage/over-shift diagnostics from r["midi"] (the original sounding
            # pitch, never mutated -- only octave_el.text is) and r["delta"].
            onset_groups: dict = {}
            for r in pitched:
                onset_groups.setdefault(r["start"], []).append(r)
            for notes in onset_groups.values():
                if max(r["midi"] for r in notes) >= 88 and not any(r["delta"] < 0 for r in notes):
                    stats["rh_high_not_covered"] += 1
                if any(r["delta"] < 0 for r in notes) and min(r["midi"] + r["delta"] for r in notes) < 67:
                    stats["rh_over_shifted"] += 1

    return stats


def _postprocess_musicxml(xml_bytes: bytes, strip_default_composer: bool, apply_ottava: bool) -> Tuple[bytes, dict]:
    """Fix up music21's raw MusicXML output before it reaches MuseScore.

    1. Renumber <voice> per measure so all staff-1 voices are 1..4 and all
       staff-2 voices are 5..8 (order of first appearance among that staff's
       notes). music21 numbers voices per-measure without staff awareness (RH
       voice 2, LH voice 3, next measure LH voice 2 again, etc.); MuseScore's
       importer instead assigns each raw voice NUMBER to one staff for the
       whole part (majority vote) and silently moves notes to match, which is
       what caused RH notes to land on the bass staff. Giving each staff its
       own disjoint voice-number range removes the ambiguity entirely.
    2. Drop music21's default "Music21" composer <creator> if we're not
       supplying our own (SCORE_COMPOSER empty).
    3. Strip every <accidental> child of <note>. music21 attaches an explicit
       accidental (even "natural") to every note built from a plain MIDI
       number, regardless of key signature, so MuseScore rendered whatever it
       was given (accidentals everywhere, even in keys with no sharps/flats).
       Without an <accidental> tag, MuseScore computes display accidentals
       itself from pitch vs. the active key signature -- correct and far fewer.
    4. Hide rests in every non-primary voice (voice != "1" on staff 1, != "5"
       on staff 2). makeRests(fillGaps=True) fills gaps in *every* voice, and
       MuseScore draws secondary-voice rests floating far above/below the
       staff; print-object="no" keeps them (and their timing) but draws
       nothing.
    5. Insert automatic 8va/8vb brackets (see _apply_ottava) if apply_ottava.

    Returns (new_xml_bytes, stats) where stats has keys max_voices,
    accidentals_stripped, hidden_rests, n_8va, n_8vb, notes_shifted, span_lines.
    """
    header_end = xml_bytes.index(b"<score-partwise")
    header = xml_bytes[:header_end]
    root = ET.fromstring(xml_bytes)

    max_voices = 0
    for measure in root.iter("measure"):
        staff_map: dict = {"1": {}, "2": {}}
        for note_el in measure.findall("note"):
            staff_el, voice_el = note_el.find("staff"), note_el.find("voice")
            if staff_el is None or voice_el is None:
                continue
            m = staff_map.get(staff_el.text)
            if m is not None and voice_el.text not in m:
                base = 1 if staff_el.text == "1" else 5
                m[voice_el.text] = base + len(m)
        max_voices = max(max_voices, len(staff_map["1"]), len(staff_map["2"]))
        assert len(staff_map["1"]) <= 4, f"measure {measure.get('number')} staff 1 needs {len(staff_map['1'])} voices (>4)"
        assert len(staff_map["2"]) <= 4, f"measure {measure.get('number')} staff 2 needs {len(staff_map['2'])} voices (>4)"

        for note_el in measure.findall("note"):
            staff_el, voice_el = note_el.find("staff"), note_el.find("voice")
            if staff_el is None or voice_el is None:
                continue
            m = staff_map.get(staff_el.text)
            if m is not None:
                voice_el.text = str(m[voice_el.text])
        for tag in ("direction", "forward", "backup"):
            for el in measure.findall(tag):
                voice_el = el.find("voice")
                if voice_el is None:
                    continue
                if voice_el.text in staff_map["1"]:
                    voice_el.text = str(staff_map["1"][voice_el.text])
                elif voice_el.text in staff_map["2"]:
                    voice_el.text = str(staff_map["2"][voice_el.text])

    if strip_default_composer:
        identification = root.find("identification")
        if identification is not None:
            for creator_el in list(identification.findall("creator")):
                if creator_el.text == "Music21":
                    identification.remove(creator_el)

    accidentals_stripped = 0
    for note_el in root.iter("note"):
        for acc_el in list(note_el.findall("accidental")):
            note_el.remove(acc_el)
            accidentals_stripped += 1

    hidden_rests = 0
    for note_el in root.iter("note"):
        if note_el.find("rest") is None:
            continue
        staff_el, voice_el = note_el.find("staff"), note_el.find("voice")
        if staff_el is None or voice_el is None:
            continue
        primary_voice = "1" if staff_el.text == "1" else "5"
        if voice_el.text != primary_voice:
            note_el.set("print-object", "no")
            hidden_rests += 1

    ottava_stats = {
        "n_8va": 0, "n_15ma": 0, "n_8vb": 0, "notes_shifted": 0, "span_lines": [], "span_info": [], "counts": {}, "ledger": {},
        "rh_high_not_covered": 0, "rh_over_shifted": 0,
        "cost_plain_total": 0.0, "cost_chosen_total": 0.0,
    }
    if apply_ottava:
        ottava_stats = _apply_ottava(root)

    stats = {
        "max_voices": max_voices,
        "accidentals_stripped": accidentals_stripped,
        "hidden_rests": hidden_rests,
        **ottava_stats,
    }
    return header + ET.tostring(root, encoding="utf-8"), stats


_SIMPLE_QLS = {4, 3, 2, 1.5, 1, 0.75, 0.5, 0.375, 0.25, 0.125}  # notatable as one notehead, <=1 dot
_REST_PIECE_SIZES = (4.0, 2.0, 1.0, 0.5, 0.25)  # plain (undotted) binary values only, for rests


def _is_simple_duration(d) -> bool:
    """True if a Duration is notatable as a single notehead with at most one dot."""
    return d.dots <= 1 and d.type != "complex" and float(d.quarterLength) in _SIMPLE_QLS


def _greedy_binary_pieces(offset_in_measure: float, ql: float) -> List[float]:
    """Break a duration into beat-aligned, undotted binary pieces.

    First cuts to the next integer-beat boundary if not already on one, then
    greedily fills the remainder with the largest size from _REST_PIECE_SIZES
    that both fits and is aligned to a multiple of its own size (so e.g. a
    2-beat piece may only start on beat 0 or 2 of the bar).
    """
    pieces: List[float] = []
    frac = offset_in_measure - math.floor(offset_in_measure + 1e-9)
    if frac > 1e-6:
        to_boundary = min(ql, 1.0 - frac)
        pieces.append(to_boundary)
        offset_in_measure += to_boundary
        ql -= to_boundary
    for size in _REST_PIECE_SIZES:
        while ql + 1e-6 >= size and abs(offset_in_measure % size) < 1e-6:
            pieces.append(size)
            offset_in_measure += size
            ql -= size
    if ql > 1e-6:
        pieces.append(ql)  # shouldn't happen on our 16th-note grid
    return pieces


def _split_complex_durations(staff: "stream.PartStaff") -> Tuple[int, int]:
    """Break up any Rest/Note/Chord with more than one dot into simple, beat-
    aligned pieces (tied, for notes/chords). makeRests(fillGaps=True) happily
    produces e.g. a 3.75-beat rest as a triple-dotted whole rest, which
    MuseScore renders exactly as given -- hard to read. Note: music21's own
    splitAtDurations() only splits durations it can't notate as one notehead
    at all; a triple-dotted value like 3.75 *is* notatable that way, so it
    doesn't split it -- hence the manual beat-aligned decomposition here.
    Returns (rests_split, notes_split).
    """
    from music21 import stream as m21stream, note as m21note, chord as m21chord, tie as m21tie

    rests_split = notes_split = 0
    for measure in staff.getElementsByClass(m21stream.Measure):
        for container in [measure] + list(measure.voices):
            for r in list(container.getElementsByClass(m21note.Rest)):
                if _is_simple_duration(r.duration):
                    continue
                orig_offset = float(r.offset)
                pieces = _greedy_binary_pieces(orig_offset, float(r.duration.quarterLength))
                if len(pieces) <= 1:
                    continue
                container.remove(r)
                off = orig_offset
                for p in pieces:
                    nr = m21note.Rest()
                    nr.duration.quarterLength = p
                    container.insert(off, nr)
                    off += p
                rests_split += 1

            for nc in list(container.getElementsByClass((m21note.Note, m21chord.Chord))):
                if _is_simple_duration(nc.duration):
                    continue
                orig_offset = float(nc.offset)
                pieces = _greedy_binary_pieces(orig_offset, float(nc.duration.quarterLength))
                if len(pieces) <= 1:
                    continue
                first_tie_in = nc.tie is not None and nc.tie.type in ("stop", "continue")
                last_tie_out = nc.tie is not None and nc.tie.type in ("start", "continue")
                new_objs = []
                for p in pieces:
                    obj = m21chord.Chord(list(nc.pitches)) if isinstance(nc, m21chord.Chord) else m21note.Note(nc.pitch)
                    obj.duration.quarterLength = p
                    new_objs.append(obj)
                for i, obj in enumerate(new_objs):
                    if i == 0:
                        obj.tie = m21tie.Tie("continue" if first_tie_in else "start")
                    elif i == len(new_objs) - 1:
                        obj.tie = m21tie.Tie("continue" if last_tie_out else "stop")
                    else:
                        obj.tie = m21tie.Tie("continue")
                container.remove(nc)
                off = orig_offset
                for obj in new_objs:
                    container.insert(off, obj)
                    off += obj.duration.quarterLength
                notes_split += 1
    return rests_split, notes_split


_BAR_LEN = Fraction(4)


def _split_at_barlines(start: Fraction, end: Fraction) -> List[Tuple[Fraction, Fraction]]:
    """Split [start, end) into pieces so none crosses a barline (multiples of 4 beats)."""
    pieces = []
    s = start
    while True:
        b = (s // _BAR_LEN + 1) * _BAR_LEN  # smallest barline strictly greater than s
        if b < end:
            pieces.append((s, b))
            s = b
        else:
            pieces.append((s, end))
            return pieces


def _validate_musicxml(xml_bytes: bytes) -> int:
    """Parse exported MusicXML and assert every voice in every measure ends at
    exactly one bar's worth of duration, and never exceeds it mid-measure --
    the very last measure of a part may end short, but no voice may ever run
    long. Prints every offending measure/voice and raises if any are found, so
    a bad file is never written. Returns the number of measures checked.
    """
    root = ET.fromstring(xml_bytes)
    problems: List[str] = []
    n_measures = 0
    for part in root.findall("part"):
        measures = part.findall("measure")
        divisions = None
        for mi, measure in enumerate(measures):
            n_measures += 1
            is_last = mi == len(measures) - 1
            for attrs in measure.findall("attributes"):
                div_el = attrs.find("divisions")
                if div_el is not None:
                    divisions = int(div_el.text)
            if divisions is None:
                continue
            cursor = 0
            positions: dict = {}
            for child in measure:
                if child.tag == "note":
                    voice_el, dur_el = child.find("voice"), child.find("duration")
                    voice = voice_el.text if voice_el is not None else "1"
                    dur = int(dur_el.text) if dur_el is not None else 0
                    if child.find("chord") is None:  # chord notes stack, don't advance
                        cursor += dur
                    positions.setdefault(voice, []).append(cursor)
                elif child.tag == "backup":
                    dur_el = child.find("duration")
                    cursor -= int(dur_el.text) if dur_el is not None else 0
                elif child.tag == "forward":
                    dur_el, voice_el = child.find("duration"), child.find("voice")
                    cursor += int(dur_el.text) if dur_el is not None else 0
                    if voice_el is not None:
                        positions.setdefault(voice_el.text, []).append(cursor)
            expected = 4 * divisions
            for voice, pos_list in positions.items():
                end, peak = pos_list[-1], max(pos_list)
                bad = peak > expected + 1e-6 or (end > expected + 1e-6) or (not is_last and abs(end - expected) > 1e-6)
                if bad:
                    problems.append(
                        f"measure {measure.get('number')}: voice {voice} -> "
                        f"(start=0.000, end={end / divisions:.3f}) beats "
                        f"[peak={peak / divisions:.3f}, expected={expected / divisions:.3f}]"
                    )
    if problems:
        print("MusicXML validator: FOUND PROBLEMS:")
        for p in problems:
            print(" ", p)
        raise AssertionError(f"MusicXML validator: {len(problems)} measure/voice timing problem(s) found")
    print(f"MusicXML validator: {n_measures} measures OK")
    return n_measures


def _write_musicxml(
    quantized: List[dict], bpm: float, beat_s: float, out_path: Path, title: str, composer: str, apply_ottava: bool
) -> None:
    """Build a 2-staff piano MusicXML score from the final quantized note list.

    One music21 Score with two PartStaff objects (right, then left) under a
    brace StaffGroup -- on export music21 merges these into a single <part>
    with <staves>2</staves> and per-note <staff> tags, which is what lets
    MuseScore show one piano instrument with a normal grand staff instead of
    re-splitting two separate MIDI tracks into four staves.
    """
    from music21 import stream, note, chord, clef, meter, tempo as m21tempo, key as m21key, layout, metadata, tie as m21tie
    from music21.musicxml import m21ToXml

    rh, lh = stream.PartStaff(), stream.PartStaff()
    rh.id, lh.id = "P1-Staff1", "P1-Staff2"
    rh.partName = "Piano"

    # Exact arithmetic: snap every onset/end to the nearest 1/32 beat as a
    # Fraction (not float) before touching music21. Float offsets like
    # 31.9999999 previously landed a note fractionally into the *next* bar,
    # and makeTies/makeMeasures mishandled the sliver -- corrupting the file.
    def _snap32(x_beats: float) -> Fraction:
        return Fraction(round(x_beats * 32), 32)

    notes_by_hand: dict = {"R": [], "L": []}
    for q in quantized:
        if q.get("hand") not in ("R", "L"):
            continue
        start = _snap32(q["start_s"] / beat_s)
        end = _snap32(q["end_s"] / beat_s)
        if end <= start:
            end = start + Fraction(1, 32)
        notes_by_hand[q["hand"]].append({"pitch": q["pitch"], "start": start, "end": end})

    def _insert_notes(staff: "stream.PartStaff", hand_notes: List[dict]) -> None:
        # Group by (start, end): notes sharing both become a Chord; notes that
        # start together but end differently must stay separate Note objects
        # at the same offset (music21/MusicXML express that as separate voices).
        # Each group is then split at barlines *as a group*, piece by piece, so
        # a chord stays a chord in every piece; pieces are tied to each other.
        groups: dict = {}
        for q in hand_notes:
            groups.setdefault((q["start"], q["end"]), []).append(q["pitch"])
        for (start, end), pitches in groups.items():
            pieces = _split_at_barlines(start, end)
            for i, (ps, pe) in enumerate(pieces):
                obj = note.Note(pitches[0]) if len(pitches) == 1 else chord.Chord(pitches)
                obj.duration.quarterLength = pe - ps
                if len(pieces) > 1:
                    obj.tie = m21tie.Tie("start" if i == 0 else "stop" if i == len(pieces) - 1 else "continue")
                staff.insert(ps, obj)

    _insert_notes(rh, notes_by_hand["R"])
    _insert_notes(lh, notes_by_hand["L"])

    rh.insert(0, clef.TrebleClef())
    lh.insert(0, clef.BassClef())
    rh.insert(0, meter.TimeSignature("4/4"))
    lh.insert(0, meter.TimeSignature("4/4"))
    rh.insert(0, m21tempo.MetronomeMark(number=bpm))

    score = stream.Score()
    score.insert(0, rh)
    score.insert(0, lh)
    score.insert(0, layout.StaffGroup([rh, lh], name="Piano", abbreviation="Pno.", symbol="brace", barTogether=True))
    score.metadata = metadata.Metadata(title=title)
    if composer:
        score.metadata.composer = composer

    n_bars = max(1, math.ceil(max(q["end_s"] for q in quantized) / beat_s / 4))
    fifths_per_bar = _fit_key_signature_bars(quantized, beat_s, n_bars)
    seg_start = 0
    for b in range(1, n_bars + 1):
        if b == n_bars or fifths_per_bar[b] != fifths_per_bar[seg_start]:
            f = fifths_per_bar[seg_start]
            rh.insert(seg_start * 4.0, m21key.KeySignature(f))
            lh.insert(seg_start * 4.0, m21key.KeySignature(f))
            print(f"bars {seg_start + 1}-{b}: {_key_name(f)} ({f})")
            seg_start = b

    total_rests_split = total_notes_split = 0
    for label, staff in (("Right", rh), ("Left", lh)):
        try:
            staff.makeMeasures(inPlace=True)
        except Exception as e:
            print(f"WARNING: makeMeasures failed for the {label} hand: {e}")
            continue
        for m in staff.getElementsByClass(stream.Measure):
            try:
                if len(m.flatten().notes) > 1:
                    m.makeVoices(inPlace=True)
            except Exception as e:
                print(f"WARNING: makeVoices failed for the {label} hand, measure {m.number}: {e}")
        try:
            # No-op in practice now: our own barline pre-split above already
            # ties every piece, so no note should still cross a measure
            # boundary. Kept as a safety net for anything we missed.
            staff.makeTies(inPlace=True)
        except Exception as e:
            print(f"WARNING: makeTies failed for the {label} hand: {e}")
        try:
            staff.makeRests(fillGaps=True, inPlace=True)
        except Exception as e:
            print(f"WARNING: makeRests failed for the {label} hand: {e}")
        try:
            r, n = _split_complex_durations(staff)
            total_rests_split += r
            total_notes_split += n
        except Exception as e:
            print(f"WARNING: splitting complex durations failed for the {label} hand: {e}")
    print(f"Complex-duration split: {total_rests_split} rests, {total_notes_split} notes/chords")

    # ScoreExporter.parse() is used directly (not Score.write()/GeneralObjectExporter)
    # because the latter's default deepcopy-and-makeNotation-per-part pass does not
    # reliably preserve the StaffGroup/PartStaff join into one <part> -- verified
    # empirically on this piece; ScoreExporter.parse() on the already-notated score
    # joins correctly and doesn't need makeNotation=False (which chokes on any note
    # whose duration doesn't map to a single notated type).
    sx = m21ToXml.ScoreExporter(score)
    sx.parse()
    raw_bytes = sx.asBytes()

    fixed_bytes, stats = _postprocess_musicxml(raw_bytes, strip_default_composer=not composer, apply_ottava=apply_ottava)
    print(f"Max voices used in any single staff/measure: {stats['max_voices']}")
    print(f"Accidentals stripped (MuseScore recomputes them from the key signature): {stats['accidentals_stripped']}")
    print(f"Secondary-voice rests hidden (print-object=no): {stats['hidden_rests']}")
    print(f"Octave brackets: {stats['n_8va']} 8va, {stats['n_15ma']} 15ma, {stats['n_8vb']} 8vb   notes shifted: {stats['notes_shifted']}")
    for line in stats["span_lines"]:
        print(f"  {line}")
    print(f"RH onset groups with top >= 88 (E6) NOT under 8va/15ma: {stats['rh_high_not_covered']}")
    print(f"RH onset groups under 8va/15ma whose written bottom < 67 (G4): {stats['rh_over_shifted']}")
    for staff, lg in sorted(stats["ledger"].items()):
        print(f"Staff {staff} ({'treble' if staff == '1' else 'bass'}): ledger lines total {lg['lines_before']} -> {lg['lines_after']}; notes with >=4 ledger lines {lg['ge4_before']} -> {lg['ge4_after']}")
    print(f"Ledger-line cost: plain {stats['cost_plain_total']:.1f}, chosen {stats['cost_chosen_total']:.1f}")
    _validate_musicxml(fixed_bytes)  # raises before anything is written if a measure/voice is malformed
    out_path.write_bytes(fixed_bytes)

    written = out_path.read_text(errors="replace")
    n_score_parts = written.count("<score-part id=")  # not "<score-part" alone: matches the "<score-partwise" root too
    has_staves2 = "<staves>2</staves>" in written
    print(f"MusicXML check: <score-part> count={n_score_parts}, <staves>2</staves> present={has_staves2}")
    assert n_score_parts == 1, f"expected exactly one <score-part>, found {n_score_parts}"
    assert has_staves2, "expected <staves>2</staves> in the exported MusicXML"


def cmd_export(args: argparse.Namespace) -> None:
    """Stage 4: quantize detected notes to a tempo grid and write a type-1 MIDI file."""
    notes_path = _resolve(args.notes)
    if not notes_path.exists():
        sys.exit(f"ERROR: {notes_path} not found -- run `python3 video2midi.py detect` first.")
    notes = json.loads(notes_path.read_text())["notes"]
    if not notes:
        sys.exit(f"ERROR: {notes_path} has no notes to export.")

    bpm, grid = args.bpm, args.grid
    beat_s = 60.0 / bpm
    step_s = beat_s * (4.0 / grid)  # seconds per grid subdivision
    print(f"Grid step: {step_s * 1000:.2f} ms ({grid}th notes @ {bpm:g} BPM)")

    starts = np.array([n["start_s"] for n in notes])
    if args.no_quantize:
        phase = 0.0
    else:
        phase, rms_s, on_grid_frac = _find_grid_phase(starts, step_s)
        print(f"Grid phase: {phase * 1000:.2f} ms   RMS residual: {rms_s * 1000:.2f} ms")
        print(f"Onsets within +/-25% of a grid point: {on_grid_frac * 100:.1f}%")
        if on_grid_frac < 0.8:
            print("WARNING: on-grid fraction < 80% -- BPM may be off, or the grid too coarse.")

    quantized = []
    for n in notes:
        start, end = n["start_s"], n["start_s"] + n["duration_s"]
        sound_end = n.get("sound_end_s")  # optional pedal-extended sounding end, same timebase as start_s
        if args.no_quantize:
            q_start, q_end, q_sound_end = start, end, sound_end
        else:
            q_start = round((start - phase) / step_s) * step_s + phase
            q_end = round((end - phase) / step_s) * step_s + phase
            q_end = max(q_end, q_start + step_s)  # minimum one grid unit
            # Same phase/grid transform as start/end, so it lands on the same grid.
            q_sound_end = round((sound_end - phase) / step_s) * step_s + phase if sound_end is not None else None
        quantized.append({
            "pitch": n["pitch"], "name": n.get("name") or note_name(n["pitch"]),
            "start_s": q_start, "end_s": q_end, "sound_end_s": q_sound_end,
        })

    # Normalize so the earliest quantized onset sits at 0; the lead-in (fixed
    # or auto-detected) is applied as a further shift once hands are known.
    baseline_shift = -min(q["start_s"] for q in quantized)
    for q in quantized:
        q["start_s"] += baseline_shift
        q["end_s"] += baseline_shift
        if q["sound_end_s"] is not None:
            q["sound_end_s"] += baseline_shift

    trimmed, dropped = _resolve_collisions(quantized, step_s)
    print(f"Same-pitch collisions: {trimmed} trimmed, {dropped} dropped")
    quantized.sort(key=lambda q: q["start_s"])

    split_hands = not args.no_split_hands
    greedy_hands = None
    if split_hands:
        fallback_count, spans_seen, info = _split_hands(quantized, beat_s, args.hand_split)
        greedy_hands = info["greedy_hands"]
        n_left = sum(1 for q in quantized if q["hand"] == "L")
        n_right = sum(1 for q in quantized if q["hand"] == "R")
        print(f"\nHands ({args.hand_split}): {n_right} notes -> Right, {n_left} notes -> Left")
        print(f"Span > {HAND_MAX_SPAN} fallbacks / events: {fallback_count}")
        top_spans = sorted(spans_seen, reverse=True)[:10]
        print(f"10 largest hand spans (semitones): {top_spans}")
        for name in ("greedy", "beam"):
            if name in info:
                mm = info[name]
                print(f"  [{name:6}] LH>=E5: {mm['lh_ge_e5']:4d}  RH<=E3: {mm['rh_le_e3']:4d}  crossings: {mm['crossings']:4d}  span>{HAND_MAX_SPAN} events: {mm['span_gt_max']:4d}")
    else:
        for q in quantized:
            q["hand"] = None

    # Lead-in: an explicit --lead-in-beats always wins; otherwise search for the
    # shift that best aligns LEFT-hand chords with downbeats (needs "hand" set).
    # The search always runs on the greedy hand assignment so the bar lines (and
    # therefore every onset) don't depend on which hand-split mode was chosen.
    if args.lead_in_beats is not None:
        lead_in_beats = args.lead_in_beats
    elif split_hands and not args.no_auto_downbeat:
        chosen_hands = [q["hand"] for q in quantized]
        for q, h in zip(quantized, greedy_hands):
            q["hand"] = h
        lead_in_beats, lead_in_score = _choose_lead_in(quantized, beat_s)
        for q, h in zip(quantized, chosen_hands):
            q["hand"] = h
        print(f"\nAuto-downbeat: lead-in={lead_in_beats:g} beats (score={lead_in_score:.2f}, from greedy hands)")
    else:
        lead_in_beats = LEAD_IN_BEATS
    lead_in_shift = lead_in_beats * beat_s
    for q in quantized:
        q["start_s"] += lead_in_shift
        q["end_s"] += lead_in_shift
        if q["sound_end_s"] is not None:
            q["sound_end_s"] += lead_in_shift

    hand_lists = [[q for q in quantized if q["hand"] == h] for h in ("L", "R")] if split_hands else [quantized]

    if not args.no_fill_rests:
        extended = sum(_fill_rests(hand_notes, beat_s, args.fill_rest_max_beats) for hand_notes in hand_lists)
        print(f"\nRests filled (note end extended to close a short gap): {extended}")

    if args.single_voice:
        shortened = lengthened = 0
        for hand_notes in hand_lists:
            s, l = _make_single_voice(hand_notes, step_s)
            shortened += s
            lengthened += l
        for hand_notes in hand_lists:
            _assert_single_voice(hand_notes)
        print(f"Single-voice: {shortened} notes shortened, {lengthened} notes lengthened")

    # Filling/single-voicing can only ever push an end up to another note's
    # start, but re-run the same-pitch guard in case that created a fresh
    # overlap. Key on (hand, pitch) when hands are split: the same pitch
    # overlapping across two different hands/tracks is legal MIDI on separate
    # staves and must not be trimmed away.
    trimmed2, dropped2 = _resolve_collisions(quantized, step_s, by_hand=split_hands)
    print(f"Same-pitch collisions after fill/single-voice: {trimmed2} trimmed, {dropped2} dropped")
    if split_hands and args.single_voice:
        # Single-voicing already made every within-hand pair identical or
        # non-overlapping, so a same-(hand,pitch) collision here would mean
        # that guarantee broke.
        assert trimmed2 == 0 and dropped2 == 0, (
            f"unexpected same-hand collision survived single-voicing: trimmed={trimmed2} dropped={dropped2}"
        )
        for hand_notes in ([q for q in quantized if q["hand"] == h] for h in ("L", "R")):
            _assert_single_voice(hand_notes)
        print("Single-voice guarantee holds after the final collision trim.")
    quantized.sort(key=lambda q: q["start_s"])

    out_path = _resolve(args.out)  # MIDI path; the MusicXML path is derived from this
    want_midi = args.format in ("midi", "both")
    want_musicxml = args.format in ("musicxml", "both")

    if want_midi:
        tempo = mido.bpm2tempo(bpm)
        mid = mido.MidiFile(type=1, ticks_per_beat=TICKS_PER_BEAT)

        def _note_events(notes_subset: List[dict], channel: int) -> List[tuple]:
            # (tick, is_note_on, message) -- sort with note_off (0) before note_on (1)
            # at equal ticks so a repeated note's off/on pair doesn't collapse to nothing.
            evts = []
            for q in notes_subset:
                start_tick = round(mido.second2tick(q["start_s"], TICKS_PER_BEAT, tempo))
                end_tick = round(mido.second2tick(q["end_s"], TICKS_PER_BEAT, tempo))
                evts.append((end_tick, 0, mido.Message("note_off", note=q["pitch"], velocity=0, channel=channel)))
                evts.append((start_tick, 1, mido.Message("note_on", note=q["pitch"], velocity=MIDI_VELOCITY, channel=channel)))
            evts.sort(key=lambda e: (e[0], e[1]))
            return evts

        def _append_events(track: mido.MidiTrack, evts: List[tuple]) -> int:
            last = 0
            for tick, _, msg in evts:
                msg.time = tick - last
                track.append(msg)
                last = tick
            track.append(mido.MetaMessage("end_of_track", time=0))
            return last

        meta = mido.MidiTrack()
        mid.tracks.append(meta)
        meta.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))
        meta.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))

        if split_hands:
            meta.append(mido.MetaMessage("end_of_track", time=0))
            # Right Hand first so MuseScore puts it on the upper (treble) staff.
            track_specs = [
                ("Right Hand", 0, [q for q in quantized if q["hand"] == "R"]),
                ("Left Hand", 1, [q for q in quantized if q["hand"] == "L"]),
            ]
        else:
            track_specs = [(TRACK_NAME, 0, quantized)]

        last_tick = 0
        for name, channel, subset in track_specs:
            tr = meta if not split_hands else mido.MidiTrack()
            if tr is not meta:
                mid.tracks.append(tr)
            tr.append(mido.MetaMessage("track_name", name=name, time=0))
            tr.append(mido.Message("program_change", program=0, channel=channel, time=0))
            last_tick = max(last_tick, _append_events(tr, _note_events(subset, channel)))

        mid.save(str(out_path))

        total_beats = last_tick / TICKS_PER_BEAT
        bars, beat_in_bar = divmod(total_beats, 4.0)
        print(f"\nNotes exported: {len(quantized)}")
        print(f"Total duration: {int(bars)} bars + {beat_in_bar:.2f} beats ({total_beats:.2f} beats, {total_beats * beat_s:.2f}s)")
        print(f"Tick range: 0 .. {last_tick}")
        print(f"Wrote {out_path}")

        # Read the file back and sanity-check each track round-trips.
        expected_counts = {name: len(subset) for name, _, subset in track_specs}
        readback = mido.MidiFile(str(out_path))
        print("\nReadback:")
        for tr in readback.tracks:
            ons = sum(1 for m in tr if m.type == "note_on" and m.velocity > 0)
            offs = sum(1 for m in tr if m.type == "note_off" or (m.type == "note_on" and m.velocity == 0))
            if ons == 0 and offs == 0:
                continue
            name = next((m.name for m in tr if m.type == "track_name"), "?")
            expected = expected_counts.get(name, -1)
            assert ons == offs == expected, f"readback mismatch on track '{name}': note_on={ons} note_off={offs} expected={expected}"
            print(f"  {name:<12} note_on={ons:5d}  note_off={offs:5d}  notes={expected}")
    else:
        total_beats = max(q["end_s"] for q in quantized) / beat_s
        print(f"\nNotes exported: {len(quantized)}  (total {total_beats:.2f} beats)")

    quant_path = out_path.parent / "output_notes_quantized.json"
    quant_path.write_text(json.dumps([
        {
            "pitch": q["pitch"], "name": q["name"],
            "start_beats": q["start_s"] / beat_s,
            "duration_beats": (q["end_s"] - q["start_s"]) / beat_s,
            "hand": q["hand"],
        }
        for q in quantized
    ], indent=2))
    print(f"Wrote {quant_path}")

    if want_musicxml:
        if not split_hands:
            print("WARNING: MusicXML export needs hand-split notes; skipping (--no-split-hands was given).")
        else:
            xml_path = out_path.with_suffix(".musicxml")
            _write_musicxml(quantized, bpm, beat_s, xml_path, args.title, SCORE_COMPOSER, OTTAVA_ENABLED and not args.no_ottava)
            print(f"Wrote {xml_path}")

    quantized.sort(key=lambda q: q["start_s"])
    print(f"\n{'beat':>8}  {'MIDI':>4}  {'hand':<4}  name")
    for q in quantized[:16]:
        print(f"{q['start_s'] / beat_s:8.3f}  {q['pitch']:>4}  {q['hand'] or '-':<4}  {q['name']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="video2midi: piano-roll video -> MIDI")
    sub = parser.add_subparsers(dest="command", required=True)

    calib = sub.add_parser("calibrate", help="Stage 1: locate keyboard keys across sample frames")
    calib.add_argument("--time", type=float, default=None, help="use only this single frame timestamp, seconds")
    calib.add_argument("--out", type=str, default="calibration.png", help="annotated PNG output path")
    calib.add_argument("--json", type=str, default="calibration.json", help="calibration JSON output path")

    detect = sub.add_parser("detect", help="Stage 2/3: detect falling notes")
    detect.add_argument("--max-seconds", type=float, default=MAX_SECONDS, help="how many seconds of video to scan (0 = whole video)")
    detect.add_argument("--step", type=int, default=FRAME_STEP, help="frame stride")
    detect.add_argument("--debug", action="store_true", help="write annotated debug frames + a mask preview")
    detect.add_argument("--out", type=str, default="notes.json", help="detected-notes JSON output path")

    export = sub.add_parser("export", help="Stage 4: quantize notes.json and export a MIDI file")
    export.add_argument("--notes", type=str, default="notes.json", help="input notes JSON path")
    export.add_argument("--out", type=str, default=MIDI_PATH, help="output MIDI file path")
    export.add_argument("--bpm", type=float, default=BPM, help="tempo in beats per minute")
    export.add_argument("--grid", type=int, default=QUANT_GRID, help="quantization grid: subdivisions per whole note (16 or 32)")
    export.add_argument("--lead-in-beats", type=float, default=None, help="beats of lead-in before the first note; given explicitly, this overrides --auto-downbeat")
    export.add_argument("--no-auto-downbeat", action="store_true", help="don't search for a metrically-sensible lead-in; use --lead-in-beats (or 0) as-is")
    export.add_argument("--no-quantize", action="store_true", help="skip quantization; still apply the lead-in shift")
    export.add_argument("--no-split-hands", action="store_true", help="keep a single 'piano' track instead of splitting into Right/Left Hand tracks")
    export.add_argument("--hand-split", choices=["greedy", "beam"], default=HAND_SPLIT_MODE, help="hand assignment: greedy EMA tracker or beam search over playability costs")
    export.add_argument("--no-fill-rests", action="store_true", help="don't extend note ends to close short inter-note gaps")
    export.add_argument("--fill-rest-max-beats", type=float, default=FILL_REST_MAX_BEATS, help="max gap (beats) to close between two notes in the same hand when filling rests")
    export.add_argument("--single-voice", action="store_true", help="force one voice per hand by cutting held notes at the next onset (default keeps every note at its own length; MuseScore then uses a 2nd voice where a hand holds a note under a moving line)")
    export.add_argument("--no-ottava", action="store_true", help="don't auto-insert 8va/8vb brackets for passages with too many ledger lines")
    export.add_argument("--format", choices=["midi", "musicxml", "both"], default=EXPORT_FORMAT, help="output format(s) to write")
    export.add_argument("--title", type=str, default=SCORE_TITLE, help="score title (MusicXML metadata)")

    args = parser.parse_args()
    if args.command == "calibrate":
        cmd_calibrate(args)
    elif args.command == "detect":
        cmd_detect(args)
    elif args.command == "export":
        cmd_export(args)


if __name__ == "__main__":
    main()
