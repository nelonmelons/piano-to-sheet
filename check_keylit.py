"""Cross-check notes.json onsets against key-lighting events on the keyboard.

Independent of bar detection: a key glows orange while pressed, so rising edges
of key brightness are ground-truth note-ons. Usage: python3 check_keylit.py [max_seconds]
"""
import json, sys
import cv2, numpy as np

MAX_S = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
TOL_S = 0.08  # onset match tolerance

c = json.load(open('calibration.json'))
keys = c['keys'] if 'keys' in c else c['key_regions']
cap = cv2.VideoCapture('video.mp4'); fps = cap.get(cv2.CAP_PROP_FPS)
prev = {k['midi']: False for k in keys}; events = []
for idx in range(int(MAX_S * fps)):
    ok, fr = cap.read()
    if not ok: break
    hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
    for k in keys:
        y = 780 if k['is_black'] else 930
        px = hsv[y-3:y+3, k['x_center']-3:k['x_center']+3]
        lit = ((px[..., 1] > 120) & (px[..., 2] > 120)).mean() > 0.5
        if lit and not prev[k['midi']]: events.append((idx / fps, k['midi'], k['name']))
        prev[k['midi']] = lit

det = [(n['start_s'], n['pitch'], n['name']) for n in json.load(open('notes.json'))['notes'] if n['start_s'] < MAX_S]
used = set(); missed = []; offs = []
for t, p, nm in events:
    m = [i for i, (t2, p2, _) in enumerate(det) if p2 == p and abs(t2 - t) < TOL_S and i not in used]
    if m: used.add(m[0]); offs.append(det[m[0]][0] - t)
    else: missed.append((round(t, 3), nm))
extra = [(round(t, 3), nm) for i, (t, p, nm) in enumerate(det) if i not in used]
print(f"key-lit onsets <{MAX_S}s: {len(events)}   detected onsets: {len(det)}")
print("key-lit WITHOUT detection:", missed)
print("detections WITHOUT key-lit:", extra)
if offs: print(f"onset offset (detected - keylit): mean {np.mean(offs):+.3f}s  sd {np.std(offs):.3f}s")
