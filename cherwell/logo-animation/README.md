# Cherwell logo animation (local build, no Seedance)

Goal: brown plate -> ampersand drawn like a pen stroke -> ring draws line by line -> letters fade in -> CHERWELL & CO. wordmark. Final frame must equal `/Volumes/Cherwell/Cherwell Media/Brand Assets/Logo/C&C Logo Brown Background_Updated.jpg` exactly.

Latest: `build_logo_latest_v56.py` -> `cherwell_logo_reveal_v56.mp4` (1080x1920, 30fps, 7.2s).
Run: `python3 build_logo_latest_v56.py out.mp4`. The script reads the logo JPG from the SSD path above and the Seedance-free pipeline needs numpy, scipy, Pillow, ffmpeg. NOTE: it still reads/writes a scratchpad path variable `S` near the top; edit if the scratchpad is gone.

Stroke order (settled with user, reference = IMG_5474 2.mov storyboard): start on the bottom-right pointed tip, go UP the diagonal, up the left stem, over the top loop, down the right side of the loop, round the lower-left bowl, then right stem, then short cross flick along the crossbar. Constant speed ~2.9s, then pen stops and drawing stays. Stroke thickness = the logo's own.
Ring: outer circle one line (bottom arc then top arc, clockwise), inner circle after, two dots, then letters fade in one by one, then wordmark.
Open issue: tiny nubs where stem/bowl/diagonal merge look "frizzy"; v56 rounds nubs (open/close r=4.5px + gaussian smooth) while drawing then blends back to exact shape.
Pending: user confirms, then render 4K (2160x3840) rebuilt from `C&Cfulllogo_transparent_nobuffer.png` (1280px emblem) so ring text stays sharp.


## v60 (2026-10-05)
Root cause of 'broken font' look: ampersand was reconstructed from the ~105px plate glyph. v60 draws the pen reveal on the 1280px emblem (`C&Cfulllogo_transparent_nobuffer.png`), no morphological smoothing, area-downsampled per frame via `emblem_to_plate.npy` (scale 0.2286 + offset, fitted RMS 0.05). Build: `python3 build_logo_latest_v60.py out.mp4`. Open: small stem-foot nub on diagonal's right edge until the stem is drawn (~2.9s).

## v62 (2026-10-05) - current
Build: `python3 build_logo_latest_v62.py out.mp4` (needs `emblem_to_plate.npy`). Pen model: round nib swept along the pen path snapped to the glyph centreline, radius = glyph's own half-thickness; a pixel appears the first time the nib covers it (replaces the region-gated Dijkstra, which froze the diagonal and popped the loop on at once). Pen timing keyed to the reference footage (waypoints in the script, ~0.1s ahead of storyboard labels since reference ink lags the nib): write-on 0.1s to 2.8s, ring/wordmark shifted 0.55s earlier, total 6.7s. Final frame crossfades to exact original pixels. Open: tiny nick on diagonal's right edge; ring letters + wordmark still low-res plate pixels; 4K pending.

## v63 (2026-10-05) - current
`python3 build_logo_latest_v63.py out.mp4`. v62 + fix for the 'nick' on the diagonal's upper edge: the nib sweep inked a small strip of the stem's foot early. The diagonal's edge now continues straight across the foot (chord between the two V corners, hardcoded emblem coords (704,754)-(724,771) for the fixed artwork) and the foot waits for the stem stroke (~1.85s). Also: nib radius follows stroke taper, small out-of-sequence patches smoothed. Open: sub-pixel kink on stem inner edge at the loop crossing; ring letters + wordmark still low-res plate pixels; 4K pending.

## 4K v70 (2026-10-05)
`python3 build_logo_4k_v70.py out.mp4` -> 2160x3840, 30fps, 6.7s, same write-on/timing as v63. Ampersand, ring arcs, dots and ring letters are drawn from the 1280px emblem (black file `C&Cfulllogo_transparent_nobuffer.png` alpha, painted white; the white file has ~7% less ink so it is NOT used) area-resampled onto the 4K grid via `emblem_to_plate.npy`. Wordmark 'CHERWELL & CO.' only exists in the 1080 plate, so it is upscaled 2x (bicubic + edge re-sharpen). Background = exact RGB(111,84,55); RGB->YUV (full-range BT.601) is done in numpy so the flat colour decodes exactly (swscale gave 110,85,54). Saved to ~/Desktop/Rendered Videos-Social Media Post/cherwell_logo_reveal_4K_v70.mp4.
Colour tags (4K v70): pixel values are exactly RGB(111,84,55) (own BT.601 full-range conversion), tagged primaries=bt709, transfer=iec61966-2-1 (sRGB), matrix=smpte170m via `setparams` (the plain -color_* flags did NOT write the tags). Earlier all-smpte170m tagging made QuickTime show the brown differently from the JPG. The 1080p builds (v55-v63) still use the old tags.
