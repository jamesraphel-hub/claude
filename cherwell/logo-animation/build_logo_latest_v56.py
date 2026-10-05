import numpy as np, subprocess, glob, sys
from PIL import Image
from scipy import ndimage as ndi

S = "/private/tmp/claude-501/-Users-jamesraphel-Documents-Claude/df569d2a-ebbe-413b-90c4-6f0fa460594d/scratchpad/"
REAL = "/Volumes/Cherwell/Cherwell Media/Brand Assets/Logo/C&C Logo Brown Background_Updated.jpg"
OUT = sys.argv[1] if len(sys.argv) > 1 else S + "logo_anim_v1.mp4"
FPS, DUR = 30, 7.2
W, H = 1080, 1920

real = np.array(Image.open(REAL).convert("RGB")).astype(np.float32)
bg = real[60:160, 60:160].reshape(-1, 3).mean(0)
L = real.mean(2)
bgL = bg.mean()
alpha = np.clip((L - bgL) / (255 - bgL), 0, 1)

# --- split elements -------------------------------------------------
lab, n = ndi.label(alpha > 0.15, structure=np.ones((3, 3)))
cy, cx = 902, 540
amp = np.zeros((H, W), bool)
ring = np.zeros((H, W), bool)
word = np.zeros((H, W), bool)
for i in range(1, n + 1):
    m = lab == i
    ys, xs = np.where(m)
    if m.sum() < 4:
        continue
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    if y0 > 1030:
        word |= m
    elif y1 < 1030 and (y1 - y0) < 140 and (x1 - x0) < 140 and abs((x0 + x1) / 2 - cx) < 30 and abs((y0 + y1) / 2 - cy) < 30 and (y1 - y0) > 80:
        amp |= m
    elif y1 < 1030:
        ring |= m
print("amp px", amp.sum(), "ring px", ring.sum(), "word px", word.sum())
grow = lambda m, k: ndi.binary_dilation(m, iterations=k)
yy0 = np.mgrid[0:H, 0:W][0]
low = alpha > 0.03
R_amp = grow(amp, 3)
R_word = grow(low & (yy0 > 1030), 3)
R_ring = grow(low & (yy0 <= 1030), 3) & ~R_amp & ~R_word
D = (real - bg[None, None, :]).astype(np.float32)          # exact original pixel minus flat bg
A_amp = R_amp.astype(np.float32)
A_ring = R_ring.astype(np.float32)
A_word = R_word.astype(np.float32)
print("region px", R_amp.sum(), R_ring.sum(), R_word.sum())
ys, xs = np.where(amp)
real_bb = (xs.min(), ys.min(), xs.max(), ys.max())
print("real amp bbox", real_bb)

# --- hand-written stroke order (pen-reference style), logo pixel coords ---
from scipy.interpolate import splprep, splev
STROKES = [
    # 1: starts on the bottom-right corner (pointed tip), goes UP the diagonal to the junction, up the stem, over the loop, down to the crossing, then round the lower-left bowl
    [(592,942),(582,943),(572,941),(562,938),(552,931),(542,920),(532,909),(525,900),(519,895),(512,888),(509,877),(509,866),(515,853),(526,848),(536,850),(541,858),(538,870),(530,881),(521,889),(508,894),(496,908),(492,923),(495,938),(506,947),(522,951),(536,948),(545,944)],
    # 2: right stem rising from its base on the diagonal
    [(565,925),(566,919),(567,913),(568,907),(569,901)],
    # 3: the short cross flick along the crossbar, left to right
    [(546,895),(558,894),(570,894),(582,893),(593,893)],
]
pts_all = []; s_all = []; stroke_id = []
tot = 0.0
lens = []
curves = []
for st in STROKES:
    P = np.array(st, float).T
    tck, u = splprep(P, s=1.5, k=3)
    uu = np.linspace(0, 1, 400)
    xs_, ys_ = splev(uu, tck)
    seg = np.hypot(np.diff(xs_), np.diff(ys_)); cum = np.r_[0, np.cumsum(seg)]
    curves.append((xs_, ys_, cum)); lens.append(cum[-1])

total = sum(lens)
import heapq
R_amp = grow(amp, 3)
yy0 = np.mgrid[0:H, 0:W][0]
low = alpha > 0.03
R_word = grow(low & (yy0 > 1030), 3)
R_ring = grow(low & (yy0 <= 1030), 3) & ~R_amp & ~R_word
D = (real - bg[None, None, :]).astype(np.float32)
A_amp = R_amp.astype(np.float32); A_ring = R_ring.astype(np.float32); A_word = R_word.astype(np.float32)

ease = lambda x: x * x * (3 - 2 * x)
easeo = lambda x: 1 - (1 - x) ** 3
clip01 = lambda x: float(min(1, max(0, x)))
yy, xx = np.mgrid[0:H, 0:W]
ang = (np.arctan2(xx - cx, -(yy - cy)) + 2 * np.pi) / (2 * np.pi) % 1.0
ringmask_soft = 0.025

# ---------- timeline ----------
WRITE_S, WRITE_E = 0.50, 3.40
RING_S, RING_E = 3.50, 6.00
WORD_S, WORD_E = 5.80, 6.60
GAP = 0.10
n = len(curves)
avail = (WRITE_E - WRITE_S) - GAP * (n - 1)
seg_t = [avail * l / total for l in lens]
seg_start = []; tc = WRITE_S
for k in range(n):
    seg_start.append(tc); tc += seg_t[k] + GAP

# ---------- gated wavefront: ink flows through the letter from the pen tip; each part of the letter opens when the pen reaches it ----------
import heapq
HI = 4
CX0, CY0, CW, CH = 478, 836, 128, 128
from PIL import Image as _I
_a = np.where(grow(amp, 3), alpha, 0).astype(np.float32)[CY0:CY0+CH, CX0:CX0+CW]
alpha_hi = np.clip(np.array(_I.fromarray(_a, mode="F").resize((CW*HI, CH*HI), _I.BICUBIC)), 0, 1)
mask_hi = np.array(_I.fromarray(R_amp[CY0:CY0+CH, CX0:CX0+CW].astype(np.uint8)*255).resize((CW*HI, CH*HI), _I.BILINEAR)) > 127
yy_h, xx_h = np.mgrid[0:CH*HI, 0:CW*HI]
X_h = CX0 + (xx_h + 0.5) / HI - 0.5; Y_h = CY0 + (yy_h + 0.5) / HI - 0.5
c0x, c0y, cum0 = curves[0]
ts0 = seg_start[0] + seg_t[0] * (cum0 / cum0[-1])
def nearest_idx(px, py, lo=0, hi=None):
    hi = len(c0x) if hi is None else hi
    return lo + int(np.argmin(np.hypot(c0x[lo:hi] - px, c0y[lo:hi] - py)))
j_junc = nearest_idx(519, 895, 0, len(c0x) // 3)
j_top = nearest_idx(541, 858)
j_cross = nearest_idx(521, 889, j_top)
j_bowl = j_cross + int(np.argmax(c0x[j_cross:] <= 511))          # first sample that has left the junction on the way round the bowl
j_apex = nearest_idx(526, 848, j_junc)
groups = [(c0x[:j_junc], c0y[:j_junc]), (c0x[j_junc:j_apex], c0y[j_junc:j_apex]), (c0x[j_apex:j_bowl], c0y[j_apex:j_bowl]), (c0x[j_bowl:], c0y[j_bowl:]),
          (curves[1][0], curves[1][1]), (curves[2][0], curves[2][1])]
gates = [ts0[0], ts0[0], ts0[j_apex] - 0.04, ts0[j_bowl], seg_start[1], seg_start[2]]
print("gates", [round(float(g), 2) for g in gates])
from scipy.spatial import cKDTree
_q = np.c_[X_h.ravel(), Y_h.ravel()]
dist_g = np.stack([cKDTree(np.c_[gx, gy]).query(_q)[0].reshape(X_h.shape) for gx, gy in groups])
dist_g[4] -= 4.5
dist_g[0] -= 3.0
dist_g[1] -= 4.0      # the stem owns a little more of the junction, so its foot is drawn by the stem stroke
region = dist_g.argmin(0)
# straighten the region borders: majority vote in a small window
votes = np.stack([ndi.uniform_filter((region == g).astype(np.float32), size=13) for g in range(len(groups))])
region = votes.argmax(0)
# per-pixel gate = the time the pen's nib reaches that cross-section of the letter (so every part is drawn in pen order, no matter how fast ink could flow)
ts1 = seg_start[1] + seg_t[1] * (curves[1][2] / curves[1][2][-1]); ts2 = seg_start[2] + seg_t[2] * (curves[2][2] / curves[2][2][-1])
g_ts = [ts0[:j_junc], ts0[j_junc:j_apex], ts0[j_apex:j_bowl], ts0[j_bowl:], ts1, ts2]
gate_img = np.zeros(region.shape)
for g_, (gx_, gy_) in enumerate(groups):
    mk_ = region == g_
    if not mk_.any(): continue
    _d, _i = cKDTree(np.c_[gx_, gy_]).query(np.c_[X_h[mk_], Y_h[mk_]])
    gate_img[mk_] = g_ts[g_][_i] - 0.03
speed = cum0[-1] / seg_t[0]                                # px per second along the main stroke
NB = [(-1,0,1.0),(1,0,1.0),(0,-1,1.0),(0,1,1.0),(-1,-1,1.4142),(-1,1,1.4142),(1,-1,1.4142),(1,1,1.4142),
      (-2,-1,2.2361),(-2,1,2.2361),(2,-1,2.2361),(2,1,2.2361),(-1,-2,2.2361),(1,-2,2.2361),(-1,2,2.2361),(1,2,2.2361)]
Hh, Wh = mask_hi.shape
T = np.full((Hh, Wh), np.inf)
sx_, sy_ = curves[0][0][0], curves[0][1][0]
seed = (np.hypot(X_h - sx_, Y_h - sy_) <= 1.2) & mask_hi
pq = []
for y_, x_ in zip(*np.where(seed)):
    T[y_, x_] = ts0[0]; heapq.heappush(pq, (ts0[0], y_, x_))
while pq:
    t_, y_, x_ = heapq.heappop(pq)
    if t_ > T[y_, x_]: continue
    for dy_, dx_, c_ in NB:
        ny, nx = y_ + dy_, x_ + dx_
        if 0 <= ny < Hh and 0 <= nx < Wh and mask_hi[ny, nx]:
            nt = max(gate_img[ny, nx], t_ + c_ / (HI * speed))
            if nt < T[ny, nx]:
                T[ny, nx] = nt; heapq.heappush(pq, (nt, ny, nx))
T_hi = np.where(np.isfinite(T) & mask_hi, T, 1e9).astype(np.float32)
print("reveal range", T_hi[mask_hi].min(), T_hi[mask_hi & (T_hi<1e8)].max(), "unreached", int((mask_hi & (T_hi>=1e8)).sum()))


# ---------- ring: one line per circle, dots, letters fade in whole ----------
RX0, RY0, RX1, RY1 = 410, 770, 675, 1035
rc_cx, rc_cy = 542.0, 902.5
_al = np.where(R_ring, alpha, 0.0)
_lab, _n = ndi.label(_al > 0.12, structure=np.ones((3, 3)))
_idx = ndi.distance_transform_edt(_lab == 0, return_distances=False, return_indices=True)
_near = _lab[_idx[0], _idx[1]]
_near = np.where(R_ring, _near, 0)                       # every ring pixel -> nearest ring component
_yy, _xx = np.mgrid[0:H, 0:W]
_th = (np.degrees(np.arctan2(_xx - rc_cx, -(_yy - rc_cy))) + 360.0) % 360.0
_rad = np.hypot(_xx - rc_cx, _yy - rc_cy)
arcs, dots, letters = [], [], []
for i in range(1, _n + 1):
    m = _lab == i
    ar = int(m.sum())
    if ar < 4: continue
    ys_, xs_ = np.where(m)
    w_, h_ = xs_.max() - xs_.min() + 1, ys_.max() - ys_.min() + 1
    rmean = float(_rad[m].mean()); rstd = float(_rad[m].std())
    if ar > 400 and rstd < 3.0: arcs.append((i, rmean))
    elif w_ <= 11 and h_ <= 11 and ar >= 60 and ar / (w_ * h_) > 0.6: dots.append(i)
    else: letters.append((i, float(xs_.mean()), float(ys_.mean())))
print("ring comps: arcs", len(arcs), "dots", len(dots), "letters", len(letters))
ring_f_shape = (RY1 - RY0, RX1 - RX0)
arc_defs = []   # (mask, u) per arc in crop coords
for (i, rmean) in arcs:
    mk = (_near == i)
    th = _th[_lab == i]
    ang_sorted = np.sort(th)
    gaps = np.diff(np.r_[ang_sorted, ang_sorted[0] + 360.0])
    g = int(np.argmax(gaps)); start = ang_sorted[(g + 1) % len(ang_sorted)]; span = 360.0 - gaps[g]
    u = ((_th - start) % 360.0) / max(span, 1e-6)
    ys_, xs_ = np.where(_lab == i); cy_ = ys_.mean()
    arc_defs.append(dict(mask=mk[RY0:RY1, RX0:RX1], u=u[RY0:RY1, RX0:RX1].astype(np.float32), outer=rmean > 100, bottom=(cy_ > rc_cy)))
dot_mask = np.isin(_near, dots)[RY0:RY1, RX0:RX1]
let_id = np.full(ring_f_shape, -1, int)
let_t = []
order = sorted(letters, key=lambda L: L[1])
top_ids = [L for L in order if L[2] < rc_cy - 20]; bot_ids = [L for L in order if L[2] >= rc_cy - 20]
for k_, L in enumerate(top_ids): let_t.append((L[0], 5.20 + 0.04 * k_))
for k_, L in enumerate(bot_ids): let_t.append((L[0], 5.30 + 0.04 * k_))
LUT = np.zeros(_n + 1, int) - 1; LT = np.zeros(len(let_t))
for j_, (i, t0_) in enumerate(let_t): LUT[i] = j_; LT[j_] = t0_
let_id = LUT[_near][RY0:RY1, RX0:RX1]
# arc timing (clockwise circuit, bottom arc first, then top arc; outer circle, then inner circle)
def arc_time(a):
    if a['outer']: return (3.98, 4.43) if not a['bottom'] else (3.50, 3.95)
    return (4.82, 5.12) if not a['bottom'] else (4.50, 4.80)
DOT_T = 5.15
D_ring = D[RY0:RY1, RX0:RX1]

p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                      "-c:v", "libx264", "-preset", "slow", "-crf", "10", "-vf", "scale=out_color_matrix=bt601:out_range=pc,format=yuvj420p", "-color_range", "pc", "-colorspace", "smpte170m", "-pix_fmt", "yuvj420p", "-movflags", "+faststart", OUT], stdin=subprocess.PIPE)
_r = 18
_yy, _xx = np.ogrid[-_r:_r+1, -_r:_r+1]
_FP = (_xx**2 + _yy**2) <= _r**2
nf = int(FPS * DUR)
for k in range(nf):
    t = k / FPS
    out = np.broadcast_to(bg[None, None, :], (H, W, 3)).copy()
    if t >= WRITE_E + 0.15:
        out += D * A_amp[..., None]                                   # exact original pixels once the stroke is done
    elif t >= WRITE_S:
        rv_h = np.clip((t - T_hi) / 0.07, 0, 1)
        a_raw = alpha_hi * rv_h
        a_o = ndi.grey_closing(ndi.grey_opening(a_raw, footprint=_FP), footprint=_FP); a_o = np.clip((ndi.gaussian_filter(a_o, 5.0) - 0.5) * 2.5 + 0.5, 0, 1)   # round off nubs and notches narrower than ~5px while the stroke is growing
        w_ = clip01((t - 3.20) / 0.35)                                                 # ... and ease back to the exact shape of the logo at the end
        a_h = a_o * (1 - w_) + a_raw * w_
        a1 = a_h.reshape(CH, HI, CW, HI).mean((1, 3))
        reg = out[CY0:CY0+CH, CX0:CX0+CW]
        reg[:] = reg * (1 - a1[..., None]) + 255.0 * a1[..., None]
    if t >= RING_S:
        f = np.zeros(ring_f_shape, np.float32)
        for a_ in arc_defs:
            t0_, t1_ = arc_time(a_)
            if t < t0_: continue
            p_ = clip01((t - t0_) / (t1_ - t0_))
            eps = 0.015
            ff = np.clip((p_ * (1 + eps) - a_['u']) / eps, 0, 1)
            f = np.where(a_['mask'], ff, f)
        if t >= DOT_T:
            f = np.where(dot_mask, ease(clip01((t - DOT_T) / 0.2)), f)
        lm = let_id >= 0
        if lm.any():
            tl = LT[np.maximum(let_id, 0)]
            fl = np.clip((t - tl) / 0.14, 0, 1); fl = fl * fl * (3 - 2 * fl)
            f = np.where(lm, fl, f)
        out[RY0:RY1, RX0:RX1] += D_ring * f[..., None]
    if t >= WORD_S:
        pw = clip01((t - WORD_S) / (WORD_E - WORD_S))
        fa = easeo(pw)
        dy = int(round((1 - easeo(pw)) * 14))
        out += np.roll(D * A_word[..., None], dy, axis=0) * fa
    p.stdin.write(np.clip(out + 0.5, 0, 255).astype(np.uint8).tobytes())
p.stdin.close(); p.wait()
print("done", OUT)
