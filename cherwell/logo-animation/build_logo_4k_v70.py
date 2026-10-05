import numpy as np, subprocess, glob, sys
from PIL import Image
from scipy import ndimage as ndi

S = "/private/tmp/claude-501/-Users-jamesraphel-Documents-Claude/df569d2a-ebbe-413b-90c4-6f0fa460594d/scratchpad/"
REAL = "/Volumes/Cherwell/Cherwell Media/Brand Assets/Logo/C&C Logo Brown Background_Updated.jpg"
OUT = sys.argv[1] if len(sys.argv) > 1 else S + "logo_4k.mp4"
FPS, DUR = 30, 6.7
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
WRITE_S, WRITE_E = 0.15, 2.85
NIB = 0.98
SH = 0.55                                   # ring / wordmark run this much earlier than in v60
RING_S, RING_E = 3.50, 6.00
WORD_S, WORD_E = 5.80, 6.60
GAP = 0.10
n = len(curves)
avail = (WRITE_E - WRITE_S) - GAP * (n - 1)
seg_t = [avail * l / total for l in lens]
seg_start = []; tc = WRITE_S
for k in range(n):
    seg_start.append(tc); tc += seg_t[k] + GAP

CX0, CY0, CW, CH = 478, 836, 128, 128
# ---------- hi-res pen write-on: the glyph is drawn in the 1280px emblem's native resolution (clean edges, no morphology), then area-downsampled onto the plate ----------
import heapq
from scipy.spatial import cKDTree
EMB = "/Volumes/Cherwell/Cherwell Media/Brand Assets/Logo/C&Cfulllogo_transparent_nobuffer.png"
RS, RTX, RTY = np.load(__file__.rsplit("/", 1)[0] + "/emblem_to_plate.npy")      # plate = RS*emblem + (RTX, RTY)
ea = np.array(Image.open(EMB))[..., 3].astype(np.float32) / 255
_l, _n2 = ndi.label(ea > 0.3, structure=np.ones((3, 3)))
_sz = ndi.sum(np.ones_like(ea), _l, range(1, _n2 + 1))
_amp_id = int(np.argmax([ (0 if z < 20000 else z) for z in _sz])) + 1                 # the big ampersand blob
amp_e = ndi.binary_dilation(_l == _amp_id, iterations=3)
ea_amp = np.where(amp_e, ea, 0).astype(np.float32)
EX0, EY0, EW, EH = 380, 360, 540, 540
ea_c = ea_amp[EY0:EY0+EH, EX0:EX0+EW]
mask_e = ea_c > 0.02
Xe, Ye = np.meshgrid(np.arange(EX0, EX0+EW, dtype=float), np.arange(EY0, EY0+EH, dtype=float))
toE = lambda x, y: ((x - RTX) / RS, (y - RTY) / RS)
curves = []
for st in STROKES:
    Pp = np.array(st, float).T
    Pe = np.array(toE(Pp[0], Pp[1]))
    tck, u = splprep(Pe, s=1.5 / RS**2, k=3)
    xs_, ys_ = splev(np.linspace(0, 1, 1200), tck)
    seg = np.hypot(np.diff(xs_), np.diff(ys_)); cum = np.r_[0, np.cumsum(seg)]
    curves.append((xs_, ys_, cum))
lens = [c[2][-1] for c in curves]; total = sum(lens)
n = len(curves)
avail = (WRITE_E - WRITE_S) - GAP * (n - 1)
seg_t = [avail * l / total for l in lens]
seg_start = []; tc = WRITE_S
for k in range(n):
    seg_start.append(tc); tc += seg_t[k] + GAP
c0x, c0y, cum0 = curves[0]
ts0 = seg_start[0] + seg_t[0] * (cum0 / cum0[-1])
def nearest_idx(px, py, lo=0, hi=None):
    px, py = toE(px, py)
    hi = len(c0x) if hi is None else hi
    return lo + int(np.argmin(np.hypot(c0x[lo:hi] - px, c0y[lo:hi] - py)))
j_junc = nearest_idx(519, 895, 0, len(c0x) // 3)
j_top = nearest_idx(541, 858)
j_cross = nearest_idx(521, 889, j_top)
j_bowl = j_cross + int(np.argmax(c0x[j_cross:] <= toE(511, 0)[0]))
j_apex = nearest_idx(526, 848, j_junc)
# pen timing keyed to the reference footage (ink in the reference lags the nib a little, so these are ~0.1s earlier than the storyboard labels)
seg_start = [0.10, 1.85, 2.30]; seg_t = [1.70, 0.35, 0.50]
_wi = [0, nearest_idx(509, 866, 0, j_junc + 400), j_apex, j_cross, nearest_idx(506, 947, j_cross), len(c0x) - 1]
_wt = [0.10, 0.75, 0.97, 1.27, 1.60, 1.80]
print("pen waypoints", _wi, _wt)
ts0 = np.interp(np.arange(len(c0x)), _wi, _wt)
groups = [(c0x[:j_junc], c0y[:j_junc]), (c0x[j_junc:j_apex], c0y[j_junc:j_apex]), (c0x[j_apex:j_bowl], c0y[j_apex:j_bowl]), (c0x[j_bowl:], c0y[j_bowl:]),
          (curves[1][0], curves[1][1]), (curves[2][0], curves[2][1])]
_q = np.c_[Xe.ravel(), Ye.ravel()]
_edt = ndi.distance_transform_edt(np.pad(mask_e, 1))[1:-1, 1:-1]            # half-thickness of the glyph at every pixel
_dg = []
for gx, gy in groups:
    _tr = cKDTree(np.c_[gx, gy])
    _pts = np.c_[gx, gy]
    _hw = np.array([_edt[min(max(int(round(y - EY0)), 0), EH - 1), min(max(int(round(x - EX0)), 0), EW - 1)] for x, y in _pts])
    _hw = np.full_like(_hw, np.median(_hw) * 1.05)                          # constant band = the stroke's typical half-width, so junction blobs don't count
    _d, _i = _tr.query(_q)
    _dg.append((_d - _hw[_i]).reshape(Xe.shape))                              # distance beyond the stroke's own half-width at the nearest nib position
dist_g = np.stack(_dg)
region = dist_g.argmin(0)
votes = np.stack([ndi.uniform_filter((region == g).astype(np.float32), size=int(13 / RS)) for g in range(len(groups))])
region = votes.argmax(0)
ts1 = seg_start[1] + seg_t[1] * (curves[1][2] / curves[1][2][-1]); ts2 = seg_start[2] + seg_t[2] * (curves[2][2] / curves[2][2][-1])
g_ts = [ts0[:j_junc], ts0[j_junc:j_apex], ts0[j_apex:j_bowl], ts0[j_bowl:], ts1, ts2]
gate_img = np.zeros(region.shape)
for g_, (gx_, gy_) in enumerate(groups):
    mk_ = region == g_
    if not mk_.any(): continue
    _d, _i = cKDTree(np.c_[gx_, gy_]).query(np.c_[Xe[mk_], Ye[mk_]])
    gate_img[mk_] = g_ts[g_][_i] - 0.03
speed = cum0[-1] / seg_t[0]                                # emblem px per second along the main stroke
NB = [(-1,0,1.0),(1,0,1.0),(0,-1,1.0),(0,1,1.0),(-1,-1,1.4142),(-1,1,1.4142),(1,-1,1.4142),(1,1,1.4142),
      (-2,-1,2.2361),(-2,1,2.2361),(2,-1,2.2361),(2,1,2.2361),(-1,-2,2.2361),(1,-2,2.2361),(-1,2,2.2361),(1,2,2.2361)]
Hh, Wh = mask_e.shape
T = np.full((Hh, Wh), np.inf)
sx_, sy_ = c0x[0], c0y[0]
seed = (np.hypot(Xe - sx_, Ye - sy_) <= 6.0) & mask_e
pq = []
for y_, x_ in zip(*np.where(seed)):
    T[y_, x_] = ts0[0]; heapq.heappush(pq, (ts0[0], y_, x_))
while pq:
    t_, y_, x_ = heapq.heappop(pq)
    if t_ > T[y_, x_]: continue
    for dy_, dx_, c_ in NB:
        ny, nx = y_ + dy_, x_ + dx_
        if 0 <= ny < Hh and 0 <= nx < Wh and mask_e[ny, nx]:
            nt = max(gate_img[ny, nx], t_ + c_ / speed)
            if nt < T[ny, nx]:
                T[ny, nx] = nt; heapq.heappush(pq, (nt, ny, nx))
# swept round-nib model: the nib is a disc of the glyph's own local half-thickness; a pixel appears the first time the nib covers it
_edt = ndi.distance_transform_edt(np.pad(mask_e, 1))[1:-1, 1:-1]
_edt5 = ndi.maximum_filter(_edt, size=5)
T_new = np.full(mask_e.shape, np.inf)
def _snap(gx, gy, R=16, sig=18):
    """move every pen sample onto the glyph's centreline (ridge of the thickness map), then smooth along the path"""
    ox, oy = np.zeros_like(gx), np.zeros_like(gy)
    for k_, (x_, y_) in enumerate(zip(gx, gy)):
        ix, iy = int(round(x_ - EX0)), int(round(y_ - EY0))
        y0_, y1_, x0_, x1_ = max(iy - R, 0), min(iy + R + 1, EH), max(ix - R, 0), min(ix + R + 1, EW)
        sc = _edt[y0_:y1_, x0_:x1_] - 0.25 * np.hypot(Xe[y0_:y1_, x0_:x1_] - x_, Ye[y0_:y1_, x0_:x1_] - y_)
        j_ = np.unravel_index(np.argmax(sc), sc.shape)
        ox[k_], oy[k_] = Xe[y0_:y1_, x0_:x1_][j_], Ye[y0_:y1_, x0_:x1_][j_]
    return ndi.gaussian_filter1d(ox, sig, mode="nearest"), ndi.gaussian_filter1d(oy, sig, mode="nearest")
_sn = [_snap(c[0], c[1]) for c in curves]
_all = [(_sn[0][0], _sn[0][1], ts0), (_sn[1][0], _sn[1][1], ts1), (_sn[2][0], _sn[2][1], ts2)]
_cover = np.zeros(mask_e.shape, bool)
for gx_, gy_, gt_ in _all:
    _ix = np.clip(np.round(gx_ - EX0).astype(int), 0, EW - 1); _iy = np.clip(np.round(gy_ - EY0).astype(int), 0, EH - 1)
    _rs = ndi.uniform_filter1d(ndi.minimum_filter1d(_edt5[_iy, _ix], 41, mode="nearest"), 25, mode="nearest")   # follows the stroke taper, ignores junction bulges        # nib radius = the stroke's typical half-thickness, junction bulges don't count
    for x_, y_, t_, rr_ in zip(gx_, gy_, gt_, _rs):
        ix, iy = int(round(x_ - EX0)), int(round(y_ - EY0))
        if not (0 <= ix < EW and 0 <= iy < EH): continue
        r_ = rr_ * NIB + 0.5
        R = int(r_) + 2
        y0_, y1_, x0_, x1_ = max(iy - R, 0), min(iy + R + 1, EH), max(ix - R, 0), min(ix + R + 1, EW)
        win = ((Xe[y0_:y1_, x0_:x1_] - x_) ** 2 + (Ye[y0_:y1_, x0_:x1_] - y_) ** 2) <= r_ * r_
        sub = T_new[y0_:y1_, x0_:x1_]
        sub[win] = np.minimum(sub[win], t_)
# anything the nib never reached (path slightly off-centre): take the time of the nearest nib position
_miss = mask_e & ~np.isfinite(T_new)
if _miss.any():
    _pp = np.c_[np.concatenate([c[0] for c in _sn]), np.concatenate([c[1] for c in _sn])]
    _tt = np.concatenate([ts0, ts1, ts2])
    _d_, _i_ = cKDTree(_pp).query(np.c_[Xe[_miss], Ye[_miss]])
    T_new[_miss] = _tt[_i_]
print("nib-uncovered px", int(_miss.sum()))
# de-nick: small patches revealed out of sequence (e.g. the stem foot / bowl end overlapping the diagonal's edge) take the time of their surroundings; real stroke-to-stroke steps are bigger and stay
_idx_in = ndi.distance_transform_edt(~mask_e, return_distances=False, return_indices=True)
for _it in range(3):
    _Tf = T_new[_idx_in[0], _idx_in[1]]
    _Tm = ndi.median_filter(_Tf, size=17)
    _bad = mask_e & (np.abs(T_new - _Tm) > 0.15)
    _lb, _nb = ndi.label(_bad, structure=np.ones((3, 3)))
    _sz = ndi.sum(np.ones_like(_lb), _lb, range(1, _nb + 1))
    _small = np.isin(_lb, 1 + np.where(np.array(_sz) < 600)[0]) & _bad
    T_new = np.where(_small, _Tm, T_new)
# stem foot: the diagonal's edge continues straight across the foot's base (chord between the two corners of the V where the stem leaves it); anything inked early beyond that chord is the stem's own foot and waits for the stem stroke
_A = np.array([704.0, 754.0]); _B = np.array([724.0, 771.0])                   # emblem coords of the V corners (fixed artwork)
_u = (_B - _A) / np.hypot(*(_B - _A)); _nrm = np.array([_u[1], -_u[0]])         # normal pointing toward the stem
_side = (Xe - _A[0]) * _nrm[0] + (Ye - _A[1]) * _nrm[1]
_near = np.hypot(Xe - (_A[0] + _B[0]) / 2, Ye - (_A[1] + _B[1]) / 2) < 30
_foot = mask_e & _near & (_side > 0.5) & (T_new < 1.7)
T_new = np.where(_foot, ts1[0] + 0.0, T_new)
print("stem-foot px moved", int(_foot.sum()))
T_e = np.where(mask_e, T_new, 1e9).astype(np.float32)
print("reveal range", T_e[mask_e].min(), T_e[mask_e & (T_e < 1e8)].max(), "unreached", int((mask_e & (T_e >= 1e8)).sum()))

# =====================================================================
# 4K render (2160x3840): everything except the wordmark is drawn from the 1280px emblem, area-resampled straight onto the 4K grid
# =====================================================================
K = 2
RX0, RY0, RX1, RY1 = 410, 770, 675, 1035
W4, H4 = W * K, H * K
BG = np.round(bg).astype(np.uint8)
INK = 255.0 - bg                                                   # ink = white on the flat brown plate
def grid(x0, y0, w, h):
    ys, xs = np.mgrid[0:h, 0:w]
    px = (xs + x0 + 0.5) / K - 0.5; py = (ys + y0 + 0.5) / K - 0.5   # 4K pixel centre -> plate coords
    return (px - RTX) / RS, (py - RTY) / RS                           # -> emblem coords
SIGMA = 0.55 / (K * RS)                                            # anti-alias before decimating the emblem
def sample(arr, ex, ey):
    return np.clip(ndi.map_coordinates(ndi.gaussian_filter(arr, SIGMA), [ey, ex], order=1, cval=0), 0, 1)

# --- ampersand window
AX0, AY0, AW, AH = CX0 * K, CY0 * K, CW * K, CH * K
aex, aey = grid(AX0, AY0, AW, AH)
# --- ring window
RX0_, RY0_ = RX0 * K, RY0 * K
RWi, RHi = (RX1 - RX0) * K, (RY1 - RY0) * K
rex, rey = grid(RX0_, RY0_, RWi, RHi)
ea_ring = (ea * (~amp_e)).astype(np.float32)
R4 = sample(ea_ring, rex, rey)
rc_x = (RS * 640.0 + RTX) * K + (K - 1) / 2 - RX0_                  # ring centre (emblem centre) in window px
rc_y = (RS * 640.5 + RTY) * K + (K - 1) / 2 - RY0_
print("ring centre (plate)", RS * 640.0 + RTX, RS * 640.5 + RTY)
_lab, _n = ndi.label(R4 > 0.12, structure=np.ones((3, 3)))
_idx = ndi.distance_transform_edt(_lab == 0, return_distances=False, return_indices=True)
_near = _lab[_idx[0], _idx[1]]
_near = np.where(R4 > 0.003, _near, 0)
_yy, _xx = np.mgrid[0:RHi, 0:RWi]
_th = (np.degrees(np.arctan2(_xx - rc_x, -(_yy - rc_y))) + 360.0) % 360.0
_rad = np.hypot(_xx - rc_x, _yy - rc_y)
arcs, dots, letters = [], [], []
for i in range(1, _n + 1):
    m = _lab == i
    ar = int(m.sum())
    if ar < 4 * K * K: continue
    ys_, xs_ = np.where(m)
    w_, h_ = xs_.max() - xs_.min() + 1, ys_.max() - ys_.min() + 1
    rmean = float(_rad[m].mean()); rstd = float(_rad[m].std())
    if max(w_, h_) > 60 * K and rstd < 4.0 * K: arcs.append((i, rmean))              # long thin curve = ring arc (letters are < 30px, dots < 12px)
    elif w_ <= 11 * K and h_ <= 11 * K and ar >= 60 * K * K and ar / (w_ * h_) > 0.6: dots.append(i)
    else: letters.append((i, float(xs_.mean()), float(ys_.mean())))
print("ring comps (4K): arcs", len(arcs), "dots", len(dots), "letters", len(letters))
arc_defs = []
for (i, rmean) in arcs:
    mk = (_near == i)
    th = _th[_lab == i]
    ang_sorted = np.sort(th)
    gaps = np.diff(np.r_[ang_sorted, ang_sorted[0] + 360.0])
    g = int(np.argmax(gaps)); start = ang_sorted[(g + 1) % len(ang_sorted)]; span = 360.0 - gaps[g]
    u = ((_th - start) % 360.0) / max(span, 1e-6)
    ys_, xs_ = np.where(_lab == i); cy_ = ys_.mean()
    arc_defs.append(dict(mask=mk, u=u.astype(np.float32), outer=rmean > 100 * K, bottom=(cy_ > rc_y)))
dot_mask = np.isin(_near, dots)
order = sorted(letters, key=lambda L: L[1])
top_ids = [L for L in order if L[2] < rc_y - 20 * K]; bot_ids = [L for L in order if L[2] >= rc_y - 20 * K]
let_t = []
for k_, L in enumerate(top_ids): let_t.append((L[0], 5.20 + 0.04 * k_))
for k_, L in enumerate(bot_ids): let_t.append((L[0], 5.30 + 0.04 * k_))
LUT = np.zeros(_n + 1, int) - 1; LT = np.zeros(max(len(let_t), 1))
for j_, (i, t0_) in enumerate(let_t): LUT[i] = j_; LT[j_] = t0_
let_id = LUT[_near]
def arc_time(a):
    if a['outer']: return (3.98, 4.43) if not a['bottom'] else (3.50, 3.95)
    return (4.82, 5.12) if not a['bottom'] else (4.50, 4.80)
DOT_T = 5.15

# --- wordmark: only exists in the 1080 plate, so upscale it 2x and re-sharpen the edges
_ys, _xs = np.where(R_word)
wy0, wy1, wx0, wx1 = _ys.min(), _ys.max() + 1, _xs.min(), _xs.max() + 1
_aw = np.where(R_word, alpha, 0).astype(np.float32)[wy0:wy1, wx0:wx1]
_aw4 = np.array(Image.fromarray(_aw, mode="F").resize(((wx1 - wx0) * K, (wy1 - wy0) * K), Image.BICUBIC))
_aw4 = np.clip((np.clip(_aw4, 0, 1) - 0.5) * 1.6 + 0.5, 0, 1).astype(np.float32)
print("wordmark box (plate)", wx0, wy0, wx1, wy1)

p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "yuvj420p", "-s", f"{W4}x{H4}", "-r", str(FPS), "-i", "-",
                      "-vf", "setparams=range=pc:colorspace=smpte170m:color_primaries=bt709:color_trc=iec61966-2-1", "-c:v", "libx264", "-preset", "medium", "-crf", "9", "-pix_fmt", "yuvj420p", "-color_range", "pc", "-colorspace", "smpte170m", "-color_primaries", "bt709", "-color_trc", "iec61966-2-1", "-movflags", "+faststart", OUT], stdin=subprocess.PIPE)
def to_yuv420(rgb):
    """full-range BT.601, converted here (not by swscale) so the flat plate colour round-trips exactly"""
    x = rgb.astype(np.float32)
    Y = 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]
    Cb = 128.0 - 0.168736 * x[..., 0] - 0.331264 * x[..., 1] + 0.5 * x[..., 2]
    Cr = 128.0 + 0.5 * x[..., 0] - 0.418688 * x[..., 1] - 0.081312 * x[..., 2]
    sub = lambda c: c.reshape(H4 // 2, 2, W4 // 2, 2).mean((1, 3))
    q = lambda c: np.clip(np.rint(c), 0, 255).astype(np.uint8)
    return q(Y).tobytes() + q(sub(Cb)).tobytes() + q(sub(Cr)).tobytes()
base = np.empty((H4, W4, 3), np.uint8); base[:] = BG
nf = int(FPS * DUR)
for k in range(nf):
    t = k / FPS
    tr = t + SH
    out = base.copy()
    # ---- ring window (float): ring, dots, letters, then the ampersand on top
    rw = np.broadcast_to(bg[None, None, :], (RHi, RWi, 3)).astype(np.float32).copy()
    if tr >= RING_S:
        f = np.zeros((RHi, RWi), np.float32)
        for a_ in arc_defs:
            t0_, t1_ = arc_time(a_)
            if tr < t0_: continue
            p_ = clip01((tr - t0_) / (t1_ - t0_))
            eps = 0.015
            ff = np.clip((p_ * (1 + eps) - a_['u']) / eps, 0, 1)
            f = np.where(a_['mask'], ff, f)
        if tr >= DOT_T:
            f = np.where(dot_mask, ease(clip01((tr - DOT_T) / 0.2)), f)
        lm = let_id >= 0
        if lm.any():
            tl = LT[np.maximum(let_id, 0)]
            fl = np.clip((tr - tl) / 0.14, 0, 1); fl = fl * fl * (3 - 2 * fl)
            f = np.where(lm, fl, f)
        rw += INK[None, None, :] * (R4 * f)[..., None]
    if t >= WRITE_S:
        rv = np.clip((t - T_e) / 0.04, 0, 1)
        a4 = sample((ea_c * rv).astype(np.float32), aex - EX0, aey - EY0)
        oy_, ox_ = AY0 - RY0_, AX0 - RX0_
        sub = rw[oy_:oy_ + AH, ox_:ox_ + AW]
        sub[:] = sub * (1 - a4[..., None]) + 255.0 * a4[..., None]
    out[RY0_:RY0_ + RHi, RX0_:RX0_ + RWi] = np.clip(rw + 0.5, 0, 255).astype(np.uint8)
    # ---- wordmark
    if tr >= WORD_S:
        pw = clip01((tr - WORD_S) / (WORD_E - WORD_S))
        fa = easeo(pw)
        dy = int(round((1 - easeo(pw)) * 14)) * K
        y0_ = wy0 * K + dy; y1_ = y0_ + _aw4.shape[0]
        reg = out[y0_:y1_, wx0 * K:wx1 * K].astype(np.float32)
        reg += INK[None, None, :] * (_aw4 * fa)[..., None]
        out[y0_:y1_, wx0 * K:wx1 * K] = np.clip(reg + 0.5, 0, 255).astype(np.uint8)
    p.stdin.write(to_yuv420(out))
p.stdin.close(); p.wait()
print("done", OUT)
