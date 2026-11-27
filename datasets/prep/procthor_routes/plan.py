"""Route planning on a house's walkable grid: kinematic camera paths (no gait, no gaze), like the training data.

Pose per frame: Unity (x, z) in m and heading psi in rad (forward = (sin psi, cos psi), psi = yaw_u in radians).
A window of T frames = 20 warm-up + (T - 21) rollout steps + 1 target; records carry PAD extra frames (pose held) so
slipstream's decoder fit rule (2-frame tail) and `eval_windows` (3-frame margin) serve the whole window from frame 0.

Validity: every frame position must be on walkable floor of the FURNISHED house: within SNAP_TOL (0.13 m, the
generator's tolerance) of a reachable grid point (0.25 m grid, which already keeps the agent capsule off objects and
walls), or inside a grid cell whose four corners are all reachable. A step that lands up to 0.20 m from the grid is
pulled back to 0.12 m (counted as a correction, as the generator snaps its Brownian steps); farther fails the attempt.
Paths are shortest paths on the grid weighted to keep ~0.75 m from walls and furniture where the floor allows.

Every scripted route: warm-up = 20 steps of walking along a path that ends at A (a point with room for the event),
the event starts at A at the first rollout step (frame 20), then the camera walks on (goal-to-goal) to frame T-1.
  goal      goal-to-goal tour (legs 2-5 m geodesic) from frame 0, walking pace
  turn      360 deg turn on a tight circle (0.05 m/step forward, radius ~0.33 m), at <= 0.15 rad/step (61: 0.157)
  outback   walk out along a path (2-3 m; 61: ~1 m), turn 180 deg (0.02 m/step forward), retrace the path back to A
  loop      one circuit of a circle at walking pace (w 0.09-0.15 rad/step, radius 0.77-1.28 m; 61: w 0.157, 0.73 m)
  brownian  the training generator (Gaussian 0.1 m steps snapped to the grid, Gaussian 0.15 rad turns)
Seeded per (house, type, length, index, try); every try is counted.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

GRID = 0.25
SNAP_TOL = 0.13
FIX_MAX, FIX_TO = 0.20, 0.12
WARMUP = 20
PAD = 4                      # held frames after the window (decoder 2-frame tail + eval_windows 3-frame margin)
V = 0.115                    # walking pace, m/step (training step median 0.115 m)
W_MAX = 0.15                 # turn limit, rad/step (training |turn| p68.6)
TYPES = ("goal", "turn", "outback", "loop", "brownian")


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def short(T):
    return T - 1 - WARMUP < 60


class House:
    def __init__(self, reach: np.ndarray):
        self.reach = reach; n = len(reach)
        self.tree = cKDTree(reach)
        self.origin = reach.min(0)
        ij = self._ij(reach); idx = {tuple(k): i for i, k in enumerate(ij)}; self.keys = set(idx)
        cand = np.unique(np.array([(a + da, b + db) for a, b in ij for da in range(-6, 7) for db in range(-6, 7)]), axis=0)
        holes = np.array([c for c in map(tuple, cand) if c not in idx])
        self.clear = cKDTree(holes * GRID).query(ij * GRID)[0] if len(holes) else np.full(n, 9.0)
        pen = 1 + 1.5 * np.clip(0.75 - self.clear, 0, None)          # prefer >= 0.75 m clearance
        rows, cols, d, dw = [], [], [], []
        for i, (a, b) in enumerate(ij):
            for da, db in ((1, 0), (0, 1), (1, 1), (1, -1)):
                j = idx.get((a + da, b + db))
                if j is None or (da and db and ((a + da, b) not in idx or (a, b + db) not in idx)):
                    continue
                L = GRID * math.hypot(da, db); Lw = L * (pen[i] + pen[j]) / 2
                rows += [i, j]; cols += [j, i]; d += [L, L]; dw += [Lw, Lw]
        self.geo = dijkstra(csr_matrix((d, (rows, cols)), shape=(n, n)), directed=False)
        _, self.pred = dijkstra(csr_matrix((dw, (rows, cols)), shape=(n, n)), directed=False, return_predecessors=True)
        self.nbrs = np.array([len(x) for x in self.tree.query_ball_point(reach, 1.0)])

    def _ij(self, p):
        return np.round((np.asarray(p) - self.origin) / GRID).astype(int)

    def valid(self, p) -> np.ndarray:
        p = np.atleast_2d(p)
        ok = self.tree.query(p)[0] <= SNAP_TOL
        if (~ok).any():
            f = np.floor((p[~ok] - self.origin) / GRID).astype(int)
            ok[~ok] = [all((a + da, b + db) in self.keys for da in (0, 1) for db in (0, 1)) for a, b in f]
        return ok

    def node(self, p) -> int:
        return int(self.tree.query(p)[1])

    def path(self, i, j) -> np.ndarray | None:
        if not np.isfinite(self.geo[i, j]):
            return None
        out = [j]
        while out[-1] != i:
            out.append(int(self.pred[i, out[-1]]))
        return self.reach[out[::-1]]


def smooth(poly: np.ndarray, it: int = 2) -> np.ndarray:
    """Chaikin corner cutting, endpoints kept."""
    for _ in range(it):
        if len(poly) < 3:
            return poly
        q = 0.75 * poly[:-1] + 0.25 * poly[1:]; r = 0.25 * poly[:-1] + 0.75 * poly[1:]
        poly = np.vstack([poly[:1], np.column_stack([q, r]).reshape(-1, 2), poly[-1:]])
    return poly


class Walker:
    """Kinematic camera: position p, heading psi; one pose per frame."""

    def __init__(self, house: House, p, psi):
        self.h = house; self.p = np.array(p, float); self.psi = float(psi)
        self.P = [self.p.copy()]; self.Psi = [self.psi]; self.ok = True; self.fixes = 0

    def step(self, dpsi, v):
        self.psi = float(wrap(self.psi + dpsi))
        p = self.p + v * np.array([math.sin(self.psi), math.cos(self.psi)])
        if not self.h.valid(p)[0]:
            d, j = self.h.tree.query(p); q = self.h.reach[j]
            if d > FIX_MAX:
                self.ok = False
            else:
                p = q + (p - q) * (FIX_TO / d); self.fixes += 1
        self.p = p; self.P.append(p.copy()); self.Psi.append(self.psi)

    @property
    def n(self):
        return len(self.P)

    def pursue(self, poly, n_max, v=V, w=W_MAX, look=0.35, stop=0.12):
        """Pure pursuit along a polyline; slows (to 0.3 v) while the heading error is large. Returns steps taken."""
        poly = smooth(poly); s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(poly, axis=0), axis=1))]
        k = 0
        while k < n_max and self.ok:
            d = np.linalg.norm(poly - self.p, axis=1); i = int(d.argmin())
            if s[-1] - s[i] < stop and d[-1] < stop + v:
                break
            tgt = poly[min(int(np.searchsorted(s, s[i] + look)), len(poly) - 1)]
            err = wrap(math.atan2(tgt[0] - self.p[0], tgt[1] - self.p[1]) - self.psi)
            self.step(float(np.clip(err, -w, w)), v * float(np.clip(math.cos(err), 0.3, 1.0)))
            k += 1
        return k

    def circle(self, total, w, v):
        """Constant-rate arc: `total` rad (signed) at |w| rad/step, v m/step."""
        n = int(math.ceil(abs(total) / abs(w) - 1e-9))
        for i in range(n):
            self.step(math.copysign(min(abs(w), abs(total) - i * abs(w)), total), v)


def circle_fits(h: House, p, psi, total, w, v) -> bool:
    n = int(math.ceil(abs(total) / abs(w) - 1e-9)); q = np.array(p, float); a = psi; pts = []
    for i in range(n):
        a += math.copysign(min(abs(w), abs(total) - i * abs(w)), total)
        q = q + v * np.array([math.sin(a), math.cos(a)]); pts.append(q.copy())
    return bool(h.valid(np.array(pts)).all())


def _goal_from(h: House, i0: int, rng, lo=2.0, hi=5.0, away_from=None, ahead=None):
    d = h.geo[i0]; c = np.where((d >= lo) & (d <= hi))[0]
    if away_from is not None and len(c):
        keep = h.geo[away_from, c] > d[c] * 0.5        # don't head straight back
        c = c[keep] if keep.any() else c
    if ahead is not None and len(c):                  # (p, psi, max angle): roughly ahead of the camera
        p, psi, amax = ahead
        ang = np.abs(wrap(np.arctan2(h.reach[c, 0] - p[0], h.reach[c, 1] - p[1]) - psi))
        c = c[ang < amax]
    return int(rng.choice(c)) if len(c) else None


def _walk_tour(h: House, wk: Walker, n_frames: int, rng, prev=None):
    """Walk goal-to-goal (2-5 m legs) until the walk has n_frames frames. Returns goals reached."""
    goals = 0; cur = h.node(wk.p); last = prev
    while wk.n < n_frames and wk.ok:
        g = _goal_from(h, cur, rng, away_from=last, ahead=(wk.p, wk.psi, math.radians(100)) if goals == 0 else None)
        if g is None:
            g = _goal_from(h, cur, rng, away_from=last)
        if g is None:
            wk.ok = False; break
        before = wk.n
        wk.pursue(h.path(h.node(wk.p), g), n_frames - wk.n)
        if wk.n == before:
            wk.ok = False; break
        if np.linalg.norm(wk.p - h.reach[g]) < 0.3:
            goals += 1
        last, cur = cur, g
    return goals


def _cut(wk: Walker, n):
    wk.P, wk.Psi = wk.P[:n], wk.Psi[:n]


def _warmup_to(h: House, iA: int, rng) -> Walker | None:
    """20 steps of walking along a path that ends at node iA (start 2.4-3.0 m geodesic away)."""
    c = np.where((h.geo[iA] >= 2.4) & (h.geo[iA] <= 3.0))[0]
    if not len(c):
        return None
    iW = int(rng.choice(c)); path = h.path(iW, iA)
    sp = smooth(path); psi0 = math.atan2(*(sp[min(3, len(sp) - 1)] - sp[0]))
    wk = Walker(h, h.reach[iW], psi0)
    wk.pursue(path, WARMUP, stop=0.0)
    return wk if wk.ok and wk.n == WARMUP + 1 else None


def plan(h: House, kind: str, T: int, rng) -> dict | None:
    """One window of T frames (+PAD held) or None. Rollout = steps from frame WARMUP to frame T-1."""
    R = T - 1 - WARMUP
    if kind == "brownian":
        return _brownian(h, T, rng)
    info = {}
    if kind == "goal":
        i0 = int(rng.choice(np.where(h.clear >= 0.5)[0] if (h.clear >= 0.5).any() else np.arange(len(h.reach))))
        g0 = _goal_from(h, i0, rng)
        if g0 is None:
            return None
        sp = smooth(h.path(i0, g0)); wk = Walker(h, h.reach[i0], math.atan2(*(sp[min(3, len(sp) - 1)] - sp[0])))
        info["goals_total"] = _walk_tour(h, wk, T, rng)
    else:
        need = {"turn": 0.6, "outback": 0.5, "loop": 1.0}[kind]
        cand = np.where(h.clear >= need)[0]
        if not len(cand):
            return None
        wk = _warmup_to(h, int(rng.choice(cand)), rng)
        if wk is None:
            return None
        A = wk.p.copy(); info["A"] = A.tolist()
        if kind == "turn":
            w = W_MAX if not short(T) else max(W_MAX, 2 * math.pi / R)
            sides = rng.permutation([-1, 1])
            side = next((s for s in sides if circle_fits(h, A, wk.psi, s * 2 * math.pi, w, 0.05)), None)
            if side is None:
                return None
            s0 = wk.n - 1; wk.circle(side * 2 * math.pi, w, 0.05)
            info |= {"turn_start": s0, "turn_end": wk.n - 1, "turn_rate": w}
        elif kind == "outback":
            w = W_MAX if not short(T) else max(W_MAX, math.pi / 20)
            n_turn = math.ceil(math.pi / w)
            d_out = float(rng.uniform(2.0, 3.0)) if not short(T) else V * (R - n_turn - 4) / 2
            iA = h.node(A); d = h.geo[iA]
            ang = np.abs(wrap(np.arctan2(h.reach[:, 0] - A[0], h.reach[:, 1] - A[1]) - wk.psi))
            c = np.where((d >= d_out + 0.15) & (d <= d_out + 0.75) & (ang < math.radians(60)))[0]
            if not len(c):
                return None
            B = int(rng.choice(c))
            s_out = wk.n - 1
            wk.pursue(h.path(iA, B), int(round(d_out / V)) + 12)
            walked = np.r_[0, np.cumsum(np.linalg.norm(np.diff(np.array(wk.P[s_out:]), axis=0), axis=1))]
            if walked[-1] < d_out:
                return None
            _cut(wk, s_out + 1 + int(np.argmax(walked >= d_out)))        # stop once d_out is walked
            wk.p, wk.psi = wk.P[-1].copy(), wk.Psi[-1]
            sides = rng.permutation([-1, 1])
            side = next((s for s in sides if circle_fits(h, wk.p, wk.psi, s * math.pi, w, 0.02)), None)
            if side is None or not wk.ok:
                return None
            s0 = wk.n - 1; wk.circle(side * math.pi, w, 0.02)
            out_track = np.array(wk.P[s_out:s0 + 1])[::-1]                # retrace the out-leg
            wk.pursue(np.vstack([wk.p, out_track]), T - wk.n, stop=0.06)
            info |= {"d_out_target": d_out, "out_start": s_out, "turn_start": s0, "turn_end": s0 + n_turn,
                     "back_end": wk.n - 1, "turn_rate": w}
        elif kind == "loop":
            ws = [2 * math.pi / R] if short(T) else list(rng.permutation([0.09, 0.11, 0.13, 0.15]))
            opts = [(w, s) for w in ws for s in rng.permutation([-1, 1])]
            pick = next(((w, s) for w, s in opts if circle_fits(h, A, wk.psi, s * 2 * math.pi, w, V)), None)
            if pick is None:
                return None
            w, side = pick; s0 = wk.n - 1
            wk.circle(side * 2 * math.pi, w, V)
            info |= {"loop_start": s0, "loop_end": wk.n - 1, "turn_rate": float(w), "radius_m": V / w}
        if wk.n < T and wk.ok:                       # walk on for the rest of the window
            _walk_tour(h, wk, T, rng)
    if not wk.ok or wk.n < T:
        return None
    _cut(wk, T); info["fixes"] = wk.fixes
    return _pack(kind, T, wk.P, wk.Psi, info)


def _pack(kind, T, P, Psi, info):
    P = np.array(P); Psi = np.array(Psi)
    P = np.vstack([P, np.repeat(P[-1:], PAD, 0)]); Psi = np.r_[Psi, np.repeat(Psi[-1], PAD)]
    return {"kind": kind, "T": T, "pos_u": P, "psi_u": Psi, "info": info}


def _brownian(h: House, T: int, rng, step_std=0.1, angle_std=0.15) -> dict:
    """The procthor-walks generator's motion (walk_and_render), from a start clear of walls."""
    pos = h.reach[rng.choice(np.where(h.nbrs >= np.quantile(h.nbrs, 0.9))[0])].astype(float)
    yaw = float(rng.uniform(-math.pi, math.pi)); P = [pos.copy()]; Psi = [yaw]
    for _ in range(T - 1):
        cand = pos + rng.normal(0.0, step_std, 2)
        d, j = h.tree.query(cand)
        pos = h.reach[j].astype(float) if d > SNAP_TOL else cand
        yaw = float(wrap(yaw + rng.normal(0.0, angle_std)))
        P.append(pos.copy()); Psi.append(yaw)
    return _pack("brownian", T, P, Psi, {})


def metrics(w: dict) -> dict:
    """Event metrics on the rollout (frames WARMUP..T-1) and per-step egomotion over the whole window."""
    T = w["T"]; P = w["pos_u"][:T]; Psi = w["psi_u"][:T]
    step = np.linalg.norm(np.diff(P, axis=0), axis=1); dpsi = wrap(np.diff(Psi))
    A = P[WARMUP]; disp = np.linalg.norm(P[WARMUP:] - A, axis=1)
    left = np.where(disp > 0.8)[0]
    rd = dpsi[WARMUP:]
    m = {"path_m": float(step[WARMUP:].sum()), "max_disp_m": float(disp.max()), "end_disp_m": float(disp[-1]),
         "net_turn_rad": float(rd.sum()), "abs_turn_rad": float(np.abs(rd).sum()),
         "max_abs_net_turn_rad": float(np.abs(np.cumsum(rd)).max()),
         "return_to_A_m": float(disp[left[0]:].min()) if len(left) else None,
         "max_step_m": float(step.max()), "max_turn_rad": float(np.abs(dpsi).max()),
         "frac_steps_lt_0.032m": float((step < 0.032).mean()),            # below training step p5 (incl. zero)
         "frac_turns_gt_0.15": float((np.abs(dpsi) > 0.15 + 1e-9).mean()),
         "frac_turns_gt_0.16": float((np.abs(dpsi) > 0.16 + 1e-9).mean()),
         "fixes": int(w["info"].get("fixes", 0))}
    i = w["info"]; k = w["kind"]; lo = T < 100
    if k == "turn":
        m["turn_rad"] = abs(float(dpsi[i["turn_start"]:i["turn_end"]].sum()))
        m["event_ok"] = m["turn_rad"] >= 2 * math.pi - 1e-6 and i["turn_end"] <= T - 1
    elif k == "outback":
        ts = i["turn_start"]
        m |= {"out_path_m": float(step[i["out_start"]:ts].sum()), "out_disp_m": float(np.linalg.norm(P[ts] - A)),
              "turn_rad": abs(float(dpsi[ts:min(i["turn_end"], T - 1)].sum())),
              "back_to_A_m": float(np.linalg.norm(P[min(i["back_end"], T - 1)] - A)), "back_frame": int(i["back_end"])}
        m["event_ok"] = (i["back_end"] <= T - 1 and m["back_to_A_m"] <= 0.3 and m["turn_rad"] >= math.pi - 1e-6
                         and m["out_path_m"] >= (0.9 if lo else 2.0))
    elif k == "loop":
        e = i["loop_end"]
        m |= {"loop_closure_m": float(np.linalg.norm(P[min(e, T - 1)] - A)),
              "loop_turn_rad": abs(float(dpsi[i["loop_start"]:min(e, T - 1)].sum())), "loop_path_m": float(step[i["loop_start"]:e].sum())}
        m["event_ok"] = e <= T - 1 and m["loop_closure_m"] <= 0.3 and m["loop_turn_rad"] >= 2 * math.pi - 1e-6
    elif k == "goal":
        m["event_ok"] = m["max_disp_m"] >= (1.5 if lo else 2.5)
    else:
        m["event_ok"] = True
    return m
