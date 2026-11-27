"""Route planning on a house's walkable grid: kinematic camera paths (no gait, no gaze), like the training data.

Pose per frame: Unity (x, z) in m and heading psi in rad (forward = (sin psi, cos psi), psi = yaw_u in radians).
A window of T frames = 20 warm-up + (T - 21) rollout steps + 1 target; records carry PAD extra frames (pose held) so
slipstream's decoder fit rule (2-frame tail) and `eval_windows` (3-frame margin) serve the whole window from frame 0.

Validity: every frame position must be on walkable floor of the FURNISHED house: within SNAP_TOL (0.13 m, the
generator's tolerance) of a reachable grid point (0.25 m grid, which already keeps the agent capsule off objects and
walls), or inside a grid cell whose four corners are all reachable.

Route types (seeded per (house, type, length, index); a failed attempt is retried with the next sub-seed, and every
attempt is counted):
  goal      tour of goals 2-5 m apart (geodesic), followed by pure pursuit at walking pace from frame 0
  turn      warm-up walk, then a 360 deg turn on a tight circle (0.05 m/step, radius ~0.33 m), then walk on
  outback   warm-up walk to A; walk out along a near-straight path; turn 180 deg (0.02 m/step); walk back to A
  loop      warm-up walk to A; one circuit of a circle (walking pace, radius v / w >= 0.73 m) back to A; walk on
  brownian  the training generator (Gaussian 0.1 m steps snapped to the grid, Gaussian 0.15 rad turns)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

GRID = 0.25
SNAP_TOL = 0.13
WARMUP = 20
PAD = 4                      # held frames after the window (decoder 2-frame tail + eval_windows 3-frame margin)
V = 0.115                    # walking pace, m/step (training step median 0.115 m, p58 = 0.13 m)
W_MAX = 0.15                 # turn limit, rad/step (training |turn| p68.6)
TYPES = ("goal", "turn", "outback", "loop", "brownian")


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


@dataclass
class House:
    reach: np.ndarray                         # [n, 2] reachable grid points (furnished house)
    tree: cKDTree = field(init=False)
    keys: set = field(init=False)
    origin: np.ndarray = field(init=False)
    geo: np.ndarray | None = None             # all-pairs geodesic distances [n, n] (m)
    pred: np.ndarray | None = None

    def __post_init__(self):
        self.tree = cKDTree(self.reach)
        self.origin = self.reach.min(0)
        self.keys = {tuple(k) for k in self._ij(self.reach)}
        n = len(self.reach)
        ij = self._ij(self.reach); idx = {tuple(k): i for i, k in enumerate(ij)}
        rows, cols, w = [], [], []
        for i, (a, b) in enumerate(ij):
            for da, db in ((1, 0), (0, 1), (1, 1), (1, -1)):
                j = idx.get((a + da, b + db))
                if j is None:
                    continue
                if da and db and ((a + da, b) not in idx or (a, b + db) not in idx):
                    continue                  # no corner cutting
                d = GRID * math.hypot(da, db)
                rows += [i, j]; cols += [j, i]; w += [d, d]
        g = csr_matrix((w, (rows, cols)), shape=(n, n))
        self.geo, self.pred = dijkstra(g, directed=False, return_predecessors=True)
        # clearance of each grid point: distance to the nearest non-reachable grid neighbour position
        cand = np.array([(a + da, b + db) for a, b in ij for da in range(-6, 7) for db in range(-6, 7)])
        cand = np.unique(cand, axis=0)
        holes = np.array([c for c in map(tuple, cand) if c not in idx])
        self.clear = cKDTree(holes * GRID).query(ij * GRID)[0] if len(holes) else np.full(n, 9.0)
        self.nbrs = np.array([len(x) for x in self.tree.query_ball_point(self.reach, 1.0)])

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

    def path(self, i, j) -> np.ndarray:
        out = [j]
        while out[-1] != i:
            out.append(int(self.pred[i, out[-1]]))
            if out[-1] < 0:
                return None
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
    """Kinematic camera: position p, heading psi; records one pose per frame."""

    def __init__(self, house: House, p, psi):
        self.h = house; self.p = np.array(p, float); self.psi = float(psi)
        self.P = [self.p.copy()]; self.Psi = [self.psi]; self.ok = True

    def step(self, dpsi, v):
        self.psi = float(wrap(self.psi + dpsi))
        self.p = self.p + v * np.array([math.sin(self.psi), math.cos(self.psi)])
        if not self.h.valid(self.p)[0]:
            self.ok = False
        self.P.append(self.p.copy()); self.Psi.append(self.psi)

    @property
    def n(self):
        return len(self.P)

    def pursue(self, poly, n_max, v=V, w=W_MAX, look=0.5, stop=0.15):
        """Pure pursuit along a polyline; slows (to 0.3 v) while the heading error is large. Returns steps taken."""
        poly = smooth(poly); seg = np.linalg.norm(np.diff(poly, axis=0), axis=1); s = np.r_[0, np.cumsum(seg)]
        k = 0
        while k < n_max and self.ok:
            d = np.linalg.norm(poly - self.p, axis=1); i = int(d.argmin())
            if s[-1] - s[i] < stop and d[-1] < stop + v:
                break
            tgt = poly[np.searchsorted(s, min(s[i] + look, s[-1]), side="left").clip(0, len(poly) - 1)]
            err = wrap(math.atan2(tgt[0] - self.p[0], tgt[1] - self.p[1]) - self.psi)
            self.step(float(np.clip(err, -w, w)), v * float(np.clip(math.cos(err), 0.3, 1.0)))
            k += 1
        return k

    def circle(self, total, w, v):
        """Constant-rate arc: `total` rad (signed) at |w| rad/step, v m/step."""
        n = int(math.ceil(abs(total) / abs(w) - 1e-9))
        for i in range(n):
            self.step(math.copysign(min(abs(w), abs(total) - i * abs(w)), total), v)


def _goal_from(h: House, i0: int, rng, lo=2.0, hi=5.0, away_from=None):
    d = h.geo[i0]; c = np.where((d >= lo) & (d <= hi))[0]
    if away_from is not None and len(c):
        keep = h.geo[away_from, c] > d[c] * 0.5        # don't go straight back
        c = c[keep] if keep.any() else c
    return int(rng.choice(c)) if len(c) else None


def _start(h: House, rng, min_clear=0.5):
    c = np.where(h.clear >= min_clear)[0]
    return int(rng.choice(c if len(c) else np.arange(len(h.reach))))


def _walk_tour(h: House, wk: Walker, i_from: int, n_steps: int, rng, prev=None):
    """Walk goal-to-goal (2-5 m legs) for n_steps."""
    goals = 0; cur = i_from; last = prev
    while wk.n < n_steps + 1 and wk.ok:
        g = _goal_from(h, cur, rng, away_from=last)
        if g is None:
            wk.ok = False; break
        before = wk.n
        wk.pursue(h.path(h.node(wk.p), g), n_steps + 1 - wk.n)
        if wk.n == before:
            wk.ok = False; break
        if np.linalg.norm(wk.p - h.reach[g]) < 0.3:
            goals += 1
        last, cur = cur, g
    return goals


def _truncate(wk: Walker, n):
    wk.P, wk.Psi = wk.P[:n], wk.Psi[:n]


def plan(h: House, kind: str, T: int, rng, w_short: float | None = None) -> dict | None:
    """One window of T frames (+PAD held) or None. Rollout = steps from frame WARMUP to frame T-1."""
    R = T - 1 - WARMUP                              # rollout steps (incl. the step to the target frame)
    if kind == "brownian":
        return _brownian(h, T, rng)
    i0 = _start(h, rng)
    g0 = _goal_from(h, i0, rng)
    if g0 is None:
        return None
    path0 = h.path(i0, g0)
    psi0 = math.atan2(*(smooth(path0)[min(2, len(path0) - 1)] - h.reach[i0]))
    wk = Walker(h, h.reach[i0], psi0)
    info = {}
    if kind == "goal":
        info["goals_total"] = _walk_tour(h, wk, i0, T, rng)
    else:
        wk.pursue(path0, WARMUP)                    # warm-up: walk toward g0
        if wk.n < WARMUP + 1:                       # reached g0 early: keep touring
            _walk_tour(h, wk, g0, WARMUP + 1, rng)
        _truncate(wk, WARMUP + 1)
        if not wk.ok or wk.n < WARMUP + 1:
            return None
        A = wk.P[WARMUP].copy(); info["A"] = A.tolist()
        if kind == "turn":
            w = W_MAX if R >= math.ceil(2 * math.pi / W_MAX) else max(W_MAX, 2 * math.pi / R)
            pre = int(rng.integers(5, max(6, R - math.ceil(2 * math.pi / w) - 10))) if R >= 60 else 0
            _walk_tour(h, wk, h.node(wk.p), WARMUP + 1 + pre, rng) if pre else None
            _truncate(wk, WARMUP + 1 + pre)
            s0 = wk.n - 1
            wk.circle(rng.choice([-1, 1]) * 2 * math.pi, w, 0.05)
            info |= {"turn_start": s0, "turn_end": wk.n - 1, "turn_rate": w}
        elif kind == "outback":
            w = W_MAX if R >= 60 else max(W_MAX, math.pi / 20)
            n_turn = math.ceil(math.pi / w)
            d_out = float(rng.uniform(2.0, 3.0)) if R >= 60 else V * (R - n_turn - 2) / 2
            # near-straight goal ahead: geodesic ~ euclidean ~ d_out, within 40 deg of the heading
            i_a = h.node(A); d = h.geo[i_a]; e = np.linalg.norm(h.reach - A, axis=1)
            ang = np.abs(wrap(np.arctan2(h.reach[:, 0] - A[0], h.reach[:, 1] - A[1]) - wk.psi))
            c = np.where((np.abs(d - d_out) < 0.2) & (d < 1.1 * e + 0.05) & (ang < math.radians(40)))[0]
            if not len(c):
                return None
            B = int(rng.choice(c))
            n_out = wk.pursue(h.path(i_a, B), int(round(d_out / V)) + 6)
            s0 = wk.n - 1
            wk.circle(rng.choice([-1, 1]) * math.pi, w, 0.02)
            info |= {"d_out_target": d_out, "out_steps": n_out, "turn_start": s0, "turn_end": wk.n - 1, "turn_rate": w}
            wk.pursue(np.vstack([wk.p, A]), T - wk.n, stop=0.05)
            info["back_end"] = wk.n - 1
        elif kind == "loop":
            w = w_short or (W_MAX if R >= 60 else 2 * math.pi / R)
            r_steps = math.ceil(2 * math.pi / w)
            if R >= 60:
                w = float(rng.uniform(0.09, W_MAX)); r_steps = math.ceil(2 * math.pi / w)
            v = V if R >= 60 else min(V, 0.115)
            side = rng.choice([-1, 1])
            s0 = wk.n - 1
            wk.circle(side * 2 * math.pi, w, v)
            info |= {"loop_start": s0, "loop_end": wk.n - 1, "turn_rate": w, "radius_m": v / w}
        if wk.n < T + 1 and wk.ok:                  # walk on for the rest of the window
            _walk_tour(h, wk, h.node(wk.p), T, rng)
    if not wk.ok or wk.n < T:
        return None
    _truncate(wk, T)
    P = np.array(wk.P); Psi = np.array(wk.Psi)
    P = np.vstack([P, np.repeat(P[-1:], PAD, 0)]); Psi = np.r_[Psi, np.repeat(Psi[-1], PAD)]
    return {"kind": kind, "T": T, "pos_u": P, "psi_u": Psi, "info": info}


def _brownian(h: House, T: int, rng, step_std=0.1, angle_std=0.15) -> dict:
    """The procthor-walks generator's motion (walk_and_render), from a start clear of walls."""
    pos = h.reach[rng.choice(np.where(h.nbrs >= np.quantile(h.nbrs, 0.9))[0])].astype(float)
    yaw = float(rng.uniform(0, 2 * math.pi)); P = [pos.copy()]; Psi = [yaw]
    for _ in range(T - 1):
        cand = pos + rng.normal(0.0, step_std, 2)
        d, j = h.tree.query(cand)
        pos = h.reach[j].astype(float) if d > SNAP_TOL else cand
        yaw = float(wrap(yaw + rng.normal(0.0, angle_std)))
        P.append(pos.copy()); Psi.append(yaw)
    P = np.array(P); Psi = np.array(Psi)
    P = np.vstack([P, np.repeat(P[-1:], PAD, 0)]); Psi = np.r_[Psi, np.repeat(Psi[-1], PAD)]
    return {"kind": "brownian", "T": T, "pos_u": P, "psi_u": Psi, "info": {}}


def metrics(w: dict) -> dict:
    """Event metrics on the rollout (frames WARMUP..T-1) and per-step egomotion over the whole window."""
    T = w["T"]; P = w["pos_u"][:T]; Psi = w["psi_u"][:T]
    step = np.linalg.norm(np.diff(P, axis=0), axis=1); dpsi = wrap(np.diff(Psi))
    r = slice(WARMUP, T); Pr = P[r]; A = Pr[0]
    disp = np.linalg.norm(Pr - A, axis=1)
    left = np.where(disp > 0.8)[0]
    ret = float(disp[left[0]:].min()) if len(left) else None
    rd = dpsi[WARMUP:]
    m = {"path_m": float(step[WARMUP:].sum()), "max_disp_m": float(disp.max()), "end_disp_m": float(disp[-1]),
         "net_turn_rad": float(rd.sum()), "abs_turn_rad": float(np.abs(rd).sum()),
         "max_abs_net_turn_rad": float(np.abs(np.cumsum(rd)).max()),
         "return_to_A_m": ret,
         "max_step_m": float(step.max()), "max_turn_rad": float(np.abs(dpsi).max()),
         "frac_steps_lt_0.032m": float((step < 0.032).mean()),            # below training step p5 (incl. zero)
         "frac_turns_gt_0.15": float((np.abs(dpsi) > 0.15 + 1e-9).mean()),
         "frac_turns_gt_0.16": float((np.abs(dpsi) > 0.16 + 1e-9).mean())}
    i = w["info"]; k = w["kind"]
    if k == "turn":
        m["event_ok"] = abs(float(dpsi[i["turn_start"]:i["turn_end"]].sum())) >= 2 * math.pi - 1e-6 and i["turn_end"] <= T - 1
    elif k == "outback":
        out = float(np.linalg.norm(P[i["turn_start"]] - A))
        m |= {"out_m": out, "end_to_A_m": float(np.linalg.norm(P[min(i["back_end"], T - 1)] - A))}
        m["event_ok"] = i["turn_end"] <= T - 1 and m["end_to_A_m"] <= 0.3 and out >= (1.0 if T < 100 else 2.0)
    elif k == "loop":
        e = i["loop_end"]
        m |= {"loop_closure_m": float(np.linalg.norm(P[min(e, T - 1)] - A)),
              "loop_turn_rad": abs(float(dpsi[i["loop_start"]:min(e, T - 1)].sum()))}
        m["event_ok"] = e <= T - 1 and m["loop_closure_m"] <= 0.3 and m["loop_turn_rad"] >= 2 * math.pi - 1e-6
    elif k == "goal":
        m["event_ok"] = m["path_m"] >= (4.0 if T < 100 else 10.0) * 0.9 and m["max_disp_m"] >= (2.0 if T < 100 else 3.0)
    else:
        m["event_ok"] = True
    return m
