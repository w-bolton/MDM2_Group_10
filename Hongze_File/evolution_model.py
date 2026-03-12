"""
Evolution Model
- Hongze Lin
--------------------------------------------------------------
Basic Coordinate Assumptions
Camera optical axis: +y
Sensor / pixel plane: x–z
Frame origin: (x=0, z=0) at the CENTER of the frame.

Variable Statement (same notation as simulate_forward / pipeline_forward):
  xp: particle x positions (width; out-of-paper direction), centered: [-Lx/2, Lx/2)
  zp: particle z positions (height), typically centered if zmin=-zmax
  y : particle y positions (depth/camera axis; used for sheet gating)
  c : dye concentration slice on x–z grid, shape (H, W)
"""

import numpy as np
from dataclasses import dataclass
from pathlib import Path
import argparse


# =========================================================
# Config / Variable Statement
# =========================================================

@dataclass
class SimConfig:
    # time / size
    T: int = 80
    dt: float = 0.01
    H: int = 384   # pixels along z
    W: int = 384   # pixels along x
    N: int = 1000

    # physical domain
    # x: width (perpendicular to drawing plane), centered at 0
    # y: length (camera optical axis)
    # z: height, typically centered at 0 if zmin=-zmax
    Lx: float = 1.0
    zmin: float = -0.5
    zmax: float = 0.5

    # placeholder velocity field on x–z slice
    A: float = 0.02
    k: float = 2 * np.pi / 1.0
    gamma: float = 0.0

    # particle dynamics noise (model error) for x–z advection
    particle_noise_sigma: float = 5e-4

    # dye dynamics on x–z light-sheet slice
    dye_kappa: float = 0.0  # diffusion

    # light-sheet gating along y (camera axis)
    enable_sheet_gating: bool = True
    sheet_center_y: float = 0.0
    sheet_thickness: float = 0.02

    # y evolution (to create appear/disappear in the sheet)
    y_noise_sigma: float = 0.005
    y_kill: float = 0.06

    seed: int = 1


@dataclass
class State:
    xp: np.ndarray  # (N,) x positions (centered)
    zp: np.ndarray  # (N,) z positions
    y: np.ndarray   # (N,) y positions (depth/camera axis)
    c: np.ndarray   # (H,W) dye slice c(x,z,t) on centered x grid


# =========================================================
# Coordinate centered periodic wrap in x
# =========================================================

def wrap_x_centered(x: np.ndarray, Lx: float) -> np.ndarray:
    """
    Wrap x into the centered periodic interval [-Lx/2, Lx/2).
    """
    return ((x + 0.5 * Lx) % Lx) - 0.5 * Lx


# =========================================================
# Velocity field on x–z slice 
# =========================================================

def vel_u_w(x, z, t, A, k, gamma):
    """
    Placeholder analytic field on x–z plane:
      u = dx/dt, w = dz/dt
    """
    decay = np.exp(-k * np.abs(z))
    phase = np.exp(1j * k * x)
    growth = np.exp(gamma * t)
    u = np.real(1j * k * A * decay * phase * growth)
    w = np.real(-k * A * np.sign(z) * decay * phase * growth)
    return u, w


# =========================================================
# Numerical advection and evolution steps
# =========================================================

def advect_particles_rk2(x, z, t, dt, cfg: SimConfig):
    """
    Particle advection on x–z slice using RK2, with optional process noise.
    - x: periodic in centered interval [-Lx/2, Lx/2)
    - z: clipped in [zmin, zmax]
    """
    u1, w1 = vel_u_w(x, z, t, cfg.A, cfg.k, cfg.gamma)
    xm = x + 0.5 * dt * u1
    zm = z + 0.5 * dt * w1

    u2, w2 = vel_u_w(xm, zm, t + 0.5 * dt, cfg.A, cfg.k, cfg.gamma)
    x_new = x + dt * u2
    z_new = z + dt * w2

    # model error / stochasticity
    if cfg.particle_noise_sigma > 0:
        x_new += np.random.normal(0.0, cfg.particle_noise_sigma, size=x_new.shape)
        z_new += np.random.normal(0.0, cfg.particle_noise_sigma, size=z_new.shape)

    # boundaries
    x_new = wrap_x_centered(x_new, cfg.Lx)
    z_new = np.clip(z_new, cfg.zmin, cfg.zmax)
    return x_new, z_new


def bilinear_sample(field, xq, zq, xs, zs, periodic_x=True):
    """
    Bilinear interpolation for field on (zs, xs) grid.
    xs is now centered: [-Lx/2, Lx/2).
    If periodic_x=True, xq is wrapped into [-Lx/2, Lx/2).
    """
    Nz, Nx = field.shape
    dx = xs[1] - xs[0]
    dz = zs[1] - zs[0]

    if periodic_x:
        xq = wrap_x_centered(xq, xs[-1] - xs[0] + dx)  # equals Lx if xs spans full period

    ix = (xq - xs[0]) / dx
    iz = (zq - zs[0]) / dz

    if periodic_x:
        ix = np.mod(ix, Nx)
    else:
        ix = np.clip(ix, 0, Nx - 1 - 1e-6)
    iz = np.clip(iz, 0, Nz - 1 - 1e-6)

    i0 = np.floor(ix).astype(int)
    j0 = np.floor(iz).astype(int)
    i1 = (i0 + 1) % Nx if periodic_x else np.minimum(i0 + 1, Nx - 1)
    j1 = np.minimum(j0 + 1, Nz - 1)

    tx = ix - i0
    tz = iz - j0

    f00 = field[j0, i0]
    f10 = field[j0, i1]
    f01 = field[j1, i0]
    f11 = field[j1, i1]

    return (1 - tx) * (1 - tz) * f00 + tx * (1 - tz) * f10 + (1 - tx) * tz * f01 + tx * tz * f11


def advect_dye_semilag(c, xs, zs, t, dt, cfg: SimConfig):
    """
    Semi-Lagrangian advection of dye concentration c(x,z,t) on the centered x–z slice.
    """
    X, Z = np.meshgrid(xs, zs)
    u, w = vel_u_w(X, Z, t, cfg.A, cfg.k, cfg.gamma)

    # backtrace
    Xb = X - u * dt
    Zb = Z - w * dt

    c_new = bilinear_sample(c, Xb, Zb, xs, zs, periodic_x=True)

    # optional diffusion
    if cfg.dye_kappa > 0:
        c_pad = np.pad(c_new, ((1, 1), (0, 0)), mode="edge")
        c_up = c_pad[0:-2, :]
        c_dn = c_pad[2:, :]
        c_lt = np.roll(c_new, 1, axis=1)
        c_rt = np.roll(c_new, -1, axis=1)
        dx = xs[1] - xs[0]
        dz = zs[1] - zs[0]
        lap = (c_lt - 2 * c_new + c_rt) / dx**2 + (c_up - 2 * c_new + c_dn) / dz**2
        c_new = np.clip(c_new + dt * cfg.dye_kappa * lap, 0.0, None)

    return c_new


def update_y_depth(y, dt, cfg: SimConfig):
    """
    Stochastic evolution of particle depth y (camera axis).
    """
    return y + np.random.normal(0.0, cfg.y_noise_sigma * np.sqrt(dt), size=y.shape)


def visible_mask_y(y, cfg: SimConfig):
    """
    Light-sheet gating along y:
      visible if |y - sheet_center_y| <= sheet_thickness/2
    """
    if not cfg.enable_sheet_gating:
        return np.ones_like(y, dtype=bool)
    return np.abs(y - cfg.sheet_center_y) <= 0.5 * cfg.sheet_thickness


def respawn(mask, state: State, cfg: SimConfig):
    """
    Respawn particles drifting too far in y.
    x is respawned uniformly on centered interval [-Lx/2, Lx/2).
    """
    idx = np.where(mask)[0]
    if idx.size == 0:
        return state

    state.xp[idx] = np.random.uniform(-0.5 * cfg.Lx, 0.5 * cfg.Lx, size=idx.size).astype(np.float32)

    zp_new = np.random.normal(0.0, 0.12 * (cfg.zmax - cfg.zmin), size=idx.size)
    state.zp[idx] = np.clip(zp_new, cfg.zmin, cfg.zmax).astype(np.float32)

    state.y[idx] = np.random.uniform(
        cfg.sheet_center_y - 0.25 * cfg.sheet_thickness,
        cfg.sheet_center_y + 0.25 * cfg.sheet_thickness,
        size=idx.size,
    ).astype(np.float32)

    return state


# =========================================================
# Initialisation (centered coordinates)
# =========================================================

def init_state(cfg: SimConfig):
    """
    Initialise (x0) with frame-centered coordinates:
      xp in [-Lx/2, Lx/2)
      zp in [zmin, zmax] (often symmetric)
      y near sheet center
      c defined on centered x grid and z grid
    """
    np.random.seed(cfg.seed)

    xp = np.random.uniform(-0.5 * cfg.Lx, 0.5 * cfg.Lx, size=cfg.N).astype(np.float32)

    zp = np.random.normal(0.0, 0.12 * (cfg.zmax - cfg.zmin), size=cfg.N).astype(np.float32)
    zp = np.clip(zp, cfg.zmin, cfg.zmax)

    y = np.random.uniform(
        cfg.sheet_center_y - 0.25 * cfg.sheet_thickness,
        cfg.sheet_center_y + 0.25 * cfg.sheet_thickness,
        size=cfg.N,
    ).astype(np.float32)

    # centered x grid for the x–z slice
    xs = np.linspace(-0.5 * cfg.Lx, 0.5 * cfg.Lx, cfg.W, endpoint=False)
    zs = np.linspace(cfg.zmin, cfg.zmax, cfg.H)

    # dye blob on centered grid (example)
    X, Z = np.meshgrid(xs, zs)
    c = np.exp(-((X - 0.0) ** 2 / (0.08 ** 2) + (Z - 0.15) ** 2 / (0.10 ** 2))).astype(np.float32)

    return State(xp=xp, zp=zp, y=y, c=c), xs, zs


# =========================================================
# One-step evolution: x_t -> x_{t+dt}
# =========================================================

def step_evolution(state: State, xs, zs, t, cfg: SimConfig):
    """
    Advance the latent state by one time step (Evolution model only).

    Order matches simulate_forward/pipeline_forward logic:
      1) advect particles in x–z (RK2 + noise)
      2) advect dye slice c(x,z,t) in x–z (semi-Lagrangian + optional diffusion)
      3) update y-depth stochastic process
      4) respawn particles beyond y_kill
      5) compute visibility mask vis

    Returns:
      state (updated)
      vis   (bool mask, shape (N,))
    """
    state.xp, state.zp = advect_particles_rk2(state.xp, state.zp, t, cfg.dt, cfg)
    state.c = advect_dye_semilag(state.c, xs, zs, t, cfg.dt, cfg)

    state.y = update_y_depth(state.y, cfg.dt, cfg)
    kill = np.abs(state.y - cfg.sheet_center_y) > cfg.y_kill
    state = respawn(kill, state, cfg)

    vis = visible_mask_y(state.y, cfg)
    return state, vis


# =========================================================
# Quick sanity-run without rendering
# =========================================================

def run_evolution_only(cfg: SimConfig):
    state, xs, zs = init_state(cfg)
    t = 0.0
    visible_frac = []

    for _ in range(cfg.T):
        state, vis = step_evolution(state, xs, zs, t, cfg)
        visible_frac.append(float(np.mean(vis)))
        t += cfg.dt

    return state, {"visible_frac": visible_frac}


# =========================================================
# Visualisation (optional, additive)
# =========================================================

def _state_to_rgb_frame(state: State, xs, zs, vis, step_idx, cfg: SimConfig):
    """
    Render one RGB frame for visualisation:
      - background: dye field c(x,z)
      - foreground: particles (visible vs hidden by y gating)
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for visualisation. "
            "Install it first (e.g. `pip install matplotlib`)."
        ) from exc

    dx = xs[1] - xs[0]
    x_right = xs[-1] + dx

    fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
    ax.imshow(
        state.c,
        extent=[xs[0], x_right, zs[0], zs[-1]],
        origin="lower",
        cmap="viridis",
        aspect="auto",
    )

    hidden = ~vis
    if np.any(hidden):
        ax.scatter(
            state.xp[hidden],
            state.zp[hidden],
            s=7,
            c="white",
            alpha=0.28,
            linewidths=0.0,
        )
    if np.any(vis):
        ax.scatter(
            state.xp[vis],
            state.zp[vis],
            s=10,
            c="red",
            alpha=0.78,
            linewidths=0.0,
        )

    ax.set_xlim(xs[0], x_right)
    ax.set_ylim(zs[0], zs[-1])
    ax.set_title(f"Evolution step={step_idx}, visible={np.mean(vis):.3f}")
    ax.set_xlabel("x (width)")
    ax.set_ylabel("z (height)")
    fig.tight_layout()

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[..., :3]
    plt.close(fig)
    return frame


def _state_to_rgb_frame_3d(state: State, xs, zs, vis, step_idx, cfg: SimConfig):
    """
    Render one 3D RGB frame:
      - dye rendered as a thin 3D volume (stacked y-slices)
      - particles rendered at true (x,y,z) coordinates
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for 3D visualisation. "
            "Install it first (e.g. `pip install matplotlib`)."
        ) from exc

    fig = plt.figure(figsize=(7.4, 6.2), dpi=120)
    ax = fig.add_subplot(111, projection="3d")

    # Downsample dye grid for faster 3D rendering.
    x_stride = max(1, cfg.W // 96)
    z_stride = max(1, cfg.H // 96)
    xs_d = xs[::x_stride]
    zs_d = zs[::z_stride]
    c_d = state.c[::z_stride, ::x_stride]
    Xd, Zd = np.meshgrid(xs_d, zs_d)

    c_norm = c_d - np.min(c_d)
    denom = np.max(c_norm)
    if denom > 0:
        c_norm = c_norm / denom
    else:
        c_norm = np.zeros_like(c_norm)

    # Volumetric dye effect: stack multiple semi-transparent slices along y.
    # Slices farther from sheet center are dimmer to emulate thickness falloff.
    y_pad = max(cfg.y_kill * 2.6, cfg.sheet_thickness * 8.0)
    dye_half_thickness = max(cfg.sheet_thickness * 2.2, 1e-4)
    n_slices = 11
    y_offsets = np.linspace(-dye_half_thickness, dye_half_thickness, n_slices)
    for y_off in y_offsets:
        y_level = cfg.sheet_center_y + y_off
        falloff = np.exp(-0.5 * (y_off / (0.45 * dye_half_thickness)) ** 2)
        rgba = plt.cm.viridis(c_norm)
        rgba[..., 3] = np.clip(0.015 + 0.20 * falloff * c_norm, 0.0, 0.35)
        Yd = np.full_like(Xd, y_level, dtype=np.float32)
        ax.plot_surface(
            Xd,
            Yd,
            Zd,
            facecolors=rgba,
            rstride=1,
            cstride=1,
            shade=False,
            linewidth=0.0,
            antialiased=False,
        )

    # Draw light-sheet center and boundaries in 3D for spatial context.
    x_line = np.array([xs[0], xs[-1] + (xs[1] - xs[0])], dtype=np.float32)
    z_line = np.array([cfg.zmin, cfg.zmax], dtype=np.float32)
    Xl, Zl = np.meshgrid(x_line, z_line)
    for y_plane, a in [
        (cfg.sheet_center_y, 0.30),
        (cfg.sheet_center_y - 0.5 * cfg.sheet_thickness, 0.18),
        (cfg.sheet_center_y + 0.5 * cfg.sheet_thickness, 0.18),
    ]:
        Yl = np.full_like(Xl, y_plane, dtype=np.float32)
        ax.plot_wireframe(Xl, Yl, Zl, color="white", linewidth=0.6, alpha=a)

    hidden = ~vis
    if np.any(hidden):
        hidden_dist = np.abs(state.y[hidden] - cfg.sheet_center_y)
        ax.scatter(
            state.xp[hidden],
            state.y[hidden],
            state.zp[hidden],
            s=14,
            c=hidden_dist,
            cmap="magma",
            vmin=0.0,
            vmax=y_pad,
            marker="o",
            alpha=0.88,
            depthshade=True,
        )
    if np.any(vis):
        ax.scatter(
            state.xp[vis],
            state.y[vis],
            state.zp[vis],
            s=16,
            c="#00E676",
            alpha=0.95,
            depthshade=True,
        )

    ax.set_xlim(-0.5 * cfg.Lx, 0.5 * cfg.Lx)
    ax.set_ylim(cfg.sheet_center_y - y_pad, cfg.sheet_center_y + y_pad)
    ax.set_zlim(cfg.zmin, cfg.zmax)
    # Stretch y display scale so particles are not visually collapsed.
    y_display_stretch = 5.5
    ax.set_box_aspect((cfg.Lx, y_display_stretch * (2.0 * y_pad), cfg.zmax - cfg.zmin))
    ax.set_xlabel("x (width)")
    ax.set_ylabel("y (depth)")
    ax.set_zlabel("z (height)")
    ax.set_title(f"3D evolution step={step_idx}, visible={np.mean(vis):.3f}")

    # Keep camera fixed so key frames share identical orientation/perspective.
    ax.set_proj_type("persp")
    ax.view_init(elev=22, azim=38)
    fig.tight_layout()

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[..., :3]
    plt.close(fig)
    return frame


def visualise_evolution(cfg: SimConfig, out_dir: Path, fps: int = 12, sample_every: int = 1):
    """
    Run evolution and export:
      - evolution.gif
      - first/middle/last key frames (PNG)
    """
    if sample_every <= 0:
        raise ValueError("sample_every must be >= 1")

    out_dir.mkdir(parents=True, exist_ok=True)
    state, xs, zs = init_state(cfg)
    t = 0.0
    visible_frac = []
    frames = []

    for step_idx in range(cfg.T):
        state, vis = step_evolution(state, xs, zs, t, cfg)
        visible_frac.append(float(np.mean(vis)))
        t += cfg.dt

        if step_idx % sample_every == 0:
            frames.append(_state_to_rgb_frame(state, xs, zs, vis, step_idx, cfg))

    if not frames:
        raise RuntimeError("No frames were generated. Check T/sample_every settings.")

    import imageio.v2 as imageio
    from PIL import Image

    gif_path = out_dir / "evolution.gif"
    imageio.mimsave(str(gif_path), frames, duration=1.0 / fps)

    key_idxs = [0, len(frames) // 2, len(frames) - 1]
    key_names = ["frame0", "frame_mid", "frame_last"]
    for idx, name in zip(key_idxs, key_names):
        Image.fromarray(frames[idx]).save(out_dir / f"{name}.png")

    return {
        "gif_path": gif_path,
        "avg_visible_frac": float(np.mean(visible_frac)),
        "num_frames": len(frames),
    }


def visualise_evolution_3d(cfg: SimConfig, out_dir: Path, fps: int = 12, sample_every: int = 1):
    """
    Run evolution and export 3D animation:
      - evolution_3d.gif
      - first/middle/last key frames (PNG)
    """
    if sample_every <= 0:
        raise ValueError("sample_every must be >= 1")

    out_dir.mkdir(parents=True, exist_ok=True)
    state, xs, zs = init_state(cfg)
    t = 0.0
    visible_frac = []
    frames = []

    for step_idx in range(cfg.T):
        state, vis = step_evolution(state, xs, zs, t, cfg)
        visible_frac.append(float(np.mean(vis)))
        t += cfg.dt

        if step_idx % sample_every == 0:
            frames.append(_state_to_rgb_frame_3d(state, xs, zs, vis, step_idx, cfg))

    if not frames:
        raise RuntimeError("No 3D frames were generated. Check T/sample_every settings.")

    import imageio.v2 as imageio
    from PIL import Image

    gif_path = out_dir / "evolution_3d.gif"
    imageio.mimsave(str(gif_path), frames, duration=1.0 / fps)

    key_idxs = [0, len(frames) // 2, len(frames) - 1]
    key_names = ["frame3d_0", "frame3d_mid", "frame3d_last"]
    for idx, name in zip(key_idxs, key_names):
        Image.fromarray(frames[idx]).save(out_dir / f"{name}.png")

    return {
        "gif_path": gif_path,
        "avg_visible_frac": float(np.mean(visible_frac)),
        "num_frames": len(frames),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Evolution-only model runner")
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Export visualisation GIF + key frames.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="Outputs_evolution",
        help="Output directory for visualisation files (relative to script dir if not absolute).",
    )
    parser.add_argument("--fps", type=int, default=12, help="GIF fps for visualisation.")
    parser.add_argument(
        "--sample_every",
        type=int,
        default=1,
        help="Render every N simulation steps.",
    )
    parser.add_argument(
        "--visualize_3d",
        action="store_true",
        help="Export 3D visualisation GIF + key frames.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = SimConfig()
    final_state, diag = run_evolution_only(cfg)
    print("Evolution-only run complete.")
    print("visible fraction (avg):", float(np.mean(diag["visible_frac"])))

    if args.visualize:
        script_dir = Path(__file__).resolve().parent
        out_dir = Path(args.out_dir)
        if not out_dir.is_absolute():
            out_dir = script_dir / out_dir

        result = visualise_evolution(
            cfg=cfg,
            out_dir=out_dir,
            fps=args.fps,
            sample_every=args.sample_every,
        )
        print("visualisation saved to:", result["gif_path"])
        print("rendered frames:", result["num_frames"])

    if args.visualize_3d:
        script_dir = Path(__file__).resolve().parent
        out_dir = Path(args.out_dir)
        if not out_dir.is_absolute():
            out_dir = script_dir / out_dir

        result3d = visualise_evolution_3d(
            cfg=cfg,
            out_dir=out_dir,
            fps=args.fps,
            sample_every=args.sample_every,
        )
        print("3D visualisation saved to:", result3d["gif_path"])
        print("3D rendered frames:", result3d["num_frames"])
