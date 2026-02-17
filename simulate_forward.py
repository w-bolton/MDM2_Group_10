# =========================================================
# simulate_forward function
# - Hongze Lin
# =========================================================

import numpy as np
from dataclasses import dataclass
from pathlib import Path
import argparse


# =========================================================
# Config / State
# =========================================================

@dataclass
class SimConfig:
    # time / size
    T: int = 80
    dt: float = 0.02
    H: int = 384
    W: int = 384
    N: int = 1000

    # physical domain (x periodic)
    Lx: float = 1.0
    zmin: float = -0.5
    zmax: float = 0.5

    # placeholder velocity field (3)(4)
    A: float = 0.02
    k: float = 2 * np.pi / 1.0
    gamma: float = 0.0

    # particle dynamics noise (model error)
    particle_noise_sigma: float = 5e-4

    # dye dynamics
    dye_kappa: float = 0.0  # diffusion

    # out-of-plane appear/disappear
    enable_out_of_plane: bool = True
    sheet_thickness: float = 0.02
    y_noise_sigma: float = 0.005
    y_kill: float = 0.06

    # rendering: particles
    psf_sigma_px: float = 1.6
    particle_amp: float = 2.0

    # rendering: dye
    dye_beta: float = 8.0
    dye_alpha: float = 0.6
    light_source_x_frac: float = 0.5   # relative to Lx
    light_source_z_above_frac: float = 1.2  # above top boundary in units of height
    dye_blur_sigma_px: float = 0.7

    # camera (physical-ish)
    use_camera_model: bool = True
    bg: float = 10.0
    gain: float = 120.0
    read_sigma: float = 1.5

    # visualization exposure (prevents "all black")
    auto_exposure: bool = True
    exposure_percentile: float = 99.7  # scale by this percentile to 0..255

    seed: int = 1


@dataclass
class State:
    xp: np.ndarray  # (N,)
    zp: np.ndarray  # (N,)
    y: np.ndarray   # (N,) out-of-plane
    c: np.ndarray   # (H,W) dye field


# =========================================================
# Velocity field Equations (3)(4)
# =========================================================

def vel_u_w(x, z, t, A, k, gamma):
    """
    Equations (3)(4) in a unified form:
      u = Re(i k A e^{-k|z|} e^{ikx} e^{γt})
      w = Re(-k A sign(z) e^{-k|z|} e^{ikx} e^{γt})
    """
    decay = np.exp(-k * np.abs(z))
    phase = np.exp(1j * k * x)
    growth = np.exp(gamma * t)
    u = np.real(1j * k * A * decay * phase * growth)
    w = np.real(-k * A * np.sign(z) * decay * phase * growth)
    return u, w


# =========================================================
# Dynamics
# =========================================================

def advect_particles_rk2(x, z, t, dt, cfg: SimConfig):
    u1, w1 = vel_u_w(x, z, t, cfg.A, cfg.k, cfg.gamma)
    xm = x + 0.5 * dt * u1
    zm = z + 0.5 * dt * w1

    u2, w2 = vel_u_w(xm, zm, t + 0.5 * dt, cfg.A, cfg.k, cfg.gamma)
    x_new = x + dt * u2
    z_new = z + dt * w2

    if cfg.particle_noise_sigma > 0:
        x_new += np.random.normal(0.0, cfg.particle_noise_sigma, size=x_new.shape)
        z_new += np.random.normal(0.0, cfg.particle_noise_sigma, size=z_new.shape)

    # boundaries
    x_new = np.mod(x_new, cfg.Lx)                 # periodic in x
    z_new = np.clip(z_new, cfg.zmin, cfg.zmax)    # clipped in z
    return x_new, z_new


def bilinear_sample(field, xq, zq, xs, zs, periodic_x=True):
    Nz, Nx = field.shape
    dx = xs[1] - xs[0]
    dz = zs[1] - zs[0]

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
    X, Z = np.meshgrid(xs, zs)
    u, w = vel_u_w(X, Z, t, cfg.A, cfg.k, cfg.gamma)
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


def update_out_of_plane(y, dt, cfg: SimConfig):
    y = y + np.random.normal(0.0, cfg.y_noise_sigma * np.sqrt(dt), size=y.shape)
    return y


def visible_mask(y, cfg: SimConfig):
    return np.abs(y) <= 0.5 * cfg.sheet_thickness


def respawn(mask, state: State, cfg: SimConfig):
    idx = np.where(mask)[0]
    if idx.size == 0:
        return state

    state.xp[idx] = np.random.uniform(0, cfg.Lx, size=idx.size)
    zp_new = np.random.normal(0.0, 0.12 * (cfg.zmax - cfg.zmin), size=idx.size)
    state.zp[idx] = np.clip(zp_new, cfg.zmin, cfg.zmax)
    state.y[idx] = np.random.uniform(-0.25 * cfg.sheet_thickness, 0.25 * cfg.sheet_thickness, size=idx.size)
    return state


# =========================================================
# Rendering
# =========================================================

def gaussian_blur_fft(img, sigma):
    if sigma <= 0:
        return img
    H, W = img.shape
    ky = np.fft.fftfreq(H) * 2 * np.pi
    kx = np.fft.fftfreq(W) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky)
    G = np.exp(-0.5 * sigma**2 * (KX**2 + KY**2))
    return np.real(np.fft.ifft2(np.fft.fft2(img) * G))


def render_particles(xp, zp, cfg: SimConfig):
    img = np.zeros((cfg.H, cfg.W), dtype=np.float32)

    px = (xp / cfg.Lx) * (cfg.W - 1)
    pz = ((zp - cfg.zmin) / (cfg.zmax - cfg.zmin)) * (cfg.H - 1)

    ix = np.rint(px).astype(int)
    iz = np.rint(pz).astype(int)

    m = (ix >= 0) & (ix < cfg.W) & (iz >= 0) & (iz < cfg.H)
    ix = ix[m]; iz = iz[m]
    np.add.at(img, (iz, ix), cfg.particle_amp)

    img = gaussian_blur_fft(img, cfg.psf_sigma_px)
    return img


def render_dye(c, xs, zs, cfg: SimConfig):
    X, Z = np.meshgrid(xs, zs)
    x0 = cfg.light_source_x_frac * cfg.Lx
    z0 = cfg.zmax + cfg.light_source_z_above_frac * (cfg.zmax - cfg.zmin)

    d = np.sqrt((X - x0) ** 2 + (Z - z0) ** 2) + 1e-6
    L = 1.0 / (d**2 + 1e-6)  # geometric attenuation

    I = cfg.dye_beta * c * L * np.exp(-cfg.dye_alpha * d * c)
    I = gaussian_blur_fft(I.astype(np.float32), cfg.dye_blur_sigma_px)
    return I


# =========================================================
# Camera / exposure
# =========================================================

def camera_model(I, cfg: SimConfig):
    I = np.clip(I + cfg.bg, 0.0, None)
    lam = np.clip(I * cfg.gain, 0.0, None)
    noisy = np.random.poisson(lam).astype(np.float32) / cfg.gain
    noisy += np.random.normal(0.0, cfg.read_sigma, size=noisy.shape).astype(np.float32)
    return np.clip(noisy, 0.0, 255.0).astype(np.uint8)


def auto_exposure_to_uint8(I, cfg: SimConfig):
    hi = np.percentile(I, cfg.exposure_percentile)
    hi = max(float(hi), 1e-6)
    J = np.clip(I / hi * 255.0, 0.0, 255.0)
    return J.astype(np.uint8)


# =========================================================
# Initialization (x0)
# =========================================================

def init_state(cfg: SimConfig):
    np.random.seed(cfg.seed)

    xp = np.random.uniform(0, cfg.Lx, size=cfg.N)
    zp = np.random.normal(0.0, 0.12 * (cfg.zmax - cfg.zmin), size=cfg.N)
    zp = np.clip(zp, cfg.zmin, cfg.zmax)

    y = np.zeros(cfg.N, dtype=np.float32)
    if cfg.enable_out_of_plane:
        y = np.random.uniform(-0.25 * cfg.sheet_thickness, 0.25 * cfg.sheet_thickness, size=cfg.N).astype(np.float32)

    xs = np.linspace(0, cfg.Lx, cfg.W, endpoint=False)
    zs = np.linspace(cfg.zmin, cfg.zmax, cfg.H)

    # dye blob example (you can replace with your own c0)
    X, Z = np.meshgrid(xs, zs)
    c = np.exp(-((X - 0.5 * cfg.Lx) ** 2 / (0.08 ** 2) + (Z - 0.15) ** 2 / (0.10 ** 2))).astype(np.float32)

    return State(xp=xp.astype(np.float32), zp=zp.astype(np.float32), y=y, c=c), xs, zs


# =========================================================
# Forward simulator internal modules
# =========================================================

def init_diagnostics():
    """
    Create per-frame diagnostic containers.
    Keeping this isolated makes the simulator loop easier to read while
    preserving the exact same metrics and accumulation order.
    """
    return {
        "I_min": [],
        "I_max": [],
        "I_mean": [],
        "visible_frac": []
    }


def step_flow_dynamics(state: State, xs, zs, t, cfg: SimConfig):
    """
    Advance physical state by one time step.
    This function only handles transport:
      1) particle advection
      2) dye semi-Lagrangian advection (+ optional diffusion)
    """
    state.xp, state.zp = advect_particles_rk2(state.xp, state.zp, t, cfg.dt, cfg)
    state.c = advect_dye_semilag(state.c, xs, zs, t, cfg.dt, cfg)
    return state


def step_out_of_plane_and_visibility(state: State, cfg: SimConfig):
    """
    Update out-of-plane coordinate and compute currently visible particles.
    Respawn is kept in the same location/order as the original logic to
    avoid changing random-number consumption and behavior.
    """
    if cfg.enable_out_of_plane:
        state.y = update_out_of_plane(state.y, cfg.dt, cfg)
        kill = np.abs(state.y) > cfg.y_kill
        state = respawn(kill, state, cfg)
        vis = visible_mask(state.y, cfg)
    else:
        vis = np.ones(cfg.N, dtype=bool)
    return state, vis


def render_total_intensity(state: State, vis, xs, zs, cfg: SimConfig):
    """
    Render both channels (particles + dye) and combine to sensor irradiance.
    """
    I_p = render_particles(state.xp[vis], state.zp[vis], cfg)
    I_d = render_dye(state.c, xs, zs, cfg)
    return I_p + I_d


def encode_frame(I, cfg: SimConfig):
    """
    Convert intensity image to uint8 using the same rule as before:
      - camera model only when camera mode is enabled AND auto-exposure is off
      - otherwise percentile-based auto exposure
    """
    if cfg.use_camera_model and not cfg.auto_exposure:
        return camera_model(I, cfg)
    # auto_exposure makes it guaranteed "not black"
    return auto_exposure_to_uint8(I, cfg)


def append_diagnostics(diag, I, vis):
    """
    Append one frame's scalar diagnostics.
    """
    diag["I_min"].append(float(I.min()))
    diag["I_max"].append(float(I.max()))
    diag["I_mean"].append(float(I.mean()))
    diag["visible_frac"].append(float(np.mean(vis)))


# =========================================================
# Forward simulator A: (x0, theta) -> video
# =========================================================

def forward_simulator(cfg: SimConfig):
    """
    Implements the forward operator A.
    Returns:
      video_u8: (T,H,W) uint8
      final_state: State
      diagnostics: dict
    """
    state, xs, zs = init_state(cfg)
    video = np.zeros((cfg.T, cfg.H, cfg.W), dtype=np.uint8)  # output video buffer
    t = 0.0
    diag = init_diagnostics()

    for n in range(cfg.T):
        # 1) Flow/state dynamics
        state = step_flow_dynamics(state, xs, zs, t, cfg)

        # 2) Out-of-plane process and visibility gate
        state, vis = step_out_of_plane_and_visibility(state, cfg)

        # 3) Rendering from current state
        I = render_total_intensity(state, vis, xs, zs, cfg)

        # 4) Sensor/exposure encoding to uint8 frame
        frame = encode_frame(I, cfg)

        video[n] = frame

        # 5) Diagnostics accumulation
        append_diagnostics(diag, I, vis)

        t += cfg.dt

    return video, state, diag


# =========================================================
# Export
# =========================================================

def export_video(video_u8, out_dir: Path, fps=20, save_gif=True, save_mp4=True, base="sim"):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Always save a few frames as PNG for quick sanity check
    try:
        import imageio.v2 as imageio
        imageio.imwrite(str(out_dir / f"{base}_frame0.png"), video_u8[0])
        imageio.imwrite(str(out_dir / f"{base}_frame_mid.png"), video_u8[len(video_u8)//2])
        imageio.imwrite(str(out_dir / f"{base}_frame_last.png"), video_u8[-1])
    except Exception:
        pass

    if save_gif:
        import imageio.v2 as imageio
        gif_path = out_dir / f"{base}.gif"
        imageio.mimsave(str(gif_path), video_u8, duration=1.0 / fps)

    if save_mp4:
        try:
            import imageio.v2 as imageio
            rgb = np.repeat(video_u8[..., None], 3, axis=-1)
            mp4_path = out_dir / f"{base}.mp4"
            imageio.mimsave(str(mp4_path), rgb, fps=fps)
        except Exception as e:
            print("[WARN] MP4 export failed (install imageio-ffmpeg). Error:", e)


# =========================================================
# CLI internal modules
# =========================================================

def parse_cli_args():
    """
    Build and parse CLI arguments.
    Isolating parser creation makes `main` focused on execution flow.
    """
    p = argparse.ArgumentParser(description="Forward simulator A(x0,theta)->video")
    p.add_argument("--out", type=str, default="", help="output directory (default: script_dir/outputs)")
    p.add_argument("--base", type=str, default="sim", help="base filename")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--gif", action="store_true")
    p.add_argument("--mp4", action="store_true")
    p.add_argument("--no-auto-exposure", action="store_true", help="disable auto exposure")
    return p.parse_args()


def resolve_output_options(args):
    """
    Resolve output folder and format flags.
    Behavior intentionally matches original implementation exactly:
      - if neither --gif nor --mp4 is set, save both
      - otherwise save only the explicitly requested formats
    """
    script_dir = Path(__file__).resolve().parent
    out_dir = Path(args.out).expanduser().resolve() if args.out else (script_dir / "outputs")
    save_gif = True if (not args.gif and not args.mp4) else args.gif
    save_mp4 = True if (not args.gif and not args.mp4) else args.mp4
    return out_dir, save_gif, save_mp4


# =========================================================
# CLI
# =========================================================

def main():
    args = parse_cli_args()

    cfg = SimConfig()
    cfg.auto_exposure = not args.no_auto_exposure

    video, state, diag = forward_simulator(cfg)

    print("video shape:", video.shape, "dtype:", video.dtype)
    print("I stats (min/max/mean):",
          f"{min(diag['I_min']):.4g}/{max(diag['I_max']):.4g}/{np.mean(diag['I_mean']):.4g}")
    print("visible fraction (avg):", np.mean(diag["visible_frac"]))

    out_dir, save_gif, save_mp4 = resolve_output_options(args)

    export_video(video, out_dir=out_dir, fps=args.fps, save_gif=save_gif, save_mp4=save_mp4, base=args.base)
    print("Saved outputs to:", out_dir)


if __name__ == "__main__":
    main()
