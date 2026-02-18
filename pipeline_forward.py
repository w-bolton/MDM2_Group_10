"""
Forward pipeline implementation 
--------------------------------------------
This script implements Goal 2 as an explicit, commented pipeline:

(1) Illumination  ->  (2) Interaction  ->  (3) Attenuation  ->  (4) Imaging  ->  (5) Sensor

It uses the same key variable names used in simulate_forward.py:
xp, zp, y, c, xs, zs, I_p, I_dye, I (and related parameters).

Output is written to: <script_dir>/outputs/  

- Hongze Lin

"""

import os
import numpy as np


# ==========================================================
# (A) Placeholder flow model: velocity field u(x,z,t), w(x,z,t)  (eqs. 3,4)
# ==========================================================
def vel_u_w(x, z, t, A=0.02, k=2*np.pi/1.0, gamma=0.0):
    """
    Implements the given placeholder velocity field:
      u(x,z,t), w(x,z,t) using a compact form with exp(-k*abs(z)).

    NOTE:
      - This is a forward (simulation) component, not an inference step.
      - x, z can be scalars or numpy arrays.
    """
    decay = np.exp(-k * np.abs(z))
    phase = np.exp(1j * k * x)
    growth = np.exp(gamma * t)

    u = np.real(1j * k * A * decay * phase * growth)
    w = np.real(-k * A * np.sign(z) * decay * phase * growth)
    return u, w


# ==========================================================
# (B) Dynamics: particle advection (RK2) and dye advection (semi-Lagrangian)
# ==========================================================
def advect_particles_rk2(xp, zp, t, dt, Lx, zmin, zmax,
                         A=0.02, k=2*np.pi/1.0, gamma=0.0,
                         noise_sigma=0.0):
    """
    Particle evolution in the light-sheet plane:
      (xp, zp)_{n+1} = (xp, zp)_n + v(xp,zp,t) * dt

    Uses RK2 (midpoint method) for stability.

    Boundaries:
      - xp is periodic (mod Lx)
      - zp is clipped to [zmin, zmax]

    noise_sigma:
      - optional process noise to mimic unresolved small-scale motion / model mismatch
    """
    # RK2 stage 1
    u1, w1 = vel_u_w(xp, zp, t, A=A, k=k, gamma=gamma)
    xm = xp + 0.5 * dt * u1
    zm = zp + 0.5 * dt * w1

    # RK2 stage 2
    u2, w2 = vel_u_w(xm, zm, t + 0.5*dt, A=A, k=k, gamma=gamma)
    xp_new = xp + dt * u2
    zp_new = zp + dt * w2

    # Optional model error / turbulence-like perturbation (simple)
    if noise_sigma > 0:
        xp_new += np.random.normal(0.0, noise_sigma, size=xp_new.shape)
        zp_new += np.random.normal(0.0, noise_sigma, size=zp_new.shape)

    # Enforce boundaries
    xp_new = np.mod(xp_new, Lx)
    zp_new = np.clip(zp_new, zmin, zmax)
    return xp_new, zp_new


def bilinear_sample(field, xq, zq, xs, zs, periodic_x=True):
    """
    Bilinear sampling of a 2D field defined on (zs, xs) grid.

    field: shape (H, W)
    xq, zq: query coordinates (same shape)
    xs: 1D x-grid of length W
    zs: 1D z-grid of length H
    """
    H, W = field.shape
    dx = xs[1] - xs[0]
    dz = zs[1] - zs[0]

    ix = (xq - xs[0]) / dx
    iz = (zq - zs[0]) / dz

    if periodic_x:
        ix = np.mod(ix, W)
    else:
        ix = np.clip(ix, 0, W - 1 - 1e-6)
    iz = np.clip(iz, 0, H - 1 - 1e-6)

    i0 = np.floor(ix).astype(int)
    j0 = np.floor(iz).astype(int)
    i1 = (i0 + 1) % W if periodic_x else np.minimum(i0 + 1, W - 1)
    j1 = np.minimum(j0 + 1, H - 1)

    tx = ix - i0
    tz = iz - j0

    f00 = field[j0, i0]
    f10 = field[j0, i1]
    f01 = field[j1, i0]
    f11 = field[j1, i1]

    return (1-tx)*(1-tz)*f00 + tx*(1-tz)*f10 + (1-tx)*tz*f01 + tx*tz*f11


def advect_dye_semilag(c, xs, zs, t, dt, A=0.02, k=2*np.pi/1.0, gamma=0.0, kappa=0.0):
    """
    Dye evolution on the light-sheet plane using semi-Lagrangian advection:
      c_{n+1}(x,z) = c_n(x - u*dt, z - w*dt)

    Optional diffusion:
      c_{n+1} += dt * kappa * Laplacian(c)

    This is the simplest stable advection method for a project MVP.
    """
    X, Z = np.meshgrid(xs, zs)  # shapes (H,W)

    u, w = vel_u_w(X, Z, t, A=A, k=k, gamma=gamma)
    Xb = X - u * dt
    Zb = Z - w * dt

    c_new = bilinear_sample(c, Xb, Zb, xs, zs, periodic_x=True)

    if kappa > 0:
        # simple Laplacian diffusion (periodic in x, "edge" in z)
        c_pad = np.pad(c_new, ((1, 1), (0, 0)), mode="edge")
        c_up = c_pad[0:-2, :]
        c_dn = c_pad[2:, :]
        c_lt = np.roll(c_new,  1, axis=1)
        c_rt = np.roll(c_new, -1, axis=1)

        dx = xs[1] - xs[0]
        dz = zs[1] - zs[0]
        lap = (c_lt - 2*c_new + c_rt)/dx**2 + (c_up - 2*c_new + c_dn)/dz**2
        c_new = c_new + dt * kappa * lap
        c_new = np.clip(c_new, 0.0, None)

    return c_new


# ==========================================================
# (C) Light-sheet "appear/disappear": out-of-plane y + visibility mask
# ==========================================================
def update_out_of_plane(y, dt, y_noise_sigma=0.005):
    """
    Simple out-of-plane motion model: random walk in y.
    This creates frame-to-frame appear/disappear in a thin light sheet.
    """
    y = y + np.random.normal(0.0, y_noise_sigma*np.sqrt(dt), size=y.shape)
    return y


def update_out_of_plane_y(y, dt, y_noise_sigma=0.005):
    """
    Backward-compatible alias.
    """
    return update_out_of_plane(y, dt, y_noise_sigma=y_noise_sigma)


def visible_mask(y, sheet_thickness=0.02):
    """Particle is visible if it lies within the light sheet thickness around y=0."""
    return np.abs(y) <= 0.5 * sheet_thickness


def visible_mask_from_sheet(y, sheet_thickness=0.02):
    """
    Backward-compatible alias.
    """
    return visible_mask(y, sheet_thickness=sheet_thickness)


def respawn(mask_respawn, xp, zp, y, Lx, zmin, zmax, sheet_thickness=0.02):
    """
    Birth/death:
      Particles that drift too far out of plane are respawned at a new random location
      (and placed near the sheet so they re-appear).
    """
    idx = np.where(mask_respawn)[0]
    if idx.size == 0:
        return xp, zp, y

    xp[idx] = np.random.uniform(0, Lx, size=idx.size)
    zp_new = np.random.normal(0.0, 0.12*(zmax - zmin), size=idx.size)
    zp[idx] = np.clip(zp_new, zmin, zmax)
    y[idx] = np.random.uniform(-0.25*sheet_thickness, 0.25*sheet_thickness, size=idx.size)
    return xp, zp, y


def respawn_particles(mask_respawn, xp, zp, y, Lx, zmin, zmax, sheet_thickness=0.02):
    """
    Backward-compatible alias.
    """
    return respawn(mask_respawn, xp, zp, y, Lx, zmin, zmax, sheet_thickness=sheet_thickness)


# ==========================================================
# (D) Imaging helpers: Gaussian PSF blur (FFT) and splatting particles
# ==========================================================
def gaussian_blur_fft(img, sigma):
    """Gaussian blur using FFT (fast, no scipy required). sigma in pixels."""
    if sigma <= 0:
        return img.astype(np.float32)

    H, W = img.shape
    ky = np.fft.fftfreq(H) * 2*np.pi
    kx = np.fft.fftfreq(W) * 2*np.pi
    KX, KY = np.meshgrid(kx, ky)
    G = np.exp(-0.5 * sigma**2 * (KX**2 + KY**2))

    return np.real(np.fft.ifft2(np.fft.fft2(img) * G)).astype(np.float32)


def render_particles(xp, zp, amps, H, W, Lx, zmin, zmax, psf_sigma_px=1.6):
    """
    Particle imaging:
      - splat each particle to nearest pixel with its amplitude
      - apply PSF blur (Gaussian) to approximate camera point spread function
    """
    img = np.zeros((H, W), dtype=np.float32)

    # physical -> pixel mapping
    px = (xp / Lx) * (W - 1)
    pz = ((zp - zmin) / (zmax - zmin)) * (H - 1)

    ix = np.rint(px).astype(int)
    iz = np.rint(pz).astype(int)

    m = (ix >= 0) & (ix < W) & (iz >= 0) & (iz < H)
    ix = ix[m]
    iz = iz[m]
    amps = amps[m].astype(np.float32)

    np.add.at(img, (iz, ix), amps)

    # PSF blur
    return gaussian_blur_fft(img, psf_sigma_px)


# ==========================================================
# (E) Goal 2 Pipeline modules:
#     (1) Illumination -> (2) Interaction -> (3) Attenuation -> (4) Imaging -> (5) Sensor
# ==========================================================
def illumination_field(xs, zs, Lx, zmin, zmax,
                       mode="point",
                       light_source_x_frac=0.5,
                       light_source_z_above_frac=1.2,
                       beam_sigma=0.18,
                       beam_depth_decay=1.5,
                       eps=1e-6,
                       light_x_frac=None,
                       light_z_above_frac=None):
    """
    (1) Illumination:
      Build an illumination intensity field L(x,z) across the light-sheet plane.

    Two simple options:
      - mode="point": point source above the domain, L ~ 1/(d^2+eps)
      - mode="gaussian_beam": beam profile in x with depth attenuation in z

    Returns:
      L_grid: shape (H,W)
      illum_at(xp,zp): function returning illumination at particle positions
    """
    X, Z = np.meshgrid(xs, zs)

    if light_x_frac is not None:
        light_source_x_frac = light_x_frac
    if light_z_above_frac is not None:
        light_source_z_above_frac = light_z_above_frac

    if mode == "point":
        x0 = light_source_x_frac * Lx
        z0 = zmax + light_source_z_above_frac * (zmax - zmin)
        d = np.sqrt((X - x0)**2 + (Z - z0)**2) + eps
        L_grid = 1.0 / (d**2 + eps)

        def illum_at(xp, zp):
            d_p = np.sqrt((xp - x0)**2 + (zp - z0)**2) + eps
            return 1.0 / (d_p**2 + eps)

        return L_grid.astype(np.float32), illum_at

    if mode == "gaussian_beam":
        x0 = light_source_x_frac * Lx
        depth = np.clip(zmax - Z, 0.0, None)  # depth from top
        L_grid = np.exp(-0.5 * ((X - x0) / beam_sigma)**2) * np.exp(-beam_depth_decay * depth)

        def illum_at(xp, zp):
            depth_p = np.clip(zmax - zp, 0.0, None)
            return np.exp(-0.5 * ((xp - x0) / beam_sigma)**2) * np.exp(-beam_depth_decay * depth_p)

        return L_grid.astype(np.float32), illum_at

    raise ValueError(f"Unknown illumination mode: {mode}")


def dye_emission(c, L_grid, dye_beta=8.0):
    """
    (2) Interaction (dye):
      Fluorescent emission before attenuation.

    Simplest model:
      emission ∝ concentration * illumination
      E_dye = dye_beta * c * L_grid
    """
    return (dye_beta * c * L_grid).astype(np.float32)


def beer_lambert_attenuation_path_integral(c, zs, dye_alpha=0.6):
    """
    (3) Attenuation (Beer–Lambert, path integral form):
      attenuation = exp(-alpha * ∫ c ds)

    Assumption for MVP:
      Light travels from top (zmax) downwards (-z direction).
      So optical depth at a pixel depends on cumulative concentration above it.

    Implementation:
      - zs is increasing from zmin -> zmax (bottom -> top)
      - integrate from top row downward by reversing and cumsum
    """
    dz = zs[1] - zs[0]
    tau_rev = dye_alpha * np.cumsum(c[::-1, :], axis=0) * dz  # integrate from top down
    tau = tau_rev[::-1, :]
    return np.exp(-tau).astype(np.float32)


def auto_exposure_to_uint8(I, p_high=99.7):
    """
    Utility: map intensity to 8-bit using percentile scaling.
    This prevents "all black" outputs while still preserving relative contrast.
    """
    hi = np.percentile(I, p_high)
    hi = max(float(hi), 1e-6)
    J = np.clip(I / hi * 255.0, 0.0, 255.0)
    return J.astype(np.uint8)


def auto_exposure_uint8(I, p_high=99.7):
    """
    Backward-compatible alias.
    """
    return auto_exposure_to_uint8(I, p_high=p_high)


def camera_model(I, bg=10.0, gain=120.0, read_sigma=1.5, clip_max=255):
    """
    (5) Sensor:
      Convert ideal intensity I(x,z) to a camera-like uint8 image.

    Includes:
      - background offset
      - Poisson shot noise
      - Gaussian read noise
      - clipping to [0,255]
    """
    I = np.clip(I + bg, 0.0, None)
    lam = np.clip(I * gain, 0.0, None)
    noisy = np.random.poisson(lam).astype(np.float32) / gain
    noisy += np.random.normal(0.0, read_sigma, size=noisy.shape).astype(np.float32)
    return np.clip(noisy, 0.0, clip_max).astype(np.uint8)


# ==========================================================
# (F) Export helpers (safe path)
# ==========================================================
def export_outputs(video, fps=20, base="sim"):
    """
    Saves:
      Outputs_pipeline/<base>.gif
      Outputs_pipeline/<base>.mp4 (if imageio-ffmpeg available)
      Outputs_pipeline/<base>_frame0.png, _mid.png, _last.png
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, "Outputs_pipeline")  # <-- changed here
    os.makedirs(out_dir, exist_ok=True)

    try:
        import imageio.v2 as imageio
        imageio.imwrite(os.path.join(out_dir, f"{base}_frame0.png"), video[0])
        imageio.imwrite(os.path.join(out_dir, f"{base}_frame_mid.png"), video[len(video)//2])
        imageio.imwrite(os.path.join(out_dir, f"{base}_frame_last.png"), video[-1])

        imageio.mimsave(os.path.join(out_dir, f"{base}.gif"), video, duration=1.0/fps)
    except Exception as e:
        print("[WARN] PNG/GIF export failed:", e)

    # MP4 often requires imageio-ffmpeg
    try:
        import imageio.v2 as imageio
        rgb = np.repeat(video[..., None], 3, axis=-1)
        imageio.mimsave(os.path.join(out_dir, f"{base}.mp4"), rgb, fps=fps)
    except Exception as e:
        print("[WARN] MP4 export failed (install imageio-ffmpeg):", e)

    print("Saved outputs to:", out_dir)



# ==========================================================
# (G) Forward simulator internal modules
# ==========================================================
def init_state(
    N, H, W, Lx, zmin, zmax, enable_out_of_plane, sheet_thickness, seed
):
    np.random.seed(seed)

    xs = np.linspace(0, Lx, W, endpoint=False)
    zs = np.linspace(zmin, zmax, H)

    xp = np.random.uniform(0, Lx, size=N).astype(np.float32)
    zp = np.random.normal(0.0, 0.12 * (zmax - zmin), size=N).astype(np.float32)
    zp = np.clip(zp, zmin, zmax)

    y = np.zeros(N, dtype=np.float32)
    if enable_out_of_plane:
        y = np.random.uniform(-0.25 * sheet_thickness, 0.25 * sheet_thickness, size=N).astype(np.float32)

    X, Z = np.meshgrid(xs, zs)
    c = np.exp(-((X - 0.5 * Lx) ** 2 / (0.08 ** 2) + (Z - 0.15) ** 2 / (0.10 ** 2))).astype(np.float32)

    state = {"xp": xp, "zp": zp, "y": y, "c": c}
    return state, xs, zs


def prepare_illumination(
    xs, zs, Lx, zmin, zmax, illum_mode,
    light_source_x_frac, light_source_z_above_frac,
    beam_sigma, beam_depth_decay
):
    L_grid, illum_at = illumination_field(
        xs, zs, Lx, zmin, zmax,
        mode=illum_mode,
        light_source_x_frac=light_source_x_frac,
        light_source_z_above_frac=light_source_z_above_frac,
        beam_sigma=beam_sigma,
        beam_depth_decay=beam_depth_decay
    )
    L_grid_max = float(np.max(L_grid)) + 1e-6
    return L_grid, illum_at, L_grid_max


def init_diagnostics():
    diag = {"I_min": [], "I_max": [], "I_mean": [], "visible_frac": []}
    # Backward-compatible aliases for previous key names.
    diag["Imin"] = diag["I_min"]
    diag["Imax"] = diag["I_max"]
    diag["Imean"] = diag["I_mean"]
    diag["vis_frac"] = diag["visible_frac"]
    return diag


def step_flow_dynamics(state, xs, zs, t, dt, Lx, zmin, zmax, A, k, gamma, particle_noise_sigma, dye_kappa):
    state["xp"], state["zp"] = advect_particles_rk2(
        state["xp"], state["zp"], t, dt, Lx, zmin, zmax,
        A=A, k=k, gamma=gamma,
        noise_sigma=particle_noise_sigma
    )
    state["c"] = advect_dye_semilag(
        state["c"], xs, zs, t, dt,
        A=A, k=k, gamma=gamma,
        kappa=dye_kappa
    )
    return state


def step_out_of_plane_and_visibility(
    state, dt, enable_out_of_plane, y_noise_sigma, y_kill, sheet_thickness, N, Lx, zmin, zmax
):
    if enable_out_of_plane:
        state["y"] = update_out_of_plane(state["y"], dt, y_noise_sigma=y_noise_sigma)
        kill = np.abs(state["y"]) > y_kill
        state["xp"], state["zp"], state["y"] = respawn(
            kill, state["xp"], state["zp"], state["y"], Lx, zmin, zmax, sheet_thickness=sheet_thickness
        )
        vis = visible_mask(state["y"], sheet_thickness=sheet_thickness)
    else:
        vis = np.ones(N, dtype=bool)
    return state, vis


def compute_interaction_and_attenuation(
    state, illum_at, L_grid, L_grid_max, particle_amp, particle_illum_power, dye_beta, zs, dye_alpha
):
    Lp = illum_at(state["xp"], state["zp"])
    Lp_norm = (Lp / L_grid_max)
    amps = (particle_amp * (Lp_norm ** particle_illum_power)).astype(np.float32)

    E_dye = dye_emission(state["c"], L_grid, dye_beta=dye_beta)
    atten = beer_lambert_attenuation_path_integral(state["c"], zs, dye_alpha=dye_alpha)
    I_dye_pre = (E_dye * atten).astype(np.float32)
    return amps, I_dye_pre


def render_total_intensity(state, vis, amps, I_dye_pre, H, W, Lx, zmin, zmax, psf_sigma_px, dye_blur_sigma_px):
    I_p = render_particles(
        state["xp"][vis], state["zp"][vis], amps[vis],
        H, W, Lx, zmin, zmax,
        psf_sigma_px=psf_sigma_px
    )
    I_dye = gaussian_blur_fft(I_dye_pre, dye_blur_sigma_px)
    return (I_p + I_dye).astype(np.float32)


def encode_frame(I, auto_exposure, exposure_percentile, use_camera_model, bg, gain, read_sigma):
    # Keep camera/exposure behavior aligned with simulate_forward.py.
    if use_camera_model and not auto_exposure:
        return camera_model(I, bg=bg, gain=gain, read_sigma=read_sigma, clip_max=255)
    return auto_exposure_to_uint8(I, p_high=exposure_percentile)


def append_diagnostics(diag, I, vis):
    diag["I_min"].append(float(I.min()))
    diag["I_max"].append(float(I.max()))
    diag["I_mean"].append(float(I.mean()))
    diag["visible_frac"].append(float(np.mean(vis)))


# ==========================================================
# (H) The NEW forward simulator: explicit Goal 2 pipeline
# ==========================================================
def simulate_video_pipeline(
    T=80, dt=0.02, N=1000,
    H=384, W=384,
    Lx=1.0, zmin=-0.5, zmax=0.5,
    # velocity field params (eqs 3,4)
    A=0.02, k=2*np.pi/1.0, gamma=0.0,
    # particle + dye dynamics
    particle_noise_sigma=5e-4, dye_kappa=0.0,
    # out-of-plane
    sheet_thickness=0.02, y_noise_sigma=0.005, y_kill=0.06, enable_out_of_plane=True,
    # pipeline params
    illum_mode="point",
    light_source_x_frac=0.5, light_source_z_above_frac=1.2,
    beam_sigma=0.18, beam_depth_decay=1.5,
    particle_amp=2.0, particle_illum_power=1.0, psf_sigma_px=1.6,
    dye_beta=8.0, dye_alpha=0.6, dye_blur_sigma_px=0.7,
    # sensor / display
    use_camera_model=True, bg=10.0, gain=120.0, read_sigma=1.5,
    auto_exposure=True, exposure_percentile=99.7,
    seed=1,
    # backward-compatible aliases (prefer canonical names above)
    particle_noise=None,
    light_x_frac=None,
    light_z_above_frac=None,
    use_camera_noise=None,
    particle_base_amp=None
):
    """
    This is the forward operator A(x0,theta) -> b0:T, with an explicit Goal 2 pipeline.

    Returns:
      video: (T,H,W) uint8
      final: dict holding final xp,zp,y,c
      diag: basic diagnostics
    """
    if particle_noise is not None:
        particle_noise_sigma = particle_noise
    if light_x_frac is not None:
        light_source_x_frac = light_x_frac
    if light_z_above_frac is not None:
        light_source_z_above_frac = light_z_above_frac
    if use_camera_noise is not None:
        use_camera_model = use_camera_noise
    if particle_base_amp is not None:
        particle_amp = particle_base_amp

    state, xs, zs = init_state(N, H, W, Lx, zmin, zmax, enable_out_of_plane, sheet_thickness, seed)
    L_grid, illum_at, L_grid_max = prepare_illumination(
        xs, zs, Lx, zmin, zmax, illum_mode,
        light_source_x_frac, light_source_z_above_frac,
        beam_sigma, beam_depth_decay
    )

    video = np.zeros((T, H, W), dtype=np.uint8)
    diag = init_diagnostics()

    t = 0.0
    for n in range(T):
        state = step_flow_dynamics(
            state, xs, zs, t, dt, Lx, zmin, zmax, A, k, gamma, particle_noise_sigma, dye_kappa
        )
        state, vis = step_out_of_plane_and_visibility(
            state, dt, enable_out_of_plane, y_noise_sigma, y_kill, sheet_thickness, N, Lx, zmin, zmax
        )
        amps, I_dye_pre = compute_interaction_and_attenuation(
            state, illum_at, L_grid, L_grid_max, particle_amp, particle_illum_power, dye_beta, zs, dye_alpha
        )
        I = render_total_intensity(
            state, vis, amps, I_dye_pre, H, W, Lx, zmin, zmax, psf_sigma_px, dye_blur_sigma_px
        )
        frame = encode_frame(I, auto_exposure, exposure_percentile, use_camera_model, bg, gain, read_sigma)

        video[n] = frame
        append_diagnostics(diag, I, vis)
        t += dt

    final = {"xp": state["xp"], "zp": state["zp"], "y": state["y"], "c": state["c"]}
    return video, final, diag


# ==========================================================
# Run
# ==========================================================
if __name__ == "__main__":
    video, final, diag = simulate_video_pipeline(
        T=80, dt=0.02, N=1000,
        H=384, W=384,
        Lx=1.0, zmin=-0.5, zmax=0.5,
        A=0.02, k=2*np.pi/1.0, gamma=0.0,
        particle_noise_sigma=5e-4, dye_kappa=0.0,
        sheet_thickness=0.02, y_noise_sigma=0.005, y_kill=0.06, enable_out_of_plane=True,
        illum_mode="point",              # try also: "gaussian_beam"
        particle_amp=2.0,
        dye_beta=8.0, dye_alpha=0.6,
        use_camera_model=True,
        auto_exposure=True,
        seed=1
    )

    print("video shape:", video.shape, "dtype:", video.dtype)
    print("I_total min/max/mean (over time):",
          f"{min(diag['I_min']):.3g}/{max(diag['I_max']):.3g}/{np.mean(diag['I_mean']):.3g}")
    print("visible fraction (avg):", np.mean(diag["visible_frac"]))

    export_outputs(video, fps=20, base="sim_pipeline")
