# MDM2_Group_10
Repository for MDM Group 10 Project 2

---
# Forward Flow-Dye Simulator (`simulate_forward.py`) — Detailed README

## 1. What This Script Does

`simulate_forward.py` implements a **forward simulator** that generates a synthetic grayscale video from a simplified flow field model and a dye field transport model.

In short, for each time step, it:

1. Advances particle positions in a 2D domain.
2. Advects a continuous dye concentration field.
3. Optionally simulates out-of-plane particle visibility loss/respawn.
4. Renders particle intensity + dye intensity to an image.
5. Converts the intensity image to `uint8` frame values.
6. Stores diagnostics and exports outputs (PNG/GIF/MP4).

The script is self-contained and intended for simulation/prototyping workflows where you need synthetic flow videos.

---

## 2. High-Level Mathematical Model

The simulator uses a velocity field `(u, w)` in `(x, z)` coordinates:

- `x`: horizontal coordinate (periodic boundary)
- `z`: vertical coordinate (clamped between `zmin` and `zmax`)

Velocity is defined from a decaying oscillatory mode:

- `u = Re(i k A exp(-k|z|) exp(i k x) exp(gamma t))`
- `w = Re(-k A sign(z) exp(-k|z|) exp(i k x) exp(gamma t))`

where:

- `A` controls amplitude
- `k` is wave number
- `gamma` is temporal growth/decay

Particles are advanced with **RK2 (midpoint)** integration.
The dye field is advanced with **semi-Lagrangian backtracing** + optional diffusion.

---

## 3. File Structure (Inside `simulate_forward.py`)

The code is organized into these sections:

1. **Config / State**
   - `SimConfig`: all simulation/render/camera parameters.
   - `State`: particle coordinates and dye field at current time.

2. **Velocity field**
   - `vel_u_w(...)`: computes flow velocity.

3. **Dynamics**
   - `advect_particles_rk2(...)`
   - `bilinear_sample(...)`
   - `advect_dye_semilag(...)`
   - out-of-plane helpers: `update_out_of_plane`, `visible_mask`, `respawn`

4. **Rendering**
   - `gaussian_blur_fft(...)`
   - `render_particles(...)`
   - `render_dye(...)`

5. **Camera / Exposure**
   - `camera_model(...)`
   - `auto_exposure_to_uint8(...)`

6. **Initialization**
   - `init_state(...)`

7. **Forward simulator orchestration**
   - internal modular steps:
     - `init_diagnostics(...)`
     - `step_flow_dynamics(...)`
     - `step_out_of_plane_and_visibility(...)`
     - `render_total_intensity(...)`
     - `encode_frame(...)`
     - `append_diagnostics(...)`
   - main loop: `forward_simulator(...)`

8. **Export and CLI**
   - `export_video(...)`
   - `parse_cli_args(...)`
   - `resolve_output_options(...)`
   - `main(...)`

---

## 4. Core Data Structures

### 4.1 `SimConfig`

`SimConfig` is the central configuration dataclass. Important groups:

- **Temporal and resolution**
  - `T`: number of frames
  - `dt`: time step
  - `H`, `W`: frame size
  - `N`: number of particles

- **Domain**
  - `Lx`: domain width in `x`
  - `zmin`, `zmax`: domain bounds in `z`

- **Flow parameters**
  - `A`, `k`, `gamma`

- **Noise and diffusion**
  - `particle_noise_sigma`
  - `dye_kappa`

- **Out-of-plane model**
  - `enable_out_of_plane`
  - `sheet_thickness`
  - `y_noise_sigma`
  - `y_kill`

- **Rendering parameters**
  - particle PSF and amplitude: `psf_sigma_px`, `particle_amp`
  - dye lighting/attenuation: `dye_beta`, `dye_alpha`, `light_source_*`, `dye_blur_sigma_px`

- **Camera and exposure**
  - `use_camera_model`, `bg`, `gain`, `read_sigma`
  - `auto_exposure`, `exposure_percentile`

- **Randomness**
  - `seed`

### 4.2 `State`

`State` stores time-varying simulation variables:

- `xp`: particle `x` positions, shape `(N,)`
- `zp`: particle `z` positions, shape `(N,)`
- `y`: out-of-plane coordinate, shape `(N,)`
- `c`: dye concentration field, shape `(H, W)`

---

## 5. Time-Step Pipeline in Detail

At each frame index `n`:

1. **Flow dynamics**
   - Particles: RK2 advection in velocity field.
   - Dye: semi-Lagrangian backtrace to sample previous field.
   - Optional diffusion (discrete Laplacian + clamping to nonnegative).

2. **Out-of-plane process** (if enabled)
   - `y` executes stochastic motion (`Brownian-like` increment).
   - Particles with `|y| > y_kill` are respawned in-domain.
   - Visibility mask is determined by `|y| <= sheet_thickness/2`.

3. **Rendering**
   - Particle image: splat particle impulses to nearest pixel + FFT Gaussian blur.
   - Dye image: geometric attenuation from a light source and exponential term.
   - Total intensity: `I = I_p + I_d`.

4. **Frame encoding**
   - If `use_camera_model` and auto-exposure is disabled:
     - apply background + Poisson shot noise + Gaussian read noise.
   - Else:
     - percentile auto-exposure to map intensity into `[0, 255]`.

5. **Diagnostics**
   - Record per-frame `I_min`, `I_max`, `I_mean`, and visible fraction.

---

## 6. Boundary and Numerical Behavior

### 6.1 Particle boundaries

- `x` is periodic: `x = mod(x, Lx)`.
- `z` is clipped: `z in [zmin, zmax]`.

### 6.2 Dye boundaries

- Semi-Lagrangian sampling uses periodicity in `x` and clamping in `z`.
- Diffusion term uses periodic neighbors in `x` and edge padding in `z`.

### 6.3 Interpolation

`bilinear_sample` computes four-corner interpolation weights on grid indices and supports periodic/non-periodic `x`.

---

## 7. Initialization Details

`init_state(cfg)` does the following:

- Seeds NumPy RNG with `cfg.seed`.
- Initializes particles:
  - `xp ~ Uniform(0, Lx)`
  - `zp ~ Normal(0, 0.12*(zmax-zmin))`, clipped to bounds
- Initializes `y`:
  - zeros if out-of-plane disabled
  - uniform inside a central slab if enabled
- Builds simulation grids:
  - `xs`: evenly spaced, periodic-like (`endpoint=False`)
  - `zs`: inclusive linear spacing in `z`
- Initializes dye concentration with a Gaussian blob example.

---

## 8. Rendering Model Explained

### 8.1 Particle rendering

- Convert physical positions to pixel coordinates.
- Nearest-pixel accumulation using `np.add.at`.
- Apply Gaussian PSF blur (`gaussian_blur_fft`).

This is computationally efficient and gives smooth bright spots from sparse particles.

### 8.2 Dye rendering

For each grid point:

- Compute distance `d` to a virtual light source.
- Compute attenuation-like factor `L = 1 / d^2`.
- Intensity term: `I = dye_beta * c * L * exp(-dye_alpha * d * c)`.
- Blur for smooth appearance.

---

## 9. Camera and Exposure

There are two frame-conversion paths:

1. **Physical-ish camera path (`camera_model`)**
   - Adds constant background.
   - Converts to Poisson parameter via `gain` (shot noise).
   - Adds Gaussian read noise.
   - Clips to `[0,255]`, casts to `uint8`.

2. **Auto-exposure path (`auto_exposure_to_uint8`)**
   - Computes high percentile (`exposure_percentile`).
   - Scales so percentile maps near 255.
   - Useful to avoid very dark outputs.

When `auto_exposure=True`, the script uses auto-exposure regardless of camera flag.

---

## 10. Outputs

### 10.1 In-memory return values

`forward_simulator(cfg)` returns:

- `video_u8`: NumPy array `(T, H, W)`, `uint8`
- `final_state`: final `State`
- `diagnostics`: dictionary with lists

### 10.2 Files written by CLI execution

By default, output directory is:

- `<script_dir>/outputs`

Files include:

- `sim_frame0.png`
- `sim_frame_mid.png`
- `sim_frame_last.png`
- `sim.gif` (default unless format flags override)
- `sim.mp4` (default unless format flags override)

If MP4 writing fails, warning suggests installing `imageio-ffmpeg`.

---

## 11. Command-Line Usage

Basic run (auto-exposure enabled by default):

```bash
python simulate_forward.py
```

Specify output directory and basename:

```bash
python simulate_forward.py --out ./results --base caseA
```

Export only GIF:

```bash
python simulate_forward.py --gif
```

Export only MP4:

```bash
python simulate_forward.py --mp4
```

Disable auto exposure (use camera model if enabled):

```bash
python simulate_forward.py --no-auto-exposure
```

Set FPS for GIF/MP4:

```bash
python simulate_forward.py --fps 30
```

Format behavior rule:

- If neither `--gif` nor `--mp4` is provided, both are saved.
- If either flag is provided, only the selected formats are saved.

---

## 12. Reproducibility Notes

- Initialization uses `np.random.seed(cfg.seed)`.
- Randomness during simulation (particle noise, out-of-plane noise, camera noise) depends on global NumPy RNG state after initialization.
- To compare outputs exactly, keep:
  - same NumPy version
  - same config values
  - same execution order

---

## 13. Performance Considerations

Main cost centers:

1. `render_particles` with full-frame blur.
2. `render_dye` with full-field evaluation + blur.
3. Per-frame semi-Lagrangian interpolation.

Ways to speed up:

- Reduce `H`, `W`, `T`, or `N`.
- Lower blur sigmas if visually acceptable.
- Profile hot paths before introducing optimization complexity.

---

## 14. Safe Extension Points

If you want to extend behavior while preserving overall architecture:

1. Replace velocity model in `vel_u_w`.
2. Replace dye initial condition in `init_state`.
3. Add external forcing/source terms in `step_flow_dynamics`.
4. Modify rendering kernels in `render_particles`/`render_dye`.
5. Add more diagnostics via `append_diagnostics`.

Because the simulator is modular, these can be changed with minimal impact on CLI/export.

---

## 15. Troubleshooting

### Problem: Output looks too dark

- Keep auto-exposure on (default).
- Increase `particle_amp` or `dye_beta`.
- Reduce `dye_alpha` if dye decays visually too fast with distance.

### Problem: Motion appears weak

- Increase `A` or adjust `k`.
- Increase `T` for longer sequence.

### Problem: Too many particles disappear

- Increase `y_kill`.
- Reduce `y_noise_sigma`.
- Increase `sheet_thickness` for visibility threshold.

### Problem: MP4 not written

- Install `imageio-ffmpeg`.
- GIF and PNG snapshots should still be available.

---

## 16. Quick API Example (Programmatic Use)

```python
from simulate_forward import SimConfig, forward_simulator, export_video
from pathlib import Path

cfg = SimConfig(T=120, H=256, W=256, N=1500, seed=42)
video, state, diag = forward_simulator(cfg)

export_video(
    video,
    out_dir=Path("./outputs_custom"),
    fps=24,
    save_gif=True,
    save_mp4=False,
    base="experiment_01"
)
```

---

## 17. Summary

This simulator provides a compact but expressive synthetic video pipeline combining:

- analytic flow advection,
- particle and field representations,
- physically inspired intensity rendering,
- optional camera/noise model,
- export utilities for quick visual inspection.

It is suitable for data generation, inverse-problem prototyping, and algorithm testing where fully real fluid simulation is unnecessary.

