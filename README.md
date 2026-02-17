# MDM2_Group_10
Repository for MDM Group 10 Project 2

---
## About Forward Flow-Dye Simulator (`simulate_forward.py`) ($Hongze$ $Lin$)
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



## 6. Boundary and Numerical Behavior

### 6.1 Particle boundaries

- `x` is periodic: `x = mod(x, Lx)`.
- `z` is clipped: `z in [zmin, zmax]`.

### 6.2 Dye boundaries

- Semi-Lagrangian sampling uses periodicity in `x` and clamping in `z`.
- Diffusion term uses periodic neighbors in `x` and edge padding in `z`.

### 6.3 Interpolation

`bilinear_sample` computes four-corner interpolation weights on grid indices and supports periodic/non-periodic `x`.



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


## 12. Reproducibility Notes

- Initialization uses `np.random.seed(cfg.seed)`.
- Randomness during simulation (particle noise, out-of-plane noise, camera noise) depends on global NumPy RNG state after initialization.
- To compare outputs exactly, keep:
  - same NumPy version
  - same config values
  - same execution order


## 13. Performance Considerations

Main cost centers:

1. `render_particles` with full-frame blur.
2. `render_dye` with full-field evaluation + blur.
3. Per-frame semi-Lagrangian interpolation.

Ways to speed up:

- Reduce `H`, `W`, `T`, or `N`.
- Lower blur sigmas if visually acceptable.
- Profile hot paths before introducing optimization complexity.


## 14. Extension Points

1. Replace velocity model in `vel_u_w`.
2. Replace dye initial condition in `init_state`.
3. Add external forcing/source terms in `step_flow_dynamics`.
4. Modify rendering kernels in `render_particles`/`render_dye`.
5. Add more diagnostics via `append_diagnostics`.

Because the simulator is modular, these can be changed with minimal impact on CLI/export.


## 15. API Example (Programmatic Use)

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
## About Forward Pipeline Simulator (`pipeline_forward.py`) ($Hongze$ $Lin$)

## 1. Overview

`pipeline_forward.py` implements a forward simulator with an explicit imaging pipeline for Goal 2:

1. Illumination
2. Interaction
3. Attenuation
4. Imaging
5. Sensor

At each frame, it updates particles and dye, applies the pipeline above, generates an 8-bit frame, stores diagnostics, and (optionally) exports files.

This script is designed as a transparent, modular baseline for synthetic data generation and inverse-problem prototyping.


## 2. Pipeline Logic (Conceptual)

The simulator combines flow dynamics with optical modeling:

- **Dynamics (state evolution)**
  - Particle advection in a prescribed velocity field.
  - Dye field advection (semi-Lagrangian) with optional diffusion.
  - Optional out-of-plane motion and respawn.

- **Optical/sensor pipeline (Goal 2)**
  - **Illumination**: create a light field over the domain.
  - **Interaction**: convert particle/dye state to emitted/scattered signal.
  - **Attenuation**: apply Beer-Lambert-style transmission for dye signal.
  - **Imaging**: convert to image-like intensity using splatting + blur.
  - **Sensor**: convert ideal intensity to `uint8` (auto exposure and/or noise model).


## 3. Main Public Entry Points

### 3.1 `simulate_video_pipeline(...)`

Primary simulation API.

Returns:

- `video`: `np.ndarray`, shape `(T, H, W)`, `uint8`
- `final`: dict with final state arrays
  - `xp`, `zp`, `y`, `c`
- `diag`: diagnostic time series dictionary
  - `Imin`, `Imax`, `Imean`, `vis_frac`

### 3.2 `export_outputs(video, fps=20, base="sim")`

Exports frames/video to:

- `<script_dir>/Outputs_pipeline/<base>_frame0.png`
- `<script_dir>/Outputs_pipeline/<base>_frame_mid.png`
- `<script_dir>/Outputs_pipeline/<base>_frame_last.png`
- `<script_dir>/Outputs_pipeline/<base>.gif`
- `<script_dir>/Outputs_pipeline/<base>.mp4` (if MP4 backend available)


## 4. Module Structure

The code is organized into these sections.

### 4.1 Dynamics core

- `vel_u_w(...)`
  - Computes velocity field `(u, w)` using the analytic placeholder model.

- `advect_particles_rk2(...)`
  - RK2 (midpoint) particle integration.
  - Boundary handling: periodic in `x`, clipped in `z`.

- `bilinear_sample(...)`
  - Bilinear interpolation for semi-Lagrangian backtracing.

- `advect_dye_semilag(...)`
  - Semi-Lagrangian dye update.
  - Optional diffusion term via Laplacian.

### 4.2 Out-of-plane visibility model

Canonical names:

- `update_out_of_plane(...)`
- `visible_mask(...)`
- `respawn(...)`

Compatibility aliases (same behavior):

- `update_out_of_plane_y(...)`
- `visible_mask_from_sheet(...)`
- `respawn_particles(...)`

### 4.3 Imaging helpers

- `gaussian_blur_fft(...)`
- `render_particles(...)`

### 4.4 Goal-2 pipeline physics helpers

- `illumination_field(...)`
  - Supports:
    - `mode="point"`
    - `mode="gaussian_beam"`

- `dye_emission(...)`
- `beer_lambert_attenuation_path_integral(...)`

### 4.5 Sensor/exposure helpers

Canonical name:

- `auto_exposure_to_uint8(...)`

Compatibility alias:

- `auto_exposure_uint8(...)`

Also includes:

- `camera_model(...)`

### 4.6 Forward-loop internal modules

- `init_state(...)`
- `prepare_illumination(...)`
- `init_diagnostics(...)`
- `step_flow_dynamics(...)`
- `step_out_of_plane_and_visibility(...)`
- `compute_interaction_and_attenuation(...)`
- `render_total_intensity(...)`
- `encode_frame(...)`
- `append_diagnostics(...)`

These modules make the main simulation loop easier to read and align structurally with `simulate_forward.py`.


## 5. Detailed Per-Frame Execution Order

Inside `simulate_video_pipeline(...)`, each time step does:

1. **Flow dynamics update**
   - Update `xp, zp` with RK2.
   - Update dye field `c` via semi-Lagrangian advection (+ optional diffusion).

2. **Out-of-plane update and visibility**
   - Random-walk update for `y`.
   - Respawn particles that move beyond `y_kill`.
   - Build visibility mask `vis` for the sheet thickness.

3. **Pipeline Step 1: Illumination**
   - Use precomputed illumination field and particle illumination samples.

4. **Pipeline Step 2: Interaction**
   - Compute particle amplitudes from local illumination.
   - Compute dye emission from `c` and illumination grid.

5. **Pipeline Step 3: Attenuation**
   - Compute Beer-Lambert attenuation based on cumulative dye concentration.

6. **Pipeline Step 4: Imaging**
   - Render visible particles with splat + PSF blur.
   - Blur dye component.
   - Combine both channels into ideal intensity `I`.

7. **Pipeline Step 5: Sensor**
   - Apply auto exposure if enabled.
   - Apply camera noise model if enabled.
   - Store final `uint8` frame.

8. **Diagnostics**
   - Append `Imin`, `Imax`, `Imean`, `vis_frac`.


## 6. Important Parameters

The function signature of `simulate_video_pipeline(...)` exposes all major controls.

### 6.1 Timeline and grid

- `T`, `dt`
- `H`, `W`
- `N`

### 6.2 Physical domain

- `Lx`
- `zmin`, `zmax`

### 6.3 Velocity field

- `A`, `k`, `gamma`

### 6.4 Dynamics and model mismatch

- `particle_noise`
- `dye_kappa`

### 6.5 Out-of-plane process

- `enable_out_of_plane`
- `sheet_thickness`
- `y_noise_sigma`
- `y_kill`

### 6.6 Pipeline-specific optics

- Illumination geometry:
  - `illum_mode`
  - `light_x_frac`
  - `light_z_above_frac`
  - `beam_sigma`
  - `beam_depth_decay`

- Particle brightness:
  - `particle_base_amp`
  - `particle_illum_power`
  - `psf_sigma_px`

- Dye imaging:
  - `dye_beta`
  - `dye_alpha`
  - `dye_blur_sigma_px`

### 6.7 Sensor/display

- `use_camera_noise`
- `bg`, `gain`, `read_sigma`
- `auto_exposure`, `exposure_percentile`

### 6.8 Reproducibility

- `seed`


## 7. Diagnostics

`diag` contains a value per frame:

- `Imin`: minimum ideal intensity of the frame
- `Imax`: maximum ideal intensity of the frame
- `Imean`: mean ideal intensity of the frame
- `vis_frac`: fraction of particles visible in the sheet

These diagnostics are useful for sanity checks and comparing parameter settings.


## 8. Running the Script

This file currently runs from its `__main__` block with built-in parameter values.

Basic run:

```bash
python pipeline_forward.py
```

Expected console outputs include:

- video shape and dtype
- intensity summary over time
- average visible fraction
- export directory path


## 9. Programmatic Usage

Use as a module from another script:

```python
from pathlib import Path
import pipeline_forward as pf

video, final, diag = pf.simulate_video_pipeline(
    T=120,
    dt=0.02,
    H=256,
    W=256,
    N=1200,
    seed=42,
)

pf.export_outputs(video, fps=24, base="pipeline_caseA")
```

## 10. Output Behavior and Path

This script exports to a fixed path inside the script directory:

- `Outputs_pipeline`

That behavior is implemented in `export_outputs(...)` and is independent of your shell working directory.


## 11. Numerical/Model Notes

- Semi-Lagrangian advection is stable and simple, suitable for MVP-level synthetic generation.
- Particle rendering uses nearest-pixel splat + FFT Gaussian blur; this is efficient and produces realistic soft spots.
- Beer-Lambert attenuation is implemented as cumulative concentration integration along the assumed light path direction.
- The camera model combines Poisson shot noise and Gaussian read noise, then clips to `[0, 255]`.


## 12. Performance Notes

Most expensive operations are:

1. Full-frame FFT blurs.
2. Full-grid dye update and attenuation.
3. Particle splatting when `N` is large.

If runtime is high, first reduce `H/W`, then `T`, then `N`.


## 13. Relationship to `simulate_forward.py`

`pipeline_forward.py` and `simulate_forward.py` now share a more consistent modular structure and naming for overlapping concepts (dynamics/visibility/render/sensor/diagnostics).

`pipeline_forward.py` adds the explicit Goal-2 optical decomposition:

- illumination
- interaction
- attenuation
- imaging
- sensor

while preserving a similar flow of internal step functions.

---
# GIF Key Frame Extractor (`extract_key_frames.py`) ($Hongze$ $Lin$)

## 1. Purpose

`extract_key_frames.py` extracts 3 key frames from each of two output GIF files:

1. Frame 0 (first frame)
2. Middle frame
3. Last frame

The script automatically creates the target folders and saves the extracted images.


## 2. Default Input Files

The script reads these two GIF files by default:

- `Outputs_simulate/sim.gif`
- `Outputs_pipeline/sim_pipeline.gif`


## 3. Output Directory Structure

After running, the script generates:

```text
Key_Outputs_image/
├── Key_simulate/
│   ├── sim_frame0.png
│   ├── sim_frame_mid.png
│   └── sim_frame_last.png
└── Key_pipeline/
    ├── sim_pipeline_frame0.png
    ├── sim_pipeline_frame_mid.png
    └── sim_pipeline_frame_last.png
```


## 4. How to Run

Run from the project root directory:

```bash
python3 extract_key_frames.py
```

## 5. Dependency

This script requires `Pillow` (`PIL`):

```bash
pip install pillow
```

## 6. Logic Summary

For each GIF:

1. Read total frame count `n_frames`
2. Compute key frame indices:
   - first frame: `0`
   - middle frame: `n_frames // 2`
   - last frame: `n_frames - 1`
3. Use `seek` to load each frame and save it as PNG

## 7. Notes

- If an input GIF path does not exist, the script raises `FileNotFoundError`.
- Existing output files with the same names will be overwritten.
- The middle frame uses integer floor division: `n_frames // 2`.

---
## About `MCS_research.md` ($Hongze$ $Lin$)
- Simple research of what is Monto Carlo simulation
- Bullet points of where to use this method and related reasons
- How to use MCS in this MDM report

