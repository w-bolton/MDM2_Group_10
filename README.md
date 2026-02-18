# Mdm2_Group_10
---
# About `forward_simulate.py` — Forward Simulator ($A(x0, θ) → b0:T$) ($Hongze$ $Lin$)

This script implements a **forward simulator** for the project: given an initial latent state (particles + dye) and a set of parameters, it generates a **synthetic camera video** (a stack of grayscale frames).

Formally, it implements the forward operator:

$\[ A(x_0,\theta)\ \mapsto\ b_{0:T} \]$

- **Latent state** $\(x_t\)$: particle positions + dye concentration field
- **Parameters** $\(\theta\)$: flow parameters, optics parameters, camera/noise parameters
- **Observations** $\(b_t\)$: camera frames (uint-8 images)

> Goal: produce a runnable end-to-end forward model that can later be used in inverse problems like  
> $\(\min_x \|A(x) - b_{\text{obs}}\|^2\)$.

## What the output GIF/video represents

The exported GIF/MP4 is the **simulated camera observation** over time:
- particles advected by a prescribed velocity field (placeholder flow)
- dye concentration advected on a grid
- a rendered image per frame from **particle intensity + dye intensity**
- optional camera noise + quantisation to 8-bit

## Code breakdown (modules inside the script)

### 1) Configuration / parameters (θ)
Typically defined near the top as constants or a config/dataclass.

Common groups:
- **Time & grid:** `T`, `dt`, `H`, `W`
- **Domain:** `Lx`, `zmin`, `zmax` (periodic in `x`, clipped in `z`)
- **Particles:** `N`, `psf_sigma_px`, `particle_amp`, process noise
- **Dye:** diffusion `kappa`, blur sigma, emission/absorption parameters
- **Optics/illumination:** illumination profile, absorption coefficient, beam geometry
- **Camera:** `bg`, `gain`, `read_sigma`, optional auto-exposure
- **Export:** `fps`, output folder name

### 2) Flow model (placeholder velocity field)
Implements the prescribed velocity field from the brief (Eqs. (3)–(4)):

- Function often named like: `vel_u_w(x, z, t, ...)`
- Returns velocity components `(u, w)`

Used by both:
- particle advection
- dye advection

### 3) Dynamics (state evolution)
#### 3.1 Particle advection (Lagrangian)
- Function: `advect_particles_rk2(...)`
- Integrator: RK2 (midpoint) or similar
- Boundaries:
  - `x` periodic: `xp = xp % Lx`
  - `z` clipped: `zp = clip(zp, zmin, zmax)`
- Optional model error: small Gaussian noise added to `(xp, zp)` per step

#### 3.2 Dye advection (Eulerian grid)
- Function: `advect_dye_semilag(...)`
- Method: semi-Lagrangian (backtrace + bilinear interpolation)
- Optional diffusion: `kappa * Laplacian(c)` for smoothing/stability

### 4) Visibility model (single light-sheet plane)
(Optional but recommended; matches the project setting)

- Variables: `y` (out-of-plane coordinate)
- `y` evolves as a random walk
- Particles are **visible** only if `|y| <= sheet_thickness/2`
- Birth/death: respawn particles that drift too far (`|y| > y_kill`)

Functions often named:
- `update_out_of_plane_y(...)`
- `visible_mask(...)`
- `respawn(...)`

### 5) Rendering (state → ideal intensity image)
The forward model typically forms an ideal intensity image:

$\[I(x,z) = I_p(x,z) + I_{\text{dye}}(x,z)\]$

#### 5.1 Particle image `I_p`
- Function: `render_particles(...)`
- Steps:
  1) splat particles onto pixel grid
  2) apply Gaussian PSF blur (`psf_sigma_px`) to create realistic blobs

#### 5.2 Dye image `I_dye`
- Function: `render_dye(...)`
- Typical ingredients:
  - illumination field \(L(x,z)\) (non-uniform brightness across the plane)
  - fluorescence emission proportional to `c * L`
  - Beer–Lambert-style attenuation via an exponential term
  - optional mild blur

### 6) Sensor model (ideal intensity → camera frame)
- Function: `camera_model(...)` (or similar)
- Adds:
  - background offset (`bg`)
  - Poisson shot noise (photon statistics)
  - Gaussian read noise (`read_sigma`)
  - clip to `[0, 255]` and cast to `uint8`

Optional:
- `auto_exposure_to_uint8(...)` (percentile scaling) to avoid all-black outputs when the raw dynamic range is small.

### 7) Forward operator assembly (the actual A)
This is usually a single function like:

- `forward_simulator(cfg)` or `simulate_forward(...)`

It loops over frames:
1. evolve state `(xp, zp, c, y)`
2. render `I_p` and `I_dye`
3. apply sensor model → frame `b_t`
4. store into `video[t]`

Return:
- `video` with shape `(T, H, W)` and dtype `uint8`
- optional diagnostics (min/max/mean intensity, visible fraction)


### 8) Export (GIF/MP4/PNG)
- Function: `export_video(...)` or `export_outputs(...)`
- Writes to a **script-relative folder** (recommended) to avoid permission errors:
  - e.g. `Outputs_simulate/`

Common exports:
- `sim.gif`
- `sim.mp4` (requires ffmpeg)
- key frames: `frame0.png`, `frame_mid.png`, `frame_last.png`
---
