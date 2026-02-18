# Mdm2_Group_10
---
# 1. About `simulate_forward.py`
## - Forward Simulator ($A(x0, θ) → b0:T$) ($Hongze$ $Lin$)

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

# 2. About `forward_pipeline.py` 
## — Explicit Forward Imaging Pipeline $(A(x0, θ) → b0:T)$ ($Hongze$ $Lin$)

This script implements a **physically structured forward pipeline** for the modelling project.  
Given an initial latent state (particles + dye) and model parameters, it generates a **synthetic camera video** (a stack of grayscale frames) by explicitly chaining:

**Illumination → Interaction → Beer–Lambert Attenuation → Imaging (PSF/blur) → Sensor (noise/quantisation)**

Formally, it implements the forward operator:

$\[A(x_0,\theta)\ \mapsto\ b_{0:T}\]$

- **Latent state** $\(x_t\)$: particle positions + dye concentration field
- **Parameters** $\(\theta\)$: flow parameters, illumination/absorption parameters, PSF parameters, camera/noise parameters
- **Observations** $\(b_t\)$: camera frames (uint8 images)

> This pipeline is designed to match the meeting emphasis:
> - avoid relying on PIV-style smoothing; instead build a forward model
> - capture **location-dependent brightness** (beam profile, attenuation)
> - provide a clean foundation for later optimisation/inference (e.g., Adam, HMC) via \(\min_x \|A(x)-b\|^2\)

## What the output GIF/video represents

The exported GIF/MP4 is the **simulated camera observation** over time:
- particles are advected in a prescribed flow (placeholder velocity field)
- dye concentration field is advected on a grid
- each frame is generated by the explicit imaging pipeline:
  - non-uniform illumination \(L(x,z)\)
  - particle brightness linked to illumination
  - dye emission linked to concentration and illumination
  - Beer–Lambert absorption (path-integral approximation)
  - PSF blur and camera noise/quantisation

## Code breakdown (modules inside the script)

### 1) Configuration / parameters (θ)
Parameters are grouped to reflect the report goals and meeting notes:

- **Time & grid:** `T`, `dt`, `H`, `W`
- **Domain:** `Lx`, `zmin`, `zmax` (periodic in `x`, confined in `z`)
- **Particles:** `N`, `psf_sigma_px`, brightness base and illumination scaling
- **Dye:** advection settings, `dye_beta` (fluorescence scale), `dye_alpha` (absorption), optional blur
- **Illumination:** mode and geometry (point source or Gaussian beam)
- **Sensor:** `bg`, `gain`, `read_sigma`, optional auto-exposure
- **Export:** `fps`, output folder name

> Variable names are kept consistent with the baseline script:  
> `xp`, `zp`, `y`, `c`, `xs`, `zs`, `I_p`, `I_dye`, `I`.

### 2) Flow model (placeholder velocity field)
- Function: typically `vel_u_w(x, z, t, ...)`
- Implements the **given** velocity field (Eqs. (3)–(4))
- Used by:
  - particle advection
  - dye advection

This matches the project brief’s “useful placeholder” approach: prioritize the forward mapping first.

### 3) Dynamics (state evolution)
#### 3.1 Particle advection (Lagrangian)
- Function: `advect_particles_rk2(...)`
- RK2 (midpoint) integration for stability
- Boundaries:
  - periodic in `x`: `xp = xp % Lx`
  - confined in `z`: `zp = clip(zp, zmin, zmax)`
- Optional small Gaussian process noise to represent model mismatch / unresolved motion

#### 3.2 Dye advection (Eulerian grid)
- Function: `advect_dye_semilag(...)`
- Semi-Lagrangian (backtrace + bilinear interpolation)
- Optional diffusion/smoothing (MVP: usually set to 0)

### 4) Visibility model (single-plane measurement)
To represent the thin laser sheet:
- State includes out-of-plane coordinate `y`
- `y` follows a random walk
- Particle is visible only if `|y| <= sheet_thickness/2`
- Particles beyond a kill distance (`|y| > y_kill`) are respawned (birth/death)

Functions often named:
- `update_out_of_plane_y(...)`
- `visible_mask_from_sheet(...)`
- `respawn_particles(...)`

This directly reflects the meeting point: camera observes a plane; particles enter/leave the sheet.

### 5) Imaging pipeline (Goal 2: explicit physical mapping)
This is the defining feature of `forward_pipeline.py`. Each frame is generated by:

#### (1) Illumination — build \(L(x,z)\)
- Function: `illumination_field(...)`
- Two options:
  - **Point source** above the domain: \(L \propto 1/(d^2+\epsilon)\)
  - **Gaussian beam** in \(x\) with depth decay in \(z\)
- Also provides illumination at particle locations \(L(x_i,z_i)\)

**Why:** matches the meeting note that a particle at the beam edge appears darker than at the beam center.

#### (2) Interaction — emission from particles and dye
- **Particles:** per-particle brightness scales with local illumination:
  - `amps = base_amp * (Lp_norm ** power)`
- **Dye:** fluorescence emission is proportional to concentration and illumination:
  - `E_dye = dye_beta * c * L_grid`

**Why:** encodes the “statistical mapping” intuition: brightness carries information about location within the beam.

#### (3) Attenuation — Beer–Lambert absorption (path-integral approximation)
- Function: `beer_lambert_attenuation_path_integral(...)`
- Uses:
  $\[\exp\Big(-\alpha \int c\, ds\Big)\]$
- Implemented as a cumulative integral along a simplified propagation direction (e.g., top-down through the domain)

**Why:** reflects the meeting point about exponential absorption and the reverse-engineering nature of dye concentration inference.

#### (4) Imaging — PSF blur / smoothing
- **Particles:** rasterisation (“splatting”) + Gaussian PSF blur → `I_p`
- **Dye:** optional mild blur → `I_dye`

**Why:** approximates camera optics without requiring expensive Monte Carlo ray tracing.

#### (5) Sensor — noise + quantisation
- Optional auto-exposure (percentile scaling) to prevent “all black” outputs
- Optional camera noise model:
  - background offset
  - Poisson shot noise
  - Gaussian read noise
  - clip to 8-bit (`uint8`)

Final intensity per frame:
$\[I = I_p + I_{dye} \quad \Rightarrow \quad b_t = \text{Sensor}(I)\]$

### 6) Forward operator assembly (the actual A)
- Function: usually `simulate_video_pipeline(...)` or similar
- Loops over frames:
  1. evolve `(xp, zp, c, y)`
  2. run pipeline steps (1)–(5)
  3. store each `frame` into `video[t]`

Returns:
- `video` with shape `(T, H, W)` and dtype `uint8`
- optional diagnostics (intensity min/max/mean, visible fraction)

### 7) Export (GIF/MP4/PNG)
Exports are written to a script-relative folder:

- `Outputs_pipline/`

Typical outputs:
- `sim_pipeline.gif`
- `sim_pipeline.mp4` (requires ffmpeg)
- key frames: `*_frame0.png`, `*_frame_mid.png`, `*_frame_last.png`
---
