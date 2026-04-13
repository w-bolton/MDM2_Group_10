# %% [markdown]
# ---
# ## Particle Simulation Evolution Model
# 
# Objective:
# - Keep the model modular so parameters and update rules can be optimised later.
# 
# Success criteria:
# - Model particle motion on the observed `x-z` slice.
# - Model particle appearance and disappearance caused by out-of-plane motion through the pixel-thickness visibility band.
# - Keep the particle cloud randomly distributed in space instead of artificially clustering near one interface.
# 

# %% [markdown]
# ## Module Skeleton 
# 
# $\rightarrow$ Particle Simulation Evolution Model
# 
# 1. Task-aligned assumptions
# 2. Configuration and latent particle state
# 3. Core particle dynamics
# 4. Initial particle-cloud generation
# 5. One-step evolution and rollout simulator
# 6. Model outputs and diagnostics
# 7. GIF visualization

# %%
# Setup: imports and reproducibility
from __future__ import annotations

import io
from dataclasses import dataclass

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image

SEED = 10
ATWOOD_REF = 0.0695
rng = np.random.default_rng(SEED)

# %% [markdown]
# ## Module 0: Assumptions
# 
# Basic Statement
# 
# - The camera sees only an `x-z` slice.
# - The camera optical axis is along `+y`, so `y` is the missing out-of-plane direction.
# - The image-frame origin is centred at `x = 0`, `z = 0`.
# - Initial particles are uniformly randomly distributed throughout the full 3D domain.
# - In-plane motion is treated as random particle motion across the plane, with only a weak placeholder drift retained from the Task velocity field.
# - Visibility is represented by a constant pixel-thickness gate in the `y` direction.

# %%
# Module 1: unified configuration and particle statement
@dataclass
class SimConfig:
    """Configuration for the particle-based reduced-order surrogate."""
    # time discretisation
    dt: float = 0.02
    accel_1_duration: float = 0.88
    decel_duration: float = 1.44
    accel_2_duration: float = 0.88

    # particle count
    N: int = 80000

    # physical domain 
    Lx: float = 0.2
    Ly: float = 0.2
    zmin: float = -0.2
    zmax: float = 0.2

    # Task / CAMPI-inspired mode settings
    atwood: float = 0.0695
    mode_number: int = 6
    initial_amplitude: float = 2.0e-4
    max_amplitude: float = 8.0e-3
    gamma_ref: float = 2.4
    unstable_accel: float = 20.0
    stable_accel: float = -30.0

    # in-plane random motion and weak background drift
    particle_noise_sigma: float = 1.6e-3
    flow_weight: float = 0.10

    # out-of-plane random motion across the full depth
    y_noise_sigma: float = 0.004

    # visibility gating: constant pixel thickness in y
    enable_visibility_gate: bool = True
    slice_center_y: float = 0.0
    pixel_thickness: float = 1.2e-3

    gif_stride: int = 4
    gif_frame_duration: float = 0.12

    seed: int = 10

    @property
    def total_time(self) -> float:
        return self.accel_1_duration + self.decel_duration + self.accel_2_duration

    @property
    def T(self) -> int:
        return int(np.ceil(self.total_time / self.dt))

    @property
    def k(self) -> float:
        return self.mode_number * np.pi / self.Lx

@dataclass
class State:
    """Latent particle state carried through the particle-only rollout."""
    xp: np.ndarray
    zp: np.ndarray
    y: np.ndarray
    vy: np.ndarray

cfg = SimConfig()

# %% [markdown]
# ## Module 2: Core Particle Dynamics
# 
# This block is intentionally simple:
# 
# - centred periodic wrapping in `x`,
# - random particle motion across the `x-z` plane,
# - a weak Task-style placeholder drift in the background,
# - random out-of-plane motion in `y`,
# - a constant pixel-thickness visibility gate.
# 
# This keeps the particle cloud spatially random rather than forcing structured interface-centred clustering.
# 

# %%
# Module 2A: coordinate helpers, acceleration history, amplitude update, velocity field, and visibility
def wrap_x_centered(x: np.ndarray, Lx: float) -> np.ndarray:
    """Wrap x positions back into the centred periodic interval [-Lx/2, Lx/2)."""
    return ((x + 0.5 * Lx) % Lx) - 0.5 * Lx


def acceleration_profile(t: float, cfg: SimConfig) -> float:
    """Return the imposed vertical acceleration at time t."""
    if t < cfg.accel_1_duration:
        return cfg.unstable_accel
    if t < cfg.accel_1_duration + cfg.decel_duration:
        return cfg.stable_accel
    if t < cfg.total_time:
        return cfg.unstable_accel
    return 0.0


def gamma_eff(t: float, cfg: SimConfig) -> float:
    """Scale the reference modal growth rate with Atwood number and acceleration."""
    return cfg.gamma_ref * (cfg.atwood / ATWOOD_REF) * acceleration_profile(t, cfg) / cfg.unstable_accel


def advance_amplitude(amplitude: float, t: float, cfg: SimConfig) -> float:
    """Advance the modal amplitude over one time step and enforce the configured cap."""
    updated = amplitude * np.exp(gamma_eff(t, cfg) * cfg.dt)
    return float(np.clip(updated, 0.0, cfg.max_amplitude))


def vel_u_w(x: np.ndarray, z: np.ndarray, amplitude: float, cfg: SimConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return the reduced-order in-plane velocity induced by the current modal amplitude."""
    decay = np.exp(-cfg.k * np.abs(z))
    sign_z = np.where(z >= 0.0, 1.0, -1.0)
    u = -cfg.flow_weight * cfg.k * amplitude * decay * np.sin(cfg.k * x)
    w = -cfg.flow_weight * sign_z * cfg.k * amplitude * decay * np.cos(cfg.k * x)
    return u, w


def visible_mask_y(y: np.ndarray, cfg: SimConfig) -> np.ndarray:
    """Keep only particles inside the out-of-plane visibility slice."""
    if not cfg.enable_visibility_gate:
        return np.ones_like(y, dtype=bool)
    return np.abs(y - cfg.slice_center_y) <= 0.5 * cfg.pixel_thickness


# %% [markdown]
# ## Module 3: Initial Particle Cloud
# 
# The initial particle cloud is uniformly random in the whole 3D domain.
# 
# - `x` is uniform across the full centred width.
# - `z` is uniform across the full height.
# - `y` is uniform across the full depth.
# 
# This matches the assumption that the initial particles are randomly scattered throughout the whole space.
# 

# %%
# Module 3: initialise the latent particle state
def init_state(cfg: SimConfig) -> State:
    """Initialise particles uniformly across the x-z-y domain."""
    rng = np.random.default_rng(cfg.seed)

    xp = rng.uniform(-0.5 * cfg.Lx, 0.5 * cfg.Lx, size=cfg.N).astype(np.float32)

    zp = rng.uniform(cfg.zmin, cfg.zmax, size=cfg.N).astype(np.float32)

    y = rng.uniform(-0.5 * cfg.Ly, 0.5 * cfg.Ly, size=cfg.N).astype(np.float32)
    vy = np.zeros(cfg.N, dtype=np.float32)

    return State(xp=xp, zp=zp, y=y, vy=vy)

state0 = init_state(cfg)


# %% [markdown]
# ## Module 4: One-Step Evolution and Rollout
# 
# Each time step does only particle-related work:
# 
# 1. update the weak background mode amplitude from the acceleration history,
# 2. move particles in `x-z` using random motion plus a small placeholder drift,
# 3. update random out-of-plane motion in `y`,
# 4. compute which particles are visible in the current slice.
# 

# %%
# Module 4A: particle update rules and one-step evolution
def reflect_interval(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """Reflect positions back into a bounded interval without discontinuous clipping."""
    span = upper - lower
    shifted = np.mod(values - lower, 2.0 * span)
    reflected = np.where(shifted <= span, shifted, 2.0 * span - shifted)
    return lower + reflected


def advect_particles_rk2(
    x: np.ndarray,
    z: np.ndarray,
    amplitude: float,
    cfg: SimConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Move particles with a midpoint Runge-Kutta step plus weak stochastic spreading."""
    u1, w1 = vel_u_w(x, z, amplitude, cfg)
    xm = x + 0.5 * cfg.dt * u1
    zm = z + 0.5 * cfg.dt * w1
    u2, w2 = vel_u_w(xm, zm, amplitude, cfg)

    x_new = x + cfg.dt * u2
    z_new = z + cfg.dt * w2
    if cfg.particle_noise_sigma > 0.0:
        x_new += rng.normal(0.0, cfg.particle_noise_sigma, size=x_new.shape)
        z_new += rng.normal(0.0, cfg.particle_noise_sigma, size=z_new.shape)

    x_new = wrap_x_centered(x_new, cfg.Lx)
    z_new = reflect_interval(z_new, cfg.zmin, cfg.zmax)
    return x_new.astype(np.float32), z_new.astype(np.float32)


def update_out_of_plane(
    y: np.ndarray,
    vy: np.ndarray,
    cfg: SimConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Diffuse particles in the hidden y direction and infer a diagnostic vy."""
    y_new = y + cfg.y_noise_sigma * np.sqrt(cfg.dt) * rng.standard_normal(y.shape)
    y_new = reflect_interval(y_new, -0.5 * cfg.Ly, 0.5 * cfg.Ly)
    vy_new = (y_new - y) / cfg.dt
    return y_new.astype(np.float32), vy_new.astype(np.float32)


def step_evolution(
    state: State,
    amplitude: float,
    t: float,
    cfg: SimConfig,
    rng: np.random.Generator,
) -> tuple[State, np.ndarray, float]:
    """Advance the full latent particle state by one time step."""
    amplitude_new = advance_amplitude(amplitude, t, cfg)
    state.xp, state.zp = advect_particles_rk2(state.xp, state.zp, amplitude_new, cfg, rng)
    state.y, state.vy = update_out_of_plane(state.y, state.vy, cfg, rng)
    vis = visible_mask_y(state.y, cfg)
    return state, vis, amplitude_new


# %%
# Module 4B: rollout simulator for particle evolution only
def simulate_particle_evolution(cfg: SimConfig) -> tuple[State, dict[str, np.ndarray], dict[str, list[float]], dict[str, list[np.ndarray]]]:
    """Roll out the particle-only surrogate and store diagnostics for plotting and GIF export."""
    rng = np.random.default_rng(cfg.seed)
    state = init_state(cfg)
    amplitude = cfg.initial_amplitude
    t = 0.0

    diag = {
        "time": [],
        "visible_frac": [],
        "amplitude": [],
        "acceleration": [],
        "x_spread": [],
        "z_spread": [],
    }
    history = {
        "time": [],
        "visible_points": [],
        "visible_frac": [],
        "acceleration": [],
    }

    visible = None
    for _ in range(cfg.T):
        state, visible, amplitude = step_evolution(state, amplitude, t, cfg, rng)

        diag["time"].append(float(t))
        diag["visible_frac"].append(float(np.mean(visible)))
        diag["amplitude"].append(float(amplitude))
        diag["acceleration"].append(float(acceleration_profile(t, cfg)))
        diag["x_spread"].append(float(np.std(state.xp)))
        diag["z_spread"].append(float(np.std(state.zp)))

        history["time"].append(float(t))
        history["visible_points"].append(np.column_stack((state.xp[visible], state.zp[visible])).astype(np.float32))
        history["visible_frac"].append(float(np.mean(visible)))
        history["acceleration"].append(float(acceleration_profile(t, cfg)))
        t += cfg.dt

    outputs = {
        "visible_mask": visible,
        "visible_points": np.column_stack((state.xp[visible], state.zp[visible])),
    }
    return state, outputs, diag, history


# %% [markdown]
# ## Module 5: Particle Outputs
# 
# The notebook exposes the main outputs of the particle model:
# 
# - `visible_points`: sparse particle positions on the observed `x-z` plane.
# - `State`: the latent particle positions and out-of-plane coordinates.
# - `diag`: summary time series for model tuning.
# - `history`: particle positions across time for animation.
# 

# %%
# Module 5: small helper accessors
def visible_particle_points(outputs: dict[str, np.ndarray]) -> np.ndarray:
    """Return the final x-z points that remain visible in the measurement slice."""
    return outputs["visible_points"]


def summary_dict(diag: dict[str, list[float]]) -> dict[str, float]:
    """Collect a compact particle summary used in markdown text and plots."""
    return {
        "final_visible_fraction": float(diag["visible_frac"][-1]),
        "max_visible_fraction": float(np.max(diag["visible_frac"])),
        "min_visible_fraction": float(np.min(diag["visible_frac"])),
        "final_amplitude": float(diag["amplitude"][-1]),
    }


# %% [markdown]
# ## Module 6: Default Run and Diagnostics
# 
# The next cells run the particle evolution model and visualise its state and summary diagnostics.
# 

# %%
# Module 6A: run the particle evolution simulator
# state, outputs, diag, history = simulate_particle_evolution(cfg)
# points = visible_particle_points(outputs)
# summary = summary_dict(diag)


# # %%
# # Module 6B: diagnostic plots for the particle module
# fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)

# axes[0].scatter(points[:, 0], points[:, 1], s=2.0, alpha=0.55, color="#1b6ca8")
# axes[0].set_title("Visible Particles on the x-z Plane")
# axes[0].set_xlabel("x (m)")
# axes[0].set_ylabel("z (m)")
# axes[0].set_xlim(-0.5 * cfg.Lx, 0.5 * cfg.Lx)
# axes[0].set_ylim(cfg.zmin, cfg.zmax)

# axes[1].plot(diag["time"], diag["visible_frac"], label="visible fraction", color="#1b6ca8", linewidth=2.0)
# axes[1].plot(diag["time"], diag["amplitude"], label="mode amplitude", color="#c84b31", linewidth=2.0)
# axes[1].set_title("Particle Evolution Diagnostics")
# axes[1].set_xlabel("time (s)")
# axes[1].grid(alpha=0.25)
# axes[1].legend()

# plt.show()


# %% [markdown]
# ## Module 7: GIF Visualisation
# 
# This module turns the particle evolution history into an in-notebook GIF.
# 
# - the animation shows visible particle positions on the `x-z` plane
# - each frame is annotated with time, visible fraction, and acceleration
# 
# The GIF is generated in memory and displayed directly inside the notebook.
# 

# %%
# Module 7: build and display an in-memory GIF
# from pathlib import Path
# from typing import cast
# import tempfile
# from matplotlib.backends.backend_agg import FigureCanvasAgg


# def gif_frame_indices(frame_count: int, stride: int) -> list[int]:
#     """Sample frames at a fixed stride and always include the final frame."""
#     indices = list(range(0, frame_count, max(stride, 1)))
#     if indices[-1] != frame_count - 1:
#         indices.append(frame_count - 1)
#     return indices


# def rgb_frame_from_figure(fig: plt.Figure) -> np.ndarray:
#     """Render a Matplotlib figure into a plain RGB array for GIF assembly."""
#     fig.canvas.draw()
#     canvas = cast(FigureCanvasAgg, fig.canvas)
#     return np.asarray(canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()


# def gif_image_from_frames(frames: list[np.ndarray], frame_duration: float) -> Image:
#     """Encode a list of RGB arrays as an in-memory GIF wrapped in IPython display output."""
#     with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp_file:
#         gif_path = Path(tmp_file.name)

#     imageio.mimsave(gif_path, frames, format="GIF", duration=float(frame_duration))
#     gif_bytes = gif_path.read_bytes()
#     gif_path.unlink(missing_ok=True)
#     return Image(data=gif_bytes, format="gif")


# def build_particle_gif(history: dict[str, list[np.ndarray]], cfg: SimConfig) -> Image:
#     """Render the particle history as a compact diagnostic GIF."""
#     frames = []

#     for idx in gif_frame_indices(len(history["time"]), cfg.gif_stride):
#         points = history["visible_points"][idx]
#         time_value = history["time"][idx]
#         visible_fraction = history["visible_frac"][idx]
#         accel = history["acceleration"][idx]

#         fig, ax = plt.subplots(figsize=(5.8, 4.6), constrained_layout=True)
#         ax.scatter(points[:, 0], points[:, 1], s=5.0, alpha=0.6, color="#1b6ca8")
#         ax.set_title(f"Visible Particles, t = {time_value:.2f} s")
#         ax.set_xlabel("x (m)")
#         ax.set_ylabel("z (m)")
#         ax.set_xlim(-0.5 * cfg.Lx, 0.5 * cfg.Lx)
#         ax.set_ylim(cfg.zmin, cfg.zmax)
#         ax.text(
#             0.02,
#             0.98,
#             f"visible fraction = {visible_fraction:.4f}\na = {accel:.1f} m/s^2",
#             transform=ax.transAxes,
#             va="top",
#             ha="left",
#             fontsize=9,
#             bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
#         )

#         frames.append(rgb_frame_from_figure(fig))
#         plt.close(fig)

#     return gif_image_from_frames(frames, cfg.gif_frame_duration)


# particle_gif = build_particle_gif(history, cfg)
# particle_gif


# %% [markdown]
# ---
# ## Dye Concentration Field Evolution Model
# 
# This notebook builds a dye-only evolution model on an `x-z` pixel slice. Each control volume stores `phi(x, z, t) in [0, 1]`, the volume fraction of the lower, dyed, low-density fluid. `phi = 1` means a pixel slice is filled with the dyed water + propanol-2 fluid, while `phi = 0` means it is filled with the upper sodium chloride solution.
# 
# In the revised scenario used here:
# - the high-density sodium chloride fluid starts above the interface and the dyed low-density fluid starts below,
# - the interface is initially sharp and slightly wavy, so the first acceleration stage can amplify a visible RT-type wave boundary,
# - after that, the model does not blend the dye field toward a prescribed inverted target profile,
# - instead, the evolving density field feeds back into a reduced-order buoyancy closure, so collapse, redistribution, diffusion, and eventual inversion come from the same transport dynamics,
# - a dedicated invariant projector is applied after every transport step so that total dyed-fluid content, total fluid mass, and domain-averaged density remain locked to their initial values,
# - by the end of the simulation, the mixed layer becomes smoother and the recovered top-light / bottom-heavy ordering emerges naturally rather than from an imposed flip.
# 
# The paper-aligned background parameters kept here are:
# - a low-Atwood configuration,
# 
# $$
# A_t = \frac{\rho_h - \rho_l}{\rho_h + \rho_l} = 0.0695,
# $$
# 
# - an `Accel-Decel-Accel` history with `+20`, `-30`, `+20 m s^-2` over `0.88 s`, `1.44 s`, `0.88 s`,
# - a field of view anchored to the paper's `0.2 m` width; to keep exactly `1000` square pixels, we use a `25 x 40` grid over a `0.20 m x 0.32 m` computational window, so `dx = dz = 8 mm`.
# 
# The dye update uses the conservative control-volume transport step,
# 
# $$
# \phi_{i,j}^{n+1} = \phi_{i,j}^n
# + \frac{F^x_{i-1/2,j} + F^z_{i,j-1/2} - F^x_{i+1/2,j} - F^z_{i,j+1/2}}{\Delta x \, \Delta z}
# + \kappa_{\mathrm{eff}} \, \Delta t \, \nabla^2 \phi_{i,j}^n,
# $$
# 
# with a linear mixture density
# 
# $$
# \rho(\phi) = \phi \, \rho_l + (1 - \phi) \, \rho_h.
# $$
# 
# The closure is density-field-driven rather than target-driven:
# - the wavy initial interface seeds horizontal density gradients,
# - those gradients generate a reduced-order baroclinic streamfunction that changes sign with the `Accel-Decel-Accel` history,
# - a local buoyancy-sorting drift lifts lighter parcels and lowers heavier parcels without prescribing a final target field,
# - a global mass-and-density constraint function then restores the reference invariants after each explicit update, so the later evolution is not contaminated by cumulative drift.
# 

# %%
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap


def integrate_trapezoid(values: np.ndarray, coordinates: np.ndarray) -> float:
    """Use np.trapezoid when available and fall back to np.trapz on older NumPy builds."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(values, coordinates))
    return float(np.trapz(values, coordinates))


# plt.rcParams.update({
#     "figure.figsize": (11.0, 8.0),
#     "axes.grid": False,
#     "image.cmap": "viridis",
# })


# %% [markdown]
# ## Module 1: Control-Volume Statement
# 
# The `x-z` slice is divided into `Nx * Nz = 1000` square control volumes. The full tank is idealized as a stack of infinitesimally thin slices in `y`, so this notebook advances one representative slice per unit thickness.
# 
# For each cell `(i, j)`, define `phi_ij^n` as the dyed low-density fluid fraction at time `t_n`. The conservative update is
# 
# $$
# \phi_{i,j}^{n+1} = \phi_{i,j}^n
# + F^x_{i-1/2,j} + F^z_{i,j-1/2}
# - F^x_{i+1/2,j} - F^z_{i,j+1/2}
# + \kappa_{\mathrm{eff}}(t) \, \Delta t \, 
# abla^2 \phi_{i,j}^n,
# $$
# 
# with upwind face fluxes
# 
# $$
# F = v_{\mathrm{face}} \, \phi_{\mathrm{upwind}} \, \Delta t \, \Delta x,
# $$
# 
# because the cells are square (`dx = dz`). This is still the pixel-wise form of your
# 
# $$
# \phi_{\mathrm{new}} = \phi_{\mathrm{old}} + F_{\mathrm{left}} + F_{\mathrm{down}} - F_{\mathrm{right}} - F_{\mathrm{up}}.
# $$
# 
# Initial condition:
# - `z > eta(x, 0)`: upper dense sodium chloride fluid, so `phi = 0`,
# - `z < eta(x, 0)`: lower dyed low-density fluid, so `phi = 1`,
# - `eta(x, 0)` is a small sinusoidal interface perturbation that seeds the first wave.
# 
# Early on, intermediate `0 < phi < 1` values mostly mean an interface cuts through a finite pixel. Later, when `\kappa_eff(t)` ramps up, they also represent genuine local mixing.
# 

# %%
@dataclass
class DyeConfig:
    """Configuration for the reduced-order dye-field transport model."""
    # square control-volume grid with exactly 1000 cells
    Nx: int = 25
    Nz: int = 40
    Lx: float = 0.20
    Lz: float = 0.32

    # time discretisation
    dt: float = 0.01
    accel_1_duration: float = 0.88
    decel_duration: float = 1.44
    accel_2_duration: float = 0.88
    settling_duration: float = 24.00

    # paper-aligned fluid and instability settings
    atwood: float = 0.0695
    rho_mean: float = 1000.0
    mode_number: int = 6
    unstable_accel: float = 20.0
    stable_accel: float = -30.0

    # reduced-order flow-strength model
    initial_flow_strength: float = 0.22
    min_flow_strength: float = 0.04
    max_flow_strength: float = 1.10
    unstable_growth_rate: float = 2.6
    stable_decay_rate: float = 0.6
    settling_decay_rate: float = 0.05

    # density-feedback transport controls
    initial_wave_amplitude: float = 0.008
    molecular_diffusivity: float = 3.0e-6
    shear_diffusivity: float = 3.0e-5
    buoyancy_diffusivity: float = 6.0e-5
    initial_sharpening_beta: float = 0.10
    vertical_decay_length: float = 0.14
    buoyancy_coupling: float = 0.42
    sorting_velocity_scale: float = 0.22
    rise_velocity_scale: float = 0.04
    pressure_iterations: int = 60

    seed: int = 12

    @property
    def dx(self) -> float:
        return self.Lx / self.Nx

    @property
    def dz(self) -> float:
        return self.Lz / self.Nz

    @property
    def zmin(self) -> float:
        return -0.5 * self.Lz

    @property
    def zmax(self) -> float:
        return 0.5 * self.Lz

    @property
    def domain_volume(self) -> float:
        return self.Lx * self.Lz

    @property
    def rho_light(self) -> float:
        return self.rho_mean * (1.0 - self.atwood)

    @property
    def rho_heavy(self) -> float:
        return self.rho_mean * (1.0 + self.atwood)

    @property
    def mixing_end_time(self) -> float:
        return self.accel_1_duration + self.decel_duration + self.accel_2_duration

    @property
    def total_time(self) -> float:
        return self.mixing_end_time + self.settling_duration

    @property
    def T(self) -> int:
        return int(np.ceil(self.total_time / self.dt))

    @property
    def k(self) -> float:
        return self.mode_number * np.pi / self.Lx


def build_grid(cfg: DyeConfig) -> dict[str, np.ndarray]:
    """Construct the square control-volume grid used by the dye solver."""
    if not np.isclose(cfg.dx, cfg.dz):
        raise ValueError("The current notebook assumes square pixels, so dx must equal dz.")

    x_edges = np.linspace(-0.5 * cfg.Lx, 0.5 * cfg.Lx, cfg.Nx + 1)
    z_edges = np.linspace(cfg.zmin, cfg.zmax, cfg.Nz + 1)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])

    return {
        "x_edges": x_edges,
        "z_edges": z_edges,
        "x_centers": x_centers,
        "z_centers": z_centers,
    }


# cfg = DyeConfig()
# grid = build_grid(cfg)


# %% [markdown]
# ## Module 2: Acceleration History and Density-Feedback Velocity Field
# 
# The paper still supplies the same `Accel-Decel-Accel` background profile with `+20`, `-30`, `+20 m s^-2` over `0.88 s`, `1.44 s`, `0.88 s`. What changes here is the closure: we no longer reverse or re-order the fluids by blending toward a hand-written target field.
# 
# The intended sequence is now:
# - during `accel_1`, the initial sinusoidal interface creates horizontal density gradients and the RT wave boundary grows,
# - during `decel`, the sign change in the imposed acceleration reverses the baroclinic circulation, so the earlier structures are pushed back, collide, and redistribute,
# - during `accel_2`, the already mixed layer is re-energised from its current state rather than from a clean single interface,
# - during `settling`, the remaining unstable density pockets continue to sort under the same buoyancy closure until a smoother top-light / bottom-heavy state appears.
# 

# %%
def phase_name(t: float, cfg: DyeConfig) -> str:
    """Return the active forcing phase for the dye-field surrogate."""
    if t < cfg.accel_1_duration:
        return "accel_1"
    if t < cfg.accel_1_duration + cfg.decel_duration:
        return "decel"
    if t < cfg.mixing_end_time:
        return "accel_2"
    if t < cfg.total_time:
        return "settling"
    return "done"


def transport_regime_name(t: float, cfg: DyeConfig) -> str:
    """Translate the phase label into a descriptive physical regime."""
    phase = phase_name(t, cfg)
    if phase == "accel_1":
        return "wave growth / bubble-spike formation"
    if phase == "decel":
        return "collapse and redistribution"
    if phase == "accel_2":
        return "mixed-layer re-acceleration"
    if phase == "settling":
        return "natural buoyancy sorting"
    return "done"


def acceleration_profile(t: float, cfg: DyeConfig) -> float:
    """Return the imposed vertical acceleration schedule used by the reduced-order model."""
    phase = phase_name(t, cfg)
    if phase == "accel_1":
        return cfg.unstable_accel
    if phase == "decel":
        return cfg.stable_accel
    if phase == "accel_2":
        return cfg.unstable_accel
    return 0.0


def growth_rate_from_acceleration(accel: float, cfg: DyeConfig) -> float:
    """Convert the signed acceleration into a growth or decay rate for flow strength."""
    if accel >= 0.0:
        return cfg.unstable_growth_rate * accel / cfg.unstable_accel
    return -cfg.stable_decay_rate * abs(accel) / abs(cfg.stable_accel)


def advance_flow_strength(flow_strength: float, t: float, cfg: DyeConfig) -> float:
    """Advance the global circulation-strength scalar by one time step."""
    phase = phase_name(t, cfg)
    if phase == "settling":
        updated = flow_strength * np.exp(-cfg.settling_decay_rate * cfg.dt)
        return float(np.clip(updated, 0.0, cfg.max_flow_strength))

    rate = growth_rate_from_acceleration(acceleration_profile(t, cfg), cfg)
    if rate >= 0.0:
        updated = flow_strength + cfg.dt * rate * (cfg.max_flow_strength - flow_strength)
    else:
        updated = flow_strength + cfg.dt * rate * flow_strength
    return float(np.clip(updated, cfg.min_flow_strength, cfg.max_flow_strength))


def wave_progress(t: float, cfg: DyeConfig) -> float:
    """Normalise time over the first acceleration phase."""
    return float(np.clip(t / max(cfg.accel_1_duration, cfg.dt), 0.0, 1.0))


def cell_centre_z_coordinates(cfg: DyeConfig) -> np.ndarray:
    """Return the one-dimensional vertical coordinates of cell centres."""
    return np.linspace(cfg.zmin + 0.5 * cfg.dz, cfg.zmax - 0.5 * cfg.dz, cfg.Nz)


def wall_damping_profile(cfg: DyeConfig) -> np.ndarray:
    """Reduce buoyancy drift close to the top and bottom boundaries."""
    z = cell_centre_z_coordinates(cfg)[:, None]
    return 0.25 + 0.75 * (1.0 - (2.0 * z / cfg.Lz) ** 2)


def central_gradient_x(phi: np.ndarray, cfg: DyeConfig) -> np.ndarray:
    """Compute the horizontal central difference with edge-value replication."""
    phi_pad = np.pad(phi, ((0, 0), (1, 1)), mode="edge")
    return (phi_pad[:, 2:] - phi_pad[:, :-2]) / (2.0 * cfg.dx)


def mixedness(phi: np.ndarray) -> float:
    """Measure how strongly the domain departs from a pure two-layer state."""
    return float(np.mean(4.0 * phi * (1.0 - phi)))


def effective_diffusivity(phi: np.ndarray, flow_strength: float, t: float, cfg: DyeConfig) -> float:
    """Assemble the reduced-order diffusivity used in the explicit update."""
    shear_level = float(np.mean(np.abs(central_gradient_x(phi, cfg))))
    unstable_jump = float(np.mean(np.clip(phi[:-1, :] - phi[1:, :], 0.0, 1.0)))
    drive_fraction = flow_strength / max(cfg.max_flow_strength, 1e-12)
    return (
        cfg.molecular_diffusivity
        + cfg.shear_diffusivity * drive_fraction * mixedness(phi)
        + cfg.buoyancy_diffusivity * unstable_jump
        + 0.15 * cfg.shear_diffusivity * shear_level * cfg.dx
    )


def sharpening_beta(phi: np.ndarray, t: float, cfg: DyeConfig) -> float:
    """Reduce interface sharpening as the domain becomes more mixed."""
    interface_energy = mixedness(phi)
    return cfg.initial_sharpening_beta * (1.0 - interface_energy) * max(0.0, 1.0 - 0.55 * wave_progress(t, cfg))


def solve_streamfunction(source: np.ndarray, cfg: DyeConfig) -> np.ndarray:
    """Solve the reduced-order streamfunction equation by Jacobi relaxation."""
    psi = np.zeros_like(source)
    coeff = 1.0 / (2.0 / cfg.dx**2 + 2.0 / cfg.dz**2)

    for _ in range(cfg.pressure_iterations):
        psi_new = psi.copy()
        psi_new[1:-1, 1:-1] = coeff * (
            (psi[1:-1, 2:] + psi[1:-1, :-2]) / cfg.dx**2
            + (psi[2:, 1:-1] + psi[:-2, 1:-1]) / cfg.dz**2
            + source[1:-1, 1:-1]
        )
        psi = psi_new

    return psi


def cell_center_velocities(phi: np.ndarray, t: float, flow_strength: float, cfg: DyeConfig) -> tuple[np.ndarray, np.ndarray]:
    """Convert the current dye field into centre-based reduced-order velocities."""
    accel_hat = acceleration_profile(t, cfg) / max(abs(cfg.unstable_accel), abs(cfg.stable_accel))
    baroclinic_source = cfg.buoyancy_coupling * flow_strength * accel_hat * central_gradient_x(phi, cfg)
    psi = solve_streamfunction(baroclinic_source, cfg)

    u = np.zeros_like(phi)
    w = np.zeros_like(phi)
    u[1:-1, :] = (psi[2:, :] - psi[:-2, :]) / (2.0 * cfg.dz)
    w[:, 1:-1] = -(psi[:, 2:] - psi[:, :-2]) / (2.0 * cfg.dx)

    z = cell_centre_z_coordinates(cfg)[:, None]
    vertical_decay = np.exp(-(z / max(cfg.vertical_decay_length, 1e-6)) ** 2)
    u *= vertical_decay
    w *= vertical_decay
    return u, w


# %% [markdown]
# ## Module 3: Wavy Initial Interface, Density-Driven Dye Transport, and Global Constraints
# 
# `phi(x, z, 0)` stores the local fraction of dyed low-density fluid. Instead of a perfectly flat interface, the initial boundary is a small sinusoid `eta(x, 0)`, so the early state already contains the horizontal density gradient needed to seed RT-type motion.
# 
# For every time step, the dye field is advanced through the same conservative five-term control-volume update:
# - left incoming face flux,
# - lower incoming face flux,
# - right outgoing face flux,
# - upper outgoing face flux,
# - `kappa_eff(phi, t) * dt * Laplacian(phi)`.
# 
# The upwind face fluxes follow
# 
# $$
# F = v_{\mathrm{face}} \, \phi_{\mathrm{upwind}} \, \Delta t \, \Delta x,
# $$
# 
# and the net flux is divided by the cell area `dx * dz` before it is added back to `phi`. This keeps the control-volume transport consistent with the grid geometry instead of suppressing the advection by a missing area normalisation.
# 
# The closure now evolves from the instantaneous state:
# - the current `phi` field sets the horizontal density gradients,
# - those gradients generate a baroclinic streamfunction rather than a prescribed target shape,
# - unstable light-below-heavy jumps contribute extra buoyancy sorting,
# - the interface sharpening is weak and fades as the layer becomes genuinely mixed, so the model can lose a clean interface when the physics says it should,
# - after the explicit update, a dedicated global constraint function projects the field back to the reference dyed-fluid content and the reference mixture density, which in turn keeps both total fluid mass and average density unchanged.
# 

# %%
def initial_interface_height(x: np.ndarray, cfg: DyeConfig) -> np.ndarray:
    """Return the cosine-perturbed initial interface height."""
    phase = cfg.k * (x + 0.5 * cfg.Lx)
    return cfg.initial_wave_amplitude * np.cos(phase)


def initial_low_density_fraction(cfg: DyeConfig, grid: dict[str, np.ndarray]) -> np.ndarray:
    """Populate each cell with the initial low-density-fluid fraction beneath the perturbed interface."""
    x_centres = grid["x_centers"]
    z_edges = grid["z_edges"]
    interface_height = initial_interface_height(x_centres, cfg)
    phi0 = np.zeros((cfg.Nz, cfg.Nx), dtype=np.float64)

    for column, eta in enumerate(interface_height):
        for row in range(cfg.Nz):
            z_lower = z_edges[row]
            z_upper = z_edges[row + 1]
            dyed_height = np.clip(min(z_upper, eta) - z_lower, 0.0, z_upper - z_lower)
            phi0[row, column] = dyed_height / (z_upper - z_lower)

    return phi0


def face_velocities(phi: np.ndarray, t: float, flow_strength: float, cfg: DyeConfig) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate centre velocities to faces and add buoyancy-driven vertical corrections."""
    u_center, w_center = cell_center_velocities(phi, t, flow_strength, cfg)
    buoyancy_drift = cfg.rise_velocity_scale * wall_damping_profile(cfg) * (phi - 0.5)

    u_faces = np.zeros((cfg.Nz, cfg.Nx + 1), dtype=np.float64)
    u_faces[:, 1:-1] = 0.5 * (u_center[:, :-1] + u_center[:, 1:])

    w_faces = np.zeros((cfg.Nz + 1, cfg.Nx), dtype=np.float64)
    w_faces[1:-1, :] = 0.5 * (w_center[:-1, :] + w_center[1:, :])
    w_faces[1:-1, :] += 0.5 * (buoyancy_drift[:-1, :] + buoyancy_drift[1:, :])

    unstable_jump = np.clip(phi[:-1, :] - phi[1:, :], 0.0, 1.0)
    w_faces[1:-1, :] += cfg.sorting_velocity_scale * flow_strength * unstable_jump

    u_faces[:, 0] = 0.0
    u_faces[:, -1] = 0.0
    w_faces[0, :] = 0.0
    w_faces[-1, :] = 0.0
    return u_faces, w_faces


def diffusion_laplacian(phi: np.ndarray, cfg: DyeConfig) -> np.ndarray:
    """Return the explicit diffusion Laplacian with replicated edge values."""
    phi_pad = np.pad(phi, ((1, 1), (1, 1)), mode="edge")
    lap_x = (phi_pad[1:-1, :-2] - 2.0 * phi + phi_pad[1:-1, 2:]) / cfg.dx**2
    lap_z = (phi_pad[:-2, 1:-1] - 2.0 * phi + phi_pad[2:, 1:-1]) / cfg.dz**2
    return lap_x + lap_z


def project_to_bounded_mass(phi_candidate: np.ndarray, target_sum: float) -> np.ndarray:
    """Project a candidate field into [0, 1] while matching the prescribed total dyed-fluid content."""
    phi_projected = np.clip(phi_candidate, 0.0, 1.0)

    for _ in range(8):
        delta = target_sum - float(np.sum(phi_projected))
        if abs(delta) < 1e-12:
            break

        if delta > 0.0:
            mask = phi_projected < 1.0 - 1e-12
            weights = 1.0 - phi_projected[mask]
        else:
            mask = phi_projected > 1e-12
            weights = phi_projected[mask]

        if weights.size == 0 or float(np.sum(weights)) <= 1e-12:
            break

        phi_projected[mask] += delta * weights / float(np.sum(weights))
        phi_projected = np.clip(phi_projected, 0.0, 1.0)

    return phi_projected


def density_field(phi: np.ndarray, cfg: DyeConfig) -> np.ndarray:
    """Map the dyed-fluid fraction to physical density."""
    return phi * cfg.rho_light + (1.0 - phi) * cfg.rho_heavy


def total_fluid_mass(phi: np.ndarray, cfg: DyeConfig) -> float:
    """Return the total mass contained in the 2D control-volume domain."""
    return float(np.sum(density_field(phi, cfg)) * cfg.dx * cfg.dz)


def average_density(phi: np.ndarray, cfg: DyeConfig) -> float:
    """Return the domain-average density implied by the current field."""
    return float(np.mean(density_field(phi, cfg)))


def target_phi_sum_from_invariants(
    target_total_mass: float,
    target_average_density: float,
    cfg: DyeConfig,
) -> float:
    """Infer the dyed-fluid inventory required by the mass and density invariants."""
    consistent_total_mass = target_average_density * cfg.domain_volume
    if not np.isclose(target_total_mass, consistent_total_mass, rtol=0.0, atol=1e-10):
        raise ValueError("For a fixed-volume domain, target total mass and target average density must be consistent.")

    mean_phi = (target_average_density - cfg.rho_heavy) / (cfg.rho_light - cfg.rho_heavy)
    return float(mean_phi * cfg.Nx * cfg.Nz)


def enforce_global_mass_and_density(
    phi_candidate: np.ndarray,
    target_phi_sum: float,
    target_total_mass: float,
    target_average_density: float,
    cfg: DyeConfig,
) -> np.ndarray:
    """Project the updated field back onto the global mass and average-density constraints."""
    density_implied_phi_sum = target_phi_sum_from_invariants(target_total_mass, target_average_density, cfg)
    if not np.isclose(target_phi_sum, density_implied_phi_sum, rtol=0.0, atol=1e-10):
        raise ValueError("The dyed-fluid content and density targets are inconsistent.")

    phi_projected = project_to_bounded_mass(phi_candidate, target_phi_sum)
    total_mass_error = target_total_mass - total_fluid_mass(phi_projected, cfg)
    density_error = target_average_density - average_density(phi_projected, cfg)

    if abs(total_mass_error) > 1e-12 or abs(density_error) > 1e-12:
        phi_projected = project_to_bounded_mass(phi_projected, density_implied_phi_sum)

    return np.clip(phi_projected, 0.0, 1.0)


def mass_preserving_sharpen(phi_candidate: np.ndarray, beta: float, target_sum: float) -> np.ndarray:
    """Apply interface sharpening without changing the dyed-fluid inventory."""
    if beta <= 1e-10:
        return project_to_bounded_mass(phi_candidate, target_sum)

    sharpened = phi_candidate + beta * (phi_candidate - 0.5) * phi_candidate * (1.0 - phi_candidate)
    return project_to_bounded_mass(np.clip(sharpened, 0.0, 1.0), target_sum)


def face_flux_increments(
    phi: np.ndarray,
    t: float,
    flow_strength: float,
    cfg: DyeConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute upwind flux increments across cell faces over one explicit time step."""
    u_faces, w_faces = face_velocities(phi, t, flow_strength, cfg)
    face_length = cfg.dx

    flux_x = np.zeros((cfg.Nz, cfg.Nx + 1), dtype=np.float64)
    u_inner = u_faces[:, 1:-1]
    flux_x[:, 1:-1] = (cfg.dt * face_length) * np.where(u_inner >= 0.0, u_inner * phi[:, :-1], u_inner * phi[:, 1:])

    flux_z = np.zeros((cfg.Nz + 1, cfg.Nx), dtype=np.float64)
    w_inner = w_faces[1:-1, :]
    flux_z[1:-1, :] = (cfg.dt * face_length) * np.where(w_inner >= 0.0, w_inner * phi[:-1, :], w_inner * phi[1:, :])
    return flux_x, flux_z, u_faces, w_faces


def conservative_upwind_step(
    phi: np.ndarray,
    t: float,
    flow_strength: float,
    cfg: DyeConfig,
    target_sum: float,
    target_total_mass: float,
    target_average_density: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Advance the dye field with conservative advection, diffusion, sharpening, and invariant projection."""
    flux_x, flux_z, u_faces, w_faces = face_flux_increments(phi, t, flow_strength, cfg)
    kappa = effective_diffusivity(phi, flow_strength, t, cfg)
    beta = sharpening_beta(phi, t, cfg)
    laplacian_phi = diffusion_laplacian(phi, cfg)
    cell_area = cfg.dx * cfg.dz

    phi_candidate = (
        phi
        + (flux_x[:, :-1] + flux_z[:-1, :] - flux_x[:, 1:] - flux_z[1:, :]) / cell_area
        + kappa * cfg.dt * laplacian_phi
    )
    phi_candidate = np.clip(phi_candidate, 0.0, 1.0)
    phi_new = mass_preserving_sharpen(phi_candidate, beta, target_sum)
    phi_new = enforce_global_mass_and_density(
        phi_new,
        target_sum,
        target_total_mass,
        target_average_density,
        cfg,
    )
    return phi_new, flux_x, flux_z, u_faces, w_faces, kappa


def cfl_number(u_faces: np.ndarray, w_faces: np.ndarray, kappa: float, cfg: DyeConfig) -> float:
    """Report the largest explicit stability indicator among advection and diffusion terms."""
    adv_x = np.max(np.abs(u_faces)) * cfg.dt / cfg.dx
    adv_z = np.max(np.abs(w_faces)) * cfg.dt / cfg.dz
    diff_scale = kappa * cfg.dt * (1.0 / cfg.dx**2 + 1.0 / cfg.dz**2)
    return float(max(adv_x, adv_z, diff_scale))


# %% [markdown]
# ## Module 4: Rollout and Dye-Field Diagnostics
# 
# To stay close to the paper, we monitor the horizontally averaged volume fraction $\bar{\phi}(z, t)$ and the interface-width proxy
# 
# $$
# h(t) = 6 \int \bar{\phi}(z, t) \left(1 - \bar{\phi}(z, t)\right) \, dz.
# $$
# 
# In the present density-feedback closure, `h(t)` is interpreted as:
# - early RT wave growth during the first acceleration stage,
# - then a broader mixed layer as deceleration collapses and redistributes the original wave boundary,
# - then the natural inversion of the mixed layer as lighter parcels rise and heavier parcels sink,
# - and finally a smoother mixed profile with recovered top-light / bottom-heavy ordering.
# 
# We also track:
# - total dyed-fluid mass, to keep the colour-field transport conservative,
# - total fluid mass and average density, which are explicitly re-imposed after each step by the global constraint projector,
# - the relative drift of those two invariants with respect to their initial values,
# - a colour-gradient index,
# 
# $$
# C(t) = \bar{\phi}(z_{\mathrm{top}}, t) - \bar{\phi}(z_{\mathrm{bottom}}, t),
# $$
# 
# which becomes positive once the inversion has really emerged from the flow,
# - the fraction of cells with `0 < phi < 1`, which measures how much of the domain is in a mixed state,
# - the state-dependent effective diffusivity, which rises with mixedness and unstable density jumps rather than with a prescribed time ramp.
# 

# %%
def horizontal_average(phi: np.ndarray) -> np.ndarray:
    """Average the dye field over x to obtain a vertical mean profile."""
    return phi.mean(axis=1)


def mixing_width(phi: np.ndarray, cfg: DyeConfig, grid: dict[str, np.ndarray]) -> float:
    """Compute the standard integral estimate of the mixed-layer thickness."""
    phi_bar = horizontal_average(phi)
    return 6.0 * integrate_trapezoid(phi_bar * (1.0 - phi_bar), grid["z_centers"])


def dyed_mass(phi: np.ndarray, cfg: DyeConfig) -> float:
    """Return the total dyed-fluid content in the 2D surrogate domain."""
    return float(np.sum(phi) * cfg.dx * cfg.dz)


def colour_gradient_index(phi: np.ndarray) -> float:
    """Measure whether the vertical mean profile is lighter at the top or at the bottom."""
    phi_bar = horizontal_average(phi)
    return float(phi_bar[-1] - phi_bar[0])


def interfacial_cell_fraction(phi: np.ndarray) -> float:
    """Estimate how much of the domain contains partially mixed cells."""
    mask = (phi > 1e-3) & (phi < 1.0 - 1e-3)
    return float(np.mean(mask))


def relative_drift(series: np.ndarray) -> np.ndarray:
    """Normalise the drift of a time series relative to its initial value."""
    reference = series[0]
    return (series - reference) / max(abs(reference), 1e-12)


def sampled_history_time(t: float, cfg: DyeConfig) -> float:
    """Clamp history queries to the active time horizon used by auxiliary closures."""
    return min(max(t, 0.0), cfg.total_time - 1e-12)


def dye_history_entry(
    phi: np.ndarray,
    t: float,
    flow_strength: float,
    cfg: DyeConfig,
    grid: dict[str, np.ndarray],
) -> dict[str, np.ndarray | float]:
    """Collect one complete diagnostic snapshot for the dye-field model."""
    sampled_t = sampled_history_time(t, cfg)
    return {
        "time": float(t),
        "phi": phi.astype(np.float32),
        "phi_bar": horizontal_average(phi).astype(np.float32),
        "mixing_width": mixing_width(phi, cfg, grid),
        "dyed_mass": dyed_mass(phi, cfg),
        "total_fluid_mass": total_fluid_mass(phi, cfg),
        "average_density": average_density(phi, cfg),
        "acceleration": acceleration_profile(sampled_t, cfg),
        "flow_strength": float(flow_strength),
        "effective_diffusivity": effective_diffusivity(phi, flow_strength, sampled_t, cfg),
        "sharpening_beta": sharpening_beta(phi, sampled_t, cfg),
        "cfl": 0.0,
        "colour_gradient_index": colour_gradient_index(phi),
        "interfacial_cell_fraction": interfacial_cell_fraction(phi),
    }


def initialise_dye_history(
    phi: np.ndarray,
    flow_strength: float,
    cfg: DyeConfig,
    grid: dict[str, np.ndarray],
) -> dict[str, list[np.ndarray | float]]:
    """Initialise the history dictionary with the t = 0 state."""
    entry = dye_history_entry(phi, 0.0, flow_strength, cfg, grid)
    return {key: [value] for key, value in entry.items()}


def append_dye_history_entry(
    history: dict[str, list[np.ndarray | float]],
    phi: np.ndarray,
    t: float,
    flow_strength: float,
    cfl: float,
    cfg: DyeConfig,
    grid: dict[str, np.ndarray],
) -> None:
    """Append one new time step of dye diagnostics to the history container."""
    entry = dye_history_entry(phi, t, flow_strength, cfg, grid)
    entry["cfl"] = cfl
    for key, value in entry.items():
        history[key].append(value)


def final_regime_indices(history: dict[str, np.ndarray], cfg: DyeConfig) -> dict[str, int]:
    """Locate representative snapshots for the main dynamical regimes."""
    wave_mask = history["time"] <= cfg.accel_1_duration + 1e-12
    wave_idx = int(np.argmax(np.where(wave_mask, history["mixing_width"], -np.inf)))
    redistribution_time = cfg.mixing_end_time + 0.25 * cfg.settling_duration
    sorting_time = cfg.mixing_end_time + 0.55 * cfg.settling_duration
    redistribution_idx = int(np.argmin(np.abs(history["time"] - redistribution_time)))
    sorting_idx = int(np.argmin(np.abs(history["time"] - sorting_time)))
    return {
        "wave_idx": wave_idx,
        "redistribution_idx": redistribution_idx,
        "sorting_idx": sorting_idx,
    }


def build_dye_summary(history: dict[str, np.ndarray], cfg: DyeConfig, max_cfl: float) -> dict[str, float]:
    """Assemble a compact summary of the rolled-out dye evolution."""
    regime_indices = final_regime_indices(history, cfg)
    return {
        "final_mixing_width": float(history["mixing_width"][-1]),
        "peak_mixing_width": float(np.max(history["mixing_width"])),
        "relative_mass_change": float((history["dyed_mass"][-1] - history["dyed_mass"][0]) / history["dyed_mass"][0]),
        "relative_total_fluid_mass_change": float(history["relative_total_fluid_mass_drift"][-1]),
        "relative_average_density_change": float(history["relative_average_density_drift"][-1]),
        "max_total_fluid_mass_drift": float(np.max(np.abs(history["relative_total_fluid_mass_drift"]))),
        "max_average_density_drift": float(np.max(np.abs(history["relative_average_density_drift"]))),
        "max_cfl": float(max_cfl),
        "final_flow_strength": float(history["flow_strength"][-1]),
        "final_colour_gradient_index": float(history["colour_gradient_index"][-1]),
        "final_interfacial_cell_fraction": float(history["interfacial_cell_fraction"][-1]),
        "wave_idx": regime_indices["wave_idx"],
        "redistribution_idx": regime_indices["redistribution_idx"],
        "sorting_idx": regime_indices["sorting_idx"],
        "final_top_phi": float(history["phi_bar"][-1, -1]),
        "final_bottom_phi": float(history["phi_bar"][-1, 0]),
        "final_average_density": float(history["average_density"][-1]),
        "final_total_fluid_mass": float(history["total_fluid_mass"][-1]),
    }


def simulate_dye_evolution(cfg: DyeConfig, grid: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Roll out the reduced-order dye-field model and record all diagnostics needed later in the notebook."""
    phi = initial_low_density_fraction(cfg, grid)
    target_phi_sum = float(np.sum(phi))
    reference_total_mass = total_fluid_mass(phi, cfg)
    reference_average_density = average_density(phi, cfg)
    flow_strength = cfg.initial_flow_strength
    time_value = 0.0
    max_cfl = 0.0

    history = initialise_dye_history(phi, flow_strength, cfg, grid)

    for _ in range(cfg.T):
        flow_strength = advance_flow_strength(flow_strength, time_value, cfg)
        phi, flux_x, flux_z, u_faces, w_faces, kappa = conservative_upwind_step(
            phi,
            time_value,
            flow_strength,
            cfg,
            target_phi_sum,
            reference_total_mass,
            reference_average_density,
        )
        time_value += cfg.dt

        cfl = cfl_number(u_faces, w_faces, kappa, cfg)
        max_cfl = max(max_cfl, cfl)
        append_dye_history_entry(history, phi, time_value, flow_strength, cfl, cfg, grid)

    history = {key: np.array(value) for key, value in history.items()}
    history["relative_total_fluid_mass_drift"] = relative_drift(history["total_fluid_mass"])
    history["relative_average_density_drift"] = relative_drift(history["average_density"])

    summary = build_dye_summary(history, cfg, max_cfl)
    return history, summary


# history, summary = simulate_dye_evolution(cfg, grid)


# %% [markdown]
# ## Module 5: Two-Colour Visualisation of the Dye Field
# 
# The colour map still distinguishes the two fluids clearly:
# - orange: mostly upper dense sodium chloride fluid (`phi` close to `0`),
# - blue: mostly lower dyed low-density fluid (`phi` close to `1`).
# 
# To make both directions of motion readable, each snapshot now shows three layers at once:
# - the dye fraction field itself,
# - dark-blue and dark-orange contours marking light-fluid-rich and heavy-fluid-rich cores,
# - sparse velocity arrows derived from the current reduced-order flow field.
# 
# The snapshot set is arranged to show a natural progression:
# - an initial RT-type wavy interface,
# - the amplified wave boundary during the first acceleration stage,
# - the redistributed mixed layer after the collapse of that boundary,
# - the later density inversion driven by buoyancy sorting,
# - and the final mixed layer with a recovered top-light / bottom-heavy ordering.
# 
# The diagnostics underneath follow the same storyline: wave growth first, then redistribution and mixing, then the eventual sign change of the colour gradient once the inversion has formed from the evolving density field itself.
# 

# %%
two_fluid_cmap = LinearSegmentedColormap.from_list(
    "two_fluid",
    ["#d97706", "#f3d9b1", "#b7d4ea", "#1d4e89"],
)


def velocity_overlay_fields(
    phi: np.ndarray,
    t: float,
    flow_strength: float,
    cfg: DyeConfig,
    grid: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return centre coordinates and overlay velocities for a snapshot plot."""
    u_faces, w_faces = face_velocities(phi.astype(np.float64), t, flow_strength, cfg)
    u_centres = 0.5 * (u_faces[:, 1:] + u_faces[:, :-1])
    w_centres = 0.5 * (w_faces[1:, :] + w_faces[:-1, :])
    X, Z = np.meshgrid(grid["x_centers"], grid["z_centers"])
    return X, Z, u_centres, w_centres


def add_snapshot_overlays(
    ax: Axes,
    phi: np.ndarray,
    t: float,
    flow_strength: float,
    cfg: DyeConfig,
    grid: dict[str, np.ndarray],
) -> None:
    """Draw core contours and velocity arrows on top of a dye-field snapshot."""
    X, Z, u_centres, w_centres = velocity_overlay_fields(phi, t, flow_strength, cfg, grid)

    ax.contour(X, Z, phi, levels=[0.20], colors=["#7c2d12"], linewidths=1.2, alpha=0.95)
    ax.contour(X, Z, phi, levels=[0.80], colors=["#1e3a8a"], linewidths=1.2, alpha=0.95)

    stride_z = 3
    stride_x = 2
    speed = np.sqrt(u_centres**2 + w_centres**2)
    speed_ref = max(float(np.max(speed)), 1e-12)
    mask = speed[::stride_z, ::stride_x] > 0.08 * speed_ref

    X_sample = X[::stride_z, ::stride_x]
    Z_sample = Z[::stride_z, ::stride_x]
    U_sample = u_centres[::stride_z, ::stride_x]
    W_sample = w_centres[::stride_z, ::stride_x]

    ax.quiver(
        X_sample[mask],
        Z_sample[mask],
        U_sample[mask],
        W_sample[mask],
        color="#111827",
        alpha=0.70,
        angles="xy",
        scale_units="xy",
        scale=3.4,
        width=0.004,
        pivot="mid",
    )


def draw_phase_break_lines(ax: Axes, phase_breaks: list[float], alpha: float = 0.35) -> None:
    """Add vertical markers that separate the imposed forcing phases."""
    for tau in phase_breaks:
        ax.axvline(tau, color="black", linestyle="--", linewidth=1.0, alpha=alpha)


def plot_snapshot_panel(
    ax: Axes,
    phi: np.ndarray,
    t: float,
    flow_strength: float,
    title: str,
    cfg: DyeConfig,
    grid: dict[str, np.ndarray],
    extent: list[float],
    cmap: LinearSegmentedColormap,
    annotation_text: str | None = None,
):
    """Plot one dye snapshot together with contour and velocity overlays."""
    image = ax.imshow(
        phi,
        origin="lower",
        extent=extent,
        vmin=0.0,
        vmax=1.0,
        cmap=cmap,
        aspect="equal",
    )
    add_snapshot_overlays(ax, phi, t, flow_strength, cfg, grid)
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.axhline(0.0, color="white", linestyle="--", linewidth=1.0, alpha=0.8)

    if annotation_text is not None:
        ax.text(
            0.03,
            0.97,
            annotation_text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8.5,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.88),
        )
    return image


# wave_idx = int(summary["wave_idx"])
# redistribution_idx = int(summary["redistribution_idx"])
# sorting_idx = int(summary["sorting_idx"])
# extent = [-0.5 * cfg.Lx, 0.5 * cfg.Lx, cfg.zmin, cfg.zmax]
# phase_breaks = [cfg.accel_1_duration, cfg.accel_1_duration + cfg.decel_duration, cfg.mixing_end_time]
# phi_bar_map = history["phi_bar"].T

# fig, axes = plt.subplots(2, 4, figsize=(19.5, 9.4), constrained_layout=True)

# im0 = plot_snapshot_panel(
#     axes[0, 0],
#     history["phi"][0],
#     history["time"][0],
#     history["flow_strength"][0],
#     "Initial RT-Type Interface",
#     cfg,
#     grid,
#     extent,
#     two_fluid_cmap,
#     "orange core contour + blue core contour + velocity arrows",
# )

# plot_snapshot_panel(
#     axes[0, 1],
#     history["phi"][wave_idx],
#     history["time"][wave_idx],
#     history["flow_strength"][wave_idx],
#     f"Wave Boundary Growth (t = {history['time'][wave_idx]:.2f} s)",
#     cfg,
#     grid,
#     extent,
#     two_fluid_cmap,
# )

# plot_snapshot_panel(
#     axes[0, 2],
#     history["phi"][redistribution_idx],
#     history["time"][redistribution_idx],
#     history["flow_strength"][redistribution_idx],
#     f"Redistributed Mixed Layer (t = {history['time'][redistribution_idx]:.2f} s)",
#     cfg,
#     grid,
#     extent,
#     two_fluid_cmap,
# )

# plot_snapshot_panel(
#     axes[0, 3],
#     history["phi"][sorting_idx],
#     history["time"][sorting_idx],
#     history["flow_strength"][sorting_idx],
#     f"Natural Density Inversion (t = {history['time'][sorting_idx]:.2f} s)",
#     cfg,
#     grid,
#     extent,
#     two_fluid_cmap,
# )

# axes[1, 0].imshow(
#     phi_bar_map,
#     origin="lower",
#     extent=[history["time"][0], history["time"][-1], cfg.zmin, cfg.zmax],
#     aspect="auto",
#     vmin=0.0,
#     vmax=1.0,
#     cmap=two_fluid_cmap,
# )
# axes[1, 0].set_title(r"Horizontally Averaged $\bar{\phi}(z,t)$")
# axes[1, 0].set_xlabel("time (s)")
# axes[1, 0].set_ylabel("z (m)")
# draw_phase_break_lines(axes[1, 0], phase_breaks, alpha=0.5)

# axes[1, 1].plot(history["time"], history["mixing_width"], color="#7c3aed", linewidth=2.4, label="$h(t)$")
# axes[1, 1].scatter(history["time"][wave_idx], history["mixing_width"][wave_idx], color="#111827", s=38, zorder=5, label="wave boundary")
# axes[1, 1].scatter(history["time"][redistribution_idx], history["mixing_width"][redistribution_idx], color="#0f766e", s=38, zorder=5, label="redistributed layer")
# axes[1, 1].scatter(history["time"][sorting_idx], history["mixing_width"][sorting_idx], color="#b91c1c", s=38, zorder=5, label="density inversion")
# draw_phase_break_lines(axes[1, 1], phase_breaks)
# axes[1, 1].set_title("Interface / Mixing Width $h(t)$")
# axes[1, 1].set_xlabel("time (s)")
# axes[1, 1].set_ylabel("$h(t)$ (m)")
# axes[1, 1].grid(alpha=0.25)
# axes[1, 1].legend(loc="best")

# ax_left = axes[1, 2]
# ax_right = ax_left.twinx()
# line1 = ax_left.plot(history["time"], history["colour_gradient_index"], color="#0f766e", linestyle="--", linewidth=2.0, label="colour-gradient index")
# line2 = ax_right.plot(history["time"], history["interfacial_cell_fraction"], color="#b91c1c", linestyle=":", linewidth=2.0, label="0 < phi < 1 fraction")
# line3 = ax_right.plot(
#     history["time"],
#     history["effective_diffusivity"] / max(np.max(history["effective_diffusivity"]), 1e-12),
#     color="#1d4ed8",
#     linestyle="-.",
#     linewidth=1.8,
#     label=r"$\kappa_{\mathrm{eff}} / \kappa_{\max}$",
# )
# draw_phase_break_lines(ax_left, phase_breaks)
# ax_left.set_title("Other Diagnostics")
# ax_left.set_xlabel("time (s)")
# ax_left.set_ylabel("colour-gradient index", color="#0f766e")
# ax_right.set_ylabel("mixed-cell / normalised diffusion", color="#b91c1c")
# ax_left.set_ylim(-1.05, 1.05)
# ax_right.set_ylim(-0.05, 1.05)
# ax_left.grid(alpha=0.25)
# lines = line1 + line2 + line3
# labels = [line.get_label() for line in lines]
# ax_left.legend(lines, labels, loc="best")
# ax_left.text(
#     0.03,
#     0.05,
#     f"max mass drift = {summary['max_total_fluid_mass_drift']:.2e}\nmax density drift = {summary['max_average_density_drift']:.2e}",
#     transform=ax_left.transAxes,
#     fontsize=9,
#     bbox=dict(boxstyle="round", facecolor="white", alpha=0.88),
# )

# plot_snapshot_panel(
#     axes[1, 3],
#     history["phi"][-1],
#     history["time"][-1],
#     history["flow_strength"][-1],
#     "Final Mixed Stable Layer",
#     cfg,
#     grid,
#     extent,
#     two_fluid_cmap,
# )
# axes[1, 3].text(
#     0.03,
#     0.97,
#     f"rho_avg = {summary['final_average_density']:.2f} kg m^-3\nM = {summary['final_total_fluid_mass']:.2f} kg m^-1",
#     transform=axes[1, 3].transAxes,
#     va="top",
#     ha="left",
#     fontsize=9,
#     bbox=dict(boxstyle="round", facecolor="white", alpha=0.88),
# )

# cbar = fig.colorbar(im0, ax=[axes[0, 0], axes[0, 1], axes[0, 2], axes[0, 3], axes[1, 3]], shrink=0.92)
# cbar.set_label(r"dyed-fluid fraction $\phi$")

# plt.show()


# %% [markdown]
# ## Module 6: GIF Visualisation (Vibe Coding Part)
# 
# This animation now follows the same density-feedback storyline as the main plots:
# - first the RT wave boundary grows,
# - then deceleration redistributes and broadens the mixed layer,
# - then the sign of the colour gradient changes because the lighter fluid rises and the heavier fluid sinks,
# - the frame overlays keep the heavy-fluid descent visible by showing core contours and instantaneous velocity arrows,
# - and finally the system relaxes toward a smoother mixed layer with recovered stable ordering.
# 

# %%
# Module 6: build and display an in-memory GIF for the dye field
def build_dye_gif(
    history: dict[str, np.ndarray],
    cfg: DyeConfig,
    cmap: LinearSegmentedColormap,
    grid: dict[str, np.ndarray],
    stride: int = 6,
    frame_duration: float = 0.12,
) -> Image:
    """Render the dye-field history into a compact in-memory GIF with diagnostics."""
    frames = []
    extent = [-0.5 * cfg.Lx, 0.5 * cfg.Lx, cfg.zmin, cfg.zmax]
    reference_total_mass = history["total_fluid_mass"][0]

    for idx in gif_frame_indices(len(history["time"]), stride):
        phi = history["phi"][idx]
        time_value = history["time"][idx]
        width = history["mixing_width"][idx]
        gradient = history["colour_gradient_index"][idx]
        mixed_fraction = history["interfacial_cell_fraction"][idx]
        regime = transport_regime_name(time_value, cfg)
        kappa_eff = history["effective_diffusivity"][idx]
        rho_avg = history["average_density"][idx]
        relative_mass_drift = (history["total_fluid_mass"][idx] - reference_total_mass) / max(reference_total_mass, 1e-12)

        fig, ax = plt.subplots(figsize=(5.9, 4.9), constrained_layout=True)
        image = ax.imshow(
            phi,
            origin="lower",
            extent=extent,
            vmin=0.0,
            vmax=1.0,
            cmap=cmap,
            aspect="equal",
        )
        add_snapshot_overlays(ax, phi, time_value, history["flow_strength"][idx], cfg, grid)
        ax.set_title(f"Dye Field Evolution, t = {time_value:.2f} s")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("z (m)")
        ax.axhline(0.0, color="white", linestyle="--", linewidth=1.0, alpha=0.8)
        ax.text(
            0.02,
            0.98,
            f"regime = {regime}\nwidth = {width:.4f} m\ncolour gradient = {gradient:.2f}\n0 < phi < 1 fraction = {mixed_fraction:.3f}\nkappa_eff = {kappa_eff:.2e}\nrho_avg = {rho_avg:.2f} kg m^-3\nrelative mass drift = {relative_mass_drift:.2e}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.88),
        )
        cbar = fig.colorbar(image, ax=ax, shrink=0.9)
        cbar.set_label("dyed-fluid fraction phi")

        frames.append(rgb_frame_from_figure(fig))
        plt.close(fig)

    return gif_image_from_frames(frames, frame_duration)


# dye_gif = build_dye_gif(history, cfg, two_fluid_cmap, grid)
# dye_gif


# %% [markdown]
# ## Module 7: Density Matrix Output at Every Second
# 
# For downstream analysis we now export a per-pixel density matrix at every integer second.
# 
# - Each matrix entry is a **normalised density** in `[0, 1]`.
# - `0` means the light-fluid density bound.
# - `1` means the heavy-fluid density bound.
# - The **first row of each matrix is the physical top** of the domain, so at `t = 0 s` the upper region starts near `1` and the lower region starts near `0`.
# - The matrices are both printed here and saved as `.csv` files under `output/density_matrices/`.
# 

# %%
from pathlib import Path


def normalised_density_matrix(phi: np.ndarray, cfg: DyeConfig) -> np.ndarray:
    """Convert the dye fraction to a normalised density matrix in [0, 1]."""
    rho = density_field(phi.astype(np.float64), cfg)
    rho_span = max(cfg.rho_heavy - cfg.rho_light, 1e-12)
    rho_norm = (rho - cfg.rho_light) / rho_span
    return np.clip(rho_norm, 0.0, 1.0).astype(np.float32)


def matrix_with_top_row_first(matrix: np.ndarray) -> np.ndarray:
    """Flip the exported matrix so the first row corresponds to the physical top boundary."""
    return np.flipud(matrix).astype(np.float32)


def density_matrices_each_second(
    history: dict[str, np.ndarray],
    cfg: DyeConfig,
) -> tuple[dict[int, np.ndarray], dict[int, float]]:
    """Sample the simulated density field at integer seconds and return matrices plus sampled times."""
    time_values = np.asarray(history["time"], dtype=np.float64)
    final_second = int(np.floor(time_values[-1] + 1e-12))
    second_marks = np.arange(0, final_second + 1, dtype=int)

    matrices: dict[int, np.ndarray] = {}
    sampled_times: dict[int, float] = {}

    for second in second_marks:
        idx = int(np.argmin(np.abs(time_values - float(second))))
        rho_matrix = normalised_density_matrix(history["phi"][idx], cfg)
        matrices[int(second)] = matrix_with_top_row_first(rho_matrix)
        sampled_times[int(second)] = float(time_values[idx])

    return matrices, sampled_times


def export_density_matrices(
    matrices: dict[int, np.ndarray],
    output_dir: Path,
) -> tuple[list[dict[str, str]], Path]:
    """Write each sampled density matrix to CSV and to one compressed NPZ archive."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, str]] = []
    archive_payload: dict[str, np.ndarray] = {}

    for second, matrix in matrices.items():
        csv_path = output_dir / f"density_matrix_t_{second:02d}s.csv"
        np.savetxt(csv_path, matrix, delimiter=",", fmt="%.6f")
        archive_payload[f"t_{second:02d}s"] = matrix
        manifest.append({
            "second": str(second),
            "csv_path": str(csv_path),
            "shape": str(matrix.shape),
        })

    archive_path = output_dir / "density_matrices_by_second.npz"
    np.savez_compressed(str(archive_path), **archive_payload)
    return manifest, archive_path


def print_exported_density_matrices(
    matrices: dict[int, np.ndarray],
    sampled_times: dict[int, float],
) -> None:
    """Print the exported integer-second matrices with a compact time summary."""
    with np.printoptions(precision=4, suppress=True, linewidth=180):
        for second, matrix in matrices.items():
            sampled_time = sampled_times[second]
            print(f"\nt = {second:02d} s (sampled at simulation time {sampled_time:.2f} s)")
            print(matrix)


# density_matrices_by_second, density_matrix_sample_times = density_matrices_each_second(history, cfg)
# density_matrix_manifest, density_matrix_archive = export_density_matrices(
#     density_matrices_by_second,
#     Path("output/density_matrices"),
# )

# print(f"Exported {len(density_matrix_manifest)} density matrices.")
# print(f"CSV directory: {Path('output/density_matrices').resolve()}")
# print(f"Compressed archive: {density_matrix_archive.resolve()}")
# print_exported_density_matrices(density_matrices_by_second, density_matrix_sample_times)


# %% [markdown]
# ---
# ## Module 8: Differentiable PyTorch Advect Operator
# 
# This section adds a **fully differentiable one-step advection operator** written in
# PyTorch. The purpose is to create a function
# 
# $$
# \rho_{t+\Delta t_{\mathrm{img}}} = \mathcal{A}_{\theta}(\rho_t),
# $$
# 
# where:
# 
# - $\rho_t$ is the density field at the current image time,
# - $\Delta t_{\mathrm{img}}$ is the same time interval as the experimental image spacing,
# - $\mathcal{A}_{\theta}$ is a parameterised, differentiable update rule,
# - and $\theta$ collects the learnable transport parameters.
# 
# The key design goal is **autograd compatibility**. Every operation is implemented with
# `torch` tensors so that gradients can flow through the full mapping from the current
# density field to the next one.
# 
# In the current notebook the primary state variable is the low-density-fluid fraction
# $\phi \in [0, 1]$, whereas image-based data are often stored as a **normalised density
# field**. To make the PyTorch operator easier to use for optimisation against images,
# the public interface of the new module accepts the normalised density field
# 
# $$
# \rho_n = \frac{\rho - \rho_{\mathrm{light}}}{\rho_{\mathrm{heavy}} - \rho_{\mathrm{light}}}
# \in [0,1].
# $$
# 
# With this convention:
# 
# - $\rho_n = 0$ means light fluid,
# - $\rho_n = 1$ means heavy fluid,
# - and the existing notebook variable is recovered from
# 
# $$
# \phi = 1 - \rho_n.
# $$
# 

# %% [markdown]
# ## Module 9: Mathematical Structure of the One-Step Operator
# 
# The new PyTorch operator mirrors the reduced-order RT-like closure already used in the
# notebook, but wraps it into a single differentiable map.
# 
# ### 1. Density-to-velocity coupling
# 
# The operator first converts the normalised density field into the notebook variable:
# 
# $$
# \phi_t = 1 - \rho_{n,t}.
# $$
# 
# A horizontal density gradient then generates a baroclinic source term:
# 
# $$
# S_{\psi}
# =
# C_b \, f \, \hat{a} \, \partial_x \phi_t,
# $$
# 
# where $C_b$ is the buoyancy-coupling coefficient, $f$ is the flow-strength amplitude,
# and $\hat{a}$ is the signed acceleration normalised by the reference experiment scale.
# 
# ### 2. Streamfunction closure
# 
# The source term drives a streamfunction solve:
# 
# $$
# \nabla^2 \psi = S_{\psi}.
# $$
# 
# Once $\psi$ is available, centre velocities are constructed as
# 
# $$
# u_c = d(z)\,\frac{\partial \psi}{\partial z},
# \qquad
# w_c = -d(z)\,\frac{\partial \psi}{\partial x},
# $$
# 
# with the vertical decay factor
# 
# $$
# d(z) = \exp\!\left[-\left(\frac{z}{L_d}\right)^2\right].
# $$
# 
# The vertical transport also includes a buoyancy drift and a local sorting correction:
# 
# $$
# w_{\mathrm{drift}} = C_r\,g_w(z)\,(\phi_t - 0.5),
# $$
# 
# $$
# w_{\mathrm{sorting}}
# =
# C_s\,f\,\max\!\left(\phi_{\mathrm{below}} - \phi_{\mathrm{above}},\,0\right).
# $$
# 
# ### 3. Differentiable one-step advection
# 
# For a single image interval $\Delta t_{\mathrm{img}}$, a semi-Lagrangian backtrace is used:
# 
# $$
# \phi_{t+\Delta t}^{\mathrm{adv}}(x,z)
# =
# \phi_t\!\left(x-u\,\Delta t,\; z-w\,\Delta t\right).
# $$
# 
# This is implemented with `torch.nn.functional.grid_sample`, which keeps the one-step map
# differentiable with respect to both the input field and the model parameters.
# 
# ### 4. Diffusive correction and output reconstruction
# 
# A state-dependent diffusive correction is then applied:
# 
# $$
# \phi_{t+\Delta t}
# =
# \mathrm{clip}\!\left(
# \phi_{t+\Delta t}^{\mathrm{adv}}
# +
# \Delta t\,
# \kappa_{\mathrm{eff}}(\phi_t)\,\nabla^2 \phi_t,\,
# 0,\,
# 1
# \right).
# $$
# 
# Finally the operator returns the normalised density field expected by image-based
# optimisation:
# 
# $$
# \rho_{n,t+\Delta t} = 1 - \phi_{t+\Delta t}.
# $$
# 
# The complete one-step operator is therefore
# 
# $$
# \rho_{n,t+\Delta t}
# =
# \mathcal{A}_{\theta}\!\left(\rho_{n,t}\right),
# $$
# 
# and can be inserted directly into a differentiable loss such as
# 
# $$
# \mathcal{L}(\theta)
# =
# \left\|
# \mathcal{A}_{\theta}(\rho_{n,t})
# -
# \rho_{n,t+\Delta t}^{\mathrm{image}}
# \right\|_2^2.
# $$
# 

# %% [markdown]
# ## Module 10: PyTorch Implementation of the Differentiable One-Step Advect Operator
# 
# This module converts the reduced-order RT-like update into a **structured PyTorch implementation**.
# Instead of placing the full implementation in one large code cell, the operator is divided into
# small functional blocks so that each part of the model can be read, checked, and modified more easily.
# 
# The implementation below is organised into five parts:
# 
# 1. helper functions and tensor-layout preparation,
# 2. base class construction and parameter registration,
# 3. discrete differential operators and elliptic closure,
# 4. transport-velocity construction,
# 5. one-step advection, diffusion, and public interface.
# 

# %% [markdown]
# ### Module 10.1: Helper Functions and Tensor Layout
# 
# The public state variable of the optimisation model is the normalised density field
# $\rho_n \in [0,1]$. The notebook closure is still expressed in terms of the low-density-fluid
# fraction $\phi$, so the first relation used by the implementation is
# 
# $$
# \phi = 1 - \rho_n.
# $$
# 
# The PyTorch code also needs a consistent tensor layout. A single image field is naturally stored
# as $[H, W]$, but differentiable image sampling with `grid_sample` expects a batch-channel layout,
# so the helper conversion maps
# 
# $$
# [H, W] \mapsto [1, 1, H, W].
# $$
# 
# The inverse conversion is applied at the end of the update so that the returned field has the same
# shape as the original input.
# 

# %%
# Module 10.1: imports and helper functions used by the differentiable operator
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _inverse_softplus(value):
    """Map a positive scalar into the raw parameter space used before softplus."""
    value = max(float(value), 1.0e-12)
    return float(np.log(np.expm1(value)))


def _to_bchw(field):
    """Convert a field to [B, C, H, W] layout for differentiable image-style processing."""
    if field.ndim == 2:
        return field.unsqueeze(0).unsqueeze(0), 2
    if field.ndim == 4 and field.shape[1] == 1:
        return field, 4
    raise ValueError("Expected a density field of shape [H, W] or [B, 1, H, W].")


def _restore_field_shape(field, original_ndim):
    """Restore the field to the same dimensionality as the original input."""
    if original_ndim == 2:
        return field[0, 0]
    return field


# %% [markdown]
# ### Module 10.2: Base Class, Grid Buffers, and Learnable Parameters
# 
# The base class stores the physical grid and the parameters that will later be optimised. The cell-centre
# coordinates follow the standard finite-volume definition
# 
# $$
# x_j = -\frac{L_x}{2} + \left(j + \frac{1}{2}\right)\Delta x,
# \qquad
# z_i = -\frac{L_z}{2} + \left(i + \frac{1}{2}\right)\Delta z.
# $$
# 
# Several transport coefficients are kept learnable. Their raw optimisation variables are transformed
# into physically admissible values by bounded or positive maps:
# 
# $$
# f = f_{\max}\,\sigma(\tilde f),
# $$
# 
# $$
# C_b = \operatorname{softplus}(\tilde C_b),
# \qquad
# C_r = \operatorname{softplus}(\tilde C_r),
# \qquad
# C_s = \operatorname{softplus}(\tilde C_s),
# $$
# 
# $$
# \kappa_m = \operatorname{softplus}(\tilde{\kappa}_m),
# \qquad
# \kappa_s = \operatorname{softplus}(\tilde{\kappa}_s),
# \qquad
# \kappa_b = \operatorname{softplus}(\tilde{\kappa}_b),
# $$
# 
# $$
# L_d = \operatorname{softplus}(\tilde L_d).
# $$
# 
# This is useful because optimisation can freely update the raw variables, while the exposed physical
# coefficients remain positive or bounded during the forward pass.
# 
# #### Python Class Concepts Used Here
# 
# | Concept | Example in this notebook | Subordination / hierarchy | Role in the model |
# |---|---|---|---|
# | `class` | `class _DifferentiableRTAdvectorBase(nn.Module):` | `_DifferentiableRTAdvectorBase` is the parent of `_DifferentiableRTAdvectorOperators`; `_DifferentiableRTAdvectorOperators` is the parent of `_DifferentiableRTAdvectorTransport`; `_DifferentiableRTAdvectorTransport` is the parent of `DifferentiableRTAdvector`. | Defines the layered blueprint of the differentiable advect model. |
# | `object` / `instance` | `torch_advector = DifferentiableRTAdvector(...)` in Module 11.2 | `torch_advector` is an instance of the final class `DifferentiableRTAdvector`. | Stores one actual configured model that can advance a density field. |
# | `base class` | `_DifferentiableRTAdvectorBase` | It sits directly under `nn.Module` and above all later subclasses. | Holds common geometry, buffers, and learnable parameters. |
# | `subclass` | `_DifferentiableRTAdvectorOperators`, `_DifferentiableRTAdvectorTransport`, `DifferentiableRTAdvector` | Each subclass inherits everything defined above it and adds a new functional layer. | Splits the implementation into logical stages instead of one giant class. |
# | `inheritance` | `class _DifferentiableRTAdvectorTransport(_DifferentiableRTAdvectorOperators)` | `_DifferentiableRTAdvectorTransport` depends on `_DifferentiableRTAdvectorOperators`, which already depends on `_DifferentiableRTAdvectorBase`. | Lets later blocks reuse gradients, Laplacians, and the streamfunction solver. |
# | `constructor` | `def __init__(self, nx, nz, lx, lz, dt_img, ...)` | The constructor belongs to `_DifferentiableRTAdvectorBase`, so all subclasses and final objects inherit its initialisation logic. | Builds the grid, stores constants, and initialises learnable parameters. |
# | `attribute` | `self.dx`, `self.dz`, `self.max_flow_strength` | These belong to each model instance, so every `DifferentiableRTAdvector` object carries its own stored configuration. | Keeps geometry and configuration attached to the model object. |
# | `buffer` | `self.register_buffer("x_centres", ...)` | Buffers belong to the model instance but are not treated as trainable parameters by `nn.Module`. | Keeps fixed coordinate fields, masks, and forcing tensors inside the model. |
# | `parameter` | `self.raw_flow_strength = nn.Parameter(...)` | Parameters belong to the model instance and are collected by PyTorch as learnable quantities. | Represents transport coefficients that gradients can adjust. |
# | `method` | `solve_streamfunction(self, source)` | This method belongs to `_DifferentiableRTAdvectorOperators`, so it is inherited by all classes below it. | Performs one model-specific operation using the stored state. |
# | `property` | `flow_strength`, `buoyancy_coupling` | These properties are defined on the class and accessed through each model instance. | Converts raw optimisation variables into physical coefficients on demand. |
# | `public interface` | `forward`, `advect`, `advect_with_diagnostics` | These methods belong to the final class `DifferentiableRTAdvector`, which is the class instantiated in Module 11. | Defines how users call the one-step operator from the notebook. |
# | `private/helper convention` | `_inverse_softplus`, `_advance_with_diagnostics` | These helper names sit underneath the public interface and support it internally. | Signals that these pieces support the main interface but are not the main public entry points. |
# | `self` | `self.dx`, `self.solve_streamfunction(...)` | `self` always refers to the current model object, for example the concrete `torch_advector` instance. | Connects each method to the data already stored in that particular model instance. |
# 
# #### Class Hierarchy in This Notebook
# 
# The subordination structure of the implementation is:
# 
# 1. `nn.Module`
# 2. `_DifferentiableRTAdvectorBase`
# 3. `_DifferentiableRTAdvectorOperators`
# 4. `_DifferentiableRTAdvectorTransport`
# 5. `DifferentiableRTAdvector`
# 6. `torch_advector = DifferentiableRTAdvector(...)`
# 
# This means that the final object used in Module 11 inherits all earlier layers:
# 
# - the base class provides geometry, buffers, and learnable coefficients,
# - the operator subclass provides gradients, the Laplacian, and the elliptic closure,
# - the transport subclass provides RT-like velocity construction,
# - the final public class provides the one-step update and user-facing interface.
# 

# %%
# Module 10.2: base class with grid construction, buffers, and learnable parameter registration
class _DifferentiableRTAdvectorBase(nn.Module):
    """Base container for geometry, fixed buffers, and learnable reduced-order coefficients."""

    accel_value: torch.Tensor
    accel_scale: torch.Tensor
    x_centres: torch.Tensor
    z_centres: torch.Tensor
    x_min: torch.Tensor
    x_max: torch.Tensor
    z_min: torch.Tensor
    z_max: torch.Tensor
    wall_damping: torch.Tensor
    interior_mask: torch.Tensor
    raw_flow_strength: torch.Tensor
    raw_buoyancy_coupling: torch.Tensor
    raw_rise_velocity_scale: torch.Tensor
    raw_sorting_velocity_scale: torch.Tensor
    raw_molecular_diffusivity: torch.Tensor
    raw_shear_diffusivity: torch.Tensor
    raw_buoyancy_diffusivity: torch.Tensor
    raw_vertical_decay_length: torch.Tensor

    def __init__(
        self,
        nx,
        nz,
        lx,
        lz,
        dt_img,
        accel_value,
        accel_scale,
        pressure_iterations,
        max_flow_strength,
        initial_flow_strength,
        initial_buoyancy_coupling,
        initial_rise_velocity_scale,
        initial_sorting_velocity_scale,
        initial_molecular_diffusivity,
        initial_shear_diffusivity,
        initial_buoyancy_diffusivity,
        initial_vertical_decay_length,
    ):
        super().__init__()
        self.nx = int(nx)
        self.nz = int(nz)
        self.lx = float(lx)
        self.lz = float(lz)
        self.dt_img = float(dt_img)
        self.dx = self.lx / self.nx
        self.dz = self.lz / self.nz
        self.pressure_iterations = int(pressure_iterations)
        self.max_flow_strength = float(max_flow_strength)

        self._register_acceleration_buffers(accel_value, accel_scale)
        self._register_geometry_buffers()
        self._register_learnable_parameters(
            initial_flow_strength,
            initial_buoyancy_coupling,
            initial_rise_velocity_scale,
            initial_sorting_velocity_scale,
            initial_molecular_diffusivity,
            initial_shear_diffusivity,
            initial_buoyancy_diffusivity,
            initial_vertical_decay_length,
        )

    def _register_acceleration_buffers(self, accel_value, accel_scale):
        """Store the one-step experimental forcing used by the differentiable operator."""
        self.register_buffer("accel_value", torch.tensor(float(accel_value), dtype=torch.float32))
        self.register_buffer("accel_scale", torch.tensor(float(accel_scale), dtype=torch.float32))

    def _register_geometry_buffers(self):
        """Create reusable coordinate, damping, and interior-mask buffers."""
        x = torch.linspace(-0.5 * self.lx + 0.5 * self.dx, 0.5 * self.lx - 0.5 * self.dx, self.nx)
        z = torch.linspace(-0.5 * self.lz + 0.5 * self.dz, 0.5 * self.lz - 0.5 * self.dz, self.nz)
        z_grid, x_grid = torch.meshgrid(z, x, indexing="ij")

        self.register_buffer("x_centres", x_grid.unsqueeze(0).unsqueeze(0))
        self.register_buffer("z_centres", z_grid.unsqueeze(0).unsqueeze(0))
        self.register_buffer("x_min", torch.tensor(float(x[0]), dtype=torch.float32))
        self.register_buffer("x_max", torch.tensor(float(x[-1]), dtype=torch.float32))
        self.register_buffer("z_min", torch.tensor(float(z[0]), dtype=torch.float32))
        self.register_buffer("z_max", torch.tensor(float(z[-1]), dtype=torch.float32))

        wall_damping = 0.25 + 0.75 * (1.0 - (2.0 * z_grid / self.lz) ** 2)
        self.register_buffer("wall_damping", wall_damping.unsqueeze(0).unsqueeze(0).float())

        interior_mask = torch.zeros((1, 1, self.nz, self.nx), dtype=torch.float32)
        if self.nz > 2 and self.nx > 2:
            interior_mask[:, :, 1:-1, 1:-1] = 1.0
        self.register_buffer("interior_mask", interior_mask)

    def _register_learnable_parameters(
        self,
        initial_flow_strength,
        initial_buoyancy_coupling,
        initial_rise_velocity_scale,
        initial_sorting_velocity_scale,
        initial_molecular_diffusivity,
        initial_shear_diffusivity,
        initial_buoyancy_diffusivity,
        initial_vertical_decay_length,
    ):
        """Register unconstrained raw parameters that are mapped into bounded physical coefficients."""
        init_fraction = np.clip(
            initial_flow_strength / max(self.max_flow_strength, 1.0e-12),
            1.0e-4,
            1.0 - 1.0e-4,
        )
        self.raw_flow_strength = nn.Parameter(
            torch.tensor(np.log(init_fraction / (1.0 - init_fraction)), dtype=torch.float32)
        )
        self.raw_buoyancy_coupling = nn.Parameter(
            torch.tensor(_inverse_softplus(initial_buoyancy_coupling), dtype=torch.float32)
        )
        self.raw_rise_velocity_scale = nn.Parameter(
            torch.tensor(_inverse_softplus(initial_rise_velocity_scale), dtype=torch.float32)
        )
        self.raw_sorting_velocity_scale = nn.Parameter(
            torch.tensor(_inverse_softplus(initial_sorting_velocity_scale), dtype=torch.float32)
        )
        self.raw_molecular_diffusivity = nn.Parameter(
            torch.tensor(_inverse_softplus(initial_molecular_diffusivity), dtype=torch.float32)
        )
        self.raw_shear_diffusivity = nn.Parameter(
            torch.tensor(_inverse_softplus(initial_shear_diffusivity), dtype=torch.float32)
        )
        self.raw_buoyancy_diffusivity = nn.Parameter(
            torch.tensor(_inverse_softplus(initial_buoyancy_diffusivity), dtype=torch.float32)
        )
        self.raw_vertical_decay_length = nn.Parameter(
            torch.tensor(_inverse_softplus(initial_vertical_decay_length), dtype=torch.float32)
        )

    @property
    def flow_strength(self):
        return torch.sigmoid(self.raw_flow_strength) * self.max_flow_strength

    @property
    def buoyancy_coupling(self):
        return F.softplus(self.raw_buoyancy_coupling)

    @property
    def rise_velocity_scale(self):
        return F.softplus(self.raw_rise_velocity_scale)

    @property
    def sorting_velocity_scale(self):
        return F.softplus(self.raw_sorting_velocity_scale)

    @property
    def molecular_diffusivity(self):
        return F.softplus(self.raw_molecular_diffusivity)

    @property
    def shear_diffusivity(self):
        return F.softplus(self.raw_shear_diffusivity)

    @property
    def buoyancy_diffusivity(self):
        return F.softplus(self.raw_buoyancy_diffusivity)

    @property
    def vertical_decay_length(self):
        return F.softplus(self.raw_vertical_decay_length)

    def rho_to_phi(self, rho_norm):
        """Convert the public density field to the internal low-density-fluid fraction."""
        return 1.0 - rho_norm

    def phi_to_rho(self, phi):
        """Convert the internal low-density-fluid fraction back to the public density field."""
        return 1.0 - phi


# %% [markdown]
# ### Module 10.3: Differential Operators and Elliptic Closure
# 
# The reduced-order solver uses discrete derivatives of the current field. The horizontal and vertical
# gradients are computed by second-order central differences:
# 
# $$
# \left(\partial_x q\right)_{i,j}
# \approx
# \frac{q_{i,j+1} - q_{i,j-1}}{2\Delta x},
# \qquad
# \left(\partial_z q\right)_{i,j}
# \approx
# \frac{q_{i+1,j} - q_{i-1,j}}{2\Delta z}.
# $$
# 
# The Laplacian used in the diffusive correction follows the standard five-point stencil:
# 
# $$
# \left(\nabla^2 q\right)_{i,j}
# \approx
# \frac{q_{i,j-1} - 2q_{i,j} + q_{i,j+1}}{\Delta x^2}
# +
# \frac{q_{i-1,j} - 2q_{i,j} + q_{i+1,j}}{\Delta z^2}.
# $$
# 
# The streamfunction is then obtained from the elliptic closure
# 
# $$
# \nabla^2 \psi = S_{\psi}
# $$
# 
# by Jacobi relaxation. The interior update used in the code is
# 
# $$
# \psi_{i,j}^{(m+1)}
# =
# \frac{
# \dfrac{\psi_{i,j+1}^{(m)} + \psi_{i,j-1}^{(m)}}{\Delta x^2}
# +
# \dfrac{\psi_{i+1,j}^{(m)} + \psi_{i-1,j}^{(m)}}{\Delta z^2}
# +
# S_{\psi,i,j}
# }{
# \dfrac{2}{\Delta x^2} + \dfrac{2}{\Delta z^2}
# }.
# $$
# 
# The same block also defines the effective reduced-order diffusivity used later in the one-step update.
# 

# %%
# Module 10.3: discrete differential operators and the reduced-order elliptic closure
class _DifferentiableRTAdvectorOperators(_DifferentiableRTAdvectorBase):
    """Add finite-difference operators, streamfunction solve, and effective diffusivity."""

    def central_gradient_x(self, field):
        """Second-order central difference in the horizontal direction."""
        field_pad = F.pad(field, (1, 1, 0, 0), mode='replicate')
        return (field_pad[:, :, :, 2:] - field_pad[:, :, :, :-2]) / (2.0 * self.dx)

    def central_gradient_z(self, field):
        """Second-order central difference in the vertical direction."""
        field_pad = F.pad(field, (0, 0, 1, 1), mode='replicate')
        return (field_pad[:, :, 2:, :] - field_pad[:, :, :-2, :]) / (2.0 * self.dz)

    def laplacian(self, field):
        """Five-point Laplacian with replicated edge values."""
        field_pad = F.pad(field, (1, 1, 1, 1), mode='replicate')
        lap_x = (field_pad[:, :, 1:-1, :-2] - 2.0 * field + field_pad[:, :, 1:-1, 2:]) / (self.dx ** 2)
        lap_z = (field_pad[:, :, :-2, 1:-1] - 2.0 * field + field_pad[:, :, 2:, 1:-1]) / (self.dz ** 2)
        return lap_x + lap_z

    def solve_streamfunction(self, source):
        """Jacobi relaxation for the reduced-order streamfunction equation."""
        psi = torch.zeros_like(source)
        coeff = 1.0 / (2.0 / (self.dx ** 2) + 2.0 / (self.dz ** 2))

        for _ in range(self.pressure_iterations):
            psi_pad = F.pad(psi, (1, 1, 1, 1), mode='constant', value=0.0)
            neighbour_sum = (
                (psi_pad[:, :, 1:-1, 2:] + psi_pad[:, :, 1:-1, :-2]) / (self.dx ** 2)
                + (psi_pad[:, :, 2:, 1:-1] + psi_pad[:, :, :-2, 1:-1]) / (self.dz ** 2)
            )
            psi = coeff * (neighbour_sum + source)
            psi = psi * self.interior_mask

        return psi

    def mean_unstable_jump(self, phi):
        """Average the positive downward-to-upward light-fluid jump over the batch."""
        return F.relu(phi[:, :, :-1, :] - phi[:, :, 1:, :]).mean(dim=(-2, -1), keepdim=True)

    def effective_diffusivity(self, phi):
        """Compute one reduced-order diffusivity per batch item and broadcast it over the domain."""
        grad_x = self.central_gradient_x(phi)
        shear_level = grad_x.abs().mean(dim=(-2, -1), keepdim=True)
        mixedness = (4.0 * phi * (1.0 - phi)).mean(dim=(-2, -1), keepdim=True)
        unstable_jump_mean = self.mean_unstable_jump(phi)
        drive_fraction = self.flow_strength / max(self.max_flow_strength, 1.0e-12)

        return (
            self.molecular_diffusivity
            + self.shear_diffusivity * drive_fraction * mixedness
            + self.buoyancy_diffusivity * unstable_jump_mean
            + 0.15 * self.shear_diffusivity * shear_level * self.dx
        )


# %% [markdown]
# ### Module 10.4: Transport Velocity Construction
# 
# The centre transport field is built from the current low-density-fluid fraction by the reduced-order closure
# 
# $$
# S_{\psi} = C_b\,f\,\hat a\,\partial_x \phi.
# $$
# 
# After solving for the streamfunction, the centre velocities are recovered from
# 
# $$
# u_c = d(z)\,\frac{\partial \psi}{\partial z},
# \qquad
# w_c = -d(z)\,\frac{\partial \psi}{\partial x},
# \qquad
# d(z) = \exp\!\left[-\left(\frac{z}{L_d}\right)^2\right].
# $$
# 
# The vertical transport is then corrected by two additional terms:
# 
# $$
# w_{\mathrm{drift}} = C_r\,g_w(z)\,(\phi - 0.5),
# $$
# 
# $$
# J_{i+\frac{1}{2},j} = \max(\phi_{i,j} - \phi_{i+1,j}, 0),
# \qquad
# w_{\mathrm{sorting}} = C_s\,f\,J.
# $$
# 
# These terms are assembled on cell faces because the reduced-order transport step is defined there.
# After the face corrections are added, the face field is averaged back to cell centres so that the
# semi-Lagrangian update can backtrace from the centre coordinates.
# 

# %%
# Module 10.4: build reduced-order transport velocities from the current density structure
class _DifferentiableRTAdvectorTransport(_DifferentiableRTAdvectorOperators):
    """Add RT-like transport-velocity construction and differentiable semi-Lagrangian advection."""

    def build_baroclinic_source(self, phi):
        """Construct the reduced-order baroclinic source from the current density structure."""
        accel_hat = self.accel_value / torch.clamp(self.accel_scale, min=1.0e-12)
        return self.buoyancy_coupling * self.flow_strength * accel_hat * self.central_gradient_x(phi)

    def base_center_velocities(self, phi):
        """Solve for the streamfunction and convert it into centre-based circulation velocities."""
        baroclinic_source = self.build_baroclinic_source(phi)
        psi = self.solve_streamfunction(baroclinic_source)
        vertical_decay = torch.exp(-((self.z_centres / torch.clamp(self.vertical_decay_length, min=1.0e-6)) ** 2))
        u_center = vertical_decay * self.central_gradient_z(psi)
        w_center = -vertical_decay * self.central_gradient_x(psi)
        diagnostics = {
            "baroclinic_source": baroclinic_source,
            "psi": psi,
        }
        return u_center, w_center, diagnostics

    def corrected_face_velocities(self, phi, u_center, w_center):
        """Interpolate centre velocities to faces and add buoyancy and sorting corrections."""
        w_drift = self.rise_velocity_scale * self.wall_damping * (phi - 0.5)

        u_faces = torch.zeros(phi.shape[0], 1, self.nz, self.nx + 1, dtype=phi.dtype, device=phi.device)
        w_faces = torch.zeros(phi.shape[0], 1, self.nz + 1, self.nx, dtype=phi.dtype, device=phi.device)
        u_faces[:, :, :, 1:-1] = 0.5 * (u_center[:, :, :, :-1] + u_center[:, :, :, 1:])
        w_faces[:, :, 1:-1, :] = 0.5 * (w_center[:, :, :-1, :] + w_center[:, :, 1:, :])
        w_faces[:, :, 1:-1, :] += 0.5 * (w_drift[:, :, :-1, :] + w_drift[:, :, 1:, :])

        unstable_jump = F.relu(phi[:, :, :-1, :] - phi[:, :, 1:, :])
        w_faces[:, :, 1:-1, :] += self.sorting_velocity_scale * self.flow_strength * unstable_jump

        diagnostics = {
            "w_drift": w_drift,
            "unstable_jump": unstable_jump,
        }
        return u_faces, w_faces, diagnostics

    def density_to_transport_velocity(self, phi):
        """Assemble the full corrected centre transport velocity used by the one-step operator."""
        u_center, w_center, base_diagnostics = self.base_center_velocities(phi)
        u_faces, w_faces, correction_diagnostics = self.corrected_face_velocities(phi, u_center, w_center)

        u_transport = 0.5 * (u_faces[:, :, :, 1:] + u_faces[:, :, :, :-1])
        w_transport = 0.5 * (w_faces[:, :, 1:, :] + w_faces[:, :, :-1, :])

        diagnostics = {
            **base_diagnostics,
            **correction_diagnostics,
            "u_center": u_transport,
            "w_center": w_transport,
            "u_transport": u_transport,
            "w_transport": w_transport,
        }
        return u_transport, w_transport, diagnostics

    def backtrace_grid(self, u_transport, w_transport):
        """Compute the backtraced sampling grid over one image-sized time interval."""
        x_back = self.x_centres - u_transport * self.dt_img
        z_back = self.z_centres - w_transport * self.dt_img

        x_norm = 2.0 * (x_back - self.x_min) / torch.clamp(self.x_max - self.x_min, min=1.0e-12) - 1.0
        z_norm = 2.0 * (z_back - self.z_min) / torch.clamp(self.z_max - self.z_min, min=1.0e-12) - 1.0
        return torch.stack((x_norm[:, 0], z_norm[:, 0]), dim=-1)

    def semi_lagrangian_advect(self, phi, u_transport, w_transport):
        """Backtrace the current field over one image-sized time interval with bilinear sampling."""
        sampling_grid = self.backtrace_grid(u_transport, w_transport)
        return F.grid_sample(phi, sampling_grid, mode="bilinear", padding_mode="border", align_corners=True)


# %% [markdown]
# ### Module 10.5: One-Step Update and Public Interface
# 
# Once the corrected transport field is known, the semi-Lagrangian backtrace evaluates the previous field at
# the departure point
# 
# $$
# x_{\mathrm{back}} = x - u\,\Delta t_{\mathrm{img}},
# \qquad
# z_{\mathrm{back}} = z - w\,\Delta t_{\mathrm{img}}.
# $$
# 
# This gives the advected intermediate field
# 
# $$
# \phi^{\mathrm{adv}}_{t+\Delta t}(x,z)
# =
# \phi_t\!\left(x_{\mathrm{back}}, z_{\mathrm{back}}\right).
# $$
# 
# A diffusive correction is then applied using the effective diffusivity from the current state:
# 
# $$
# \phi_{t+\Delta t}
# =
# \operatorname{clip}
# \left(
# \phi^{\mathrm{adv}}_{t+\Delta t}
# +
# \Delta t_{\mathrm{img}}\,\kappa_{\mathrm{eff}}(\phi_t)\,\nabla^2 \phi_t,
# 0,
# 1
# \right).
# $$
# 
# The external interface of the class finally returns
# 
# $$
# \rho_{n,t+\Delta t} = 1 - \phi_{t+\Delta t}.
# $$
# 
# Both a standard `forward` method and readable aliases `advect` and `advect_with_diagnostics` are exposed so
# that the notebook can use the task wording directly while remaining compatible with `nn.Module` conventions.
# 

# %%
# Module 10.5: final one-step update and the public interface used by the notebook
class DifferentiableRTAdvector(_DifferentiableRTAdvectorTransport):
    """Final public class combining the reduced-order closure with a differentiable one-step update."""

    def _clamp_unit_interval(self, field):
        """Keep all scalar fields inside the admissible [0, 1] interval."""
        return torch.clamp(field, 0.0, 1.0)

    def _diffusive_update(self, phi_now, phi_advected):
        """Apply the reduced-order diffusive correction after the semi-Lagrangian step."""
        kappa_eff = self.effective_diffusivity(phi_now)
        phi_next = phi_advected + self.dt_img * kappa_eff * self.laplacian(phi_now)
        return self._clamp_unit_interval(phi_next), kappa_eff

    def _advance_with_diagnostics(self, rho_norm):
        """Advance the public density field by one image-sized interval and keep intermediate diagnostics."""
        rho_norm, original_ndim = _to_bchw(rho_norm)
        rho_norm = self._clamp_unit_interval(rho_norm.to(dtype=torch.float32))

        phi_now = self.rho_to_phi(rho_norm)
        u_transport, w_transport, diagnostics = self.density_to_transport_velocity(phi_now)
        phi_advected = self.semi_lagrangian_advect(phi_now, u_transport, w_transport)
        phi_next, kappa_eff = self._diffusive_update(phi_now, phi_advected)
        rho_next = self._clamp_unit_interval(self.phi_to_rho(phi_next))

        diagnostics["kappa_eff"] = kappa_eff
        diagnostics["phi_now"] = phi_now
        diagnostics["phi_advected"] = phi_advected
        diagnostics["phi_next"] = phi_next
        diagnostics["rho_now"] = rho_norm
        diagnostics["rho_next"] = rho_next

        rho_next = _restore_field_shape(rho_next, original_ndim)
        return rho_next, diagnostics

    def forward(self, rho_norm):
        """Standard nn.Module entry point returning only the next density field."""
        rho_next, _ = self._advance_with_diagnostics(rho_norm)
        return rho_next

    def forward_with_diagnostics(self, rho_norm):
        """Return both the next density field and intermediate reduced-order diagnostics."""
        return self._advance_with_diagnostics(rho_norm)

    def advect(self, rho_norm):
        """Readable alias matching the task wording used in the notebook."""
        return self.forward(rho_norm)

    def advect_with_diagnostics(self, rho_norm):
        """Readable alias returning both the next density field and diagnostic quantities."""
        return self.forward_with_diagnostics(rho_norm)


# %% [markdown]
# ## Module 11: One-Step Example and Autograd Check
# 
# This module turns the abstract operator introduced above into a **step-by-step,
# optimisation-oriented demonstration**.
# 
# The goal is to construct a differentiable map
# 
# $$
# \rho_{n,t+\Delta t_{\mathrm{img}}}
# =
# \mathcal{A}_{\theta}\!\left(\rho_{n,t}\right),
# $$
# 
# evaluate it on a concrete density field from the notebook, compare the predicted
# next state against a target next frame, and verify that gradients with respect to
# the learnable parameters are available.
# 
# Instead of placing all of these actions into one Python block, the workflow is
# separated below into:
# 
# 1. preparation of the input and target fields,
# 2. model construction,
# 3. forward one-step evolution,
# 4. loss evaluation and autograd,
# 5. visual interpretation of the result.
# 

# %% [markdown]
# ### Module 11.1: Preparing the Current and Target Density Fields
# 
# The notebook stores the scalar field $\phi$, which is the volume fraction of the
# low-density dyed fluid. The PyTorch operator, however, is written in terms of the
# **normalised density field**
# 
# $$
# \rho_n = 1 - \phi.
# $$
# 
# Therefore:
# 
# - the current optimisation input is
#   $$
#   \rho_{n,t} = 1 - \phi_t,
#   $$
# - and the target next frame for a one-step demonstration is
#   $$
#   \rho_{n,t+\Delta t}^{\mathrm{target}} = 1 - \phi_{t+\Delta t}.
#   $$
# 
# In this notebook example, the target next field is simply taken from the next stored
# simulation frame so that the forward pass and the gradient path can be checked in a
# controlled setting.
# 

# %%
# Module 11.1: prepare the current field and the target next field
import torch

# torch.manual_seed(cfg.seed)

# # Convert the notebook state variable phi into the optimisation-facing density field rho_n.
# rho_norm_0 = torch.tensor(1.0 - history["phi"][0], dtype=torch.float32)
# target_next = torch.tensor(1.0 - history["phi"][1], dtype=torch.float32)

# print("Current density-field shape:", tuple(rho_norm_0.shape))
# print("Target next density-field shape:", tuple(target_next.shape))


# %% [markdown]
# ### Module 11.2: Constructing the Differentiable Advect Model
# 
# The model is instantiated using the same grid size, physical domain, and image-sized
# time interval as the notebook. Several internal parameters are learnable. Their raw
# values are transformed so that the physical coefficients remain positive or bounded:
# 
# $$
# f = f_{\max}\,\sigma(\tilde{f}),
# $$
# 
# $$
# C_b = \operatorname{softplus}(\tilde{C}_b),
# \qquad
# C_r = \operatorname{softplus}(\tilde{C}_r),
# \qquad
# C_s = \operatorname{softplus}(\tilde{C}_s),
# $$
# 
# $$
# \kappa_m = \operatorname{softplus}(\tilde{\kappa}_m),
# \qquad
# \kappa_s = \operatorname{softplus}(\tilde{\kappa}_s),
# \qquad
# \kappa_b = \operatorname{softplus}(\tilde{\kappa}_b),
# $$
# 
# where:
# 
# - $f$ is the global flow-strength amplitude,
# - $C_b$ is the baroclinic coupling coefficient,
# - $C_r$ is the buoyancy-drift scale,
# - $C_s$ is the unstable-jump sorting scale,
# - and the $\kappa$ terms control the reduced-order diffusion level.
# 
# This parameterisation is useful for optimisation because it allows the solver to
# adjust internal transport strength while still preserving physically admissible signs.
# 

# %%
# Module 11.2: instantiate the differentiable one-step advect model
# torch_advector = DifferentiableRTAdvector(
#     nx=cfg.Nx,
#     nz=cfg.Nz,
#     lx=cfg.Lx,
#     lz=cfg.Lz,
#     dt_img=cfg.dt,
#     accel_value=cfg.unstable_accel,
#     accel_scale=max(abs(cfg.unstable_accel), abs(cfg.stable_accel)),
#     pressure_iterations=cfg.pressure_iterations,
#     max_flow_strength=cfg.max_flow_strength,
#     initial_flow_strength=cfg.initial_flow_strength,
#     initial_buoyancy_coupling=cfg.buoyancy_coupling,
#     initial_rise_velocity_scale=cfg.rise_velocity_scale,
#     initial_sorting_velocity_scale=cfg.sorting_velocity_scale,
#     initial_molecular_diffusivity=cfg.molecular_diffusivity,
#     initial_shear_diffusivity=cfg.shear_diffusivity,
#     initial_buoyancy_diffusivity=cfg.buoyancy_diffusivity,
#     initial_vertical_decay_length=cfg.vertical_decay_length,
# )

# print(f"Image-sized time step dt_img = {torch_advector.dt_img:.4f} s")
# print(f"Initial bounded flow strength = {torch_advector.flow_strength.item():.4f}")
# print(f"Initial buoyancy coupling = {torch_advector.buoyancy_coupling.item():.4f}")
# print(f"Initial rise velocity scale = {torch_advector.rise_velocity_scale.item():.4f}")
# print(f"Initial sorting velocity scale = {torch_advector.sorting_velocity_scale.item():.4f}")


# %% [markdown]
# ### Module 11.3: Forward One-Step Evolution
# 
# The actual one-step prediction is obtained by evaluating
# 
# $$
# \rho_{n,t+\Delta t}^{\mathrm{pred}}
# =
# \mathcal{A}_{\theta}\!\left(\rho_{n,t}\right).
# $$
# 
# Internally, the model carries out the following sequence:
# 
# $$
# \rho_{n,t}
# \;\Longrightarrow\;
# \phi_t
# \;\Longrightarrow\;
# S_{\psi}
# \;\Longrightarrow\;
# \psi
# \;\Longrightarrow\;
# (u,w)
# \;\Longrightarrow\;
# \phi_{t+\Delta t}
# \;\Longrightarrow\;
# \rho_{n,t+\Delta t}^{\mathrm{pred}}.
# $$
# 
# The diagnostic dictionary returned by the solver makes intermediate reduced-order
# quantities available, including:
# 
# - the baroclinic source term,
# - the streamfunction,
# - the corrected centre transport velocities,
# - and the effective diffusivity used in the one-step update.
# 

# %%
# Module 11.3: run one differentiable forward step and inspect key diagnostics
# rho_norm_1, transport_diagnostics = torch_advector.advect_with_diagnostics(rho_norm_0)

# mean_kappa_eff = float(transport_diagnostics["kappa_eff"].mean().detach().cpu())
# mean_baroclinic_source = float(transport_diagnostics["baroclinic_source"].abs().mean().detach().cpu())
# mean_transport_speed = float(
#     torch.sqrt(
#         transport_diagnostics["u_center"] ** 2 + transport_diagnostics["w_center"] ** 2
#     ).mean().detach().cpu()
# )

# print("Predicted next density-field shape:", tuple(rho_norm_1.shape))
# print(f"Mean effective diffusivity = {mean_kappa_eff:.6e}")
# print(f"Mean absolute baroclinic source = {mean_baroclinic_source:.6e}")
# print(f"Mean centre transport speed = {mean_transport_speed:.6e}")


# %% [markdown]
# ### Module 11.4: Loss Function and Autograd
# 
# To verify differentiability, a simple one-step loss is defined against the target next
# frame:
# 
# $$
# \mathcal{L}(\theta)
# =
# \frac{1}{N_x N_z}
# \sum_{i=1}^{N_z}
# \sum_{j=1}^{N_x}
# \left(
# \rho_{n,t+\Delta t,ij}^{\mathrm{pred}}
# -
# \rho_{n,t+\Delta t,ij}^{\mathrm{target}}
# \right)^2.
# $$
# 
# Because every stage of the map is written with PyTorch tensor operations, the gradient
# can be propagated back through the full update:
# 
# $$
# \nabla_{\theta}\mathcal{L}
# =
# \frac{\partial \mathcal{L}}{\partial \rho_{n,t+\Delta t}^{\mathrm{pred}}}
# \frac{\partial \rho_{n,t+\Delta t}^{\mathrm{pred}}}{\partial \theta}.
# $$
# 
# The cell below computes this loss and checks that gradients are produced for several
# representative learnable parameters.
# 

# %%
# Module 11.4: define a one-step loss and verify that autograd works
import torch.nn.functional as F

# loss = F.mse_loss(rho_norm_1, target_next)
# loss.backward()

# buoyancy_grad = torch_advector.raw_buoyancy_coupling.grad
# rise_grad = torch_advector.raw_rise_velocity_scale.grad
# sorting_grad = torch_advector.raw_sorting_velocity_scale.grad

# if buoyancy_grad is None or rise_grad is None or sorting_grad is None:
#     raise RuntimeError("Expected autograd gradients to be available after loss.backward().")

# print(f"One-step MSE loss = {loss.item():.6e}")
# print(f"Gradient of raw buoyancy coupling = {buoyancy_grad.item():.6e}")
# print(f"Gradient of raw rise velocity scale = {rise_grad.item():.6e}")
# print(f"Gradient of raw sorting velocity scale = {sorting_grad.item():.6e}")


# %% [markdown]
# ### Module 11.5: Visual Interpretation of the One-Step Update
# 
# It is useful to visualise three fields together:
# 
# 1. the current density field $\rho_{n,t}$,
# 2. the predicted next density field $\rho_{n,t+\Delta t}^{\mathrm{pred}}$,
# 3. and the one-step increment
#    $$
#    \Delta \rho_n
#    =
#    \rho_{n,t+\Delta t}^{\mathrm{pred}} - \rho_{n,t}.
#    $$
# 
# This comparison shows not only whether the operator runs, but also where the model
# predicts local rearrangement during a single image-sized time interval.
# 

# %%
# Module 11.5: visualise the current field, the predicted next field, and the one-step change
# import matplotlib.pyplot as plt

# fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2), constrained_layout=True)

# im0 = axes[0].imshow(
#     rho_norm_0.detach().cpu().numpy(),
#     origin="lower",
#     vmin=0.0,
#     vmax=1.0,
#     cmap="magma",
# )
# axes[0].set_title("Current Normalised Density $\\rho_n(t)$")
# axes[0].set_xlabel("x-index")
# axes[0].set_ylabel("z-index")

# im1 = axes[1].imshow(
#     rho_norm_1.detach().cpu().numpy(),
#     origin="lower",
#     vmin=0.0,
#     vmax=1.0,
#     cmap="magma",
# )
# axes[1].set_title("Predicted One-Step Density $\\rho_n(t+\\Delta t)$")
# axes[1].set_xlabel("x-index")
# axes[1].set_ylabel("z-index")

# delta_rho = (rho_norm_1 - rho_norm_0).detach().cpu().numpy()
# im2 = axes[2].imshow(delta_rho, origin="lower", cmap="coolwarm")
# axes[2].set_title("One-Step Change $\\Delta \\rho_n$")
# axes[2].set_xlabel("x-index")
# axes[2].set_ylabel("z-index")

# fig.colorbar(im0, ax=axes[0], shrink=0.85)
# fig.colorbar(im1, ax=axes[1], shrink=0.85)
# fig.colorbar(im2, ax=axes[2], shrink=0.85)
# plt.show()



