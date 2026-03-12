# Evolution Model Code Guide (`evolution_model.py`)

This document focuses on **how the code is organized and how data flows through it**.
It is intended as a developer-oriented introduction to the implementation.

## 1. Module Purpose

`evolution_model.py` implements the latent state evolution stage only:

- particle motion in the `x-z` plane
- scalar dye transport on an `x-z` grid
- depth-axis (`y`) random walk for light-sheet visibility events

The file can be viewed as a compact simulation core that is reusable before any optical/camera rendering stage.

## 2. Core Data Structures

### `SimConfig` (parameter container)

`SimConfig` groups all simulation hyperparameters in one dataclass so function signatures stay stable.
Parameters are organized by role:

- temporal and resolution controls: `T`, `dt`, `H`, `W`, `N`
- physical domain: `Lx`, `zmin`, `zmax`
- velocity field shape: `A`, `k`, `gamma`
- stochastic and transport settings: `particle_noise_sigma`, `dye_kappa`
- light-sheet gating settings: `enable_sheet_gating`, `sheet_center_y`, `sheet_thickness`
- out-of-plane behavior: `y_noise_sigma`, `y_kill`
- reproducibility: `seed`

### `State` (runtime latent variables)

`State` stores one snapshot of the latent system:

- `xp`: particle x positions, shape `(N,)`
- `zp`: particle z positions, shape `(N,)`
- `y`: particle depth positions, shape `(N,)`
- `c`: dye field on the z-x grid, shape `(H, W)`

This separation (`SimConfig` vs `State`) makes simulation steps pure with respect to configuration and mutable only on state.

## 3. Spatial Conventions and Boundaries

The coordinate system is centered on the frame:

- `x` is periodic in `[-Lx/2, Lx/2)` using `wrap_x_centered`
- `z` is bounded and clipped to `[zmin, zmax]`
- `y` is depth along camera axis and used only for sheet visibility logic

The periodic treatment of `x` is applied consistently in both particle and dye updates.

## 4. Dynamics Building Blocks

### 4.1 Velocity field: `vel_u_w`

`vel_u_w(x, z, t, A, k, gamma)` returns in-plane velocity components:

- `u = dx/dt` (x direction)
- `w = dz/dt` (z direction)

The implementation uses a decaying analytic mode:

- vertical decay through `exp(-k*abs(z))`
- oscillation through complex phase `exp(i*k*x)`
- optional temporal growth/decay `exp(gamma*t)`

This function is the single source of flow dynamics used by both particle and dye transport.

### 4.2 Particle update: `advect_particles_rk2`

Particle trajectories are updated with midpoint RK2:

1. evaluate velocity at current position
2. estimate midpoint state
3. evaluate velocity at midpoint
4. advance full step with midpoint velocity

Then the code injects Gaussian process noise (`particle_noise_sigma`) and enforces boundaries:

- wrap `x` periodically
- clip `z` to domain limits

### 4.3 Dye update: `advect_dye_semilag`

Dye transport is semi-Lagrangian:

1. build grid coordinates `(X, Z)`
2. compute velocity on grid
3. backtrace to source points `(Xb, Zb)`
4. sample old field at backtraced points via `bilinear_sample`

`bilinear_sample` handles periodic indexing in `x`, so scalar transport matches particle periodicity.

If `dye_kappa > 0`, an explicit diffusion correction is added:

- finite-difference Laplacian in `x` and `z`
- periodic stencil in `x` via `np.roll`
- edge-stable treatment in `z` via `np.pad(..., mode="edge")`

### 4.4 Depth evolution and gating

`update_y_depth` applies Brownian-like depth increments:

- `y <- y + Normal(0, y_noise_sigma * sqrt(dt))`

`visible_mask_y` computes whether particles are inside sheet thickness around `sheet_center_y`.

`respawn` handles particles with large depth deviation:

- kill rule in caller: `abs(y - sheet_center_y) > y_kill`
- reset `x` uniformly over full periodic width
- reset `z` from a centered Gaussian then clipped
- reset `y` near light-sheet center

This creates continual enter/exit behavior while preserving particle count.

## 5. Step Orchestration

`step_evolution` is the key state transition function.  
Its fixed update order is:

1. `advect_particles_rk2`
2. `advect_dye_semilag`
3. `update_y_depth`
4. `respawn` for killed particles
5. `visible_mask_y` generation

It returns:

- updated `State`
- visibility mask `vis` for current step

Because this ordering is centralized in one function, downstream simulators can treat it as the evolution operator.

## 6. Initialization and Diagnostics

### `init_state`

`init_state` builds a reproducible initial condition from `seed`:

- particles are distributed in centered `x` and bounded `z`
- `y` starts near sheet center
- dye field `c` starts as a Gaussian blob on the `x-z` grid

### `run_evolution_only`

This function repeatedly applies `step_evolution` for `T` steps and collects:

- `visible_frac` time series (`mean(vis)` each step)

It is the cleanest entry point when debugging only the latent dynamics.

## 7. Visualization Subsystem (Code-Level View)

Two render helpers consume `State` snapshots:

- `_state_to_rgb_frame`: 2D frame with dye background and particle overlays
- `_state_to_rgb_frame_3d`: 3D frame with volumetric-like dye slices and true `(x, y, z)` particles

Two wrappers run simulation + frame capture:

- `visualise_evolution` (2D)
- `visualise_evolution_3d` (3D)

Both wrappers share the same evolution core (`step_evolution`), so visualization never diverges from simulation logic.

## 8. Call Graph

```text
init_state
  -> State(xp, zp, y, c), xs, zs

run_evolution_only / visualise_evolution / visualise_evolution_3d
  -> step_evolution
       -> advect_particles_rk2
            -> vel_u_w
       -> advect_dye_semilag
            -> vel_u_w
            -> bilinear_sample
       -> update_y_depth
       -> respawn
       -> visible_mask_y
```

## 9. Extension Points for Future Work

Natural places to customize behavior:

- replace `vel_u_w` with measured or learned flow fields
- switch dye diffusion to implicit solvers for larger `dt`
- modify `respawn` distribution to encode boundary inflow physics
- add anisotropic or state-dependent `y` dynamics
- expose more internals from `run_evolution_only` (for inversion/assimilation pipelines)
