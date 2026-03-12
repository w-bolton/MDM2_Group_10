"""
Forward simulator implementation
--------------------------------

Pipeline:

Evolution Model  ->  Optical Model  ->  Camera Model  ->  Export

Frame origin is at the CENTER of the pixel frame.

Output folder:
Outputs_simulate/
"""

"""
Forward Simulator
=================

This script implements a forward model for generating synthetic camera
observations from an underlying flow and dye field.

The forward operator follows the pipeline:

    Evolution Model  →  Optical Model  →  Camera Model  →  Export


Coordinate System
-----------------
The pixel frame uses a centered coordinate system:

    x : width direction  (perpendicular to drawing plane)
    y : depth direction  (camera optical axis)
    z : height direction

The frame origin (0,0) is located at the center of the x–z image plane.


Overall Model Structure
-----------------------

The simulator is organised into three main modelling components:

    1) Evolution Model
    2) Optical Model
    3) Camera Model

These are executed sequentially at each timestep.


---------------------------------------------------------------------
1. Evolution Model
---------------------------------------------------------------------

Responsible for updating the latent physical state:

    x_t  →  x_{t+1}

State variables:

    xp : particle x positions
    zp : particle z positions
    y  : particle out-of-plane coordinate
    c  : dye concentration field c(x,z,t)

Evolution Model Submodules:

    • Velocity Field
        vel_u_w()

    • Particle Advection
        advect_particles_rk2()

    • Dye Advection (semi-Lagrangian)
        advect_dye_semilag()

    • Out-of-Plane Dynamics
        update_out_of_plane()

    • Visibility Gating (light sheet thickness)
        visible_mask()

    • Particle Respawn
        respawn()

    • Evolution Step Wrapper
        step_evolution()


---------------------------------------------------------------------
2. Optical Model
---------------------------------------------------------------------

Maps the physical state to an optical intensity field:

    state  →  image intensity I(x,z)

Optical Model Submodules:

    • Particle Scattering / Rendering
        render_particles()

    • Dye Fluorescence / Emission
        render_dye()

    • Optical Blur (Point Spread Function)
        gaussian_blur_fft()

    • Optical Combination
        render_total_intensity()


---------------------------------------------------------------------
3. Camera Model
---------------------------------------------------------------------

Transforms the optical intensity into a camera sensor frame.

Camera Model Submodules:

    • Camera Noise Model
        camera_model()

    • Auto Exposure Mapping
        auto_exposure_to_uint8()

    • Frame Encoding Wrapper
        encode_frame()


---------------------------------------------------------------------
4. Diagnostics
---------------------------------------------------------------------

Collects per-frame statistics during simulation.

    init_diagnostics()
    append_diagnostics()


---------------------------------------------------------------------
5. Export
---------------------------------------------------------------------

Handles saving outputs (GIF / MP4 / images).

    export_video()


---------------------------------------------------------------------
6. CLI Interface
---------------------------------------------------------------------

Command line interface and runtime configuration.

    parse_cli_args()
    resolve_output_options()
    main()


---------------------------------------------------------------------
Forward Operator Representation
---------------------------------------------------------------------

The simulator implements the forward operator:

        A(x₀, θ)

which can be decomposed as

        A = C ∘ O ∘ E

where

    E : Evolution Model
    O : Optical Model
    C : Camera Model


---------------------------------------------------------------------
Outputs
---------------------------------------------------------------------

Generated outputs include:

    • Synthetic video sequence
    • GIF / MP4 export
    • Diagnostic statistics
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
    # Time / Size
    T: int = 80
    dt: float = 0.02
    H: int = 384
    W: int = 384
    N: int = 1000

    # Physical domain
    Lx: float = 1.0
    zmin: float = -0.5
    zmax: float = 0.5

    # Placeholder velocity field on x-z slice
    A: float = 0.02
    k: float = 2*np.pi
    gamma: float = 0.0

    # particle dynamics noise for x-z advection 
    particle_noise_sigma: float = 5e-4
    dye_kappa: float = 0.0

    # keep for backward compatibility; when False, y random walk/respawn is skipped
    enable_out_of_plane: bool = True
    enable_sheet_gating: bool = True
    sheet_center_y: float = 0.0
    sheet_thickness: float = 0.02
    y_noise_sigma: float = 0.005
    y_kill: float = 0.06

    psf_sigma_px: float = 1.6
    particle_amp: float = 2.0

    dye_beta: float = 8.0
    dye_alpha: float = 0.6

    light_source_x_frac: float = 0.5
    light_source_z_above_frac: float = 1.2
    dye_blur_sigma_px: float = 0.7

    use_camera_model: bool = True
    bg: float = 10.0
    gain: float = 120.0
    read_sigma: float = 1.5

    auto_exposure: bool = True
    exposure_percentile: float = 99.7

    seed: int = 1


@dataclass
class State:

    xp: np.ndarray
    zp: np.ndarray
    y: np.ndarray
    c: np.ndarray


# =========================================================
# Evolution Model
# =========================================================

'''
Coordinate helpers (NEW): centered periodic wrap in x
'''
def wrap_x_centered(x, Lx):

    return ((x + 0.5*Lx) % Lx) - 0.5*Lx

'''
Velocity field on x–z slice
'''
def vel_u_w(x, z, t, A, k, gamma):

    decay = np.exp(-k*np.abs(z))
    phase = np.exp(1j*k*x)
    growth = np.exp(gamma*t)

    u = np.real(1j*k*A*decay*phase*growth)
    w = np.real(-k*A*np.sign(z)*decay*phase*growth)

    return u, w

'''
Numercial advection and evolution steps
'''
def advect_particles_rk2(x, z, t, dt, cfg):

    u1,w1 = vel_u_w(x,z,t,cfg.A,cfg.k,cfg.gamma)

    xm = x + 0.5*dt*u1
    zm = z + 0.5*dt*w1

    u2,w2 = vel_u_w(xm,zm,t+0.5*dt,cfg.A,cfg.k,cfg.gamma)

    x_new = x + dt*u2
    z_new = z + dt*w2

    if cfg.particle_noise_sigma > 0:

        x_new += np.random.normal(0,cfg.particle_noise_sigma,x.shape)
        z_new += np.random.normal(0,cfg.particle_noise_sigma,z.shape)

    x_new = wrap_x_centered(x_new,cfg.Lx)
    z_new = np.clip(z_new,cfg.zmin,cfg.zmax)

    return x_new,z_new


def bilinear_sample(field, xq, zq, xs, zs, periodic_x=True):

    Nz, Nx = field.shape
    dx = xs[1] - xs[0]
    dz = zs[1] - zs[0]

    if periodic_x:
        xq = wrap_x_centered(xq, xs[-1] - xs[0] + dx)

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


def advect_dye_semilag(c, xs, zs, t, dt, cfg):

    X, Z = np.meshgrid(xs, zs)
    u, w = vel_u_w(X, Z, t, cfg.A, cfg.k, cfg.gamma)

    Xb = X - u * dt
    Zb = Z - w * dt

    c_new = bilinear_sample(c, Xb, Zb, xs, zs, periodic_x=True)

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


def update_y_depth(y,dt,cfg):

    return y + np.random.normal(0,cfg.y_noise_sigma*np.sqrt(dt),size=y.shape)


def update_out_of_plane(y,dt,cfg):

    return update_y_depth(y,dt,cfg)


def visible_mask_y(y,cfg):

    if not cfg.enable_sheet_gating:
        return np.ones_like(y,dtype=bool)
    return np.abs(y - cfg.sheet_center_y) <= 0.5*cfg.sheet_thickness


def visible_mask(y,cfg):

    return visible_mask_y(y,cfg)


def respawn(mask,state,cfg):

    idx = np.where(mask)[0]

    if idx.size == 0:
        return state

    state.xp[idx] = np.random.uniform(-0.5*cfg.Lx,0.5*cfg.Lx,idx.size)

    zp_new = np.random.normal(0,0.12*(cfg.zmax-cfg.zmin),idx.size)
    state.zp[idx] = np.clip(zp_new,cfg.zmin,cfg.zmax)

    state.y[idx] = np.random.uniform(
        cfg.sheet_center_y - 0.25*cfg.sheet_thickness,
        cfg.sheet_center_y + 0.25*cfg.sheet_thickness,
        idx.size
    )

    return state


def step_evolution(state,xs,zs,t,cfg):

    state.xp,state.zp = advect_particles_rk2(
        state.xp,state.zp,t,cfg.dt,cfg
    )

    state.c = advect_dye_semilag(state.c,xs,zs,t,cfg.dt,cfg)

    if cfg.enable_out_of_plane:

        state.y = update_y_depth(state.y,cfg.dt,cfg)

        kill = np.abs(state.y - cfg.sheet_center_y) > cfg.y_kill
        state = respawn(kill,state,cfg)
    
    vis = visible_mask_y(state.y,cfg)

    return state,vis


# =========================================================
# Optical Model
# =========================================================

def gaussian_blur_fft(img,sigma):

    if sigma <= 0:
        return img

    H,W = img.shape

    ky = np.fft.fftfreq(H)*2*np.pi
    kx = np.fft.fftfreq(W)*2*np.pi

    KX,KY = np.meshgrid(kx,ky)

    G = np.exp(-0.5*sigma**2*(KX**2+KY**2))

    return np.real(np.fft.ifft2(np.fft.fft2(img)*G))


def render_particles(xp,zp,cfg):

    img = np.zeros((cfg.H,cfg.W),dtype=np.float32)

    px = (xp + 0.5*cfg.Lx)/cfg.Lx*(cfg.W-1)
    pz = (zp - cfg.zmin)/(cfg.zmax-cfg.zmin)*(cfg.H-1)

    ix = np.rint(px).astype(int)
    iz = np.rint(pz).astype(int)

    m = (ix>=0)&(ix<cfg.W)&(iz>=0)&(iz<cfg.H)

    np.add.at(img,(iz[m],ix[m]),cfg.particle_amp)

    img = gaussian_blur_fft(img,cfg.psf_sigma_px)

    return img


def render_dye(c,xs,zs,cfg):

    X,Z = np.meshgrid(xs,zs)

    x0 = cfg.light_source_x_frac*cfg.Lx - 0.5*cfg.Lx
    z0 = cfg.zmax + cfg.light_source_z_above_frac*(cfg.zmax-cfg.zmin)

    d = np.sqrt((X-x0)**2+(Z-z0)**2)+1e-6

    L = 1.0/(d**2+1e-6)

    I = cfg.dye_beta*c*L*np.exp(-cfg.dye_alpha*d*c)

    return gaussian_blur_fft(I,cfg.dye_blur_sigma_px)


def render_total_intensity(state,vis,xs,zs,cfg):

    I_p = render_particles(state.xp[vis],state.zp[vis],cfg)
    I_d = render_dye(state.c,xs,zs,cfg)

    return I_p + I_d


# =========================================================
# Camera Model
# =========================================================

def camera_model(I,cfg):

    I = np.clip(I+cfg.bg,0,None)

    lam = np.clip(I*cfg.gain,0,None)

    noisy = np.random.poisson(lam)/cfg.gain

    noisy += np.random.normal(0,cfg.read_sigma,I.shape)

    return np.clip(noisy,0,255).astype(np.uint8)


def auto_exposure_to_uint8(I,cfg):

    hi = np.percentile(I,cfg.exposure_percentile)

    hi = max(hi,1e-6)

    J = np.clip(I/hi*255,0,255)

    return J.astype(np.uint8)


def encode_frame(I,cfg):

    if cfg.use_camera_model and not cfg.auto_exposure:
        return camera_model(I,cfg)

    return auto_exposure_to_uint8(I,cfg)


# =========================================================
# Initialization
# =========================================================

def init_state(cfg):

    np.random.seed(cfg.seed)

    xp = np.random.uniform(-0.5*cfg.Lx,0.5*cfg.Lx,cfg.N)
    zp = np.random.normal(0,0.12*(cfg.zmax-cfg.zmin),cfg.N)
    zp = np.clip(zp,cfg.zmin,cfg.zmax)

    y = np.random.uniform(
        cfg.sheet_center_y - 0.25*cfg.sheet_thickness,
        cfg.sheet_center_y + 0.25*cfg.sheet_thickness,
        cfg.N
    )

    xs = np.linspace(-0.5*cfg.Lx,0.5*cfg.Lx,cfg.W,endpoint=False)
    zs = np.linspace(cfg.zmin,cfg.zmax,cfg.H)

    X,Z = np.meshgrid(xs,zs)

    c = np.exp(-(X**2+(Z-0.15)**2)/0.02)

    return State(xp,zp,y,c),xs,zs


# =========================================================
# Diagnostics
# =========================================================

def init_diagnostics():

    return dict(
        I_min=[],
        I_max=[],
        I_mean=[],
        visible_frac=[]
    )


def append_diagnostics(diag,I,vis):

    diag["I_min"].append(float(I.min()))
    diag["I_max"].append(float(I.max()))
    diag["I_mean"].append(float(I.mean()))
    diag["visible_frac"].append(float(np.mean(vis)))


# =========================================================
# Forward Simulator
# =========================================================

def forward_simulator(cfg):

    state,xs,zs = init_state(cfg)

    video = np.zeros((cfg.T,cfg.H,cfg.W),dtype=np.uint8)

    diag = init_diagnostics()

    t = 0.0

    for n in range(cfg.T):

        state,vis = step_evolution(state,xs,zs,t,cfg)

        I = render_total_intensity(state,vis,xs,zs,cfg)

        frame = encode_frame(I,cfg)

        video[n] = frame

        append_diagnostics(diag,I,vis)

        t += cfg.dt

    return video,state,diag


# =========================================================
# Export
# =========================================================

def export_video(video,out_dir,fps=20,base="sim"):

    out_dir.mkdir(parents=True,exist_ok=True)

    import imageio.v2 as imageio

    imageio.mimsave(
        str(out_dir/f"{base}.gif"),
        video,
        duration=1.0/fps
    )


# =========================================================
# CLI
# =========================================================

def main():

    cfg = SimConfig()

    video,state,diag = forward_simulator(cfg)

    script_dir = Path(__file__).resolve().parent
    out_dir = script_dir/"Outputs_simulate"

    export_video(video,out_dir)

    print("video shape:",video.shape)
    print("visible fraction:",np.mean(diag["visible_frac"]))
    print("saved to:",out_dir)


if __name__ == "__main__":
    main()
