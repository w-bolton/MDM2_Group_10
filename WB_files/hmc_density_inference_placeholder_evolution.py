"""
hmc_density_inference_v2.py
============================
Weak-Constraint HMC Inference for Density Field rho(x, z, t)
--------------------------------------------------------------
Illuminati Ltd. — MDM2 Project

WHAT THIS DOES
--------------
Infers the dye/density field rho(x, z, t) over a sequence of T+1 frames
given a sequence of observed images b_0, b_1, ..., b_T.

The unknown parameter vector is:
    q = [ rho_0, rho_1, ..., rho_T ]   shape: ((T+1) * N * N,)

Each rho_t is the full 50x50 density field at frame t.
There are no global flow parameters (A, k, gamma) in q — they are fixed
constants in the designed forward model.

THE FORWARD MODEL  A(x) = b
----------------------------
A(x) is the composition of two parts (both fixed, not parameterised):

    1. Evolution model  (this file — team slot available)
       rho_t_predicted = advect(rho_{t-1}; u, w, dt)
       Uses semi-Lagrangian advection with the placeholder R-T velocity field
       at FIXED parameters A_FIXED, K_FIXED, GAMMA_FIXED.

    2. Optical model  (team slot in optical_model())
       b_t = optical_model(rho_t)
       Currently identity. Replace with Lambert-Beer model.

THE POSTERIOR  (potential energy U(q))
---------------------------------------
U(q) = -log p(q | b_0, ..., b_T)

Three terms:

    Likelihood: each rho_t must explain its observed image b_t
        sum_t  (1 / 2*sigma_obs^2) * || b_t - optical_model(rho_t) ||^2

    Dynamical consistency prior: consecutive rho_t must be physically
    consistent — rho_{t+1} should be close to what the evolution model
    predicts from rho_t. This is the soft physics constraint. sigma_dyn
    controls how strictly the physics is enforced.
        sum_{t=0}^{T-1}  (1 / 2*sigma_dyn^2) * || rho_{t+1} - advect(rho_t) ||^2

    Initial condition prior: rho_0 should be close to the known initial
    profile (dense fluid above z=0, light below).
        (1 / 2*sigma_rho^2) * || rho_0 - rho_prior ||^2

HMC STRUCTURE  (following Bradley Gram-Hansen, 2019)
------------------------------------------------------
    Potential  — computes U(q) and grad U(q) via autograd
    Kinetic    — computes K(p) = (1/2)||p||^2 and its gradient
    Integrator — leapfrog steps using Potential and Kinetic
    Metropolis — one full HMC proposal + M-H accept/reject
    HMCSampler — runs the full chain, collects samples, reports diagnostics

TEAM SLOTS
----------
    evolve_density()  — swap in your Lagrangian/advection model
    optical_model()   — swap in your Lambert-Beer optical model

REFERENCE
---------
    Neal, R.M. (2011). MCMC using Hamiltonian dynamics. Ch. 5.
    https://arxiv.org/abs/1206.1901

    Bradley Gram-Hansen (2019). The beginners guide to HMC.
    https://bayesianbrad.github.io/posts/2019_hmc.html
"""

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from Optical_Model import geometry, A_image_torch
import time

# ── Reproducibility ───────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)


# =============================================================================
# SECTION 1 — GRID AND PHYSICAL CONSTANTS
# =============================================================================

GRID_N = 32       # spatial grid is GRID_N x GRID_N - cells per dimension (match optical model)

# Physical domain
# x ∈ [-π, π]  (horizontal)
# z ∈ [-1,  1] (vertical, z=0 is the interface)
x_vals = torch.linspace(-np.pi, np.pi, GRID_N)
z_vals = torch.linspace(-1.0,   1.0,  GRID_N)
# x_vals = torch.linspace(0, 0.2, GRID_N)
# z_vals = torch.linspace(0,   0.4,  GRID_N)
# Physical domain — must match the optical model
TANK_WIDTH  = 0.2   # metres
TANK_HEIGHT = 0.4   # metres

delta_x = TANK_WIDTH / GRID_N # individual cell width
delta_z = TANK_HEIGHT / GRID_N   # individual cell height

grid = grid = {
    "dx": delta_x,
    "dz": delta_z,
    "nx": GRID_N,
    "nz": GRID_N
}
# Precompute geometry once at module load time.
# This is expensive but only needs to happen once.
GEOM = geometry(grid, TANK_WIDTH, TANK_HEIGHT)

# Build 2D coordinate grids.  indexing='ij' means:
#   XX[i, j] = x_vals[i]  (varies along first axis)
#   ZZ[i, j] = z_vals[j]  (varies along second axis)
# So XX and ZZ have shape (N_x, N_z) = (50, 50).
XX, ZZ = torch.meshgrid(x_vals, z_vals, indexing='ij')

DT = 0.05   # timestep Δt between frames

# ── Fixed forward model parameters ───────────────────────────────────────────
# These are NOT unknowns. They define the designed forward model.
# In a real deployment these would be determined from physical knowledge
# of the experiment (e.g. from separate calibration measurements).
A_FIXED     = 1.0
K_FIXED     = 1.0
GAMMA_FIXED = 0.5

# ── Noise hyperparameters ─────────────────────────────────────────────────────
SIGMA_OBS = 0.0003 # 0.05   # observation noise std (camera noise)
SIGMA_DYN = 0.008   # dynamical consistency std (how strictly physics is enforced)
SIGMA_RHO = 0.10   # initial condition prior std (how close rho_0 must be to prior)


# =============================================================================
# SECTION 2 — FORWARD MODEL COMPONENTS
# =============================================================================

# ── 2a. Prior mean for rho_0 ──────────────────────────────────────────────────
def make_prior_density() -> torch.Tensor:
    """
    The PRIOR MEAN used by the inference engine.
    Represent what you belive rho_0 looks like before seeing any images. Should be 
    uninformative - the images should do the work of recovering the true structure
    """
    #return torch.ones(GRID_N, GRID_N) * 0.5
    return 0.5 * (1.0 + torch.tanh(10.0 * ZZ))
def make_true_initial_density() -> torch.Tensor:
    """
    Returns the prior mean for the initial density field rho_0.

    Physics: dense fluid (rho ~ 1) sits above the interface z=0,
    light fluid (rho ~ 0) below.  A smooth tanh profile encodes this.
    A small sinusoidal perturbation seeds the Rayleigh-Taylor instability.

    Shape: (GRID_N, GRID_N) = (N_x, N_z)
    """
    rho_prior = 0.5 * (1.0 + torch.tanh(10.0 * ZZ))
    rho_prior = rho_prior + 0.15 * torch.exp(-5.0 * ZZ**2) * torch.cos(K_FIXED * XX)
    return rho_prior


# ── 2b. Placeholder velocity field  (equations 3-4 from brief) ───────────────

def velocity_field(t: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns the Rayleigh-Taylor placeholder velocity field at time t.

    This is the FIXED designed forward model — A_FIXED, K_FIXED, GAMMA_FIXED
    are constants, not unknowns.

    Equations (3)-(4) from the brief:
        u(x,z,t) = Re( i·k·A · e^{±kz} · e^{ikx} · e^{γt} )
        w(x,z,t) = Re( ∓k·A · e^{±kz} · e^{ikx} · e^{γt} )
    upper sign for z > 0, lower for z < 0.

    Parameters
    ----------
    t : physical time at which to evaluate the velocity field

    Returns
    -------
    u : (GRID_N, GRID_N) horizontal velocity
    w : (GRID_N, GRID_N) vertical velocity
    """
    A     = torch.tensor(A_FIXED,     dtype=torch.float32)
    k     = torch.tensor(K_FIXED,     dtype=torch.float32)
    gamma = torch.tensor(GAMMA_FIXED, dtype=torch.float32)

    growth  = torch.exp(gamma * t)
    cos_kx  = torch.cos(k * XX)
    sin_kx  = torch.sin(k * XX)

    # Vertical structure: exponential decay away from interface
    mask_above = (ZZ >= 0).float()
    mask_below = 1.0 - mask_above
    e_kz = mask_above * torch.exp(-k * ZZ) + mask_below * torch.exp(k * ZZ)

    # u = Re(ik·A·e^{±kz}·e^{ikx}·e^{γt}) = -A·sin(kx)·e^{±kz}·e^{γt}
    u = -A * sin_kx * e_kz * growth

    # w = Re(∓k·A·e^{±kz}·e^{ikx}·e^{γt}) = ∓k·A·cos(kx)·e^{±kz}·e^{γt}
    w_above = -k * A * cos_kx * torch.exp(-k * ZZ) * growth * mask_above
    w_below =  k * A * cos_kx * torch.exp( k * ZZ) * growth * mask_below
    w = w_above + w_below

    return u, w


# ── 2c. Evolution model  ← TEAM SLOT ─────────────────────────────────────────

def evolve_density(rho: torch.Tensor, t: float = 0.0) -> torch.Tensor:
    """
    Advects the density field forward by one timestep DT using the fixed
    velocity field evaluated at time t.

    Method: semi-Lagrangian back-tracing.
    At each grid cell (x_i, z_j), find the source position by tracing
    back along the velocity field, then interpolate rho there:

        x_src = x_i - u(x_i, z_j, t) · DT
        z_src = z_j - w(x_i, z_j, t) · DT
        rho_new(x_i, z_j) = interpolate(rho, x_src, z_src)

    This is mass-conservative to first order.

    TEAM SLOT: Replace the body of this function with your preferred
    advection scheme (e.g. RK4 back-tracing, mass-conserving remapping).
    Keep the signature:  rho (N_x, N_z) -> rho_new (N_x, N_z).

    Parameters
    ----------
    rho : torch.Tensor, shape (N_x, N_z)
        Density field at current time.
    t   : float
        Current physical time (used to evaluate the velocity field).

    Returns
    -------
    rho_new : torch.Tensor, shape (N_x, N_z)
        Advected density field at time t + DT.
    """
    u, w = velocity_field(t=t)

    # Back-trace source positions
    x_src = XX - u * DT     # where did the parcel at (x_i, z_j) come from?
    z_src = ZZ - w * DT

    # Normalise to [-1, 1] for torch grid_sample
    x_norm = (x_src / np.pi).clamp(-1.0, 1.0)
    z_norm = (z_src / 1.0 ).clamp(-1.0, 1.0)

    # ── grid_sample axis convention ───────────────────────────────────────
    # grid_sample treats its input as (batch, channel, HEIGHT, WIDTH),
    # i.e. dim -2 = z-axis (rows), dim -1 = x-axis (columns).
    # Our rho is stored as (N_x, N_z), so we MUST transpose before passing
    # it in, and transpose the result back.
    # The grid argument has shape (1, N_z, N_x, 2) where the last dim is
    # [x_coord, z_coord] in grid_sample's convention (x first, despite
    # the spatial layout being (H=z, W=x)).
    # ─────────────────────────────────────────────────────────────────────
    rho_input = rho.T.unsqueeze(0).unsqueeze(0)          # (1, 1, N_z, N_x)
    grid      = torch.stack([x_norm.T, z_norm.T], dim=-1).unsqueeze(0)  # (1, N_z, N_x, 2)

    rho_new_zx = F.grid_sample(
        rho_input, grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    ).squeeze(0).squeeze(0)                              # (N_z, N_x)

    return rho_new_zx.T                                  # (N_x, N_z)


# ── 2d. Optical model  ← TEAM SLOT ───────────────────────────────────────────

def optical_model(rho: torch.Tensor) -> torch.Tensor:
    """
    Maps a density/dye-concentration field to observed pixel intensities.

    TEAM SLOT: Replace with the Lambert-Beer absorption model.
    The full model should account for:
        1. Laser intensity decay with distance from source
        2. Lambert-Beer absorption along each light ray:
               I(s) = I_0 · exp(-mu · integral_0^s c(r) dr)
           where c(r) is dye concentration and mu is absorption coefficient
        3. Camera response (linear CCD: output proportional to incident I)

    Keep the signature: rho (N_x, N_z) -> image (N_x, N_z)

    Parameters
    ----------
    rho : torch.Tensor, shape (N_x, N_z)

    Returns
    -------
    image : torch.Tensor, shape (N_x, N_z)
    """
    """
    Maps dye concentration field to observed pixel intensities.
    Uses the Lambert-Beer fluorescence model with ray geometry.

    Parameters
    ----------
    rho : torch.Tensor, shape (N_x, N_z)  — your internal (x, z) convention

    Returns
    -------
    image : torch.Tensor, shape (N_x, N_z)
    """
    # Their model expects (nz, nx) — transpose from your (nx, nz)
    state_nz_nx = rho.T    # shape (N_z, N_x) = (nz, nx)

    # Run their differentiable forward model
    image_nz_nx = A_image_torch(state_nz_nx, GEOM)   # shape (nz, nx)

    # Transpose back to your (nx, nz) convention
    return image_nz_nx.T   # shape (N_x, N_z)


# ── 2e. Helper: run full forward trajectory ───────────────────────────────────

def forward_trajectory(rho_0: torch.Tensor,
                        n_frames: int) -> list[torch.Tensor]:
    """
    Runs the evolution model forward for n_frames steps from rho_0,
    using gradient checkpointing to prevent stack overflow when n_frames
    is large.

    Gradient checkpointing (torch_checkpoint) breaks the computation graph
    into segments: rather than storing the full chain of intermediate
    activations for backprop, it recomputes them on the backward pass.
    This trades compute for memory and avoids C-stack overflow from deeply
    chained grid_sample operations.

    Parameters
    ----------
    rho_0    : torch.Tensor, shape (N_x, N_z)
    n_frames : int, number of steps to take (produces n_frames+1 frames
               including rho_0)

    Returns
    -------
    trajectory : list of torch.Tensor, length n_frames+1
                 trajectory[t] = rho_t, shape (N_x, N_z)
    """
    trajectory = [rho_0]
    rho = rho_0

    for step in range(n_frames):
        t_val = float(step * DT)

        # Wrap evolve_density in a closure so torch_checkpoint can call it.
        # torch_checkpoint requires a function and its inputs as separate args.
        def make_step(t):
            def step_fn(r):
                return evolve_density(r, t=t)
            return step_fn

        rho = torch_checkpoint(make_step(t_val), rho, use_reentrant=False)
        trajectory.append(rho)

    return trajectory


# =============================================================================
# SECTION 3 — POTENTIAL ENERGY  U(q)
# =============================================================================

class Potential:
    """
    Computes the potential energy U(q) = -log p(q | b_0, ..., b_T)
    and its gradient with respect to q.

    Following Bradley's article, the potential plays the role of the
    negative log-posterior. The HMC sampler minimises U(q) to find
    high-probability regions of the posterior.

    Parameter vector q layout
    --------------------------
    q is a flat 1D tensor of length (T+1) * N * N.
    Internally it is reshaped to (T+1, N, N) where q[t] = rho_t.

    U(q) has three terms
    ---------------------
    1. Observation likelihood  (each rho_t must explain its image b_t)
           L(q) = sum_t  (1/2 sigma_obs^2) * || b_t - optical_model(rho_t) ||^2

    2. Dynamical consistency prior  (consecutive frames must be physically
       consistent — soft enforcement of the evolution model)
           D(q) = sum_t  (1/2 sigma_dyn^2) * || rho_{t+1} - advect(rho_t) ||^2

    3. Initial condition prior  (rho_0 should be near the known prior profile)
           P(q) = (1/2 sigma_rho^2) * || rho_0 - rho_prior ||^2

    So  U(q) = L(q) + D(q) + P(q)

    Note on the dynamical consistency prior
    ----------------------------------------
    This is the key term that makes the inference physically meaningful.
    Without it, each rho_t is inferred independently from its image alone —
    which is an extremely ill-posed problem (many density fields can produce
    the same image). The dynamical prior couples the frames: rho_{t+1} must
    be close to what the evolution model predicts from rho_t, allowing only
    small deviations that represent sub-grid-scale flow the model cannot
    resolve.

    sigma_dyn controls the physics enforcement strength:
        sigma_dyn → 0   : evolution model obeyed exactly (strong constraint)
        sigma_dyn → inf : each frame inferred independently (no physics)
        sigma_dyn ~ 0.1 : frames guided but not rigidly constrained (recommended)
    """

    def __init__(self,
                 b_obs_sequence: list[torch.Tensor],
                 rho_prior:      torch.Tensor,
                 sigma_obs:      float = SIGMA_OBS,
                 sigma_dyn:      float = SIGMA_DYN,
                 sigma_rho:      float = SIGMA_RHO):
        """
        Parameters
        ----------
        b_obs_sequence : list of T+1 tensors, each shape (N_x, N_z)
                         The observed images at each frame.
        rho_prior      : torch.Tensor, shape (N_x, N_z)
                         Prior mean for the initial density field rho_0.
        sigma_obs      : float, observation noise std
        sigma_dyn      : float, dynamical consistency std
        sigma_rho      : float, initial condition prior std
        """
        self.b_obs_sequence = b_obs_sequence
        self.rho_prior      = rho_prior
        self.sigma_obs      = sigma_obs
        self.sigma_dyn      = sigma_dyn
        self.sigma_rho      = sigma_rho
        self.T              = len(b_obs_sequence) - 1  # number of intervals
        self.D              = (self.T + 1) * GRID_N * GRID_N  # total parameter dim

    def unpack(self, q: torch.Tensor) -> list[torch.Tensor]:
        """
        Reshapes the flat parameter vector q into a list of density fields.

        q has shape ((T+1)*N*N,). We split it into T+1 chunks of N*N each,
        then reshape each to (N_x, N_z).

        Parameters
        ----------
        q : torch.Tensor, shape (D,) where D = (T+1)*N*N

        Returns
        -------
        rho_list : list of T+1 tensors, each shape (N_x, N_z)
        """
        return [q[t * GRID_N**2 : (t+1) * GRID_N**2].reshape(GRID_N, GRID_N)
                for t in range(self.T + 1)]

    def eval(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluates the potential energy U(q) and its gradient grad_q U(q).

        Following Bradley's convention: we compute log_joint = log p(q | b)
        and return both -log_joint (the potential U) and +grad_q log_joint
        (the gradient of the log posterior, positive sign).

        The leapfrog uses:
            p ← p + (ε/2) * grad_log_joint    [half momentum step]
        which equals:
            p ← p - (ε/2) * grad_U            [in standard notation]
        Both are equivalent; we follow Bradley's positive-gradient convention.

        Parameters
        ----------
        q : torch.Tensor, shape (D,), requires_grad should be True for autograd

        Returns
        -------
        U          : scalar tensor, the potential energy -log p(q | b)
        grad_U_neg : tensor shape (D,), the gradient of log_posterior w.r.t. q
                     (i.e. -∂U/∂q = ∂log_p/∂q, positive sign following Bradley)
        """
        # Ensure q has gradient tracking so autograd can compute ∂U/∂q
        q_leaf = q.detach().requires_grad_(True)

        rho_list = self.unpack(q_leaf)

        # ── Term 1: Observation likelihood ───────────────────────────────────
        # For each frame t, compute the predicted image from rho_t via the
        # optical model and compare to the observed image b_obs_t.
        # Each observed frame constrains its own density field independently.
        log_likelihood = torch.tensor(0.0)
        for t, (rho_t, b_obs_t) in enumerate(zip(rho_list, self.b_obs_sequence)):
            b_pred_t   = optical_model(rho_t)
            residual_t = b_obs_t - b_pred_t
            log_likelihood = log_likelihood - (
                0.5 / self.sigma_obs**2 * (residual_t**2).sum()
            )

        # ── Term 2: Dynamical consistency prior ───────────────────────────────
        # For each pair of consecutive frames (rho_t, rho_{t+1}), check that
        # rho_{t+1} is close to what the evolution model predicts from rho_t.
        # This is the key term that enforces temporal physical consistency.
        #
        # We use gradient checkpointing on the advection step to prevent
        # stack overflow when backpropagating through many chained grid_sample
        # operations.
        log_dyn_prior = torch.tensor(0.0)
        for t in range(self.T):
            rho_t      = rho_list[t]
            rho_t1     = rho_list[t + 1]
            t_val      = float(t * DT)

            # rho_t1_predicted: what the evolution model says rho_{t+1} should be
            def make_step(time):
                def step_fn(r):
                    return evolve_density(r, t=time)
                return step_fn

            rho_t1_pred = torch_checkpoint(
                make_step(t_val), rho_t, use_reentrant=False
            )

            # Penalise discrepancy between predicted and actual rho_{t+1}
            discrepancy = rho_t1 - rho_t1_pred
            log_dyn_prior = log_dyn_prior - (
                0.5 / self.sigma_dyn**2 * (discrepancy**2).sum()
            )

        # ── Term 3: Initial condition prior ──────────────────────────────────
        # rho_0 should be close to the known physical prior profile.
        rho_0          = rho_list[0]
        deviation      = rho_0 - self.rho_prior
        log_prior_rho0 = -0.5 / self.sigma_rho**2 * (deviation**2).sum()

        # ── Total log posterior ───────────────────────────────────────────────
        log_joint = log_likelihood + log_dyn_prior + log_prior_rho0

        # ── Gradient via autograd ─────────────────────────────────────────────
        # torch.autograd.grad computes ∂log_joint/∂q_leaf in one backward pass
        # through the entire computation graph (likelihood + dynamical prior +
        # initial condition prior), including through the chained advection steps.
        #
        # create_graph=False : we do not need second-order gradients
        # retain_graph=False : free the graph after this backward pass
        grad_log_joint = torch.autograd.grad(
            outputs=log_joint,
            inputs=q_leaf,
            create_graph=False,
            retain_graph=False
        )[0].detach()

        U = -log_joint.detach()

        # Return (U, grad_log_joint) following Bradley's convention:
        # the leapfrog adds grad_log_joint to momentum (not subtracts grad_U).
        return U, grad_log_joint

    def generate(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Generates the initial state for the HMC sampler.

        Initialises q at the prior mean: each rho_t is set to rho_prior,
        which represents our best guess before looking at any images.

        Returns
        -------
        log_joint_init : scalar tensor, log p(q_0 | b)
        q_init         : tensor shape (D,), initial parameter vector
        grad_init      : tensor shape (D,), initial gradient of log posterior
        dim            : int, dimension of q
        """
        # Initialise all frames at the prior mean density profile
        q_init = self.rho_prior.flatten().repeat(self.T + 1).clone()

        U_init, grad_init = self.eval(q_init)
        log_joint_init    = -U_init

        return log_joint_init, q_init, grad_init, self.D


# =============================================================================
# SECTION 4 — KINETIC ENERGY  K(p)
# =============================================================================

class Kinetic:
    """
    Computes the kinetic energy K(p) and its gradient.

    Following Bradley's article, we use the standard Gaussian kinetic energy:
        K(p) = (1/2) * p^T * M^{-1} * p

    With M = I (identity mass matrix), this simplifies to:
        K(p) = (1/2) * ||p||^2

    This corresponds to sampling momentum from p ~ N(0, M) = N(0, I).

    The mass matrix M controls the scale of momentum in each dimension.
    With M = I, all dimensions are treated equally. A diagonal M with
    different entries per parameter could improve mixing if the posterior
    has very different scales in different directions — this is a future
    tuning option.

    Parameters
    ----------
    p : torch.Tensor, shape (D,), the current momentum vector
    M : torch.Tensor or None
        Mass matrix (positive definite). Defaults to identity.
    """

    def __init__(self, p: torch.Tensor, M: torch.Tensor = None):
        D = p.shape[0]
        if M is not None:
            # Store M^{-1} since K = (1/2) p^T M^{-1} p
            self.M_inv = torch.inverse(M)
        else:
            # M = I, so M^{-1} = I
            self.M_inv = torch.eye(D)

    def gauss_ke(self, p: torch.Tensor,
                 grad: bool = False) -> torch.Tensor:
        """
        Evaluates the Gaussian kinetic energy K(p) = (1/2) p^T M^{-1} p.

        With M = I: K(p) = (1/2) ||p||^2
        Gradient:  ∂K/∂p = M^{-1} p = p   (when M = I)

        In the leapfrog, the position update is:
            q ← q + ε * ∂K/∂p = q + ε * p    (since M = I)

        Parameters
        ----------
        p    : torch.Tensor, shape (D,)
        grad : bool, if True return ∂K/∂p instead of K

        Returns
        -------
        K or ∂K/∂p : scalar tensor or tensor shape (D,)
        """
        p_var = p.detach().requires_grad_(True)
        # K = (1/2) * p^T * M^{-1} * p
        # With M=I and 1D vector: K = 0.5 * dot(p, p)
        K = 0.5 * p_var @ self.M_inv @ p_var

        if grad:
            # ∂K/∂p via autograd — gives M^{-1} p = p when M = I
            grad_K = torch.autograd.grad(
                outputs=K,
                inputs=p_var,
                create_graph=False,
                retain_graph=False
            )[0].detach()
            return grad_K
        else:
            return K.detach()


# =============================================================================
# SECTION 5 — LEAPFROG INTEGRATOR
# =============================================================================

class Integrator:
    """
    Leapfrog (Störmer-Verlet) integrator for Hamiltonian dynamics.

    The leapfrog discretises Hamilton's equations:
        dq/dt =  ∂H/∂p =  ∂K/∂p
        dp/dt = -∂H/∂q = -∂U/∂q = +∂log_p/∂q

    Three-step leapfrog (Neal 2011, equations 10-12):
        1. Half momentum step:
               p(t + ε/2) = p(t) + (ε/2) * ∂log_p/∂q(t)
        2. Full position step:
               q(t + ε)   = q(t) + ε * ∂K/∂p(t + ε/2)
                           = q(t) + ε * p(t + ε/2)    [when M = I]
        3. Half momentum step:
               p(t + ε)   = p(t + ε/2) + (ε/2) * ∂log_p/∂q(t + ε)

    For L > 1 leapfrog steps, steps 1-2 are combined as:
        full momentum step: p ← p + ε * ∂log_p/∂q
        full position step: q ← q + ε * p

    with a half-step at the start and end.

    This integrator is:
        - Symplectic (volume-preserving in phase space)
        - Time-reversible (required for detailed balance in M-H step)
        - Second-order accurate: local error O(ε^3), global error O(ε^2)

    Following Bradley's article, we randomise step_size and traj_size within
    bounds at each call to reduce sensitivity to these hyperparameters.

    Parameters
    ----------
    potential  : Potential object
    min_step   : float, minimum leapfrog step size ε
    max_step   : float, maximum leapfrog step size ε
    min_traj   : int,   minimum number of leapfrog steps L
    max_traj   : int,   maximum number of leapfrog steps L
    """

    def __init__(self,
                 potential: Potential,
                 min_step: float = 0.004,
                 max_step: float = 0.010,
                 min_traj: int   = 8,
                 max_traj: int   = 15):
        self.potential = potential
        self.min_step  = min_step
        self.max_step  = max_step
        self.min_traj  = min_traj
        self.max_traj  = max_traj

    def _sample_step_traj(self) -> tuple[float, int]:
        """
        Randomly samples step size and trajectory length within bounds.
        Randomisation reduces the chance of resonance — a failure mode
        where a fixed step size causes the leapfrog to retrace the same
        path, reducing effective exploration.
        """
        step_size = float(np.random.uniform(self.min_step, self.max_step))
        traj_size = int(np.random.uniform(self.min_traj, self.max_traj))
        return step_size, traj_size

    def leapfrog(self,
                 p_init:    torch.Tensor,
                 q_init:    torch.Tensor,
                 grad_init: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Performs L leapfrog steps from (q_init, p_init).

        The momentum update uses grad_log_joint (= -∂U/∂q = +∂log_p/∂q),
        so momentum is *added* (not subtracted), following Bradley's
        convention:
            p ← p + (ε/2) * grad_log_joint

        Parameters
        ----------
        p_init    : tensor (D,), initial momentum
        q_init    : tensor (D,), initial position (current rho trajectory)
        grad_init : tensor (D,), ∂log_p/∂q at q_init (from previous eval)

        Returns
        -------
        q_new : tensor (D,), proposed new position
        p_new : tensor (D,), proposed new momentum (negated for reversibility)
        """
        step_size, traj_size = self._sample_step_traj()

        # Initialise kinetic energy object with identity mass matrix
        kinetic = Kinetic(p_init, M=None)

        p = p_init.clone()
        q = q_init.clone()

        # ── Step 1: Half momentum step using initial gradient ─────────────
        # p(t + ε/2) = p(t) + (ε/2) * ∂log_p/∂q(t)
        p = p + 0.5 * step_size * grad_init

        # ── Step 2: Full position step ────────────────────────────────────
        # q(t + ε) = q(t) + ε * ∂K/∂p(t + ε/2)
        #          = q(t) + ε * p(t + ε/2)   [since M = I → ∂K/∂p = p]
        q = q + step_size * kinetic.gauss_ke(p, grad=True)

        # ── Steps 3 to L: Alternate full momentum and position steps ─────
        # For each intermediate step (all but the last):
        #   full momentum: p ← p + ε * ∂log_p/∂q(q)
        #   full position: q ← q + ε * p
        for i in range(traj_size - 1):
            # Recompute gradient at new q
            _, grad_q = self.potential.eval(q)

            # Full momentum step
            p = p + step_size * grad_q

            # Full position step
            q = q + step_size * kinetic.gauss_ke(p, grad=True)

        # ── Final half momentum step ──────────────────────────────────────
        # p(t + L·ε) = p(t + (L-½)·ε) + (ε/2) * ∂log_p/∂q(q_final)
        _, grad_final = self.potential.eval(q)
        p = p + 0.5 * step_size * grad_final

        # Negate momentum for time-reversibility.
        # This ensures that the reverse trajectory (starting from (q*, -p*))
        # traces back exactly to (q, p), which is required for the M-H
        # acceptance step to satisfy detailed balance.
        return q, -p


# =============================================================================
# SECTION 6 — METROPOLIS ACCEPTANCE STEP
# =============================================================================

class Metropolis:
    """
    Metropolis-Hastings acceptance step for HMC.

    Given the current state (q, p), proposes a new state (q*, p*) via
    the leapfrog integrator, then accepts or rejects based on the change
    in the Hamiltonian H = U(q) + K(p).

    In exact Hamiltonian dynamics H is conserved, so ΔH = 0 and α = 1
    always. In the discretised leapfrog there is a small error, so ΔH ≠ 0.
    The M-H step corrects for this, ensuring the chain has the correct
    stationary distribution p(q) ∝ exp(-U(q)).

    Acceptance probability:
        α = min(1, exp(H(q, p) - H(q*, p*)))
          = min(1, exp(-U(q) + U(q*) - K(p) + K(p*)))

    Parameters
    ----------
    potential  : Potential object
    integrator : Integrator object
    M          : mass matrix (None for identity)
    """

    def __init__(self,
                 potential:  Potential,
                 integrator: Integrator,
                 M:          torch.Tensor = None):
        self.potential  = potential
        self.integrator = integrator
        self.M          = M
        self.n_accepted = 0
        self.n_total    = 0

    def sample_momentum(self, q: torch.Tensor) -> torch.Tensor:
        """
        Samples a fresh momentum vector p ~ N(0, M) = N(0, I).

        Momentum is sampled independently at each HMC iteration and is
        completely independent of q. This is the 'refreshment' step that
        makes the chain ergodic — without it the chain would follow the
        same energy contour forever.

        Parameters
        ----------
        q : tensor (D,), current position (used only for shape)

        Returns
        -------
        p : tensor (D,), sampled momentum
        """
        return torch.randn_like(q)

    def hamiltonian(self,
                    U:       torch.Tensor,
                    p:       torch.Tensor,
                    kinetic: Kinetic) -> torch.Tensor:
        """
        Computes the Hamiltonian H = U(q) + K(p).

        U is the potential energy = -log_posterior(q).
        K is the kinetic energy   = (1/2)||p||^2.

        H is the total energy of the fictitious physical system that HMC
        simulates. The leapfrog approximately conserves H, and ΔH is used
        in the M-H acceptance criterion.

        Parameters
        ----------
        U       : scalar tensor, potential energy at current q
        p       : tensor (D,), current momentum
        kinetic : Kinetic object

        Returns
        -------
        H : scalar tensor
        """
        K = kinetic.gauss_ke(p, grad=False)
        return U + K

    def acceptance(self,
                   q_current:    torch.Tensor,
                   U_current:    torch.Tensor,
                   grad_current: torch.Tensor
                   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs one complete HMC transition:
            1. Sample fresh momentum
            2. Compute current Hamiltonian H_current
            3. Propose (q*, p*) via leapfrog
            4. Compute proposed Hamiltonian H_proposed
            5. Accept/reject via M-H criterion

        Parameters
        ----------
        q_current    : tensor (D,), current position
        U_current    : scalar tensor, current potential energy
        grad_current : tensor (D,), current gradient ∂log_p/∂q

        Returns
        -------
        q_next    : tensor (D,), next state (proposed or current)
        U_next    : scalar tensor, potential energy at next state
        grad_next : tensor (D,), gradient at next state
        """
        self.n_total += 1

        # ── Sample momentum ───────────────────────────────────────────────
        p_current = self.sample_momentum(q_current)
        kinetic   = Kinetic(p_current, self.M)

        # ── Current Hamiltonian ───────────────────────────────────────────
        H_current = self.hamiltonian(U_current, p_current, kinetic)

        # ── Leapfrog proposal ─────────────────────────────────────────────
        q_proposed, p_proposed = self.integrator.leapfrog(
            p_current, q_current, grad_current
        )

        # ── Proposed Hamiltonian ──────────────────────────────────────────
        U_proposed, grad_proposed = self.potential.eval(q_proposed)
        H_proposed = self.hamiltonian(U_proposed, p_proposed, kinetic)

        # ── Metropolis-Hastings acceptance ────────────────────────────────
        # Accept with probability min(1, exp(H_current - H_proposed)).
        # In log space: accept if log(u) < H_current - H_proposed,
        # where u ~ Uniform(0, 1).
        #
        # Note the sign: we accept if H has DECREASED (energy fell),
        # but due to the min(1, ...) we also always accept if it fell.
        # Proposals that increase H are accepted with probability < 1.
        log_alpha = (H_current - H_proposed).item()
        log_u     = float(np.log(np.random.uniform()))

        if log_u < log_alpha:
            self.n_accepted += 1
            return q_proposed.detach(), U_proposed.detach(), grad_proposed.detach()
        else:
            return q_current.detach(), U_current.detach(), grad_current.detach()

    @property
    def acceptance_rate(self) -> float:
        """Running acceptance rate since initialisation."""
        if self.n_total == 0:
            return 0.0
        return self.n_accepted / self.n_total


# =============================================================================
# SECTION 7 — HMC SAMPLER
# =============================================================================

class HMCSampler:
    """
    Full HMC sampler for density field inference.

    Orchestrates Potential, Kinetic, Integrator, and Metropolis to run
    a complete Markov chain over the density trajectory
    q = [rho_0, rho_1, ..., rho_T].

    Following Bradley's run_sampler structure:
        1. Initialise q at the prior mean
        2. Run n_warmup + n_samples HMC steps
        3. Discard warmup samples
        4. Return posterior samples and diagnostics

    Parameters
    ----------
    b_obs_sequence : list of T+1 observed images, each shape (N_x, N_z)
    n_samples      : number of posterior samples to collect after warmup
    n_warmup       : number of burn-in / warmup steps (discarded)
    min_step       : minimum leapfrog step size ε
    max_step       : maximum leapfrog step size ε
    min_traj       : minimum leapfrog trajectory length L
    max_traj       : maximum leapfrog trajectory length L
    M              : mass matrix (None for identity)
    """

    def __init__(self,
                 b_obs_sequence: list[torch.Tensor],
                 n_samples:  int   = 200,
                 n_warmup:   int   = 100,
                 min_step:   float = 0.004,
                 max_step:   float = 0.010,
                 min_traj:   int   = 8,
                 max_traj:   int   = 15,
                 M:          torch.Tensor = None):

        self.n_samples = n_samples
        self.n_warmup  = n_warmup


        # Build the prior density profile
        rho_prior = make_prior_density()

        # Potential encapsulates the entire posterior
        self.potential  = Potential(b_obs_sequence, rho_prior)

        # Integrator performs the leapfrog steps
        self.integrator = Integrator(
            self.potential, min_step, max_step, min_traj, max_traj
        )

        # Metropolis performs the accept/reject
        self.metropolis = Metropolis(self.potential, self.integrator, M)

        self.T = self.potential.T
        self.D = self.potential.D

    def run_sampler(self) -> dict:
        """
        Runs the full HMC chain.

        Structure follows Bradley's run_sampler:
            - Initialise from prior
            - Iterate: compute grad → leapfrog → accept/reject → store
            - Report acceptance rate and posterior statistics

        Returns
        -------
        results : dict with keys
            'samples'        : tensor (n_samples, D), all posterior samples
            'rho_mean'       : tensor (T+1, N, N), posterior mean density trajectory
            'rho_std'        : tensor (T+1, N, N), posterior std of density trajectory
            'accept_rate'    : float, fraction of proposals accepted
            'log_post_trace' : list of float, log-posterior at each sample
        """
        print('\n' + '='*60)
        print('HMC Sampler — Density Field Inference')
        print('='*60)
        print(f'  Frames (T+1)   : {self.T + 1}')
        print(f'  Parameter dim  : {self.D:,}  ({self.T+1} x {GRID_N}x{GRID_N})')
        print(f'  Warmup steps   : {self.n_warmup}')
        print(f'  Sample steps   : {self.n_samples}')
        print(f'  sigma_obs      : {SIGMA_OBS}')
        print(f'  sigma_dyn      : {SIGMA_DYN}')
        print(f'  sigma_rho      : {SIGMA_RHO}')
        print('='*60 + '\n')

        # ── Initialise from prior ─────────────────────────────────────────
        # Following Bradley: generate() returns the initial log_joint,
        # parameter vector, gradient, and dimension.
        log_joint_init, q, grad, dim = self.potential.generate()
        U = -log_joint_init

        total_steps = self.n_warmup + self.n_samples
        samples      = []
        log_post_trace = []

        # ── Main loop ─────────────────────────────────────────────────────
        for i in range(total_steps):
            # One full HMC step: leapfrog + M-H accept/reject
            q, U, grad = self.metropolis.acceptance(q, U, grad)

            # Store samples after warmup
            if i >= self.n_warmup:
                samples.append(q.clone())
                log_post_trace.append(-U.item())

            # Progress report
            if (i + 1) % max(1, total_steps // 10) == 0:
                phase = 'warmup' if i < self.n_warmup else 'sampling'
                print(f'  Step {i+1:4d}/{total_steps}  [{phase}]'
                      f'  accept_rate={self.metropolis.acceptance_rate:.3f}'
                      f'  log_post={-U.item():.1f}')

        # ── Collect results ───────────────────────────────────────────────
        samples_tensor = torch.stack(samples, dim=0)   # (n_samples, D)

        # Reshape samples to (n_samples, T+1, N, N)
        samples_4d = samples_tensor.reshape(
            self.n_samples, self.T + 1, GRID_N, GRID_N
        )

        rho_mean = samples_4d.mean(dim=0)   # (T+1, N, N)
        rho_std  = samples_4d.std(dim=0)    # (T+1, N, N)

        acc_rate = self.metropolis.acceptance_rate
        print(f'\n  Final acceptance rate: {acc_rate:.3f}')
        print(f'  (Target: 0.60-0.80. '
              f'Too low -> increase min/max_step. '
              f'Too high -> decrease min/max_step.)')

        return {
            'samples':        samples_tensor,    # (n_samples, D)
            'rho_mean':       rho_mean,           # (T+1, N, N)
            'rho_std':        rho_std,            # (T+1, N, N)
            'accept_rate':    acc_rate,
            'log_post_trace': log_post_trace,
        }


# =============================================================================
# SECTION 8 — SYNTHETIC DATA GENERATION
# =============================================================================

def generate_synthetic_sequence(n_frames:   int   = 4,
                                 noise_std:  float = SIGMA_OBS
                                 ) -> tuple[list[torch.Tensor],
                                            list[torch.Tensor]]:
    """
    Generates a synthetic observed image sequence for testing.

    Runs the forward model forward for n_frames steps from the prior
    density profile using the fixed parameters (A_FIXED, K_FIXED,
    GAMMA_FIXED), then adds Gaussian observation noise.

    This lets us verify inference: we know the true density trajectory,
    so we can check whether HMC recovers it.

    Parameters
    ----------
    n_frames  : int, number of frames to generate (produces n_frames+1
                images including the initial frame)
    noise_std : float, observation noise standard deviation

    Returns
    -------
    b_obs_sequence  : list of n_frames+1 noisy observed images
    rho_true_sequence : list of n_frames+1 true (noiseless) density fields
    """
    rho0 = make_true_initial_density()
    rho_true_sequence = forward_trajectory(rho0, n_frames=n_frames)

    b_obs_sequence = []
    for rho_t in rho_true_sequence:
        b_clean = optical_model(rho_t)
        b_noisy = b_clean + noise_std * torch.randn_like(b_clean)
        b_obs_sequence.append(b_noisy.detach())

    return b_obs_sequence, [r.detach() for r in rho_true_sequence]


# =============================================================================
# SECTION 9 — DIAGNOSTICS AND PLOTTING
# =============================================================================

def plot_results(results:           dict,
                 b_obs_sequence:    list[torch.Tensor],
                 rho_true_sequence: list[torch.Tensor],
                 save_path:         str = 'hmc_v2_results.png'):
    """
    Produces a diagnostic figure with three rows:

    Row 1: True density trajectory rho_t (ground truth)
    Row 2: Posterior mean density trajectory E[rho_t | b]
    Row 3: Posterior std sigma[rho_t | b]  (uncertainty)

    Each column is one timestep t = 0, 1, ..., T.

    A correct inference should show:
        Row 1 ≈ Row 2  (posterior mean close to truth)
        Row 3 small    (low uncertainty)
        Row 3 largest near the interface (z=0) — this is physically expected,
        as the density gradient is steepest there and the optical model is
        most nonlinear.
    """
    rho_mean = results['rho_mean']    # (T+1, N, N)
    rho_std  = results['rho_std']     # (T+1, N, N)
    T_plus_1 = rho_mean.shape[0]

    fig = plt.figure(figsize=(4 * T_plus_1, 19))
    fig.suptitle(
        f'HMC Density Field Inference  |  '
        f'Accept rate: {results["accept_rate"]:.2f}  |  '
        f'T+1={T_plus_1} frames',
        fontsize=12, fontweight='bold'
    )
    gs = gridspec.GridSpec(6, T_plus_1, figure=fig,
                           hspace=0.35, wspace=0.25)

    def _show(ax, data, title, cmap='viridis', vmin=None, vmax=None):
        arr = data.numpy() if isinstance(data, torch.Tensor) else data
        im = ax.imshow(arr.T, origin='lower', cmap=cmap, aspect='auto',
                       extent=[-np.pi, np.pi, -1, 1],
                       vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('x', fontsize=8)
        ax.set_ylabel('z', fontsize=8)
        ax.axhline(0, color='white', lw=0.8, ls='--', alpha=0.5)
        return im

    vmin_rho = 0.0
    vmax_rho = 1.0
    best_idx = int(np.argmax(results['log_post_trace']))
    rho_map  = results['samples'][best_idx].reshape(T_plus_1, GRID_N, GRID_N)
    for t in range(T_plus_1):
        # Row 1: true density
        ax1 = fig.add_subplot(gs[0, t])
        im1 = _show(ax1, rho_true_sequence[t],
                    f'True rho_t  t={t}', vmin=vmin_rho, vmax=vmax_rho)
        if t == 0:
            ax1.set_ylabel('True rho', fontsize=9) 
        if t == T_plus_1 - 1:
            plt.colorbar(im1, ax=ax1, fraction=0.046)

        # Row 2: posterior mean
        ax2 = fig.add_subplot(gs[1, t])
        im2 = _show(ax2, rho_mean[t],
                    f'Post. mean  t={t}', vmin=vmin_rho, vmax=vmax_rho)
        if t == 0:
            ax2.set_ylabel('Post. mean', fontsize=9)
        if t == T_plus_1 - 1:
            plt.colorbar(im2, ax=ax2, fraction=0.046)

        # Row 3: posterior std
        ax3 = fig.add_subplot(gs[2, t])
        im3 = _show(ax3, rho_std[t],
                    f'Post. std  t={t}', cmap='hot', vmin=0)
        if t == 0:
            ax3.set_ylabel('Post. std', fontsize=9)
        if t == T_plus_1 - 1:
            plt.colorbar(im3, ax=ax3, fraction=0.046)
        # Row 4: observed image b_obs
        ax4 = fig.add_subplot(gs[3, t])
        im4 = _show(ax4, b_obs_sequence[t], f'Observed b  t={t}',
                    cmap='inferno')
        if t == T_plus_1 - 1:
            plt.colorbar(im4, ax=ax4, fraction=0.046)
        ax5 = fig.add_subplot(gs[4, t])
        with torch.no_grad():
            b_pred_t = optical_model(results['rho_mean'][t])
        im5 = _show(ax5, b_pred_t, f'Predicted b  t={t}', cmap='inferno')
        if t == T_plus_1 - 1:
            plt.colorbar(im5, ax=ax5, fraction=0.046)
        if t == 0:
            ax5.set_ylabel('Predicted b\n(post. mean → optical model)', fontsize=9)
        ax6 = fig.add_subplot(gs[5, t])
        im6 = _show(ax6, rho_map[t], f'MAP  t={t}', vmin=0, vmax=1)
        if t == 0:
            ax6.set_ylabel('MAP', fontsize=9)
        if t == T_plus_1 - 1:
            plt.colorbar(im6, ax=ax6, fraction=0.046)
        

    # Row labels
    # for row, label in enumerate(['True rho', 'Post. mean', 'Post. std']):
    #     fig.add_subplot(gs[row, 0]).set_ylabel(label, fontsize=9)

    plt.savefig(save_path, dpi=140, bbox_inches='tight')
    print(f'\n  Figure saved to {save_path}')

def test_optical_model():
    rho = make_prior_density().requires_grad_(True)
    loss = optical_model(rho).sum()
    loss.backward()

    assert rho.grad is not None, "FAIL: no gradient"
    assert not torch.isnan(rho.grad).any(), "FAIL: NaNs in gradient"
    print(f"Check 1 PASSED — grad norm: {rho.grad.norm():.4f}")

    # ── Check 2: Image looks physically correct ───────────────────────────────────
    with torch.no_grad():
        image = optical_model(make_prior_density())

    plt.figure(figsize=(5, 8))
    plt.imshow(image.numpy().T, origin='lower', cmap='inferno', aspect='auto')
    plt.colorbar(label='Fluorescence intensity')
    plt.title('Check 2: optical model output\n'
            'Expect: bright top (high dye), dark bottom, shadowing from absorption')
    plt.xlabel('x'); plt.ylabel('z')
    plt.savefig('optical_model_check.png', dpi=120, bbox_inches='tight')
    print("Check 2 DONE — saved optical_model_check.png")
# =============================================================================
# SECTION 10 — MAIN
# =============================================================================

if __name__ == '__main__':

    print('='*60)
    print('Illuminati Ltd. — HMC Density Field Inference v2')
    print('='*60)
    test_optical_model()
    b_obs_seq, rho_true_seq = generate_synthetic_sequence(n_frames=2)
    pot = Potential(b_obs_seq, make_prior_density())
    _, q, grad, _ = pot.generate()
    start = time.time()
    U, g = pot.eval(q)
    print(f"Single gradient eval: {time.time()-start:.2f}s")

    # ── Step 1: Generate synthetic observed image sequence ────────────────
    # n_frames=3 gives 4 frames (t=0,1,2,3), 4 * 2500 = 10,000 dimensions.
    # On a full machine increase to n_frames=5 or more.
    print('\n[1] Generating synthetic observation sequence...')
    N_FRAMES = 2
    b_obs_sequence, rho_true_sequence = generate_synthetic_sequence(
        n_frames=N_FRAMES, noise_std=SIGMA_OBS
    )
    print(f'    Generated {len(b_obs_sequence)} frames')
    print(f'    Density range: '
          f'[{rho_true_sequence[0].min():.3f}, '
          f'{rho_true_sequence[0].max():.3f}]')

    # ── Step 2: Run HMC sampler ───────────────────────────────────────────
    # n_samples and n_warmup are small for a quick test run.
    # Increase to n_samples=500, n_warmup=200 for production quality.
    print('\n[2] Running HMC sampler...')
    sampler = HMCSampler(
        b_obs_sequence = b_obs_sequence,
        n_samples  = 50,    # increase for production
        n_warmup   = 50,     # increase for production
        min_step   = 0.001,  # tune: target accept rate 0.60-0.80
        max_step   = 0.003,
        min_traj   = 5,      # tune: longer = better mixing, more compute
        max_traj   = 10,
    )
    results = sampler.run_sampler()

    # ── Step 3: Plot diagnostics ──────────────────────────────────────────
    print('\n[3] Plotting diagnostics...')
    plot_results(results, b_obs_sequence, rho_true_sequence,
                 save_path='hmc_v2_results.png')

    # ── Step 4: Summary ───────────────────────────────────────────────────
    print('\n[4] Summary')
    print(f'  Acceptance rate : {results["accept_rate"]:.3f}')
    print(f'  rho_mean shape  : {tuple(results["rho_mean"].shape)}')
    print(f'  rho_std shape   : {tuple(results["rho_std"].shape)}')
    print()
    print('  Output: results["rho_mean"][t]  -> posterior mean of rho at frame t')
    print('  Output: results["rho_std"][t]   -> posterior uncertainty at frame t')
    print('  Output: results["samples"]      -> full posterior sample matrix')

