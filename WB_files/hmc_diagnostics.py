"""
hmc_diagnostics.py
==================
Accuracy and Multimodality Diagnostics for HMC Density Field Inference
-----------------------------------------------------------------------
Illuminati Ltd. — MDM2 Project

Runs hmc_density_inference_v2.py and produces four diagnostic plots:

    PLOT 1 — Accuracy vs Ground Truth
        RMSE and coverage probability per frame.
        Tells you quantitatively how well HMC recovered the true density.

    PLOT 2 — Posterior Predictive Check
        Runs the posterior mean back through the forward model and compares
        the predicted images to the observations.
        Residuals should look like structureless noise at scale sigma_obs.

    PLOT 3 — Multimodality Check: Two Independent Chains
        Runs two HMC chains from different starting points and compares
        their log-posterior traces and posterior means.
        If the posterior is unimodal, both chains should converge to the
        same solution. If they find different solutions, the posterior is
        multimodal.

    PLOT 4 — Gelman-Rubin R-hat Statistic
        The standard convergence diagnostic for multiple chains.
        R-hat < 1.1 for all parameters means chains have converged to
        the same distribution (consistent with unimodality).
        R-hat >> 1.1 suggests chains found different modes.

IMPORTANT: This file uses a PERTURBED initial condition for the true
density, so that rho_0 is genuinely different from the prior mean.
This makes the t=0 frame informative and testable.

Usage
-----
    python hmc_diagnostics.py

Adjust HMC_KWARGS below to match the settings you used in your main run.
"""

import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Import from the main inference file ──────────────────────────────────────
#sys.path.insert(0, '/home/claude')
from hmc_density_inference_v2 import (
    GRID_N, DT, SIGMA_OBS, SIGMA_DYN, SIGMA_RHO,
    x_vals, z_vals, XX, ZZ,
    make_prior_density, make_true_initial_density, forward_trajectory, optical_model,
    Potential, Integrator, Metropolis, HMCSampler, GEOM, generate_synthetic_sequence,
)

torch.manual_seed(99)
np.random.seed(99)
# Verify immediately
print(f"GEOM dx={GEOM['grid']['dx']:.6f}, nx={GEOM['grid']['nx']}")
print(f"Number of rays: {len(GEOM['rays'])}")

# Quick sanity check — apply optical model to prior density
with torch.no_grad():
    test_image = optical_model(make_prior_density())
print(f"Test image max: {test_image.max():.6f}  (should be ~0.007 for 32x32)")
print(f"Test image min: {test_image.min():.6f}")

# =============================================================================
# SETTINGS — adjust to match your main run
# =============================================================================

N_FRAMES   = 4       # matches your main run (3 frames total)
N_SAMPLES  = 300
N_WARMUP   = 100    # enough warmup for tanh initialisation
MIN_STEP   = 0.002
MAX_STEP   = 0.007
MIN_TRAJ   = 10
MAX_TRAJ   = 20


OUTDIR = 'WB_files/outputs'


# =============================================================================
# SECTION 1 — PERTURBED SYNTHETIC DATA GENERATION
# =============================================================================

def generate_perturbed_sequence(n_frames:        int   = N_FRAMES,
                                 noise_std:       float = SIGMA_OBS,
                                 perturbation_amp: float = 0.15
                                 ) -> tuple[list[torch.Tensor],
                                            list[torch.Tensor]]:
    """
    Generates a synthetic image sequence from a PERTURBED initial condition.

    The true rho_0 is the prior mean PLUS a smooth sinusoidal perturbation.
    This means the t=0 frame is genuinely different from the prior, giving
    HMC something real to infer at t=0 rather than just recovering the prior.

    The perturbation is spatially smooth (low-wavenumber sine waves) so
    it looks physically plausible — not random pixel noise.

    Parameters
    ----------
    n_frames         : number of evolution steps after t=0
    noise_std        : observation noise std added to images
    perturbation_amp : amplitude of the initial perturbation (0 = no perturbation)

    Returns
    -------
    b_obs_sequence   : list of n_frames+1 noisy observed images
    rho_true_sequence: list of n_frames+1 true (noiseless) density fields
    rho_true_0       : the true initial density (for reference)
    """
    # Start from the standard prior profile
    # rho_prior = make_prior_density()

    # # Add a smooth spatial perturbation so rho_0 != prior mean.
    # # We use low-wavenumber modes so the perturbation is physically smooth.
    # # sin(x) * exp(-z^2) concentrates the perturbation near the interface.
    # perturbation = (
    #     perturbation_amp * torch.sin(1.0 * XX) * torch.exp(-3.0 * ZZ**2)
    #     + 0.5 * perturbation_amp * torch.sin(2.0 * XX) * torch.exp(-5.0 * ZZ**2)
    # )
    # rho_true_0 = (rho_prior + perturbation).clamp(0.0, 1.0)
    rho_true_0 = make_true_initial_density()
    # Evolve forward from the perturbed initial condition
    rho_true_sequence = forward_trajectory(rho_true_0, n_frames=n_frames)

    # Apply optical model and add noise
    b_obs_sequence = []
    for rho_t in rho_true_sequence:
        b_clean = optical_model(rho_t)
        b_noisy = b_clean + noise_std * torch.randn_like(b_clean)
        b_obs_sequence.append(b_noisy.detach())

    return (b_obs_sequence,
            [r.detach() for r in rho_true_sequence],
            rho_true_0.detach())


# =============================================================================
# SECTION 2 — RUN A SINGLE HMC CHAIN
# =============================================================================

def run_chain(b_obs_sequence: list[torch.Tensor],
              seed:           int,
              init_noise:     float = 0.0,
              label:          str   = 'Chain'
              ) -> dict:
    """
    Runs one HMC chain with a given random seed and optional initialisation noise.

    For the multimodality test we run two chains:
        Chain 1: initialised at the prior mean (init_noise=0)
        Chain 2: initialised at a randomly perturbed starting point
                 (init_noise > 0)

    If the posterior is unimodal, both chains should find the same solution
    regardless of where they start. If they find different solutions, the
    posterior has multiple modes.

    Parameters
    ----------
    b_obs_sequence : observed image sequence
    seed           : random seed for reproducibility
    init_noise     : std of Gaussian noise added to the initial q
                     (0 = start exactly at prior mean)
    label          : name printed during run

    Returns
    -------
    results dict with keys from HMCSampler.run_sampler(), plus:
        'log_post_trace' : list of log-posterior values at each sample
        'label'          : chain label string
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f'\n{"="*55}')
    print(f'Running {label}  (seed={seed}, init_noise={init_noise})')
    print(f'{"="*55}')

    sampler = HMCSampler(
        b_obs_sequence = b_obs_sequence,
        n_samples = N_SAMPLES,
        n_warmup  = N_WARMUP,
        min_step  = MIN_STEP,
        max_step  = MAX_STEP,
        min_traj  = MIN_TRAJ,
        max_traj  = MAX_TRAJ,
    )

    # If init_noise > 0, perturb the starting point of the chain.
    # We do this by adding noise to q before the first step.
    # This tests whether different starting points converge to the same posterior.
    if init_noise > 0.0:
        _, q_init, grad_init, _ = sampler.potential.generate()
        q_perturbed = q_init + init_noise * torch.randn_like(q_init)
        U_perturbed, grad_perturbed = sampler.potential.eval(q_perturbed)

        # Manually set the starting state of the metropolis object
        # by running one step from the perturbed initialisation
        sampler.metropolis.n_accepted = 0
        sampler.metropolis.n_total    = 0

        # Patch generate() to return perturbed start
        _orig_generate = sampler.potential.generate
        def _perturbed_generate():
            return -U_perturbed, q_perturbed, grad_perturbed, sampler.D
        sampler.potential.generate = _perturbed_generate

    results = sampler.run_sampler()
    results['label'] = label
    return results


# =============================================================================
# SECTION 3 — ACCURACY DIAGNOSTICS
# =============================================================================

def compute_accuracy(results:           dict,
                     rho_true_sequence: list[torch.Tensor]
                     ) -> dict:
    """
    Computes quantitative accuracy metrics against ground truth.

    Metrics
    -------
    RMSE per frame
        Root mean squared error between posterior mean and true density.
            RMSE_t = sqrt( mean( (rho_mean_t - rho_true_t)^2 ) )
        Smaller is better. Compare to sigma_obs as a baseline — if RMSE
        is comparable to sigma_obs the inference is close to the noise floor.

    Prior RMSE per frame
        RMSE of the prior mean against the truth. This is the baseline
        you would achieve without any inference at all. HMC should do
        better than this.

    Coverage probability per frame
        Fraction of grid cells where the true value falls inside the
        95% posterior credible interval [mean - 2*std, mean + 2*std].
        Well-calibrated inference gives coverage ~ 0.95.
        Lower means the posterior is overconfident.
        Higher means the posterior is too wide (underconfident).

    Returns
    -------
    accuracy dict with keys:
        'rmse_per_frame'       : list of float, RMSE at each timestep
        'prior_rmse_per_frame' : list of float, prior RMSE at each timestep
        'coverage_per_frame'   : list of float, 95% coverage at each timestep
        'mean_rmse'            : float, mean RMSE across frames
        'mean_coverage'        : float, mean coverage across frames
    """
    rho_mean = results['rho_mean']   # (T+1, N, N)
    rho_std  = results['rho_std']    # (T+1, N, N)
    rho_prior = make_prior_density()  # (N, N)
    T_plus_1  = rho_mean.shape[0]

    rmse_list       = []
    prior_rmse_list = []
    coverage_list   = []

    for t in range(T_plus_1):
        rho_true_t = rho_true_sequence[t]
        rho_mean_t = rho_mean[t]
        rho_std_t  = rho_std[t]

        # RMSE of posterior mean vs truth
        rmse = float(torch.sqrt(((rho_mean_t - rho_true_t)**2).mean()))
        rmse_list.append(rmse)

        # RMSE of prior mean vs truth (baseline — no inference)
        prior_rmse = float(torch.sqrt(((rho_prior - rho_true_t)**2).mean()))
        prior_rmse_list.append(prior_rmse)

        # 95% credible interval coverage
        # Interval: [mean - 1.96*std,  mean + 1.96*std]
        lo = rho_mean_t - 1.96 * rho_std_t
        hi = rho_mean_t + 1.96 * rho_std_t
        covered = ((rho_true_t >= lo) & (rho_true_t <= hi)).float()
        coverage_list.append(float(covered.mean()))

    return {
        'rmse_per_frame':       rmse_list,
        'prior_rmse_per_frame': prior_rmse_list,
        'coverage_per_frame':   coverage_list,
        'mean_rmse':            float(np.mean(rmse_list)),
        'mean_coverage':        float(np.mean(coverage_list)),
    }


# =============================================================================
# SECTION 4 — GELMAN-RUBIN R-HAT
# =============================================================================

def compute_rhat(samples_chain1: torch.Tensor,
                 samples_chain2: torch.Tensor) -> torch.Tensor:
    """
    Computes the Gelman-Rubin R-hat statistic for each parameter.

    R-hat measures whether two (or more) chains have converged to the
    same distribution. It compares between-chain variance to within-chain
    variance. If both chains are sampling from the same distribution,
    these should be equal, giving R-hat = 1.

    Interpretation:
        R-hat < 1.01  : chains have converged (excellent)
        R-hat < 1.05  : chains have converged (good)
        R-hat < 1.10  : marginal convergence (acceptable)
        R-hat > 1.10  : chains have NOT converged — possible multimodality
                        or insufficient samples

    Reference: Gelman & Rubin (1992), Statistical Science.

    Parameters
    ----------
    samples_chain1 : tensor (n_samples, D)
    samples_chain2 : tensor (n_samples, D)

    Returns
    -------
    rhat : tensor (D,), R-hat for each parameter
    """
    # Both chains must have the same number of samples
    n = min(samples_chain1.shape[0], samples_chain2.shape[0])
    s1 = samples_chain1[:n].double()   # (n, D)
    s2 = samples_chain2[:n].double()   # (n, D)

    # Within-chain means
    mu1 = s1.mean(dim=0)   # (D,)
    mu2 = s2.mean(dim=0)   # (D,)

    # Grand mean across both chains
    mu  = 0.5 * (mu1 + mu2)   # (D,)

    # Between-chain variance B (scaled by n so it estimates the target variance)
    # B = n / (m-1) * sum_j (mu_j - mu)^2   where m = number of chains = 2
    B = n * (((mu1 - mu)**2 + (mu2 - mu)**2) / 1.0)   # (D,)

    # Within-chain variance W (mean of per-chain sample variances)
    W = 0.5 * (s1.var(dim=0) + s2.var(dim=0))   # (D,)

    # Pooled variance estimate: weighted average of W and B/n
    var_hat = (1.0 - 1.0/n) * W + (1.0/n) * B   # (D,)

    # R-hat: ratio of pooled variance to within-chain variance
    # R-hat = 1 when chains have converged (W = var_hat)
    rhat = torch.sqrt(var_hat / (W + 1e-10))   # (D,), add epsilon for stability

    return rhat.float()


# =============================================================================
# SECTION 5 — PLOT 1: ACCURACY VS GROUND TRUTH
# =============================================================================

def plot_accuracy(accuracy:          dict,
                  results:           dict,
                  rho_true_sequence: list[torch.Tensor]):
    """
    Four-panel accuracy plot.

    Panel A  RMSE per frame vs prior RMSE
        Bar chart showing how much better HMC is than just using the prior.
        HMC bars should be shorter (smaller RMSE) than prior bars.
        The dashed line marks sigma_obs — the noise floor below which
        you cannot do better regardless of method.

    Panel B  Coverage probability per frame
        Should be close to 0.95 (the 95% credible interval should contain
        the truth 95% of the time).
        Too low: posterior is overconfident.
        Too high: posterior is too wide.

    Panel C  True vs posterior mean at final frame (t=T)
        Visual comparison of what HMC recovered vs ground truth.

    Panel D  Error map at final frame: |rho_mean - rho_true|
        Spatially shows where the inference was most and least accurate.
        Error should be small everywhere and largest near the interface.
    """
    T_plus_1 = results['rho_mean'].shape[0]
    t_vals   = list(range(T_plus_1))

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Plot 1: Accuracy vs Ground Truth', fontsize=13, fontweight='bold')
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── Panel A: RMSE bar chart ───────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    bar_w = 0.35
    x_pos = np.arange(T_plus_1)
    bars_hmc   = ax_a.bar(x_pos - bar_w/2, accuracy['rmse_per_frame'],
                          bar_w, label='HMC posterior mean', color='steelblue',
                          alpha=0.85, edgecolor='white')
    bars_prior = ax_a.bar(x_pos + bar_w/2, accuracy['prior_rmse_per_frame'],
                          bar_w, label='Prior mean (no inference)', color='grey',
                          alpha=0.65, edgecolor='white')
    # ax_a.axhline(SIGMA_OBS, color='crimson', lw=1.8, ls='--',
    #              label=f'σ_obs = {SIGMA_OBS} (noise floor)')
    ax_a.set_xticks(x_pos)
    ax_a.set_xticklabels([f't={t}' for t in t_vals], fontsize=9)
    ax_a.set_ylabel('RMSE', fontsize=9)
    ax_a.set_title('A: RMSE per frame\n(HMC bar should be shorter than Prior bar)',
                   fontsize=10, fontweight='bold')
    ax_a.legend(fontsize=7)
    ax_a.grid(True, alpha=0.2, axis='y')

    # Annotate with values
    for bar in bars_hmc:
        ax_a.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                  f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=7)

    # ── Panel B: Coverage probability ────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.bar(x_pos, accuracy['coverage_per_frame'], color='seagreen',
             alpha=0.75, edgecolor='white')
    ax_b.axhline(0.95, color='crimson', lw=1.8, ls='--', label='Target: 0.95')
    ax_b.axhline(0.90, color='orange',  lw=1.2, ls=':',  label='Lower bound: 0.90')
    ax_b.set_xticks(x_pos)
    ax_b.set_xticklabels([f't={t}' for t in t_vals], fontsize=9)
    ax_b.set_ylabel('Coverage probability', fontsize=9)
    ax_b.set_ylim(0, 1.05)
    ax_b.set_title('B: 95% Credible Interval Coverage\n'
                   '(should be ≈ 0.95 — too low = overconfident)',
                   fontsize=10, fontweight='bold')
    ax_b.legend(fontsize=8)
    ax_b.grid(True, alpha=0.2, axis='y')

    for i, cov in enumerate(accuracy['coverage_per_frame']):
        ax_b.text(i, cov + 0.01, f'{cov:.2f}', ha='center', va='bottom', fontsize=8)

    # ── Panel C: True vs posterior mean (final frame) ─────────────────────
    t_final   = T_plus_1 - 1
    rho_true  = rho_true_sequence[t_final].numpy()
    rho_mean  = results['rho_mean'][t_final].numpy()

    def _imshow(ax, data, title, cmap='viridis', vmin=0, vmax=1):
        im = ax.imshow(data.T, origin='lower', cmap=cmap, aspect='auto',
                       extent=[-np.pi, np.pi, -1, 1], vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlabel('x', fontsize=8); ax.set_ylabel('z', fontsize=8)
        ax.axhline(0, color='white', lw=0.8, ls='--', alpha=0.5)
        return im

    ax_c = fig.add_subplot(gs[1, 0])
    im_c = _imshow(ax_c, rho_true, f'C: True rho  (t={t_final})')
    plt.colorbar(im_c, ax=ax_c, fraction=0.046)

    ax_d = fig.add_subplot(gs[1, 1])
    im_d = _imshow(ax_d, rho_mean, f'D: Posterior mean  (t={t_final})')
    plt.colorbar(im_d, ax=ax_d, fraction=0.046)

    # ── Panel E: Error map ────────────────────────────────────────────────
    ax_e = fig.add_subplot(gs[1, 2])
    err  = np.abs(rho_mean - rho_true)
    im_e = _imshow(ax_e, err, f'E: |Posterior mean − True|  (t={t_final})',
                   cmap='hot', vmin=0, vmax=None)
    im_e.set_clim(0, err.max())
    plt.colorbar(im_e, ax=ax_e, fraction=0.046)
    ax_e.set_title(f'E: Error map  (t={t_final})\n'
                   f'max error = {err.max():.4f},  '
                   f'mean error = {err.mean():.4f}',
                   fontsize=9, fontweight='bold')

    # ── Summary text ──────────────────────────────────────────────────────
    ax_f = fig.add_subplot(gs[0, 2])
    ax_f.axis('off')
    summary = (
        f"ACCURACY SUMMARY\n"
        f"{'─'*30}\n"
        f"Mean RMSE (HMC)  : {accuracy['mean_rmse']:.4f}\n"
        f"Mean RMSE (prior): {np.mean(accuracy['prior_rmse_per_frame']):.4f}\n"
        f"Noise floor σ_obs: {SIGMA_OBS:.4f}\n"
        f"Mean coverage    : {accuracy['mean_coverage']:.3f}\n"
        f"Target coverage  : 0.950\n"
        f"{'─'*30}\n\n"
        f"INTERPRETATION\n"
        f"{'─'*30}\n"
        f"HMC RMSE < Prior RMSE\n  → inference is adding information ✓\n\n"
        f"HMC RMSE ≈ σ_obs\n  → near the noise floor ✓\n\n"
        f"Coverage ≈ 0.95\n  → well-calibrated uncertainty ✓"
    )
    ax_f.text(0.05, 0.95, summary, transform=ax_f.transAxes,
              fontsize=9, verticalalignment='top', fontfamily='monospace',
              bbox=dict(boxstyle='round', facecolor='lightyellow',
                        alpha=0.8, edgecolor='grey'))

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = f'{OUTDIR}/diag_01_accuracy.png'
    fig.savefig(path, dpi=140, bbox_inches='tight')
    print(f'  Saved: {path}')


# =============================================================================
# SECTION 6 — PLOT 2: POSTERIOR PREDICTIVE CHECK
# =============================================================================

def plot_posterior_predictive(results:        dict,
                               b_obs_sequence: list[torch.Tensor]):
    """
    Runs the posterior mean back through the forward model and compares
    predicted images to observed images.

    If inference is correct:
        - Predicted images should look nearly identical to observations
        - Residuals should be structureless Gaussian noise with std ≈ sigma_obs
        - Any spatial pattern in the residuals indicates a systematic model error

    One column per frame. Three rows: observed, predicted, residual.
    """
    T_plus_1 = results['rho_mean'].shape[0]

    fig, axes = plt.subplots(3, T_plus_1, figsize=(4.5 * T_plus_1, 9))
    if T_plus_1 == 1:
        axes = axes[:, np.newaxis]
    fig.suptitle('Plot 2: Posterior Predictive Check\n'
                 'Row 3 residuals should be structureless noise at scale σ_obs',
                 fontsize=12, fontweight='bold')

    residual_stds = []

    for t in range(T_plus_1):
        rho_mean_t = results['rho_mean'][t]
        b_obs_t    = b_obs_sequence[t]
        b_pred_t   = optical_model(rho_mean_t).detach()
        residual_t = b_obs_t - b_pred_t
        residual_stds.append(float(residual_t.std()))

        def _show(ax, data, title, cmap='viridis', vmin=None, vmax=None):
            arr = data.numpy() if isinstance(data, torch.Tensor) else data
            im = ax.imshow(arr.T, origin='lower', cmap=cmap, aspect='auto',
                           extent=[-np.pi, np.pi, -1, 1], vmin=vmin, vmax=vmax)
            ax.set_title(title, fontsize=9, fontweight='bold')
            ax.set_xlabel('x', fontsize=7); ax.set_ylabel('z', fontsize=7)
            return im

        _show(axes[0, t], b_obs_t, f'Observed  t={t}')
        _show(axes[1, t], b_pred_t, f'Predicted  t={t}')

        res_np = residual_t.numpy()
        vmax_r = max(abs(res_np.max()), abs(res_np.min()), 3*SIGMA_OBS)
        im_r = _show(axes[2, t], residual_t,
                     f'Residual  t={t}\nstd={residual_t.std():.4f}  '
                     f'(σ_obs={SIGMA_OBS})',
                     cmap='RdBu_r', vmin=-vmax_r, vmax=vmax_r)

        # Flag if residual std is much larger than sigma_obs
        colour = 'green' if residual_t.std() < 3 * SIGMA_OBS else 'crimson'
        axes[2, t].text(0.03, 0.05,
                        'OK' if residual_t.std() < 3 * SIGMA_OBS
                        else 'CHECK: large residual',
                        transform=axes[2, t].transAxes, fontsize=8,
                        color=colour, fontweight='bold',
                        bbox=dict(facecolor='white', alpha=0.7, pad=2))

    # Row labels
    for row, label in enumerate(['Observed b_obs', 'Predicted b_pred',
                                  'Residual b_obs − b_pred']):
        axes[row, 0].set_ylabel(label, fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = f'{OUTDIR}/diag_02_posterior_predictive.png'
    fig.savefig(path, dpi=140, bbox_inches='tight')
    print(f'  Saved: {path}')
    print(f'  Residual stds per frame: '
          f'{[f"{s:.4f}" for s in residual_stds]}  (target ≈ {SIGMA_OBS})')


# =============================================================================
# SECTION 7 — PLOT 3: TWO-CHAIN MULTIMODALITY CHECK
# =============================================================================

def plot_multimodality(results1: dict, results2: dict):
    """
    Compares two independent HMC chains to check for multimodality.

    If the posterior is UNIMODAL:
        - Both log-posterior traces should fluctuate around the same level
        - Posterior means from both chains should look nearly identical
        - The difference map |mean1 - mean2| should be close to zero everywhere

    If the posterior is MULTIMODAL:
        - Traces converge to different log-posterior levels
        - Posterior means look different (chains found different configurations)
        - Difference map shows large values — the chains found different modes

    Panels:
        A  Log-posterior traces for both chains
        B  Posterior mean from chain 1 (final frame)
        C  Posterior mean from chain 2 (final frame)
        D  Difference |mean1 - mean2| at final frame
        E  Posterior std comparison (both chains, final frame)
        F  Summary statistics
    """
    T_final  = results1['rho_mean'].shape[0] - 1

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Plot 3: Multimodality Check — Two Independent Chains\n'
                 'Chains starting from different points should find the same solution '
                 'if the posterior is unimodal',
                 fontsize=11, fontweight='bold')
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    # ── Panel A: Log-posterior traces ─────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    lp1 = results1['log_post_trace']
    lp2 = results2['log_post_trace']
    ax_a.plot(lp1, lw=0.8, color='steelblue',  alpha=0.9,
              label=results1['label'])
    ax_a.plot(lp2, lw=0.8, color='darkorange', alpha=0.9,
              label=results2['label'])
    ax_a.set_xlabel('Post-warmup sample', fontsize=8)
    ax_a.set_ylabel('Log-posterior', fontsize=8)
    ax_a.set_title('A: Log-posterior traces\n'
                   '(both should fluctuate at same level if unimodal)',
                   fontsize=9, fontweight='bold')
    ax_a.legend(fontsize=8)
    ax_a.grid(True, alpha=0.2)

    # Annotate whether they appear to agree
    mean_diff = abs(np.mean(lp1) - np.mean(lp2))
    converged = mean_diff < 50  # heuristic threshold
    ax_a.text(0.97, 0.05,
              f'Mean difference: {mean_diff:.1f}\n'
              f'{"→ Likely unimodal ✓" if converged else "→ Possible multimodality ⚠"}',
              transform=ax_a.transAxes, ha='right', va='bottom', fontsize=8,
              color='green' if converged else 'crimson',
              bbox=dict(facecolor='white', alpha=0.8, pad=3))

    # ── Panels B, C, D: Posterior mean comparison at final frame ──────────
    mean1 = results1['rho_mean'][T_final].numpy()
    mean2 = results2['rho_mean'][T_final].numpy()
    diff  = np.abs(mean1 - mean2)

    def _imshow(ax, data, title, cmap='viridis', vmin=0, vmax=1):
        im = ax.imshow(data.T, origin='lower', cmap=cmap, aspect='auto',
                       extent=[-np.pi, np.pi, -1, 1], vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.set_xlabel('x', fontsize=7); ax.set_ylabel('z', fontsize=7)
        ax.axhline(0, color='white', lw=0.6, ls='--', alpha=0.5)
        return im

    ax_b = fig.add_subplot(gs[0, 1])
    im_b = _imshow(ax_b, mean1, f'B: {results1["label"]} mean  (t={T_final})')
    plt.colorbar(im_b, ax=ax_b, fraction=0.046)

    ax_c = fig.add_subplot(gs[0, 2])
    im_c = _imshow(ax_c, mean2, f'C: {results2["label"]} mean  (t={T_final})')
    plt.colorbar(im_c, ax=ax_c, fraction=0.046)

    ax_d = fig.add_subplot(gs[1, 0])
    im_d = _imshow(ax_d, diff,
                   f'D: |Chain1 − Chain2|  (t={T_final})\n'
                   f'max={diff.max():.4f},  mean={diff.mean():.4f}\n'
                   f'{"→ Chains agree ✓" if diff.mean() < 0.05 else "→ Chains disagree ⚠"}',
                   cmap='hot', vmin=0, vmax=None)
    im_d.set_clim(0, max(diff.max(), 0.01))
    plt.colorbar(im_d, ax=ax_d, fraction=0.046)

    # ── Panel E: Posterior std comparison ─────────────────────────────────
    ax_e = fig.add_subplot(gs[1, 1])
    std1 = results1['rho_std'][T_final].numpy().flatten()
    std2 = results2['rho_std'][T_final].numpy().flatten()
    ax_e.scatter(std1, std2, s=1, alpha=0.3, color='steelblue')
    lo = min(std1.min(), std2.min())
    hi = max(std1.max(), std2.max())
    ax_e.plot([lo, hi], [lo, hi], 'r--', lw=1.5, label='y = x (perfect agreement)')
    ax_e.set_xlabel(f'{results1["label"]} posterior std', fontsize=8)
    ax_e.set_ylabel(f'{results2["label"]} posterior std', fontsize=8)
    ax_e.set_title('E: Posterior std scatter  (t=final)\n'
                   '(points on diagonal = same uncertainty in both chains)',
                   fontsize=9, fontweight='bold')
    ax_e.legend(fontsize=7)

    # ── Panel F: Summary ──────────────────────────────────────────────────
    ax_f = fig.add_subplot(gs[1, 2])
    ax_f.axis('off')
    summary = (
        f"MULTIMODALITY SUMMARY\n"
        f"{'─'*32}\n"
        f"Chain 1 mean log-post : {np.mean(lp1):.1f}\n"
        f"Chain 2 mean log-post : {np.mean(lp2):.1f}\n"
        f"Difference            : {mean_diff:.1f}\n\n"
        f"Mean |chain1-chain2|  : {diff.mean():.4f}\n"
        f"Max  |chain1-chain2|  : {diff.max():.4f}\n\n"
        f"Chain 1 accept rate   : {results1['accept_rate']:.3f}\n"
        f"Chain 2 accept rate   : {results2['accept_rate']:.3f}\n"
        f"{'─'*32}\n\n"
        f"VERDICT\n"
        f"{'─'*32}\n"
        f"{'Chains agree → posterior likely unimodal ✓' if diff.mean() < 0.05 else 'Chains differ → possible multimodality ⚠'}"
    )
    ax_f.text(0.05, 0.95, summary, transform=ax_f.transAxes,
              fontsize=8.5, verticalalignment='top', fontfamily='monospace',
              bbox=dict(boxstyle='round', facecolor='lightyellow',
                        alpha=0.8, edgecolor='grey'))

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path = f'{OUTDIR}/diag_03_multimodality.png'
    fig.savefig(path, dpi=140, bbox_inches='tight')
    print(f'  Saved: {path}')


# =============================================================================
# SECTION 8 — PLOT 4: GELMAN-RUBIN R-HAT
# =============================================================================

def plot_rhat(results1: dict, results2: dict):
    """
    Plots the Gelman-Rubin R-hat statistic across all parameters and frames.

    R-hat is computed for every parameter (every cell in every frame).
    We show it as a histogram across all parameters, and as a spatial
    map per frame so you can see WHERE the chains disagree most.

    R-hat < 1.1 for all parameters is the standard convergence criterion.

    Panels:
        A  Histogram of R-hat across all parameters
        B  Spatial R-hat map per frame (one subplot per frame)
           — shows WHERE in the density field the chains disagree
    """
    rhat = compute_rhat(results1['samples'], results2['samples'])   # (D,)
    T_plus_1 = results1['rho_mean'].shape[0]

    # Reshape R-hat to (T+1, N, N) for spatial plotting
    rhat_spatial = rhat.reshape(T_plus_1, GRID_N, GRID_N)

    max_rhat  = float(rhat.max())
    mean_rhat = float(rhat.mean())
    frac_ok   = float((rhat < 1.1).float().mean())

    fig = plt.figure(figsize=(5 + 4 * T_plus_1, 8))
    fig.suptitle(
        f'Plot 4: Gelman-Rubin R-hat Convergence Diagnostic\n'
        f'R-hat < 1.1 for all parameters = chains converged  |  '
        f'Max R-hat = {max_rhat:.3f}  |  '
        f'{100*frac_ok:.1f}% of parameters have R-hat < 1.1',
        fontsize=11, fontweight='bold')
    gs = gridspec.GridSpec(2, 1 + T_plus_1, figure=fig,
                           hspace=0.45, wspace=0.35)

    # ── Panel A: R-hat histogram ──────────────────────────────────────────
    ax_a = fig.add_subplot(gs[:, 0])
    rhat_np = rhat.numpy()
    ax_a.hist(rhat_np, bins=60, color='steelblue', edgecolor='white',
              alpha=0.8, orientation='horizontal', density=True)
    ax_a.axhline(1.1, color='crimson', lw=2, ls='--',
                 label='R-hat = 1.1\n(convergence threshold)')
    ax_a.axhline(1.0, color='green',   lw=1.5, ls='-',
                 label='R-hat = 1.0\n(perfect convergence)')
    ax_a.set_ylabel('R-hat value', fontsize=9)
    ax_a.set_xlabel('Density of parameters', fontsize=9)
    ax_a.set_title(f'A: R-hat histogram\n'
                   f'mean={mean_rhat:.3f}  max={max_rhat:.3f}\n'
                   f'{100*frac_ok:.1f}% have R-hat < 1.1',
                   fontsize=9, fontweight='bold')
    ax_a.legend(fontsize=7)
    ax_a.grid(True, alpha=0.2, axis='y')

    # Colour the verdict
    verdict_colour = 'green' if max_rhat < 1.1 else ('orange' if max_rhat < 1.5 else 'crimson')
    verdict_text   = ('All converged ✓' if max_rhat < 1.1
                      else 'Some parameters not converged ⚠'
                      if max_rhat < 1.5 else 'Poor convergence ✗')
    ax_a.text(0.97, 0.97, verdict_text, transform=ax_a.transAxes,
              ha='right', va='top', fontsize=9, color=verdict_colour,
              fontweight='bold',
              bbox=dict(facecolor='white', alpha=0.8, pad=3))

    # ── Panels B: Spatial R-hat maps per frame ────────────────────────────
    vmax_rhat = min(float(rhat_spatial.max()), 2.0)  # cap at 2 for readability

    for t in range(T_plus_1):
        ax_top = fig.add_subplot(gs[0, t + 1])
        ax_bot = fig.add_subplot(gs[1, t + 1])

        rhat_t = rhat_spatial[t].numpy()

        im_top = ax_top.imshow(rhat_t.T, origin='lower', cmap='RdYlGn_r',
                               aspect='auto', extent=[-np.pi, np.pi, -1, 1],
                               vmin=1.0, vmax=vmax_rhat)
        ax_top.set_title(f'B: R-hat map  t={t}\n'
                         f'max={rhat_t.max():.3f}',
                         fontsize=9, fontweight='bold')
        ax_top.set_xlabel('x', fontsize=7)
        ax_top.set_ylabel('z', fontsize=7)
        ax_top.axhline(0, color='white', lw=0.6, ls='--', alpha=0.5)
        plt.colorbar(im_top, ax=ax_top, fraction=0.046)

        # Bottom: show which cells exceed R-hat = 1.1
        bad_mask = (rhat_t > 1.1).astype(float)
        im_bot = ax_bot.imshow(bad_mask.T, origin='lower', cmap='RdYlGn_r',
                               aspect='auto', extent=[-np.pi, np.pi, -1, 1],
                               vmin=0, vmax=1)
        frac_bad = bad_mask.mean()
        ax_bot.set_title(f'R-hat > 1.1: {100*frac_bad:.1f}% of cells\n'
                         f'(red = not converged, green = converged)',
                         fontsize=8, fontweight='bold')
        ax_bot.set_xlabel('x', fontsize=7)
        ax_bot.set_ylabel('z', fontsize=7)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    path = f'{OUTDIR}/diag_04_rhat.png'
    fig.savefig(path, dpi=140, bbox_inches='tight')
    print(f'  Saved: {path}')
    print(f'  R-hat summary: mean={mean_rhat:.4f}  max={max_rhat:.4f}  '
          f'{100*frac_ok:.1f}% < 1.1')

def plot_density_evolution(results: dict,
                           rho_true_sequence: list[torch.Tensor]):
    """
    Plot 5: Density field evolution from HMC inference.

    Row 1: True density at each frame
    Row 2: MAP estimate at each frame
    Row 3: Change in MAP density between consecutive frames
           with quiver arrows showing the direction of density change
    """
    T_plus_1 = results['rho_mean'].shape[0]
    rho_mean = results['rho_mean']   # (T+1, N, N) — already computed

    from hmc_density_inference_v2 import TANK_WIDTH, TANK_HEIGHT
    extent = [-TANK_WIDTH/2, TANK_WIDTH/2, -TANK_HEIGHT/2, TANK_HEIGHT/2]

    x_phys = np.linspace(-TANK_WIDTH/2,  TANK_WIDTH/2,  GRID_N)
    z_phys = np.linspace(-TANK_HEIGHT/2, TANK_HEIGHT/2, GRID_N)

    fig, axes = plt.subplots(3, T_plus_1, figsize=(4.5 * T_plus_1, 11))
    if T_plus_1 == 1:
        axes = axes[:, np.newaxis]

    fig.suptitle('Plot 5: Inferred Density Field Evolution\n'
                 'Row 3: Δrho with arrows showing direction of density change',
                 fontsize=12, fontweight='bold')

    # Shared colour scale for Δrho rows
    all_diffs = [abs((rho_mean[t] - rho_mean[t-1]).numpy()).max()
                 for t in range(1, T_plus_1)]
    vmax_diff = max(max(all_diffs) if all_diffs else 0.01, 0.01)

    for t in range(T_plus_1):
        arr_true = rho_true_sequence[t].numpy()
        arr_map  = rho_mean[t].numpy()

        # ── Row 1: True density ───────────────────────────────────────────
        im1 = axes[0, t].imshow(arr_true.T, origin='lower', cmap='viridis',
                                 aspect='auto', extent=extent, vmin=0, vmax=1)
        axes[0, t].set_title(f'True rho  t={t}', fontsize=9, fontweight='bold')
        axes[0, t].set_xlabel('x (m)', fontsize=7)
        axes[0, t].set_ylabel('z (m)', fontsize=7)
        axes[0, t].axhline(0, color='white', lw=0.8, ls='--', alpha=0.5)
        if t == 0:
            axes[0, t].set_ylabel('True rho', fontsize=9)
        if t == T_plus_1 - 1:
            plt.colorbar(im1, ax=axes[0, t], fraction=0.046)

        # ── Row 2: MAP density ────────────────────────────────────────────
        im2 = axes[1, t].imshow(arr_map.T, origin='lower', cmap='viridis',
                                 aspect='auto', extent=extent, vmin=0, vmax=1)
        axes[1, t].set_title(f'Post. Mean  t={t}', fontsize=9, fontweight='bold')
        axes[1, t].set_xlabel('x (m)', fontsize=7)
        axes[1, t].axhline(0, color='white', lw=0.8, ls='--', alpha=0.5)
        if t == 0:
            axes[1, t].set_ylabel('Post. Mean', fontsize=9)
        if t == T_plus_1 - 1:
            plt.colorbar(im2, ax=axes[1, t], fraction=0.046)

        # ── Row 3: Δrho with quiver ───────────────────────────────────────
        if t == 0:
            diff = np.zeros((GRID_N, GRID_N))
            axes[2, t].set_title('Δrho  t=0\n(reference)', fontsize=9,
                                  fontweight='bold')
        else:
            diff = arr_map - rho_mean[t-1].numpy()
            mean_abs = float(np.abs(diff).mean())
            axes[2, t].set_title(f'Δrho  t={t-1}→{t}\n'
                                  f'mean|Δ|={mean_abs:.4f}',
                                  fontsize=9, fontweight='bold')

        im3 = axes[2, t].imshow(diff.T, origin='lower', cmap='RdBu_r',
                                  aspect='auto', extent=extent,
                                  vmin=-vmax_diff, vmax=vmax_diff)
        axes[2, t].axhline(0, color='black', lw=0.8, ls='--', alpha=0.5)
        axes[2, t].set_xlabel('x (m)', fontsize=7)
        if t == 0:
            axes[2, t].set_ylabel('Δrho  Blue=decrease\nRed=increase',
                                   fontsize=8)
        if t == T_plus_1 - 1:
            plt.colorbar(im3, ax=axes[2, t], fraction=0.046)

        # Quiver: spatial gradient of Δrho shows direction density is shifting
        if t > 0:
            stride = max(1, GRID_N // 8)
            # np.gradient returns [d/drow, d/dcol] = [d/dz, d/dx] for arr.T
            # We want gradients in physical (x,z) space
            grad_x, grad_z = np.gradient(diff, axis=0), np.gradient(diff, axis=1)

            xi = x_phys[::stride]
            zi = z_phys[::stride]
            X_q, Z_q = np.meshgrid(xi, zi, indexing='ij')
            gx = grad_x[::stride, ::stride]
            gz = grad_z[::stride, ::stride]

            mag = np.sqrt(gx**2 + gz**2)
            max_mag = max(float(mag.max()), 1e-12)

            axes[2, t].quiver(
                X_q, Z_q,
                gx / max_mag,   # normalised direction
                gz / max_mag,
                color='black',
                alpha=0.6,
                scale=15,
                width=0.004,
                pivot='mid'
            )

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = f'{OUTDIR}/diag_05_density_evolution.png'
    fig.savefig(path, dpi=140, bbox_inches='tight')
    print(f'  Saved: {path}')
    print('  Density change summary (Posterior Mean):')
    for t in range(1, T_plus_1):
        d = (rho_mean[t] - rho_mean[t-1]).abs().mean().item()
        print(f'    t={t-1}→{t}: mean|Δrho| = {d:.4f}')
# =============================================================================
# SECTION 9 — MAIN
# =============================================================================

if __name__ == '__main__':
    print('=' * 60)
    print('HMC Diagnostics — Accuracy and Multimodality')
    print('=' * 60)
    print(f'N_FRAMES={N_FRAMES}, N_SAMPLES={N_SAMPLES}, N_WARMUP={N_WARMUP}')
    print(f'step=[{MIN_STEP},{MAX_STEP}], traj=[{MIN_TRAJ},{MAX_TRAJ}]')

    # ── Step 1: Generate perturbed synthetic data ─────────────────────────
    print('\n[1] Generating perturbed synthetic data...')
    b_obs_seq, rho_true_seq = generate_synthetic_sequence(
        n_frames=N_FRAMES)
    rho_true_0 = rho_true_seq[0]
    print(f'    {len(b_obs_seq)} frames generated')
    print(f'    True rho_0 range: [{rho_true_0.min():.3f}, {rho_true_0.max():.3f}]')
    print(f'    Prior  rho_0 range: [{make_prior_density().min():.3f}, '
          f'{make_prior_density().max():.3f}]')
    print(f'    Perturbation RMSE from prior: '
          f'{float(torch.sqrt(((rho_true_0 - make_prior_density())**2).mean())):.4f}')

    # ── Step 2: Run chain 1 (from prior mean) ────────────────────────────
    print('\n[2] Running Chain 1 (initialised at prior mean)...')
    results1 = run_chain(b_obs_seq, seed=42, init_noise=0.0, label='Chain 1')

    # ── Step 3: Run chain 2 (from perturbed start) ───────────────────────
    # init_noise=0.1 shifts the starting point by ~0.1 per parameter,
    # enough to be a genuinely different starting configuration without
    # being so extreme that it never converges.
    print('\n[3] Running Chain 2 (initialised from perturbed start)...')
    results2 = run_chain(b_obs_seq, seed=123, init_noise=0.1, label='Chain 2')

    # ── Step 4: Accuracy diagnostics ─────────────────────────────────────
    print('\n[4] Computing accuracy metrics...')
    accuracy = compute_accuracy(results1, rho_true_seq)
    print(f'    Mean RMSE (HMC)  : {accuracy["mean_rmse"]:.4f}')
    print(f'    Mean RMSE (prior): {np.mean(accuracy["prior_rmse_per_frame"]):.4f}')
    print(f'    Noise floor σ_obs: {SIGMA_OBS:.4f}')
    print(f'    Mean coverage    : {accuracy["mean_coverage"]:.3f}  (target 0.95)')
    print(f'    RMSE per frame   : {[f"{r:.4f}" for r in accuracy["rmse_per_frame"]]}')

    # ── Step 5: Generate plots ────────────────────────────────────────────
    print('\n[5] Generating diagnostic plots...')
    plot_accuracy(accuracy, results1, rho_true_seq)
    plot_posterior_predictive(results1, b_obs_seq)
    plot_multimodality(results1, results2)
    plot_density_evolution(results1, rho_true_seq)
    plot_rhat(results1, results2)
    
    # ── Step 6: Final summary ─────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('DIAGNOSTIC SUMMARY')
    print('=' * 60)
    print(f'  RMSE improvement over prior: '
          f'{100*(1 - accuracy["mean_rmse"]/np.mean(accuracy["prior_rmse_per_frame"])):.1f}%')
    print(f'  Coverage probability: {accuracy["mean_coverage"]:.3f}  (target 0.95)')
    print(f'  Chain 1 accept rate: {results1["accept_rate"]:.3f}')
    print(f'  Chain 2 accept rate: {results2["accept_rate"]:.3f}')
    print()
    print('  Output files:')
    for i, name in enumerate(['accuracy', 'posterior_predictive',
                           'multimodality', 'rhat', 'density_evolution'], 1):
        print(f'    diag_0{i}_{name}.png')
    print('=' * 60)