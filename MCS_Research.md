# MCS_Research (Hongze Lin)
---
## **About Monte Carlo simulation (MCS)**
---
- A family of numerical methods that **estimate quantities by repeated random sampling**.
- Replaces hard-to-compute integrals/probabilities/expected values with **sample averages**:
  - Draw samples $\(X_i \sim p(X)\)$, compute $\(f(X_i)\)$, then average $\(\frac{1}{N}\sum f(X_i)\)$.
- Accuracy improves as the number of samples increases; typical error decreases like **$\(1/\sqrt{N}\)$**.
- Produces an estimate **with statistical noise** (variance), which can be reduced by more samples or variance-reduction techniques.

## **Where to use it and why**
---
- **Complex light transport / optics (ray tracing)**
  - *Where:* Particle reflection/scattering, lens acceptance, imperfect optics, multiple bounces, partial occlusion.
  - *Why:* The physics becomes a high-dimensional integral over directions/paths; **Monte Carlo approximates it by sampling rays/paths**.
- **Beer–Lambert with complicated geometry (path integrals)**
  - *Where:* Attenuation depending on **integrals along rays** through spatially varying dye concentration.
  - *Why:* Exact evaluation can be expensive/analytic forms may not exist; sampling can approximate contributions efficiently in complex scenes.
- **Uncertainty quantification (UQ)**
  - *Where:* Propagating uncertainty in parameters (illumination profile, absorption coefficient, PSF width) to uncertainty in images.
  - *Why:* MCS gives **distributions** of outputs, not just a single prediction, which supports a statistical inference view.
- **Stochastic models and noise-driven processes**
  - *Where:* Random-walk particle motion out of plane, shot noise, random measurement errors.
  - *Why:* When randomness is intrinsic to the model/measurement, Monte Carlo naturally simulates the variability.
- **Inference/sampling methods (Bayesian posterior sampling)**
  - *Where:* Particle filters, MCMC/HMC variants that draw samples from a posterior over states/parameters.
  - *Why:* When the posterior is complex, sampling-based approaches approximate it via **random draws**.

## **Practical note for this project**
---
- Monte Carlo is justified for **high-fidelity optics** (scattering + lensing) and **full uncertainty quantification**.
  
