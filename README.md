# Estimating Extreme Credit Portfolio Risk with Importance Sampling

This repository estimates one-year loss distributions for three 50-issuer corporate-credit portfolios. It compares plain Monte Carlo with importance sampling (IS) of the systemic Gaussian factor and asks whether IS can reduce the sampling variance of 99% expected shortfall (ES99) without materially changing point estimates.

The workbook contains previously prepared issuer-level LQD/HYG portfolio data. No Bloomberg connection, credential, or Excel add-in is needed to reproduce the simulation.

## Portfolios and model

The portfolios allocate 30%/70%, 50%/50%, and 70%/30% to investment-grade (LQD) and high-yield (HYG) sleeves. The simulation reads `Sheet1`, `Sheet2`, and `Sheet3`, respectively, from `data/raw/issuer_portfolios_lqd_hyg.xlsx`; it intentionally does not use the similarly named `30_70`, `50_50`, and `70_30` sheets because their PD formulas may be unresolved.

### One-factor Gaussian credit model

For issuer \(i\), let \(PD_i\) be its one-year marginal probability of default and define the default threshold

\[
c_i=\Phi^{-1}(PD_i),
\]

where \(\Phi\) is the standard-normal CDF. Its latent asset variable is

\[
X_i=\sqrt{\rho}\,Z+\sqrt{1-\rho}\,\varepsilon_i,
\]

with mutually independent \(Z,\varepsilon_1,\ldots,\varepsilon_n\sim N(0,1)\). The shared factor \(Z\) represents economy-wide credit conditions and \(\varepsilon_i\) is issuer-specific risk. Because

\[
\operatorname{Var}(X_i)=\rho+(1-\rho)=1,
\]

each \(X_i\) is standard normal and the threshold construction preserves the input marginal probability:

\[
\Pr(X_i<c_i)=\Phi(c_i)=PD_i.
\]

For two different issuers, \(\operatorname{Corr}(X_i,X_j)=\rho\), since their only common random component is \(Z\). Conditional on \(Z\), defaults are independent; unconditionally, adverse systemic realizations can cause many defaults together.

Issuer \(i\) defaults when \(X_i<c_i\). If \(a_i\) is its portfolio exposure share, \(V\) is total portfolio value, and LGD is the deterministic loss-given-default fraction, one simulated portfolio loss is

\[
L=\sum_{i=1}^n a_iV\,LGD\,\mathbf 1\{X_i<c_i\}.
\]

The baseline uses \(V=\$100\) million, \(LGD=40\%\), \(\rho=20\%\), 20,000 paths per replication, 50 replications, and seed 42.

### Risk measures

For a loss random variable \(L\), the project estimates expected loss (EL), value at risk (VaR), and expected shortfall (ES). At confidence level \(\alpha\),

\[
EL=\mathbb E[L],\qquad
VaR_\alpha=\inf\{\ell:F_L(\ell)\geq\alpha\},\qquad
ES_\alpha=\mathbb E[L\mid L\geq VaR_\alpha].
\]

The reported levels are 95% and 99%. Credit-portfolio losses are discrete because each simulated issuer either defaults or survives, so many paths can have the same loss. Consistent with the implementation, ES uses the inclusive tail \(L\geq VaR_\alpha\). This convention can include more than exactly \(1-\alpha\) of the observations when there are ties at VaR.

## Importance sampling theory

### Why importance sampling helps

Plain Monte Carlo draws every scenario from the original model. This is effective near the center of the loss distribution, but a 99% tail metric depends on relatively few adverse paths. With 20,000 paths, only roughly 200 lie beyond the empirical 99th percentile, so ES99 can vary substantially from one replication to another.

Importance sampling changes the simulation distribution to generate adverse scenarios more often, then corrects those draws with likelihood-ratio weights. For any integrable loss functional \(h(L)\), the target expectation under the original density \(p\) can be written using a proposal density \(q\):

\[
\theta=\mathbb E_p[h(L)]
=\int h(L(x))p(x)\,dx
=\int h(L(x))\frac{p(x)}{q(x)}q(x)\,dx
=\mathbb E_q[W(x)h(L)],
\]

where

\[
W(x)=\frac{p(x)}{q(x)}
\]

is the likelihood ratio. Thus sampling more tail observations does not change the target distribution as long as the likelihood ratio is applied correctly and \(q(x)>0\) wherever the target integrand matters.

An ideal proposal would devote simulation effort to the scenarios that contribute most to the target while keeping the weighted observations stable. The formally variance-minimizing proposal for a nonnegative expectation is proportional to \(h(L(x))p(x)\), but it depends on the unknown quantity being estimated and is generally unavailable. This project therefore uses a simple parametric proposal: a mean shift of the systemic factor.

### Shifting the systemic factor

Under plain Monte Carlo,

\[
Z\sim N(0,1).
\]

Under importance sampling, only the shared factor is shifted:

\[
Z\sim N(\mu_{IS},1),
\qquad
\varepsilon_i\sim N(0,1).
\]

Negative values of \(\mu_{IS}\) represent worse systemic conditions. Because default occurs when \(X_i<c_i\), shifting \(Z\) downward moves many issuers' latent variables toward their default thresholds simultaneously and produces large portfolio losses more frequently. The idiosyncratic factors are not changed.

### Derivation of the likelihood-ratio weight

Let \(p(z)\) denote the original \(N(0,1)\) density and \(q_\mu(z)\) the shifted \(N(\mu,1)\) density:

\[
p(z)=\frac{1}{\sqrt{2\pi}}\exp\left(-\frac{z^2}{2}\right),
\qquad
q_\mu(z)=\frac{1}{\sqrt{2\pi}}\exp\left(-\frac{(z-\mu)^2}{2}\right).
\]

Their density ratio is

\[
\begin{aligned}
W(z)
&=\frac{p(z)}{q_\mu(z)}\\
&=\exp\left[-\frac{z^2}{2}+\frac{(z-\mu)^2}{2}\right]\\
&=\exp\left(-\mu z+\frac{\mu^2}{2}\right).
\end{aligned}
\]

Substituting \(\mu=\mu_{IS}\) gives the weight used by `likelihood_ratio` in `simulation.py`:

\[
W(Z)=\exp\left(-\mu_{IS}Z+\frac{\mu_{IS}^2}{2}\right).
\]

Only the density of \(Z\) changes. The original and proposal densities of every \(\varepsilon_i\) are identical and therefore cancel from the joint density ratio:

\[
\frac{p_Z(z)\prod_i p_{\varepsilon_i}(\varepsilon_i)}
     {q_Z(z)\prod_i p_{\varepsilon_i}(\varepsilon_i)}
=\frac{p_Z(z)}{q_Z(z)}.
\]

This is why one likelihood-ratio weight per simulated path corrects the entire portfolio loss on that path. Omitting the weights would estimate risk under the deliberately worsened proposal economy rather than under the original credit model.

### Self-normalized estimators

If \((L_j,W_j)\), \(j=1,\ldots,N\), are losses and likelihood ratios generated under the proposal, the ordinary importance-sampling estimator of an expectation is

\[
\widehat\theta_{IS}=\frac{1}{N}\sum_{j=1}^N W_jh(L_j).
\]

The implementation uses the self-normalized form

\[
\widehat\theta_{SNIS}
=\frac{\sum_{j=1}^N W_jh(L_j)}{\sum_{j=1}^N W_j}.
\]

For EL, \(h(L)=L\). Self-normalization makes the sampled weights sum to one and provides a convenient weighted empirical distribution. It is generally slightly biased at finite \(N\), because it is a ratio of random quantities, but it is consistent as \(N\) grows and can be more stable when the sampled weights do not sum close to their theoretical mean of one.

Define normalized weights \(\widetilde W_j=W_j/\sum_k W_k\). The weighted empirical loss CDF is

\[
\widehat F_{IS}(\ell)=\sum_{j=1}^N \widetilde W_j\mathbf 1\{L_j\leq\ell\}.
\]

The weighted VaR estimator is the smallest observed loss whose weighted CDF reaches \(\alpha\):

\[
\widehat{VaR}_{\alpha,IS}
=\inf\{\ell:\widehat F_{IS}(\ell)\geq\alpha\}.
\]

Given that estimated threshold, the project computes weighted ES using the inclusive tail:

\[
\widehat{ES}_{\alpha,IS}
=\frac{\sum_{j=1}^N W_jL_j
\mathbf 1\{L_j\geq\widehat{VaR}_{\alpha,IS}\}}
{\sum_{j=1}^N W_j
\mathbf 1\{L_j\geq\widehat{VaR}_{\alpha,IS}\}}.
\]

### Weight concentration and effective sample size

Generating more tail paths is useful only if a few paths do not dominate the weighted estimate. The project reports Kish effective sample size (ESS):

\[
ESS=\frac{\left(\sum_{j=1}^N W_j\right)^2}
          {\sum_{j=1}^N W_j^2}.
\]

Equal weights give \(ESS=N\); increasingly unequal weights reduce ESS toward 1. The ratio \(ESS/N\) makes results comparable across simulation sizes. ESS is a diagnostic rather than a guarantee of accuracy: a proposal can have a lower ESS yet still improve a particular tail estimator by placing substantially more paths in the relevant region.

### Selecting the shift and measuring variance reduction

There is a trade-off in \(\mu_{IS}\). A shift near zero barely changes the number of tail observations, while an overly negative shift can generate highly variable likelihood ratios and severe weight concentration. The project evaluates

```text
0.0, -0.1, -0.2, ..., -1.4
```

using 50 independent selection replications per candidate. If \(S^2_{plain,ES99}\) and \(S^2_{IS,ES99}(\mu)\) are the across-replication sample variances of the ES99 estimators, the empirical variance-reduction ratio is

\[
VRR_{ES99}(\mu)
=\frac{S^2_{plain,ES99}}{S^2_{IS,ES99}(\mu)}.
\]

A value above 1 means IS has lower estimated sampling variance for ES99; for example, \(VRR=5\) means that the observed estimator variance is one fifth of the plain-MC variance. The nonzero candidate with the largest empirical ratio is selected separately for each portfolio. Final plain-MC and IS comparisons are then generated with new replications using that selected shift.

The `Variance_Reduction_Ratio` repeated in each portfolio summary CSV is specifically this **selection-stage ES99** ratio. It is portfolio-level metadata, not a separate variance-reduction calculation for every method and metric row. Likewise, `Selected_mu_IS` records the shift used by the portfolio's IS run even when it appears on a Plain MC row.

Across the final 50 replications, each replication produces one estimate of every risk metric. Their sample SD measures the run-to-run variability of a single 20,000-path estimator. The reported \(SE=SD/\sqrt{50}\) instead measures the uncertainty in the displayed mean across those 50 estimates.

## Setup and use

Python 3.10 or later is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
pytest
python main.py --quick
python main.py
```

`--quick` retains the complete grid and workflow but caps paths at 2,000 and both replication counts at 5. Optional arguments are `--input`, `--seed`, `--paths`, `--replications`, and `--selection-replications`.

Launch the clean notebook from the repository root after installing the requirements:

```bash
jupyter notebook notebooks/analysis.ipynb
```

## Outputs

Generated CSV files are written to `outputs/tables/`; 15 parameter-selection and replication-diagnostic figures are written to `outputs/figures/`. Per-portfolio exports include the shift sweep, plain replications, IS replications, and a tidy summary. Combined exports include selected shifts, both replication panels, and the all-portfolio summary. Generated files are ignored by Git; `.gitkeep` retains the output directories.

The root-level modules separate workbook validation, simulation, metrics, experiment orchestration, and plotting. `main.py` is the batch entry point. `notebooks/analysis.ipynb` is the concise reproducibility narrative and imports those modules—it does not reimplement the model.

```text
config.py                         central settings
data.py                           workbook loading and validation
simulation.py                     Gaussian-copula simulation
metrics.py                        VaR, ES, and ESS calculations
experiment.py                     replication-study workflow
plotting.py                       analysis figures
main.py                           command-line pipeline
requirements.txt                  Python dependencies
data/raw/                         immutable workbook
references/original_analysis.ipynb preserved behavioral reference
notebooks/analysis.ipynb          clean presentation notebook
tests/                            fast deterministic tests
outputs/{tables,figures}/         generated artifacts
```

## Baseline findings and reference comparison

The preserved original notebook selects \(\mu_{IS}=-1.2,-1.3,-0.7\) for the 30/70, 50/50, and 70/30 portfolios. Its principal seed-42 results are:

| Portfolio | Method | EL mean | VaR99 mean | ES99 mean | ES99 SD | Average ESS |
|---|---|---:|---:|---:|---:|---:|
| 30/70 | Plain MC | $428,153 | $3,578,162 | $4,519,999 | $107,258 | — |
| 30/70 | IS | $428,372 | $3,568,271 | $4,507,943 | $44,287 | 4,835 |
| 50/50 | Plain MC | $277,379 | $2,748,532 | $3,568,088 | $97,781 | — |
| 50/50 | IS | $279,423 | $2,766,512 | $3,567,801 | $31,687 | 3,740 |
| 70/30 | Plain MC | $147,971 | $2,110,395 | $2,651,536 | $58,478 | — |
| 70/30 | IS | $147,570 | $2,103,735 | $2,636,178 | $36,000 | 12,236 |

The refactor preserves the legacy `np.random` stream, portfolio/grid order, draw shapes, NumPy quantile convention, inclusive ES tail, supplied EAD weights, and maximum-VR selection rule. On NumPy 2.3.5/pandas 2.3.3, its baseline CSV values match the executed reference calculation (the generated `outputs/tables/original_vs_refactored.csv` records field-level differences). The variance-reduction ratios are approximately 5.53, 8.04, and 3.55: IS substantially tightens ES99 estimates while method means remain close. As shifts become more aggressive, ESS generally falls.

One intentional validation improvement does not affect the valid workbook: after missing/error rows are reported and dropped, any remaining out-of-range PD now raises an actionable error instead of being silently filtered. Materially incorrect portfolio weights also raise rather than being renormalized.

`references/original_analysis.ipynb` is the unchanged working notebook used as the behavioral reference. `notebooks/analysis.ipynb` is the maintained, presentation-oriented replacement. Because the original notebook derives paths from its working directory, it was executed from the repository root for comparison without altering its contents.

## Reproducibility and limitations

The default implementation deliberately uses NumPy's legacy seeded stream because switching to `Generator` changes the benchmark. Runs with identical inputs and settings are deterministic; exact last digits can still differ across future NumPy, pandas, Excel-reader, or Matplotlib versions.

The model assumes homogeneous deterministic LGD, constant asset correlation, a one-year static horizon, Gaussian dependence, and default-only losses. It uses self-normalized IS estimators and simulation-based shift selection, and it includes neither parameter uncertainty nor a model-calibration study.

Natural extensions include heterogeneous or stochastic LGD, sector-specific systematic factors, heavier-tailed copulas, adaptive importance sampling, ESS-constrained shift selection, parameter uncertainty, and multi-period credit migration.
