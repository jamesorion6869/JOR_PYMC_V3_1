# JOR-Bayesian Fusion Framework (v3.1)
### Probabilistic UAP Analysis Engine

📌 DOI: https://doi.org/10.5281/zenodo.18157347  
📦 Zenodo Record: https://zenodo.org/records/18157347

The **James Orion Report (JOR) Bayesian Fusion** framework is a probabilistic analysis system designed to evaluate Unidentified Anomalous Phenomena (UAP) using structured evidentiary scoring combined with **Markov Chain Monte Carlo (MCMC)** sampling via **PyMC**.

The system integrates qualitative observation rubrics with quantitative Bayesian-style inference to produce conservative estimates of whether a case is consistent with a non-human origin hypothesis.

---

## 🚀 Key Improvements in v3.1

* **Unified Parameter Sync:** Global constants (K-calibration, priors, and weights) are centrally defined in `jor_fusion.py` and inherited across all execution layers, ensuring consistency between CLI scoring and Bayesian inference.
* **Evidence-Layer Flight Modulation:** Flight-related anomalies are now treated as a probabilistic modifier on physical evidence quality rather than a post-hoc score adjustment. This ensures anomaly information is incorporated at the evidentiary interpretation stage.
* **Numerical Stability Improvements:** Variable clipping (0.001–0.999) prevents invalid probability states, and fixed random seeds ensure reproducible MCMC sampling.
* **Dynamic Scorer Logic:** The interactive CLI includes conditional scoring adjustments for high-confidence observational conditions (e.g., clear daytime visibility), reducing under-weighting of high-quality data.

---

## 📂 Repository Structure

| File | Description |
| :--- | :--- |
| `jor-bayesian-fusion-V3.pdf` | **Methodology:** Original theoretical framework and evidentiary rubric definitions. |
| `jor_fusion.py` | **Scorer:** Interactive CLI tool for structured case input and deterministic scoring. |
| `jor_pymc.py` | **Bayesian Engine:** PyMC-based probabilistic model for posterior estimation via MCMC sampling. |
| `jor_pymc_runner.py` | **Batch Orchestrator:** Executes multiple case evaluations and generates confidence intervals. |
| `jor_scores.csv` | **Outputs:** Model-generated posterior estimates, means, and credible intervals for evaluated cases. |

---

## 🧭 Version Philosophy (v3 vs v3.1)

- **v3:** Additive feature-weight scoring model where anomaly indicators influence posterior estimates through weighted contributions.  
- **v3.1:** Evidence-conditioned inference model where observational reliability (e.g., flight anomalies) modifies how physical evidence is interpreted within the likelihood structure.  

This distinction is critical: v3.1 does not simply re-weight outputs—it modifies how evidence is conditioned before probabilistic inference.

---

## 🧪 The Bayesian Logic

The framework evaluates competing hypotheses:

- **H (Human/Prosaic Origin)**  
- **NH (Non-Human / Anomalous Origin)**  

Priors are initialized as:

- P(H) = 0.80  
- P(NH) = 0.20  

These priors represent conservative baseline assumptions under conditions of uncertainty.

---

### Stochastic Flight Modeling

A key component of the model is the **Flight Effect**, which introduces a probabilistic modulation of physical evidence strength based on observed kinematic anomalies and assumed measurement uncertainty.

This adjustment is applied at the evidence interpretation stage:

$$
P_{Anomalous} = \text{clip}(P \times (1 + \text{Flight Effect}), 0.0, 0.95)
$$

Where the Flight Effect is modeled as a truncated distribution:

- TruncatedNormal(μ = mod, σ = 0.03, lower = 0.0, upper = 0.10)

This formulation ensures that anomalous flight characteristics contribute proportionally to evidence strength while remaining bounded by observational uncertainty.

---

### K-Calibration Constant (Heuristic Likelihood Proxy)

To maintain conservative inference behavior aligned with uncertainty constraints, the framework includes a calibration constant:

$$
P(E|H) = \min(1, 1 - NHP + K \cdot SOP)
$$

Where:

- K = 0.20 (calibration constant)  
- SOP = Solid Object Probability  
- NHP = Non-Human Probability (intermediate estimate)  

This expression is a **heuristic likelihood proxy**, not a strict probabilistic identity. It is used to stabilize inference under incomplete observational data.

---

## 🛠️ Usage

### Requirements and Execution

Install dependencies and run the framework:

```bash
pip install pymc pytensor pandas numpy colorama
python jor_fusion.py
python jor_pymc_runner.py

```
