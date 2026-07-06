# JOR-Bayesian Fusion Framework (v3.1)
### Probabilistic UAP Analysis Engine & Aerospace Safety Command Center

![Status](https://img.shields.io/badge/Status-Project_Complete-brightgreen.svg)  
**Version:** 3.1 Final Release

📌 **DOI:** [https://doi.org/10.5281/zenodo.18088931](https://doi.org/10.5281/zenodo.18088931)
📦 **Zenodo Record:** [https://zenodo.org/records/18088931](https://zenodo.org/records/18088931)  
📟 **Operational App Interface:** `Collision_Risk/Safety_Command_Center.py` (Streamlit Safety Command Center UI)

The **James Orion Report (JOR) Bayesian Fusion** framework is a probabilistic analysis system designed to evaluate Unidentified Anomalous Phenomena (UAP) using structured evidentiary scoring combined with **Markov Chain Monte Carlo (MCMC)** sampling via **PyMC**.

The system integrates qualitative observation rubrics with quantitative Bayesian-style inference to produce conservative estimates of whether a case is consistent with a non-human origin hypothesis.

---

## 🚀 Key Improvements in v3.1

* **Unified Parameter Sync:** Global constants ($K$-calibration, priors, and weights) are centrally defined in `jor_fusion.py` and inherited across all execution layers, ensuring consistency between CLI scoring, the Streamlit GUI, and Bayesian inference.
* **Evidence-Layer Flight Modulation:** Flight-related anomalies are now treated as a probabilistic modifier on physical evidence quality rather than a post-hoc score adjustment. This ensures anomaly information is incorporated at the evidentiary interpretation stage.
* **Numerical Stability Improvements:** Variable clipping (0.001–0.999) prevents invalid probability states, and fixed random seeds ensure reproducible MCMC sampling.
* **Dynamic Scorer Logic:** The interactive CLI includes conditional scoring adjustments for high-confidence observational conditions (e.g., clear daytime visibility), reducing under-weighting of high-quality data.
* **Deterministic Data Exports:** Implemented a targeted `.round(3)` truncation limit on the `Uncertainty` column to permanently strip binary floating-point tail artifacts (e.g., converting `0.11900000000000005` to a clean `0.119`), ensuring flawless downstream spreadsheet integration.
* **Syntax Bug Resolution:** Optimized the dashboard's `ax.errorbar` data arrays by splitting nested `np.vstack` calculations into discrete, standalone variables, eliminating trailing parenthesis compilation panics.
* **Environmental Clarification:** Clarified Environmental Strong-category wording to explicitly treat clear daytime and clear nighttime atmospheric visibility equivalently. This change improves scoring consistency and reduces interpretive variance; no mathematical weighting or posterior logic changed.

---

## 📟 Streamlit GUI Command Center Features (`Safety_Command_Center.py`)

* **🌌 Kinetic Hazard Frontier:** A dynamic bubble chart using a high-visibility `plasma` color map to plot Solid Object Probability (`SOP_Mean`) against Flight Modifiers (`Flight_Mod`). Bubble sizing automatically scales according to MCMC data uncertainty span.
* **📊 Threat Intelligence Quadrant Map:** A crosshair-synced matrix mapping physical target data against strict risk metrics, highlighting critical action limits ($>0.75$ CRI).
* **🌲 Evidentiary Uncertainty Forest:** An MCMC posterior uncertainty range check that isolates active target profiles with a prominent target lock overlay against background context fleets (95% Confidence Intervals).
* **🔍 Active Track Auditor & Sidebar Controls:** Real-time dropdown isolation engines that immediately focus all three graphical tracking maps onto a single selected asset.

> ⚠️ **IMPORTANT INTERACTIVE UI NOTE:** To launch and run the Streamlit Command Center dashboard (`Safety_Command_Center.py`), you must navigate into the `Collision_Risk` directory and use the specialized versions of `jor_fusion.py`, `jor_pymc_runner.py`, and `jor_pymc.py` stored inside that specific folder. Running the dashboard utilizing the main root folder's python files will cause dependency and path alignment errors.

---

## 📂 Repository Structure

| File / Folder | Description |
| :--- | :--- |
| **Full Methodology Report** | [Original theoretical framework and evidentiary rubric definitions](https://doi.org/10.5281/zenodo.20368678) |
| `Collision_Risk/` | **Dedicated UI Folder:** Contains `Safety_Command_Center.py` and the localized backend scripts explicitly modified to drive the Command Center interface. |
| `jor_fusion.py` | **Root Scorer:** Interactive CLI tool for standard, standalone terminal-based scoring. |
| `jor_pymc.py` | **Root Bayesian Engine:** Core PyMC-based probabilistic model for standalone MCMC sampling. |
| `jor_pymc_runner.py` | **Root Batch Orchestrator:** Standalone terminal tool for executing multiple case evaluations. |
| `jor_scores.csv` | **Outputs:** Model-generated posterior estimates, means, and credible intervals for evaluated cases. |

---

## 📊 Example Dataset

A curated dataset of 50 evaluated UAP cases using the v3.1 framework is included:

- `data/v3_1/jor_uap_cases_50_v3_1.csv`

This dataset contains structured inputs and model outputs, including:
- Evidence scores ($C$, $E$, $P$)
- Flight anomaly modifiers
- SOP / NHP intermediate values
- Posterior probabilities and credible intervals

It can be used to:
- Reproduce published results
- Validate model behavior
- Test parameter sensitivity and calibration

---

## 🧭 Version Philosophy (v3 vs v3.1)

* **v3:** Additive feature-weight scoring model where anomaly indicators influence posterior estimates through weighted contributions.  
* **v3.1:** Evidence-conditioned inference model where observational reliability (e.g., flight anomalies) modifies how physical evidence is interpreted within the likelihood structure.  

This distinction is critical: v3.1 does not simply re-weight outputs—it modifies how evidence is conditioned before probabilistic inference occurs.

---

## 🧪 The Bayesian Logic

The framework evaluates competing hypotheses:
* $H$ (Human / Prosaic Origin)
* $NH$ (Non-Human / Anomalous Origin)

Priors are initialized as:
$$P(H) = 0.80$$
$$P(NH) = 0.20$$

These priors represent conservative baseline assumptions under conditions of uncertainty.

### Stochastic Flight Modeling

A key component of the model is the *Flight Effect*, which introduces a probabilistic modulation of physical evidence strength based on observed kinematic anomalies and assumed measurement uncertainty. 

This adjustment is applied at the evidence interpretation stage:
$$P_{\text{Anomalous}} = \text{clip}(P \times (1 + \text{Flight Effect}), 0.0, 0.95)$$

Where the Flight Effect is modeled as a truncated distribution:
$$\text{TruncatedNormal}(\mu = \text{mod}, \sigma = 0.03, \text{lower} = 0.0, \text{upper} = 0.10)$$

This formulation ensures that anomalous flight characteristics contribute proportionally to evidence strength while remaining bounded by observational uncertainty.

### K-Calibration Constant (Heuristic Likelihood Proxy)

To maintain conservative inference behavior aligned with uncertainty constraints, the framework includes a calibration constant:
$$P(E|H) = \min(1, 1 - \text{NHP} + K \cdot \text{SOP})$$

Where:
* $K = 0.20$ (calibration constant aligned with AARO Standards)
* $\text{SOP}$ = Solid Object Probability
* $\text{NHP}$ = Non-Human Probability (intermediate estimate)

This expression is a *heuristic likelihood proxy*, not a strict probabilistic identity. It is used to stabilize inference under incomplete observational data.

---

## 🛠️ Usage & Execution

First, ensure you have all core framework dependencies installed:

> `pip install pymc pytensor pandas numpy colorama streamlit matplotlib seaborn`

---

> ⚠️ **CRITICAL ARCHITECTURE NOTE:** This repository contains two entirely separate, self-contained environments that use matching file names. Running a file in the root directory has absolutely nothing to do with the files inside the `Collision_Risk` folder. They do not share data, inputs, or outputs.

Choose one distinct path below depending on which application you want to run:

---

### 🔹 Option 1: Main Root Framework
Runs the primary standalone CLI scoring and Bayesian processing engine. Uses the files located strictly in the root directory.

> **Step 1:** Ensure your terminal is in the root directory.
>
> **Step 2:** Execute the root pipeline:
> `python jor_fusion.py`  
> `python jor_pymc_runner.py`

---

### 🔹 Option 2: Collision Risk Index Module (`Collision_Risk/`)
Runs the independent interactive safety dashboard and its localized backend scripts. This module operates entirely within its own directory.

> **Step 1:** Change your terminal directory into the subfolder:
> `cd Collision_Risk`
>
> **Step 2:** Launch the interactive command center dashboard:
> `streamlit run Safety_Command_Center.py`
>
> *(Note: If you need to run terminal pipeline variations for this specific safety module, use the localized `jor_fusion.py` or read the separate `readme.md` located inside this subfolder).*

