# JOR-V3.1: Collision Risk Index (CRI) Module
### 📟 Operational Safety Management System (SMS) for UAP Intelligence

The **Collision Risk Index (CRI)** is an operational aerospace safety overlay for the JOR Bayesian Fusion Framework. While the core root engine calculates the probability of an anomalous origin (Non-Human Hypothesis), this dedicated module translates physical sensor certainty and flight dynamics into a quantified **Operational Hazard Score**.

By focusing strictly on the **Kinetic Reality** of a track—regardless of the object's origin—the CRI provides aviation stakeholders and airspace intelligence with a standardized, actionable metric for threat mitigation and active risk management.

---

## 📊 Safety Risk Calculation Logic

The `Aero_Safety_Risk` (CRI) is calculated by treating the physical certainty of an asset as the baseline and its anomalous flight behavior as a **Kinetic Multiplier**.

### The Formula
$$CRI = \text{SOP}_{\text{Mean}} \times (1 + \text{FlightMod})$$

Where:
* **$\text{SOP}_{\text{Mean}}$ (Solid Object Probability):** Derived directly from the mean of the sampled distribution processed through the MCMC engine. It represents the aggregate probability of a physical, solid-state presence based on the baseline fusion of witness, environmental, and physical evidence. It remains independent of your Bayesian origin hypothesis adjustments.
* **`Flight_Mod` / $\text{FlightMod}$ (Kinetic Multiplier):** A scalar value ($0.00$ to $0.10$) representing aerodynamic behaviors that defy standard propulsion expectations (e.g., instantaneous acceleration, hypersonic velocity without signatures). In this module, it acts as a direct multiplier to evaluate safety hazard frontiers.

---

## 🚨 Hazard Level Classifications

Based on the calculated CRI score, tracking assets are automatically sorted into specific threat tiers to provide immediate situational awareness inside the dashboard:

| Hazard Level | Risk Score (CRI) | Operational Action |
| :--- | :--- | :--- |
| **🔴 Critical** | $> 0.75$ | Immediate safety-of-flight concern. Priority for active sensor-fusion audit and track containment. |
| **🟡 Elevated** | $0.45 - 0.75$ | Confirmed physical presence with kinematic anomalies; secondary track verification suggested. |
| **🟢 Low** | $< 0.45$ | Low physical certainty or standard aerodynamic characteristics; flagged for routine monitoring. |

---

## 🏗️ Localized Architecture & File Profiles

This directory operates as a **completely self-contained ecosystem** distinct from the root folder. It utilizes localized versions of the following files:

* **`app.py` (Safety Command Center UI):** The frontend Streamlit dashboard. Features a global dark theme aesthetic, dynamic background contour lines to graph the *Kinetic Hazard Frontier*, a crosshair-synced *Threat Intelligence Matrix*, and an *Active Track Auditor* sidebar that injects a prominent target lock overlay across all maps. Includes a built-in automated workable example generator if data is missing.
* **`jor_pymc_runner.py` (Operational Bridge):** Imports your tuned constants (`PRIOR_NH`, `CALIBRATION_K`, weights) and coordinates the active sampling process. It executes the aerospace safety evaluation loop, maps the `pd.cut` hazard thresholds, and saves results locally.
* **`jor_pymc.py` (Vectorized Bayesian Engine):** A highly optimized, vectorized PyMC sampling pipeline. It applies a `TruncatedNormal` distribution to your flight mechanics and extracts the precise `SOP_Mean` trace arrays required for the safety module calculations.
* **`jor_fusion.py` (Interactive Interface):** The localized configuration baseline used to manage scoring scripts and define global parameters ($K = 0.20$, priors, and evidence weights) explicitly for this folder.
* **`jor_scores.csv` (Core Telemetry Ledger):** The localized data exchange spreadsheet. It securely logs inputs and saves finalized MCMC outputs rounded to a clean `.round(3)` decimal limit to eliminate floating-point tail leakage.

---

## 🛠️ Module Execution

To run this specific safety command dashboard, you must step inside this directory so that the relative file paths align correctly. Run these commands sequentially in your terminal:

> **Step 1:** Step into the module folder  
> `cd Collision_Risk`
>
> **Step 2:** Initialize the dashboard interface  
> `streamlit run app.py`

*(Note: If you need to perform raw data modifications or check local sensitivity calibration metrics for this safety environment, execute `python jor_fusion.py` or `python jor_pymc_runner.py` while your terminal is remaining inside this subfolder).*

---
**Methodology:** The James Orion Report (JOR) Framework  
**Implementation:** V3.1 - Operational Safety Overlay  
