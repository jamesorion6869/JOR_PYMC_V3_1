import pymc as pm
import pandas as pd
import numpy as np
import pytensor.tensor as pt


def calc_beta_params_vec(mu, sigma):
    """Vectorized calculation of Beta parameters."""
    var = sigma ** 2

    # Ensure mu is within (0, 1) to avoid math errors
    mu = np.clip(mu, 0.001, 0.999)

    alpha = ((1 - mu) / var - 1 / mu) * mu ** 2
    beta = alpha * (1 / mu - 1)

    return alpha, beta


def run_jor_pymc_safe(
    data,
    prior_mu=0.20,
    k_val=0.20,
    weights=[0.4, 0.3, 0.3],
    draws=1000,
    tune=1000,
    chains=4,
    cores=4,
    target_accept=0.95
):
    """
    JOR-V3 Vectorized Bayesian Engine.
    Now receives tuned parameters from the runner script.
    """

    if isinstance(data, str):
        df = pd.read_csv(data)
    else:
        df = data.copy()

    # Vectorize inputs
    c_scores = df['C_score'].values
    e_scores = df['E_score'].values
    p_scores = df['P_score'].values
    f_mods = df['flight_mod'].values
    num_cases = len(df)

    # 1. Pre-calculate Beta parameters using the PASSED prior_mu
    # We use a standard 0.02 sigma for evidentiary uncertainty per JOR-V3
    c_a, c_b = calc_beta_params_vec(c_scores, 0.02)
    e_a, e_b = calc_beta_params_vec(e_scores, 0.02)
    p_a, p_b = calc_beta_params_vec(p_scores, 0.02)
    prior_a, prior_b = calc_beta_params_vec(prior_mu, 0.02)

    with pm.Model() as model:

        # Priors (Vectors for each case)
        C = pm.Beta('Witness', alpha=c_a, beta=c_b, shape=num_cases)
        E = pm.Beta('Environment', alpha=e_a, beta=e_b, shape=num_cases)
        P = pm.Beta('Physical', alpha=p_a, beta=p_b, shape=num_cases)

        # Flight modifier defined as a sampled Normal distribution
        Flight_Effect = pm.TruncatedNormal('Flight_Effect', mu=f_mods, sigma=0.03, lower=0.0, upper=0.10, shape=num_cases)

        # 2. SOP (Baseline) uses the PASSED weights
        SOP = pm.Deterministic('SOP', weights[0] * C + weights[1] * E + weights[2] * P)

        # 3. FLIGHT FIX: Apply Flight_Effect to Physical (P) specifically
        # This ensures the anomaly is moderated by the Physical weight (weights[2])
        P_Anomalous = pt.clip(P * (1 + Flight_Effect), 0.0, 0.95)

        # 4. NHP (Anomalous) uses the boosted Physical score and PASSED weights
        NHP = pm.Deterministic(
            'NHP',
            weights[0] * C + weights[1] * E + weights[2] * P_Anomalous
        )

        # 5. Likelihood equivalents using the PASSED k_val
        # P(E|H) = 1 - NHP + (K * SOP)
        like_h = pm.Deterministic(
            'Like_H',
            pt.minimum(1.0, 1.0 - NHP + (k_val * SOP))
        )

        like_nh = NHP

        # Bayesian update logic
        prior = pm.Beta('Prior_Dist', alpha=prior_a, beta=prior_b)

        posterior = pm.Deterministic(
            'Posterior_NH',
            (prior * like_nh) /
            ((prior * like_nh) + ((1 - prior) * like_h))
        )

        # Execute Sampling
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            target_accept=target_accept,
            progressbar=True,
            random_seed=42
        )

    # Extract and calculate results
    post_samples = trace.posterior['Posterior_NH'].values
    stacked_samples = post_samples.reshape(-1, num_cases)

    df_results = pd.DataFrame({
        'case_name': df['case_name'],
        'Posterior_Mean': np.mean(stacked_samples, axis=0),
        'CI_2.5%': np.percentile(stacked_samples, 2.5, axis=0),
        'CI_97.5%': np.percentile(stacked_samples, 97.5, axis=0)
    })

    return df_results