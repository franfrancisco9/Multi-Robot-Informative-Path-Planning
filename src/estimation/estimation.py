# src/estimation/estimation.py
"""
Estimate Sources
Citation:
M. Morelande, B. Ristic and A. Gunatilaka, "Detection and parameter estimation of multiple radioactive sources,"
2007 10th International Conference on Information Fusion, Quebec, QC, Canada, 2007, pp. 1-7, doi: 10.1109/ICIF.2007.4408094.

- Created by: Francisco Fonseca on July 2024
"""
import numpy as np
from typing import List, Tuple
from scipy.stats import uniform, multivariate_normal, poisson
from src.point_source.point_source import PointSourceField

def poisson_log_likelihood(
    theta: np.ndarray, 
    obs_wp: np.ndarray, 
    obs_vals: np.ndarray, 
    lambda_b: float, 
    M: int
) -> float:
    """
    Calculate the Poisson log-likelihood for given source parameters and observations.
    
    Parameters:
    - theta: Source parameters (flattened array of [x, y, alpha] for each source).
    - obs_wp: Observation waypoints, array of shape (num_observations, 2).
    - obs_vals: Observed values (counts), array of shape (num_observations,).
    - lambda_b: Background radiation rate.
    - M: Number of sources.

    Returns:
    - Log-likelihood value.
    """
    obs_wp = np.array(obs_wp)  # Ensure obs_wp is a NumPy array
    obs_vals = np.round(obs_vals).astype(int)  # Scale down observed values to ensure reasonable counts
    sources = theta.reshape((M, 3))
    
    d_ji = (obs_wp[:, None, 0] - sources[:, 0])**2 + (obs_wp[:, None, 1] - sources[:, 1])**2
    alpha_i = sources[:, 2]
    lambda_j = lambda_b + np.sum(alpha_i / np.maximum(d_ji, 1e-6), axis=1)
    
    # Ensure lambda_j is positive to avoid log(0) issues
    lambda_j = np.maximum(lambda_j, 1e-10)
    
    log_pmf = poisson.logpmf(obs_vals, lambda_j)
    total_log_likelihood = np.sum(log_pmf)
    # print(f"Total log-likelihood: {total_log_likelihood}")
    return total_log_likelihood

def importance_sampling_with_progressive_correction(
    obs_wp: np.ndarray, 
    obs_vals: np.ndarray, 
    lambda_b: float, 
    M: int, 
    n_samples: int, 
    s_stages: int, 
    prior_dist: List[callable], 
    scenario: 'Scenario', 
    alpha: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform importance sampling with progressive correction to estimate source parameters.
    
    Parameters:
    - obs_wp: Observation waypoints, array of shape (num_observations, 2).
    - obs_vals: Observed values (counts), array of shape (num_observations,).
    - lambda_b: Background radiation rate.
    - M: Number of sources.
    - n_samples: Number of samples.
    - s_stages: Number of stages.
    - prior_dist: List of prior distributions.
    - scenario: Scenario object containing workspace size and intensity range.
    - alpha: Perturbation factor.
    
    Returns:
    - theta_estimate: Estimated source parameters.
    - theta_samples: Samples of source parameters.
    """
    gammas = np.linspace(0.1, 1.0, s_stages)
    theta_samples = np.column_stack([dist.rvs(n_samples) for dist in prior_dist])
    
    def calc_weights(
        theta: np.ndarray, 
        gamma: float, 
        obs_wp: np.ndarray, 
        obs_vals: np.ndarray, 
        lambda_b: float, 
        M: int
    ) -> np.ndarray:
        sample_likelihood = np.array([poisson_log_likelihood(theta[i], obs_wp, obs_vals, lambda_b, M) for i in range(len(theta))])
        # print(f"Sample likelihood: {sample_likelihood}")
        log_weights = gamma * sample_likelihood
        max_log_weights = np.max(log_weights)

        weights = np.exp(log_weights - max_log_weights)  # Avoid underflow
        # print(f"Weights: {weights}")
        sum_weights = np.sum(weights)
        weights /= sum_weights
        if np.any(np.isnan(weights)):
            print("Warning: NaN weights encountered. Returning uniform weights.")
            return np.full(n_samples, 1 / n_samples)
        return weights

    for gamma in gammas:
        weights = calc_weights(theta_samples, gamma, obs_wp, obs_vals, lambda_b, M)
        if np.any(np.isnan(weights)):
            print("Warning: NaN weights encountered. Repeating the sampling process.")
            return importance_sampling_with_progressive_correction(obs_wp, obs_vals, lambda_b, M, n_samples, s_stages, prior_dist, scenario, alpha)
        
        indices = np.random.choice(n_samples, size=n_samples, p=weights)
        resampled_samples = theta_samples[indices]
        means = np.mean(resampled_samples, axis=0)
        covariances = np.cov(resampled_samples, rowvar=False)
        
        perturbations = multivariate_normal.rvs(mean=means, cov=covariances * alpha, size=n_samples)
        perturbations[:, 0] = np.clip(perturbations[:, 0], 0, scenario.workspace_size[0])
        perturbations[:, 1] = np.clip(perturbations[:, 1], 0, scenario.workspace_size[1])
        perturbations[:, 2] = np.clip(perturbations[:, 2], scenario.intensity_range[0], scenario.intensity_range[1])
        
        theta_samples = perturbations

    theta_estimate = np.mean(theta_samples, axis=0)
    return theta_estimate, theta_samples

def calculate_bic(log_likelihood: float, num_params: int, num_data_points: int) -> float:
    """Calculate the Bayesian Information Criterion."""
    return np.log(-2 * log_likelihood) + num_params * np.log(num_data_points)

def estimate_sources_bayesian(
    obs_wp: np.ndarray, 
    obs_vals: np.ndarray, 
    lambda_b: float, 
    max_sources: int, 
    n_samples: int, 
    s_stages: int, 
    scenario: PointSourceField
) -> Tuple[np.ndarray, int, float]:
    """
    Estimate the number and parameters of radioactive sources using Bayesian approach.
    
    Parameters:
    - obs_wp: Observation waypoints, array of shape (num_observations, 2).
    - obs_vals: Observed values (counts), array of shape (num_observations,).
    - lambda_b: Background radiation rate.
    - max_sources: Maximum number of sources to consider.
    - n_samples: Number of samples for importance sampling.
    - s_stages: Number of stages for progressive correction.
    - scenario: Scenario object containing workspace size and intensity range.
    
    Returns:
    - best_estimate: Best estimate of source parameters.
    - best_M: Best estimate of the number of sources.
    - best_bic: Best Bayesian Information Criterion value.
    """
    best_bic = -np.inf
    best_estimate = None
    best_M = 0

    for M in range(1, max_sources + 1):
        prior_x = uniform(loc=0, scale=scenario.workspace_size[0])
        prior_y = uniform(loc=0, scale=scenario.workspace_size[1])
        prior_intensity = uniform(loc=scenario.intensity_range[0], scale=scenario.intensity_range[1])
        prior_dist = [prior_x, prior_y, prior_intensity] * M
        
        theta_estimate, theta_samples = importance_sampling_with_progressive_correction(
            obs_wp, obs_vals, lambda_b, M, n_samples, s_stages, prior_dist, scenario
        )
        
        log_likelihood = poisson_log_likelihood(theta_estimate, obs_wp, obs_vals, lambda_b, M)
        num_params = 3 * M
        bic = calculate_bic(log_likelihood, num_params, len(obs_vals))
        
        if bic > best_bic:
            # print(f"\nEstimated {M} sources: {theta_estimate}")
            # print(f"Log-likelihood: {log_likelihood}")
            # print(f"BIC: {bic}")
            best_bic = bic
            best_estimate = theta_estimate
            best_M = M

    return best_estimate, best_M, best_bic
