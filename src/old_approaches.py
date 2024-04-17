# A collection of old approaches, or code to be preserved for future reference until it is no longer needed.

# Old Estimation Approach
def poisson_log_likelihood(theta, obs_wp, obs_vals, lambda_b, M, r_s = 0.5, T = 100, r_d = 0.5):
    converted_obs_vals = np.round(obs_vals).astype(int)

    log_likelihood = 0.0
    sources = theta.reshape((M, 3)) if len(theta) == 3 * M else np.array([theta])
    for obs_index, (x_obs, y_obs) in enumerate(obs_wp):
        lambda_j = lambda_b  # Start with background intensity
        
        for source in sources:
            x_source, y_source, source_intensity = source
            d_ji = np.sqrt((x_obs - x_source)**2 + (y_obs - y_source)**2)
            
            # Calculate the intensity contribution from each source
            if d_ji <= r_s:
                intensity = source_intensity / (4 * np.pi * r_s**2)
            else:
                intensity = source_intensity * T / (4 * np.pi * d_ji**2)
                
            # Calculate the response if within detectable range
            if d_ji <= r_d:
                theta = np.arcsin(min(r_d / d_ji, 1))
                response = 0.5 * source_intensity * (1 - np.cos(theta))
                intensity += 50 * response
            
            lambda_j += intensity  # Sum the contributions for each source

        # Use converted_obs_vals which are now in the appropriate count format
        log_pmf = poisson.logpmf(converted_obs_vals[obs_index], lambda_j)
        log_likelihood += log_pmf

    return -log_likelihood  # Minimization in optimization routines

def estimate_parameters(obs_wp, obs_vals, lambda_b, M):
    # Initial guess and bounds adjusted for using the log of intensity values
    sigma = 0.5  # Step size
    lower_bounds = [0, 0, 1e3] * M  # Log of intensity lower bound
    upper_bounds = [40, 40, 1e5] * M  # Log of intensity upper bound
    initial_guess = np.random.uniform(lower_bounds, upper_bounds)
    es = cma.CMAEvolutionStrategy(initial_guess, sigma, {'bounds': [lower_bounds, upper_bounds]})
    es.optimize(lambda x: poisson_log_likelihood(x, obs_wp, obs_vals, lambda_b, M))

    # Best solution with exponentiated intensities
    xbest_transformed = es.result.xbest
    return xbest_transformed

def estimate_parameters_nelder_mead(obs_wp, obs_vals, lambda_b, M):
    initial_guess = np.concatenate([np.random.uniform(0, 40, 2*M), np.random.uniform(1e3, 1e5, M)])
    result = minimize(lambda theta: -poisson_log_likelihood(theta, obs_wp, obs_vals, lambda_b, M),
                      initial_guess, method='l-bfgs-b', bounds=[(0, 40)] * 2*M + [(1e3, 1e5)] * M)
    if result.success:
        estimated_theta = result.x
        return estimated_theta
    else:
        return None
    
def compute_FIM(obs_wp, estimated_theta, lambda_b, M):
    FIM = np.zeros((3 * M, 3 * M))
    epsilon = 1e-6
    # Extract source positions and intensities from estimated_theta
    source_positions = estimated_theta[:2*M].reshape((M, 2))
    source_intensities = estimated_theta[2*M:]

    for j, obs_point in enumerate(obs_wp):
        for i in range(M):
            x_i, y_i = source_positions[i]
            alpha_i = source_intensities[i]
            d_ji = max(epsilon, np.sqrt((obs_point[0] - x_i)**2 + (obs_point[1] - y_i)**2))
            d_ji_squared = d_ji**2

            # Compute partial derivatives as per the paper
            d_lambda_j_d_xi = (2 * alpha_i * (obs_point[0] - x_i)) / d_ji_squared
            d_lambda_j_d_yi = (2 * alpha_i * (obs_point[1] - y_i)) / d_ji_squared
            d_lambda_j_d_alpha_i = 1 / d_ji

            # Stack the derivatives for all sources to form the gradient
            gradient = np.zeros(3 * M)
            gradient[3*i] = d_lambda_j_d_xi
            gradient[3*i + 1] = d_lambda_j_d_yi
            gradient[3*i + 2] = d_lambda_j_d_alpha_i

            # Calculate lambda_j for the current observation and source parameters
            lambda_j = lambda_b
            for k, (x_k, y_k) in enumerate(source_positions):
                alpha_k = source_intensities[k]
                d_jk = max(epsilon, np.sqrt((obs_point[0] - x_k)**2 + (obs_point[1] - y_k)**2))
                lambda_j += alpha_k / d_jk**2

            # Update the FIM with the outer product of the gradient, scaled by 1/lambda_j
            FIM += np.outer(gradient, gradient) / lambda_j

    return FIM

def estimate_sources(obs_wp, obs_vals, lambda_b, M_max):
    best_score = -np.inf  # Initialize to negative infinity for maximization
    best_model = None
    best_M = None
    epsilon = 1e-6
    for M in range(1, M_max + 1):
        bounds = [(0, np.max(obs_wp)), (0, np.max(obs_wp))] * M + [(1e3, 1e5)] * M
        estimated_theta = estimate_parameters_nelder_mead(obs_wp, obs_vals, lambda_b, M)
        if estimated_theta is None:
            estimated_theta = np.array([0, 0, 1e3] * M)
        print(f"Estimated theta: {estimated_theta}")
        if estimated_theta is not None:
            # Compute the likelihood at the estimated parameters, ensure it's the likelihood, not negative log-likelihood
            # Since the poisson_log_likelihood function returns the negative log-likelihood for minimization, negate its result.
            log_likelihood = -(poisson_log_likelihood(estimated_theta, obs_wp, obs_vals, lambda_b, M))
            
            # Calculate the Fisher Information Matrix
            FIM = compute_FIM(obs_wp, estimated_theta, lambda_b, M)

            # Ensure the determinant of FIM is positive to avoid taking log of a non-positive number
            det_FIM = np.linalg.det(FIM)
            if det_FIM <= 0:
                print("Warning: Determinant of FIM is non-positive, adjusting to epsilon.")
                det_FIM = -det_FIM + epsilon

            # Compute the penalty term using the determinant of the Fisher Information Matrix
            penalty = -0.5 * np.log(det_FIM)

            # Calculate beta_r according to the formula
            beta_r = log_likelihood + penalty  # Note: log_likelihood is already the log of the probability, not negative

            print(f"Number of sources: {M}, Beta_r: {beta_r}")
            if beta_r > best_score:
                best_score = beta_r
                best_model = estimated_theta
                best_M = M
        else:
            estimated_theta = np.array([0, 0, 1e3] * M)

    return best_model, best_M

