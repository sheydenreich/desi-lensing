import sys
from tqdm import tqdm,trange
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from scipy.optimize import minimize,minimize_scalar

# function definitions

def minimizing_fnc(sigma_sys,amplitudes,statistical_errors,target_value):
    """
    function that gives the squared difference between the chisq and the target value
    input:
    sigma_sys: float, the systematic error
    amplitudes: np.array, the amplitudes
    statistical_errors: np.array, the statistical errors
    target_value: float, the target value for the chisq
    """
    errors = np.sqrt(statistical_errors**2+sigma_sys**2)
    mean_amplitude = np.average(amplitudes,weights=1/errors**2)
    chisq = np.sum((amplitudes-mean_amplitude)**2/errors**2)
    return (chisq-target_value)**2

def find_best_fit_and_errors_brute(log_likelihood_func, n_sigma=1):
    from scipy.optimize import minimize,root_scalar
    from scipy.integrate import quad
    # from scipy.stats import chisq
    # Find the best-fit value by minimizing the negative log-likelihood
    result = minimize(lambda x: -log_likelihood_func(x), x0=1)
    best_fit = result.x[0]
    max_log_likelihood = log_likelihood_func(best_fit)
    
    # Define the likelihood ratio test statistic
    def likelihood_ratio_test(x):
        return -2 * (log_likelihood_func(x) - max_log_likelihood)
    
    # Find the critical value for the desired number of standard deviations
    # For a 1-dimensional likelihood function, the critical value is the square of the number of standard deviations
    critical_value = [0,0.68,0.95,0.997][n_sigma]
    
    # Find the parameter values that correspond to the critical value
    def find_error(critical_value):
        likelihood_fnc = lambda x: np.exp(log_likelihood_func(x))
        norm = quad(likelihood_fnc,-np.inf,np.inf)[0]
        likelihood_fnc = lambda x: np.exp(log_likelihood_func(x))/norm
        def objective_low(x):
            return quad(likelihood_fnc,-np.inf,x)[0] - (1-critical_value)/2
        def objective_up(x):
            return quad(likelihood_fnc,x,np.inf)[0] - (1-critical_value)/2
        try:
            lower_bound = root_scalar(objective_low,bracket = [-100,best_fit], method='brentq').root
        except ValueError:
            print(f"Failed to find lower bound for target value {critical_value}")
            lower_bound = 0
        try:
            upper_bound = root_scalar(objective_up,bracket = [best_fit,100], method='brentq').root
        except ValueError:
            print(f"Failed to find upper bound for target value {critical_value}")
            upper_bound = 100
        return lower_bound, upper_bound


    lower_bound, upper_bound = find_error(critical_value)
    
    return best_fit, lower_bound, upper_bound

def find_best_fit_and_errors(log_likelihood_func, n_sigma=1):
    from scipy.optimize import minimize,root_scalar
    # Find the best-fit value by minimizing the negative log-likelihood
    result = minimize(lambda x: -log_likelihood_func(x), x0=1)
    best_fit = result.x[0]
    max_log_likelihood = log_likelihood_func(best_fit)
    
    # Define the likelihood ratio test statistic
    def likelihood_ratio_test(x):
        return -2 * (log_likelihood_func(x) - max_log_likelihood)
    
    # Find the critical value for the desired number of standard deviations
    # For a 1-dimensional likelihood function, the critical value is the square of the number of standard deviations
    critical_value = n_sigma ** 2
    
    # Find the parameter values that correspond to the critical value
    def find_error(target_value):
        def objective(x):
            return likelihood_ratio_test(x) - target_value
        sol_upper = root_scalar(objective,bracket = [best_fit,best_fit+10], method='brentq')
        try:
            sol_lower = root_scalar(objective,bracket = [best_fit-10 , best_fit], method='brentq')
        except ValueError:
            print(f"Failed to find lower bound for target value {target_value}")
            return np.nan, sol_upper.root
        return sol_lower.root, sol_upper.root
    
    lower_bound, upper_bound = find_error(critical_value)
    
    return best_fit, lower_bound, upper_bound

def calculate_sigma_sys_bayesian(amplitudes,errors,confidence_sigma = [1,2],brute=False):
    mean_amplitude = np.average(amplitudes,weights=1/errors**2)

    def log_likelihood_fnc(sigma_sys):
        total_error = np.sqrt(errors**2+sigma_sys**2)
        chisq = np.sum((amplitudes-mean_amplitude)**2/total_error**2)
        covariance_det = np.prod(total_error**2)
        return -0.5*(chisq + np.log(covariance_det))
    
    if(brute):
        best_fit, lower_bound, upper_bound = find_best_fit_and_errors_brute(log_likelihood_fnc, n_sigma=confidence_sigma[0])
    else:
        best_fit, lower_bound, upper_bound = find_best_fit_and_errors(log_likelihood_fnc, n_sigma=confidence_sigma[0])
    lower_bounds = [lower_bound]
    upper_bounds = [upper_bound]
    for sigma in confidence_sigma[1:]:
        if(brute):
            _,lower_bound, upper_bound = find_best_fit_and_errors_brute(log_likelihood_fnc, n_sigma=sigma)
        else:
            _,lower_bound, upper_bound = find_best_fit_and_errors(log_likelihood_fnc, n_sigma=sigma)
        lower_bounds.append(lower_bound)
        upper_bounds.append(upper_bound)

    reduced_chisq = np.sum((amplitudes-mean_amplitude)**2/errors**2)/(len(amplitudes)-1)
    return reduced_chisq,lower_bounds[::-1]+[best_fit]+upper_bounds


def calculate_sigma_sys_chisqpdf(amplitudes,errors,confidence_regions = [0.68,0.95]):
    assert len(amplitudes)==len(errors), "Amplitudes and errors must have the same length"
    n_measurements = len(amplitudes)
    mean_amplitude = np.average(amplitudes,weights=1/errors**2)
    chisq = np.sum((amplitudes-mean_amplitude)**2/errors**2)
    reduced_chisq = chisq/(n_measurements-1)

    target_chisqs_upper = [chi2.ppf(0.5+confidence_region/2,n_measurements-1) for confidence_region in confidence_regions]
    target_chisqs_lower = [chi2.ppf(0.5-confidence_region/2,n_measurements-1) for confidence_region in confidence_regions[::-1]]
    target_chisqs = np.array(target_chisqs_lower + [n_measurements-1] + target_chisqs_upper)
    target_values = np.zeros_like(target_chisqs)
    for i,target_chisq in enumerate(target_chisqs):
        if (target_chisq >= chisq) or np.isnan(target_chisq):
            target_values[i] = np.nan
        else:
            # minval = minimize(minimizing_fnc,x0=[0.1],args=(amplitudes,errors,target_chisq),bounds=[(0,10)])
            # target_values[i] = minval.x[0] 
            minval = minimize_scalar(minimizing_fnc,args=(amplitudes,errors,target_chisq),bounds=[0,10],tol=1e-8)
            target_values[i] = minval.x

            assert minimizing_fnc(target_values[i],amplitudes,errors,target_chisq)<1e-4, "Minimization failed!"
                # print("Minimization failed! Amplitudes and errors:",amplitudes,errors)
                # assert minimizing_fnc(target_values[i],amplitudes,errors,target_chisq)<1e-1, "Minimization failed!"
    # print(target_chisqs,chisq,n_measurements)
    # print(target_values)
    return reduced_chisq,target_values[::-1]


def calculate_sigma_sys(amplitudes,errors,method="chisqpdf"):
    """
    wrapper for the three methods to calculate the systematic error
    input:
    amplitudes: np.array, the amplitudes
    errors: np.array, the statistical errors
    method: str, the method to use, either "chisqpdf" or "bayesian" or "bayesian_brute"
    returns:
    reduced_chisq: float, the reduced chisq
    sigma_sys: np.array, [-2sigma, -1sigma, best fit, 1sigma, 2sigma] array
    """
    if method=="chisqpdf":
        return calculate_sigma_sys_chisqpdf(amplitudes,errors)
    elif method=="bayesian":
        return calculate_sigma_sys_bayesian(amplitudes,errors)
    elif method=="bayesian_brute":
        return calculate_sigma_sys_bayesian(amplitudes,errors,brute=True)
    else:
        raise ValueError(f"Method {method} not recognized")


# Run the test 1000 times
sigma_sys_arr = np.zeros(1000)
sigma_sys_error_arr = np.zeros(1000)

sigma_sys_bayesian_arr = np.zeros(1000)
sigma_sys_bayesian_error_arr = np.zeros(1000)

sigma_sys_input_arr = np.zeros(1000)

for i in trange(1000):
    errors = np.random.uniform(0.1,0.5,10)
    sigma_sys_input = np.random.uniform(0,1)

    amplitudes = np.random.normal(1,np.sqrt(errors**2+sigma_sys_input**2))

    _,sigma_sys = calculate_sigma_sys(amplitudes,errors)
    _,sigma_sys_bayesian = calculate_sigma_sys(amplitudes,errors,method='bayesian')

    sigma_sys_arr[i] = sigma_sys[2]
    sigma_sys_bayesian_arr[i] = sigma_sys_bayesian[2]

    sigma_sys_error_arr[i] = (sigma_sys[3]-sigma_sys[1])/2
    sigma_sys_bayesian_error_arr[i] = (sigma_sys_bayesian[3]-sigma_sys_bayesian[1])/2

    sigma_sys_input_arr[i] = sigma_sys_input

# mask out all nan-elements and check how many are left
mask = np.isfinite(sigma_sys_arr)
print(np.sum(mask))
mask &= np.isfinite(sigma_sys_bayesian_arr)
print(np.sum(mask))
mask &= np.isfinite(sigma_sys_error_arr)
print(np.sum(mask))
mask &= np.isfinite(sigma_sys_bayesian_error_arr)
print(np.sum(mask))

sigma_sys_arr = sigma_sys_arr[mask]
sigma_sys_bayesian_arr = sigma_sys_bayesian_arr[mask]
sigma_sys_error_arr = sigma_sys_error_arr[mask]
sigma_sys_bayesian_error_arr = sigma_sys_bayesian_error_arr[mask]
sigma_sys_input_arr = sigma_sys_input_arr[mask]

# Check if the errors are correct
print(np.mean((sigma_sys_input_arr-sigma_sys_arr)**2/sigma_sys_error_arr**2),np.mean((sigma_sys_input_arr-sigma_sys_bayesian_arr)**2/sigma_sys_bayesian_error_arr**2))
print(np.mean((sigma_sys_input_arr-sigma_sys_arr)**2),np.mean((sigma_sys_input_arr-sigma_sys_bayesian_arr)**2))

# Compare best-fit and errors for the three methods for one random realization

errors = np.random.uniform(0.1,0.5,10)
sigma_sys_input = np.random.uniform(0,1)

amplitudes = np.random.normal(1,np.sqrt(errors**2+sigma_sys_input**2))

_,sigma_sys = calculate_sigma_sys(amplitudes,errors)
_,sigma_sys_bayesian = calculate_sigma_sys(amplitudes,errors,method='bayesian')
_,sigma_sys_bayesian_brute = calculate_sigma_sys(amplitudes,errors,method='bayesian_brute')



plt.errorbar(sigma_sys[2],0,xerr=np.array(sigma_sys[2]-sigma_sys[0],sigma_sys[4]-sigma_sys[2]).reshape((1,1)),fmt='o',color='blue')
plt.errorbar(sigma_sys[2],0,xerr=np.array(sigma_sys[2]-sigma_sys[1],sigma_sys[3]-sigma_sys[2]).reshape((1,1)),fmt='o',label='Frequentist',color='red')

plt.errorbar(sigma_sys_bayesian[2],1,xerr=np.array(sigma_sys_bayesian[2]-sigma_sys_bayesian[0],sigma_sys_bayesian[4]-sigma_sys_bayesian[2]).reshape((1,1)),fmt='o',color='blue')
plt.errorbar(sigma_sys_bayesian[2],1,xerr=np.array(sigma_sys_bayesian[2]-sigma_sys_bayesian[1],sigma_sys_bayesian[3]-sigma_sys_bayesian[2]).reshape((1,1)),fmt='o',label='Bayesian',color='red')

plt.errorbar(sigma_sys_bayesian_brute[2],2,xerr=np.array(sigma_sys_bayesian_brute[2]-sigma_sys_bayesian_brute[0],sigma_sys_bayesian_brute[4]-sigma_sys_bayesian_brute[2]).reshape((1,1)),fmt='o',color='blue')
plt.errorbar(sigma_sys_bayesian_brute[2],2,xerr=np.array(sigma_sys_bayesian_brute[2]-sigma_sys_bayesian_brute[1],sigma_sys_bayesian_brute[3]-sigma_sys_bayesian_brute[2]).reshape((1,1)),fmt='o',label='Bayesian brute force',color='red')

plt.axvline(sigma_sys_input,color='black',ls='--')


ax = plt.gca()
ax.set_yticks([0,1,2])
ax.set_yticklabels(['Frequentist','Bayesian','Bayesian brute'])
