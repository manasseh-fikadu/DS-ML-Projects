from scipy.stats import ttest_ind
from statsmodels.stats.weightstats import ztest
from statsmodels.stats.weightstats import DescrStatsW

def perform_statistical_tests(population1, population2, alpha=0.05):
    '''
    Perform the t-test and z-test for two independent samples.

    Parameters
    ----------
    population1 : array_like
        First sample.
    population2 : array_like
        Second sample.
    alpha : float, optional
        Significance level. The default is 0.05.

    Returns
    -------
    results : dict
    '''
    # Perform the t-test
    t, p_ttest = ttest_ind(population1, population2, equal_var=False)

    # Perform the z-test
    z, p_ztest = ztest(population1, x2=population2, value=0, alternative='two-sided')

    # Calculate confidence intervals
    ci_population1 = DescrStatsW(population1).tconfint_mean()
    ci_population2 = DescrStatsW(population2).tconfint_mean()

    # Check the p-values and print results
    results = {
        't-test': {
            't-statistic': t,
            'p-value': p_ttest,
            'hypothesis': 'Rejected' if p_ttest < alpha else 'Accepted'
        },
        'z-test': {
            'z-statistic': z,
            'p-value': p_ztest,
            'hypothesis': 'Rejected' if p_ztest < alpha else 'Accepted'
        },
        'confidence_intervals': {
            'population1': ci_population1,
            'population2': ci_population2
        }
    }
    
    return results

def print_results(results):
    '''
    Print the results of the statistical tests.

    Parameters
    ----------
    results : dict

    Returns
    -------
    None.
    '''
    print('t-test results:')
    print(results['t-test'])

    print('\nz-test results:')
    print(results['z-test'])

    print('\nConfidence interval for population1:', results['confidence_intervals']['population1'])
    print('Confidence interval for population2:', results['confidence_intervals']['population2'])