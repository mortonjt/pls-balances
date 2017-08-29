"""
This file generates feature tables for benchmarking
"""
import pandas as pd
import numpy as np
from skbio.stats.composition import closure
from scipy.stats import norm, expon


def generate_table(reps, n_species_class1, n_species_class2,
                   n_species_shared, effect_size, library_size=10000):
    """
    Parameters
    ----------
    reps : int
        Number of replicate samples per test.
    n_species_class1 : int
        Number of species changing in class1.
    n_species_class2 : int
        Number of species changing in class2.
    n_species_shared: int
        Number of species shared between classes.
    effect_size : int
        The effect size difference between the feature abundances.
    library_size : np.array
        A vector specifying the library sizes per sample.

    Returns
    -------
    generator of
        pd.DataFrame
           Ground truth tables.
        pd.DataFrame
           Metadata group categories, n_diff and effect_size
        pd.Series
           Species actually differentially abundant.
    """
    data = []
    metadata = []
    for _ in range(reps):
        data.append([effect_size]*n_species_class1 + [1]*(n_species_class1+n_species_shared) )
        metadata += [0]

    for _ in range(reps):
        data.append([1]*(n_species_class1+n_species_shared) + [effect_size]*n_species_class2)
        metadata += [1]

    data = closure(np.vstack(data))

    metadata = pd.DataFrame({'group': metadata})
    metadata['n_diff'] = n_species_class1 + n_species_class2
    metadata['effect_size'] = effect_size
    metadata['library_size'] = library_size
    metadata.index = ['S%d' % i for i in range(len(metadata.index))]
    table = pd.DataFrame(data)
    table.index = ['S%d' % i for i in range(len(table.index))]
    table.columns = ['F%d' % i for i in range(len(table.columns))]
    ground_truth = (list(table.columns[:n_species_class1]) +
                    list(table.columns[-n_species_class2:]))

    return table, metadata, ground_truth


def compositional_effect_size_generator(max_alpha, reps,
                                        intervals, n_species, n_diff):
    """
    Parameters
    ----------
    max_alpha : float
        Maximum effect size represented as log fold change
    reps : int
        Number of replicate samples per test.
    intervals : int
        Number of effect size intervals.  This corresponds to the
        number of experiments to run.
    n_species : int
        Number of species.
    n_diff : int
        Number of differentially abundant species in each group.

    Returns
    -------
    generator of
        pd.DataFrame
           Ground truth tables.
        pd.Series
           Metadata group categories.
        pd.Series
           Species actually differentially abundant.
    """
    for a in np.logspace(0, max_alpha, intervals):
        yield generate_table(reps,
                             n_species_class1=n_diff,
                             n_species_class2=n_diff,
                             n_species_shared=n_species-2*n_diff,
                             effect_size=a)


def compositional_variable_features_generator(max_changing, fold_change, reps,
                                              intervals, n_species):
    """
    Parameters
    ----------
    max_changing : float
        Maximum number of changing species.
    fold_change : float
        The fold change of the altered features.
    reps : int
        Number of replicate samples per test.
    intervals : int
        Number of effect size intervals.  This corresponds to the
        number of experiments to run.
    n_species : int
        Number of species.

    Returns
    -------
    generator of
        pd.DataFrame
           Ground truth tables.
        pd.Series
           Metadata group categories, and sample information used
           for benchmarking.
        pd.Series
           Species actually differentially abundant.
    """
    for a in np.linspace(0, max_changing, intervals):
        a_ = int(a)
        yield generate_table(reps,
                             n_species_class1=a_,
                             n_species_class2=a_,
                             n_species_shared=n_species - 2*a_,
                             effect_size=fold_change)


def generate_band_table(mu, sigma, gradient, n_species,
                        lam, n_contaminants, library_size=10000):
    """ Generates a band table with normal variables.

    Parameters
    ----------
    mu : pd.Series
        Vector of species optimal positions along gradient.
    sigma : float
        Variance of the species normal distribution.
    gradient : array
        Vector of gradient values.
    n_species : int
        Number of species to simulate.
    n_contaminants : int
       Number of contaminant species.
    lam : float
       Decay constant for contaminant urn (assumes that the contaminant urn
       follows an exponential distribution).

    Returns
    -------
    generator of
        pd.DataFrame
           Ground truth tables.
        pd.Series
           Metadata group categories, and sample information used
           for benchmarking.
        pd.Series
           Species actually differentially abundant.
    """
    xs = [norm.pdf(gradient, loc=mu[i], scale=sigma)
          for i in range(len(mu))]

    table = closure(np.vstack(xs).T)
    x = np.linspace(0, 1, n_contaminants)
    contaminant_urn = closure(expon.pdf(x, scale=lam))
    contaminant_urns = np.repeat(np.expand_dims(contaminant_urn, axis=0),
                                 table.shape[0], axis=0)
    table = np.hstack((table, contaminant_urns))
    s_ids = ['F%d' % i for i in range(n_species)]
    c_ids = ['X%d' % i for i in range(n_contaminants)]
    table = closure(table)

    metadata = pd.DataFrame({'gradient': gradient})
    metadata['n_diff'] = len(mu)
    metadata['library_size'] = library_size
    metadata.index = ['S%d' % i for i in range(len(metadata.index))]

    table = pd.DataFrame(table)
    table.index = ['S%d' % i for i in range(len(table.index))]
    table.columns = s_ids + c_ids
    ground_truth = list(table.columns)[:n_species]
    return table, metadata, ground_truth


def compositional_regression_prefilter_generator(
        max_gradient, gradient_intervals, sigma,
        n_species, lam, max_contaminants,
        contaminant_intervals):
    """ Generates tables with increasing regression effect sizes.

    Parameters
    ----------
    max_gradient : float
       Maximum value along the gradient (assumes that the gradient
       starts at zero.
    gradient_intervals : int
       Number of sampling intervals along the gradient.
    sigma : float
       Variance for normal distribution used to choose coefficients.
    n_species : int
       Number of species to simulate in the ground truth.
    lam : float
       Decay constant for contaminant urn (assumes that the contaminant urn
       follows an exponential distribution).
    max_contaminants : int
       Maximum number of contaminants.
    contaminant_intervals : int
       The number of intervals for benchmarking contaminants.

    Returns
    -------
    generator of
        pd.DataFrame
           Ground truth tables.
        pd.Series
           Metadata group categories, and sample information used
           for benchmarking.
        pd.Series
           Species actually differentially abundant.
    """
    gradient = np.linspace(0, max_gradient, gradient_intervals)
    mu = np.linspace(0, max_gradient, n_species)
    i = np.linspace(2, max_contaminants, contaminant_intervals)
    for a in i:
        yield generate_band_table(mu, sigma, gradient, n_species,
                                  lam, n_contaminants=int(a),
                                  library_size=10000)

def library_size_difference_generator():
    pass

def missing_at_random_generator():
    pass

def missing_not_at_random_generator():
    pass
