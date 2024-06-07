import copy
import math
import random
import statistics

from numpy.random import choice

from src.completeness.metrics import chao2, get_singletons, get_doubletons, estimate_shannon_entropy_abundance, estimate_simpson_diversity_abundance, \
    estimate_shannon_entropy_incidence, estimate_simpson_diversity_incidence


def get_bootstrap_probabilities_abundance(reference_sample, sample_size):
    s_obs = len(reference_sample)

    f_1 = get_singletons(reference_sample)
    f_2 = get_doubletons(reference_sample)
    f_0 = 0
    if f_2 > 0:
        f_0 = ((sample_size - 1) / sample_size) * f_1 ** 2 / (2 * f_2)
    else:
        f_0 = ((sample_size - 1) / sample_size) * f_1 * (f_1 - 1) / 2
    f_0 = math.ceil(f_0)
    # print(s_obs,f_0)
    probabilities = []
    c = get_c_n_abundance(reference_sample, sample_size)
    factor = factor_abundance(reference_sample, sample_size, c)

    for x_i in reference_sample.values():
        adapted_p = (x_i / sample_size) * (1 - factor * ((1 - (x_i / sample_size)) ** sample_size))
        probabilities.append(adapted_p)
    [probabilities.append((1 - c) / f_0) for i in range(0, f_0)]
    return probabilities


def get_bootstrap_probabilities_incidence(reference_sample, sample_size):
    s_obs = len(reference_sample)
    f_0 = 0

    f_1 = get_singletons(reference_sample)
    f_2 = get_doubletons(reference_sample)

    if f_2 > 0:
        f_0 = ((sample_size - 1) / sample_size) * f_1 ** 2 / (2 * f_2)
    else:
        f_0 = ((sample_size - 1) / sample_size) * f_1 * (f_1 - 1) / 2
    f_0 = math.ceil(f_0)
    # print(f_0, s_obs)
    # print(s_obs,f_0)

    probabilities = []
    c = get_c_n_incidence(reference_sample, sample_size)
    u = sum(reference_sample.values())

    factor = factor_incidence(reference_sample, sample_size, u, c)

    for y_i in reference_sample.values():
        adapted_p = (y_i / sample_size) * (1 - factor * ((1 - (y_i / sample_size)) ** sample_size))
        probabilities.append(adapted_p)
    [probabilities.append((u / sample_size) * (1 - c) / f_0) for i in range(0, f_0)]
    return probabilities


def generate_sample_abundance(reference_sample, sample_size, probabilities):
    s_obs = len(reference_sample)
    f_1 = get_singletons(reference_sample)
    f_2 = get_doubletons(reference_sample)
    f_0 = 0
    if f_2 > 0:
        f_0 = ((sample_size - 1) / sample_size) * f_1 ** 2 / (2 * f_2)
    else:
        f_0 = ((sample_size - 1) / sample_size) * f_1 * (f_1 - 1) / 2
    f_0 = math.ceil(f_0)
    selected_species = choice([x for x in range(0, s_obs + f_0)], sample_size, p=probabilities)
    bootstrap_sample = {}
    for s in selected_species:
        bootstrap_sample[s] = bootstrap_sample.get(s, 0) + 1

    return bootstrap_sample


def generate_sample_incidence(reference_sample, sample_size, probabilities):
    s_obs = len(reference_sample)
    f_0 = 0

    f_1 = get_singletons(reference_sample)
    f_2 = get_doubletons(reference_sample)

    if f_2 > 0:
        f_0 = ((sample_size - 1) / sample_size) * f_1 ** 2 / (2 * f_2)
    else:
        f_0 = ((sample_size - 1) / sample_size) * f_1 * (f_1 - 1) / 2
    f_0 = math.ceil(f_0)

    bootstrap_sample = {}
    for n in range(0, sample_size):
        for s, p in enumerate(probabilities):
            if random.random() <= p:
                bootstrap_sample[s] = bootstrap_sample.get(s, 0) + 1

    # print(bootstrap_sample)
    # print(len(bootstrap_sample), sum(bootstrap_sample.values()))
    return bootstrap_sample


# generate multiple bootstrap sample for a set of sample sizes up to the given sample size:
def generate_bootstrap_sequence_abundance(reference_sample, sample_size, step_size=10):
    s_obs = len(reference_sample)

    f_1 = get_singletons(reference_sample)
    f_2 = get_doubletons(reference_sample)
    f_0 = 0
    if f_2 > 0:
        f_0 = ((sample_size - 1) / sample_size) * f_1 ** 2 / (2 * f_2)
    else:
        f_0 = ((sample_size - 1) / sample_size) * f_1 * (f_1 - 1) / 2
    f_0 = math.ceil(f_0)
    # print(s_obs,f_0)
    probabilities = []
    c = get_c_n_abundance(reference_sample, sample_size)
    factor = factor_abundance(reference_sample, sample_size, c)
    # print(c,factor)

    for x_i in reference_sample.values():
        # print(x_i, sample_size)
        #    print((x_i / sample_size), (1 - factor * (1 - (x_i / sample_size) ** sample_size)))
        adapted_p = (x_i / sample_size) * (1 - factor * ((1 - (x_i / sample_size)) ** sample_size))
        probabilities.append(adapted_p)
    [probabilities.append((1 - c) / f_0) for i in range(0, f_0)]

    # print(probabilities)
    # print(sum(probabilities))

    sample_sequence = []
    sample_steps = []
    bootstrap_sample = {}
    steps = 0
    while steps + step_size < sample_size:
        selected_species = choice([x for x in range(0, s_obs + f_0)], step_size, p=probabilities)
        for s in selected_species:
            bootstrap_sample[s] = bootstrap_sample.get(s, 0) + 1
        steps = steps + step_size
        sample_steps.append(steps)
        sample_sequence.append(copy.deepcopy(bootstrap_sample))
        # print(bootstrap_sample)
        # print(steps,len(bootstrap_sample), sum(bootstrap_sample.values()))

    if sample_size - steps > 0:
        selected_species = choice([x for x in range(0, s_obs + f_0)], sample_size - steps, p=probabilities)
        for s in selected_species:
            bootstrap_sample[s] = bootstrap_sample.get(s, 0) + 1
        steps = steps + (sample_size - steps)
        sample_steps.append(steps)
        sample_sequence.append(copy.deepcopy(bootstrap_sample))
        # print(bootstrap_sample)
        # print(steps, len(bootstrap_sample), sum(bootstrap_sample.values()))

    return sample_sequence, sample_steps


# generate a single bootstrap sample for the given sample size
def generate_bootstrap_sample_abundance(reference_sample, sample_size):
    s_obs = len(reference_sample)

    f_1 = get_singletons(reference_sample)
    f_2 = get_doubletons(reference_sample)
    f_0 = 0
    if f_2 > 0:
        f_0 = ((sample_size - 1) / sample_size) * f_1 ** 2 / (2 * f_2)
    else:
        f_0 = ((sample_size - 1) / sample_size) * f_1 * (f_1 - 1) / 2
    f_0 = math.ceil(f_0)
    # print(s_obs,f_0)
    probabilities = []
    c = get_c_n_abundance(reference_sample, sample_size)
    factor = factor_abundance(reference_sample, sample_size, c)

    for x_i in reference_sample.values():
        adapted_p = (x_i / sample_size) * (1 - factor * ((1 - (x_i / sample_size)) ** sample_size))
        probabilities.append(adapted_p)
    [probabilities.append((1 - c) / f_0) for i in range(0, f_0)]

    # print(probabilities)
    # print(sum(probabilities))

    selected_species = choice([x for x in range(0, s_obs + f_0)], sample_size, p=probabilities)
    bootstrap_sample = {}
    for s in selected_species:
        bootstrap_sample[s] = bootstrap_sample.get(s, 0) + 1

    # print(bootstrap_sample)
    # print(len(bootstrap_sample), sum(bootstrap_sample.values()))
    return bootstrap_sample


def generate_bootstrap_sequence_incidence(reference_sample, sample_size, step_size=10):
    s_obs = len(reference_sample)
    f_0 = 0

    f_1 = get_singletons(reference_sample)
    f_2 = get_doubletons(reference_sample)

    if f_2 > 0:
        f_0 = ((sample_size - 1) / sample_size) * f_1 ** 2 / (2 * f_2)
    else:
        f_0 = ((sample_size - 1) / sample_size) * f_1 * (f_1 - 1) / 2
    f_0 = math.ceil(f_0)
    # print(s_obs,f_0)

    probabilities = []
    c = get_c_n_incidence(reference_sample, sample_size)
    u = sum(reference_sample.values())

    factor = factor_incidence(reference_sample, sample_size, u, c)

    for y_i in reference_sample.values():
        adapted_p = (y_i / sample_size) * (1 - factor * ((1 - (y_i / sample_size)) ** sample_size))
        probabilities.append(adapted_p)
    [probabilities.append((u / sample_size) * (1 - c) / f_0) for i in range(0, f_0)]
    print("PROBABILITIES")
    print(probabilities)
    print(len(probabilities))
    # print(sum(probabilities))

    sample_sequence = []
    sample_steps = []
    steps = 0
    bootstrap_sample = {}
    for n in range(0, sample_size):
        steps = steps + 1
        for s, p in enumerate(probabilities):
            if random.random() <= p:
                bootstrap_sample[s] = bootstrap_sample.get(s, 0) + 1
        if steps % step_size == 0:
            sample_sequence.append(copy.deepcopy(bootstrap_sample))
            sample_steps.append(steps)
            # print(bootstrap_sample)
    # print(len(bootstrap_sample), steps,sum(bootstrap_sample.values()),sum(reference_sample.values()))

    if steps % step_size > 0:
        sample_sequence.append(copy.deepcopy(bootstrap_sample))
        sample_steps.append(steps)

        # print(bootstrap_sample)
        # print(len(bootstrap_sample), steps)
    return sample_sequence, sample_steps


def generate_bootstrap_sample_incidence(reference_sample, sample_size):
    s_obs = len(reference_sample)
    f_0 = 0

    f_1 = get_singletons(reference_sample)
    f_2 = get_doubletons(reference_sample)

    if f_2 > 0:
        f_0 = ((sample_size - 1) / sample_size) * f_1 ** 2 / (2 * f_2)
    else:
        f_0 = ((sample_size - 1) / sample_size) * f_1 * (f_1 - 1) / 2
    f_0 = math.ceil(f_0)
    # print(s_obs,f_0)

    probabilities = []
    c = get_c_n_incidence(reference_sample, sample_size)
    u = sum(reference_sample.values())

    factor = factor_incidence(reference_sample, sample_size, u, c)

    for y_i in reference_sample.values():
        adapted_p = (y_i / sample_size) * (1 - factor * (1 - (y_i / sample_size) ** sample_size))
        probabilities.append(adapted_p)
    [probabilities.append((u / sample_size) * (1 - c) / f_0) for i in range(0, f_0)]

    # print(probabilities)
    # print(sum(probabilities))

    # selected_species = choice([x for x in range(0, s_obs + f_0)], sample_size, p=probabilities)
    bootstrap_sample = {}
    for n in range(0, sample_size):
        for s, p in enumerate(probabilities):
            if random.random() <= p:
                bootstrap_sample[s] = bootstrap_sample.get(s, 0) + 1

    # print(bootstrap_sample)
    # print(len(bootstrap_sample), sum(bootstrap_sample.values()))
    return bootstrap_sample


def get_bootstrap_stderr(reference_sample, sample_size, q=0, abundance=True, bootstrap_repetitions=200):
    estimates_q0 = []
    estimates_q1 = []
    estimates_q2 = []

    probabilities = []
    if abundance:
        probabilities = get_bootstrap_probabilities_abundance(reference_sample, sample_size)
    else:
        probabilities = get_bootstrap_probabilities_incidence(reference_sample, sample_size)

    for i in range(0, bootstrap_repetitions):
        sample = None
        if abundance:
            sample = generate_sample_abundance(reference_sample, sample_size, probabilities)
        else:
            sample = generate_sample_incidence(reference_sample, sample_size, probabilities)

        sample_estimate = None
        sample_estimate_q0 = chao2(sample, obs_species_count=sample_size)
        sample_estimate_q1 = estimate_shannon_entropy_abundance(sample,
                                                                sample_size) if abundance else estimate_shannon_entropy_incidence(
            sample, sample_size)
        sample_estimate_q2 = estimate_simpson_diversity_abundance(sample,
                                                                  sample_size) if abundance else estimate_simpson_diversity_incidence(
            sample, sample_size)
        estimates_q0.append(sample_estimate_q0)
        estimates_q1.append(sample_estimate_q1)
        estimates_q2.append(sample_estimate_q2)

    if q == 0:
        return statistics.stdev(estimates_q0)
    if q == 1:
        return statistics.stdev(estimates_q1)
    if q == 2:
        return statistics.stdev(estimates_q2)
    if q == -1:
        return (statistics.stdev(estimates_q0), statistics.stdev(estimates_q1), statistics.stdev(estimates_q2))


def factor_abundance(reference_sample, n, c):
    # print(reference_sample, n, c)
    # print(sum([(x_i / n) * (1 - (x_i / n))**n for x_i in reference_sample.values()]))
    # print()
    if c == 1:
        return 0
    return (1 - c) / (sum([(x_i / n) * (1 - (x_i / n)) ** n for x_i in reference_sample.values()]))


def factor_incidence(reference_sample, n, u, c):
    if c == 1:
        return 0
    return ((u / n) * (1 - c)) / (sum([(x_i / n) * (1 - (x_i / n)) ** n for x_i in reference_sample.values()]))


def get_c_n_abundance(reference_sample, n):
    f_1 = get_singletons(reference_sample)
    f_2 = get_doubletons(reference_sample)

    if f_2 > 0:
        return 1 - (f_1 / n) * (((n - 1) * f_1) / ((n - 1) * f_1 + 2 * f_2))
    else:
        return 1 - (f_1 / n) * (((n - 1) * (f_1 - 1)) / ((n - 1) * (f_1 - 1) + 2))


def get_c_n_incidence(reference_sample, n):
    u = sum(reference_sample.values())
    f_1 = get_singletons(reference_sample)
    f_2 = get_doubletons(reference_sample)
    if f_2 > 0:
        return 1 - (f_1 / u) * (((n - 1) * f_1) / ((n - 1) * f_1 + 2 * f_2))
    else:
        return 1 - (f_1 / u) * (((n - 1) * (f_1 - 1)) / ((n - 1) * (f_1 - 1) + 2))
