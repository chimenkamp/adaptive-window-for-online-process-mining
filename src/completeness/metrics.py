import math

import scipy
from gmpy2 import gmpy2


def get_incidence_count(obs_species_counts, i):
    return list(obs_species_counts.values()).count(i)


def get_singletons(obs_species_counts):
    return list(obs_species_counts.values()).count(1)


def get_doubletons(obs_species_counts):
    return list(obs_species_counts.values()).count(2)


def hill_q(q, obs_species_counts, no_species):
    # sample-based species richness
    if q == 0:
        return observed_species(obs_species_counts)
    # sample-based shannon diversity
    if q == 1:
        return entropy_exp(obs_species_counts, no_species)
    # sample-based simpson diversity
    if q == 2:
        return simpson_diversity(obs_species_counts, no_species)


def observed_species(obs_species_counts):
    return len(obs_species_counts.keys())


def entropy_exp(obs_species_counts, no_species):
    # total = sum([obs_species_counts.values])
    return math.exp(-1 * sum(
        [x / no_species * math.log(x / no_species) for x in obs_species_counts.values()]))


def simpson_diversity(obs_species_counts, number_species):
    # total = sum([obs_species_counts])
    a = sum([(x / number_species) ** 2 for x in obs_species_counts.values()])
    return a ** (1 / (1 - 2)) if a > 0 else 1  # TODO is return 1 reasonable?


def hill_q_asymptotic(q, obs_species_counts, sample_size, abundance=True, Q1=None, Q2=None, obs_species_count=None):
    # if get_singletons(obs_species_counts) == 0 and get_doubletons(obs_species_counts) == 0:
    #    return hill_q(q, obs_species_counts, sample_size)

    # asymptotic species richness
    # for species richness, there is no differentiation between abundance and incidence
    if q == 0:
        return chao2(obs_species_counts, Q1, Q2, obs_species_count)
    # asymptotic shannon entropy
    if q == 1:
        if abundance:
            return math.exp(estimate_shannon_entropy_abundance(obs_species_counts, sample_size, Q1, Q2))
        # incidence
        else:
            return math.exp(estimate_shannon_entropy_incidence(obs_species_counts, sample_size, Q1, Q2))
    # asymptotic simpson diversity
    if q == 2:
        if abundance:
            return estimate_simpson_diversity_abundance(obs_species_counts, sample_size)
        else:  # incidence
            return estimate_simpson_diversity_incidence(obs_species_counts, sample_size)


def chao2(obs_species_counts, Q1=None, Q2=None, obs_species_count=None):
    if Q1 is None:
        Q1 = get_singletons(obs_species_counts)
    if Q2 is None:
        Q2 = get_doubletons(obs_species_counts)
    if obs_species_count is None:
        obs_species_count = observed_species(obs_species_counts)

    if Q2 != 0:
        return obs_species_count + Q1 ** 2 / (2 * Q2)
    else:
        return obs_species_count + Q1 * (Q1 - 1) / 2


def chao2_corrected(obs_species_counts, Q1=None, Q2=None, obs_species_count=None):
    if Q1 is None:
        Q1 = get_singletons(obs_species_counts)
    if Q2 is None:
        Q2 = get_doubletons(obs_species_counts)
    if obs_species_count is None:
        obs_species_count = observed_species(obs_species_counts)

    if Q2 != 0:
        return ((obs_species_count - 1) / obs_species_count) * obs_species_count + Q1 ** 2 / (2 * Q2)
    else:
        return ((obs_species_count - 1) / obs_species_count) * obs_species_count + Q1 * (Q1 - 1) / 2


def estimate_shannon_entropy_abundance(obs_species_counts, sample_size, f_1=None, f_2=None):
    raise ValueError('A very specific bad thing happened.')

    if f_1 is None:
        f_1 = get_singletons(obs_species_counts)
    if f_2 is None:
        f_2 = get_doubletons(obs_species_counts)

    first_sum = 0

    TEST = False
    # This code part runs into division by infinity :(
    if TEST:
        for k in range(1, sample_size):
            norm_factor = 1.0 / k
            s = 0
            for x_i in obs_species_counts.values():
                if x_i <= sample_size - k:
                    s = s + (x_i / sample_size) * ((gmpy2.comb(sample_size - x_i, k) / gmpy2.comb(sample_size - 1, k)))
            s = s * norm_factor
            first_sum = first_sum + s

    else:
        for x_i in obs_species_counts.values():
            if x_i <= sample_size - 1:
                norm_factor = x_i / sample_size
                first_sum = first_sum + norm_factor * sum([1 / k for k in range(x_i, sample_size)])

    a = 0
    if f_2 > 0:
        a = (2 * f_2) / ((sample_size - 1) * f_1 + 2 * f_2)
    if f_2 == 0 and f_1 > 0:
        a = 2 / ((sample_size - 1) * (f_1 - 1) + 2)
    else:
        a = 1

    second_sum = 0
    if a == 1:
        return first_sum  # TODO does this make sense?

    second_sum = (f_1 / sample_size) * ((1 - a) ** (-sample_size + 1)) * (
            -math.log(a) - sum([(1 / r) * ((1 - a) ** r) for r in range(1, sample_size)]))

    return first_sum + second_sum


def estimate_shannon_entropy_incidence(obs_species_counts, sample_size, f_1=None, f_2=None):
    # term h_o is structurally equivalent to abundance based entropy estimation, see eq H7 in appendix H of Hill number paper

    u = sum(obs_species_counts.values())
    h_o = estimate_shannon_entropy_abundance(obs_species_counts, sample_size, f_1, f_2)
    # print("SAMPLE ENTROPY: " +str(entropy_exp(obs_species_counts, sample_size)))
    # print("SAMPLE: " + str(obs_species_counts.items()))
    # print("SAMPLE SIZE: " + str(sample_size))
    # print(sample_size, u, h_o, (sample_size / u), math.log(u / sample_size))
    return (sample_size / u) * h_o + math.log(u / sample_size)


def estimate_simpson_diversity_abundance(obs_species_counts, sample_size):
    denom = 0
    for x_i in obs_species_counts.values():
        if x_i >= 2:
            denom = denom + (x_i * (x_i - 1))
    if denom == 0:
        return 0
    return (sample_size * (sample_size - 1)) / denom

    # ensure s=0 returns 0
    # s = 0
    # for x_i in obs_species_counts.values():
    #    if x_i >= 2:
    #        s = s + (x_i ** 2) / (sample_size ** 2)
    # return s ** (1 / (1 - 2))


# T= number of sampling units = sample_size
# U= total number of incidences
def estimate_simpson_diversity_incidence(obs_species_counts, sample_size):
    u = sum(obs_species_counts.values())
    s = 0

    nom = ((1 - (1 / sample_size)) * u) ** 2

    for y_i in obs_species_counts.values():
        if y_i > 1:
            #    s = s + (sample_size ** 2 * y_i ** 2) / (u ** 2 * sample_size ** 2)
            s = s + (y_i * (y_i - 1))
    if s == 0:
        return 0
    # return s ** (1 / (1 - 2))
    return nom / s


def completeness(obs_species_counts, obs_species_count=None, s_P=None):
    if s_P is None:
        s_P = chao2(obs_species_counts)
    if obs_species_count is None:
        obs_species_count = observed_species(obs_species_counts)
    if s_P == 0:
        return 0
    return obs_species_count / s_P


def coverage(observation_count, obs_species_counts, Q1=None, Q2=None, Y=None):
    if Q1 is None:
        Q1 = get_singletons(obs_species_counts)
    if Q2 is None:
        Q2 = get_doubletons(obs_species_counts)

    # get all incidences
    if Y is None:
        Y = sum(list(obs_species_counts.values()))

    if Q2 == 0 and observation_count == 1:
        return 0
    if Q1 == 0 and Q2 == 0:
        return 1

    return 1 - Q1 / Y * (((observation_count - 1) * Q1) / ((observation_count - 1) * Q1 + 2 * Q2))


def sampling_effort_abundance(l, obs_species_counts, no_observations, comp=None, Q1=None, Q2=None):
    if comp is None:
        comp = completeness(obs_species_counts)
    if Q1 is None:
        Q1 = get_singletons(obs_species_counts)
    if Q2 is None:
        Q2 = get_doubletons(obs_species_counts)

    if l <= comp:
        return 0
    if Q2 == 0:
        return 0

    # chao2
    obs_species_count = observed_species(obs_species_counts)

    chao2 = 0
    if Q2 != 0:
        chao2 = Q1 ** 2 / (2 * Q2)
    else:
        chao2 = Q1 * (Q1 - 1) / 2

    return ((no_observations * Q1) / (2 * Q2)) * math.log(chao2 / ((1 - l) * (chao2 + obs_species_count)))


def sampling_effort_incidence(l, obs_species_counts, no_observations_incidence, comp=None, Q1=None, Q2=None):
    if comp is None:
        comp = completeness(obs_species_counts)
    if Q1 is None:
        Q1 = get_singletons(obs_species_counts)
    if Q2 is None:
        Q2 = get_doubletons(obs_species_counts)
    if l <= comp or no_observations_incidence < 2:
        return 0
    if Q2 == 0:
        return 0

    obs_species_count = observed_species(obs_species_counts)
    # for small sample sizes, correction term is introduced, otherwise math error
    chao2 = 0
    if Q2 != 0:
        chao2 = obs_species_count + (1 - 1 / no_observations_incidence) * Q1 ** 2 / (2 * Q2)
    else:
        chao2 = obs_species_count + (1 - 1 / no_observations_incidence) * Q1 * (Q1 - 1) / 2

    # TODO double check if this makes sense

    nom1 = (no_observations_incidence / (no_observations_incidence - 1))
    nom2 = ((2 * Q2) / (Q1 ** 2))
    nom3 = (l * chao2 - obs_species_count)
    nom = (math.log(1 - nom1 * nom2 * nom3))
    denom = (math.log(1 - ((2 * Q2) / ((no_observations_incidence - 1) * Q1 + 2 * Q2))))

    return nom / denom
