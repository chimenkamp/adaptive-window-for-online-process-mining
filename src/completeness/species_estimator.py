from typing import List, Dict, Any

import pandas as pd
import pm4py
from pm4py.objects.log.obj import EventLog, Trace

import src.completeness.bootstrap as bootstrap
from src.completeness.metrics import get_singletons, get_doubletons, chao2, completeness, coverage, observed_species, hill_q, \
    sampling_effort_abundance, sampling_effort_incidence, hill_q_asymptotic
from src.utils.methods import group_traces_by_case_id

METRICS = ["observation_ids", "Q1_abundance", "Q1_incidence", "Q2_abundance", "Q2_incidence",
           "total_species_count_abundance", "total_species_count_incidence", "dag",
           "observed_species_count", "chao2_abundance", "chao2_abundance_stddev", "chao2_incidence",
           "chao2_incidence_stddev",
           "D1_sample_abundance", "D1_sample_incidence", "D1_estimated_abundance", "D1_abundance_stddev",
           "D1_estimated_incidence", "D1_incidence_stddev",
           "D2_sample_abundance", "D2_sample_incidence", "D2_estimated_abundance", "D2_abundance_stddev",
           "D2_estimated_incidence", "D2_incidence_stddev",
           "completeness_abundance", "completeness_incidence",
           "coverage_abundance", "coverage_incidence",
           "l80_abundance", "l90_abundance", "l95_abundance", "l99_abundance",
           "l80_incidence", "l90_incidence", "l95_incidence", "l99_incidence"]

METRICS_q1 = ["observation_ids", "Q1_abundance", "Q1_incidence", "Q2_abundance", "Q2_incidence",
              "total_species_count_abundance", "total_species_count_incidence", "dag",
              "observed_species_count", "chao2_abundance", "chao2_abundance_stddev", "chao2_incidence",
              "chao2_abundance_stddev",
              "completeness_abundance", "completeness_incidence",
              "coverage_abundance", "coverage_incidence",
              "l80_abundance", "l90_abundance", "l95_abundance", "l99_abundance",
              "l80_incidence", "l90_incidence", "l95_incidence", "l99_incidence"]

METRICS_reduced = ["observation_ids", "Q1_abundance", "Q1_incidence", "Q2_abundance", "Q2_incidence",
                   "total_species_count_abundance", "total_species_count_incidence", "dag",
                   "observed_species_count", "chao2_abundance", "chao2_abundance_stddev",
                   # "chao2_incidence", "chao2_incidence_stddev",
                   "D1_sample_abundance", "D1_sample_incidence", "D1_estimated_abundance", "D1_abundance_stddev",
                   # "D1_estimated_incidence", "D1_incidence_stddev",
                   "D2_sample_abundance", "D2_sample_incidence", "D2_estimated_abundance", "D2_abundance_stddev",
                   # "D2_estimated_incidence", "D2_incidence_stddev",
                   "completeness_abundance",  # , "completeness_incidence",
                   "coverage_abundance"]  # "coverage_incidence"]


# "l80_abundance", "l90_abundance", "l95_abundance", "l99_abundance",
# "l80_incidence", "l90_incidence", "l95_incidence", "l99_incidence"]

class SpeciesEstimator:

    def __init__(self, species_retrieval_function, step_size=1, quantify_all=False, bs=False, hill="all"):
        self.quantify_all = quantify_all
        self.step_size = step_size
        self.hill = hill
        self.bs = bs

        self.observation_ids = []
        self.observation_ids_abundance = []

        self.species_retrieval = species_retrieval_function
        # reference samples
        self.reference_sample_abundance = {}
        self.reference_sample_incidence = {}

        # number of observations
        self.number_observations_abundance = 0
        self.number_observations_incidence = 0

        # number of observed species
        self.total_number_species_abundances = 0
        self.total_number_species_incidences = 0
        self.total_species_count_abundance = []
        self.total_species_count_incidence = []
        self.dag = []

        # degree of disagreement between abundace and incidence-based data
        self.degree_spatial_aggregation = 0

        # sample_properties
        # singletons
        self.Q1_abundance = []
        self.Q1_incidence = []
        # doubletons
        self.Q2_abundance = []
        self.Q2_incidence = []

        # diversity profile
        # q0
        # TODO make naming of variables of both profiles consistent
        self.observed_species_count = []

        self.chao2_abundance = []
        self.chao2_abundance_stddev = []

        self.chao2_incidence = []
        self.chao2_incidence_stddev = []

        # q1
        self.D1_sample_abundance = []
        self.D1_sample_incidence = []

        self.D1_estimated_abundance = []
        self.D1_abundance_stddev = []

        self.D1_estimated_incidence = []
        self.D1_incidence_stddev = []

        # q2
        self.D2_sample_abundance = []
        self.D2_sample_incidence = []

        self.D2_estimated_abundance = []
        self.D2_abundance_stddev = []

        self.D2_estimated_incidence = []
        self.D2_incidence_stddev = []

        # completeness profile
        # q0
        self.completeness_abundance = []
        self.completeness_incidence = []
        # q1
        self.coverage_abundance = []
        self.coverage_incidence = []
        # q2 is ignored

        # extrapolation of sampling effort
        self.l80_abundance = []
        self.l90_abundance = []
        self.l95_abundance = []
        self.l99_abundance = []

        self.l80_incidence = []
        self.l90_incidence = []
        self.l95_incidence = []
        self.l99_incidence = []

    def profile_log(self, log: List[Dict[str, Any]]):
        tr: list[dict[str, Any]]
        for tr in group_traces_by_case_id(log):
            self.add_observation(tr)
        self.profile()
        if self.bs:
            print("Calculating Bootstrap Standard Errors")
            self.update_stderrors()

    def add_observation(self, trace: list[dict[str, Any]]):

        # retrieve species from current observation
        trace_retrieved_species_abundance = self.species_retrieval(trace)
        trace_retrieved_species_incidence = set(trace_retrieved_species_abundance)

        self.number_observations_abundance = self.number_observations_abundance + len(trace_retrieved_species_abundance)
        self.number_observations_incidence = self.number_observations_incidence + 1

        # update number of species observed so far
        self.total_number_species_abundances = self.total_number_species_abundances + len(
            trace_retrieved_species_abundance)
        self.total_number_species_incidences = self.total_number_species_incidences + len(
            trace_retrieved_species_incidence)
        self.degree_spatial_aggregation = 1 - self.total_number_species_incidences / self.total_number_species_abundances

        # update abundance and incidence counts
        for s in trace_retrieved_species_abundance:
            self.reference_sample_abundance[s] = self.reference_sample_abundance.get(s, 0) + 1

        for s in trace_retrieved_species_incidence:
            self.reference_sample_incidence[s] = self.reference_sample_incidence.get(s, 0) + 1

        if self.quantify_all:
            return
        else:
            if self.number_observations_incidence % self.step_size == 0:
                self.profile()

    # TODO utilize same sample for each repetition!
    def update_stderrors(self):
        values_a = bootstrap.get_bootstrap_stderr(self.reference_sample_abundance,
                                                  self.number_observations_abundance, q=-1,
                                                  abundance=True, bootstrap_repetitions=100)
        values_i = bootstrap.get_bootstrap_stderr(self.reference_sample_abundance,
                                                  self.number_observations_abundance, q=-1,
                                                  abundance=False, bootstrap_repetitions=100)
        self.chao2_abundance_stddev.append(values_a[0])
        self.D1_abundance_stddev.append(values_a[1])
        self.D2_abundance_stddev.append(values_a[2])

        self.chao2_incidence_stddev.append(values_i[0])
        self.D1_incidence_stddev.append(values_i[1])
        self.D2_incidence_stddev.append(values_i[2])
        #
        # self.chao2_abundance_stddev.append( bootstrap.get_bootstrap_stderr(self.reference_sample_abundance,
        #                                                              self.number_observations_abundance, q=0,
        #                                                              abundance=True, bootstrap_repetitions=100))
        # self.chao2_incidence_stddev.append(bootstrap.get_bootstrap_stderr(self.reference_sample_incidence,
        #                                                              self.number_observations_incidence, q=0,
        #                                                              abundance=False, bootstrap_repetitions=100))
        #
        # self.D1_abundance_stddev.append(bootstrap.get_bootstrap_stderr(self.reference_sample_abundance,
        #                                                           self.number_observations_abundance, q=1,
        #                                                           abundance=True, bootstrap_repetitions=100))
        # self.D1_incidence_stddev.append(bootstrap.get_bootstrap_stderr(self.reference_sample_incidence,
        #                                                           self.number_observations_incidence, q=1,
        #                                                           abundance=False, bootstrap_repetitions=100))
        #
        # self.D2_abundance_stddev.append(bootstrap.get_bootstrap_stderr(self.reference_sample_abundance,
        #                                                           self.number_observations_abundance, q=1,
        #                                                           abundance=True, bootstrap_repetitions=100))
        # self.D2_incidence_stddev.append(bootstrap.get_bootstrap_stderr(self.reference_sample_incidence,
        #                                                           self.number_observations_incidence, q=1,
        #                                                           abundance=False, bootstrap_repetitions=100))

    def profile(self):
        self.chao2_abundance_stddev.append(-1)
        self.chao2_incidence_stddev.append(-1)
        self.D1_abundance_stddev.append(-1)
        self.D1_incidence_stddev.append(-1)
        self.D2_abundance_stddev.append(-1)
        self.D2_incidence_stddev.append(-1)

        self.observation_ids.append(self.number_observations_incidence)
        self.observation_ids_abundance.append(self.number_observations_abundance)
        # updated abundance-based metrics
        # abundance stats are calculated after each trace

        self.total_species_count_abundance.append(self.total_number_species_abundances)
        self.total_species_count_incidence.append(self.total_number_species_incidences)
        self.dag.append(1 - self.total_number_species_incidences / self.total_number_species_abundances)

        self.Q1_abundance.append(get_singletons(self.reference_sample_abundance))
        self.Q2_abundance.append(get_doubletons(self.reference_sample_abundance))

        self.observed_species_count.append(observed_species(self.reference_sample_abundance))
        self.chao2_abundance.append(
            chao2(self.reference_sample_abundance, Q1=self.Q1_abundance[-1], Q2=self.Q2_abundance[-1],
                  obs_species_count=self.observed_species_count[-1]))

        # if self.hill == "q1" or self.hill == "all":
        #     self.D1_sample_abundance.append(
        #         hill_q(1, self.reference_sample_abundance, self.total_number_species_abundances))
        #     self.D1_estimated_abundance.append(
        #         hill_q_asymptotic(1, self.reference_sample_abundance, self.number_observations_abundance,
        #                           Q1=self.Q1_abundance[-1], Q2=self.Q2_abundance[-1],
        #                           obs_species_count=self.observed_species_count[-1]))
        # if self.hill == "q2" or self.hill == "all":
        #     self.D2_sample_abundance.append(
        #         hill_q(2, self.reference_sample_abundance, self.total_number_species_abundances))
        #     self.D2_estimated_abundance.append(
        #         hill_q_asymptotic(2, self.reference_sample_abundance, self.number_observations_abundance,
        #                           Q1=self.Q1_abundance[-1], Q2=self.Q2_abundance[-1],
        #                           obs_species_count=self.observed_species_count[-1]))

        self.completeness_abundance.append(
            completeness(self.reference_sample_abundance, obs_species_count=self.observed_species_count[-1],
                         s_P=self.chao2_abundance[-1]))
        self.coverage_abundance.append(
            coverage(self.number_observations_abundance, self.reference_sample_abundance, Q1=self.Q1_abundance[-1],
                     Q2=self.Q2_abundance[-1], Y=self.total_number_species_abundances))
        self.l80_abundance.append(
            sampling_effort_abundance(.80, self.reference_sample_abundance, self.number_observations_abundance,
                                      comp=self.completeness_abundance[-1], Q1=self.Q1_abundance[-1],
                                      Q2=self.Q2_abundance[-1]))
        self.l90_abundance.append(
            sampling_effort_abundance(.90, self.reference_sample_abundance, self.number_observations_abundance,
                                      comp=self.completeness_abundance[-1], Q1=self.Q1_abundance[-1],
                                      Q2=self.Q2_abundance[-1]))
        self.l95_abundance.append(
            sampling_effort_abundance(.95, self.reference_sample_abundance, self.number_observations_abundance,
                                      comp=self.completeness_abundance[-1], Q1=self.Q1_abundance[-1],
                                      Q2=self.Q2_abundance[-1]))
        self.l99_abundance.append(
            sampling_effort_abundance(.99, self.reference_sample_abundance, self.number_observations_abundance,
                                      comp=self.completeness_abundance[-1], Q1=self.Q1_abundance[-1],
                                      Q2=self.Q2_abundance[-1]))

        # update incidence-based metrics
        self.Q1_incidence.append(get_singletons(self.reference_sample_incidence))
        self.Q2_incidence.append(get_doubletons(self.reference_sample_incidence))

        self.chao2_incidence.append(
            chao2(self.reference_sample_incidence, Q1=self.Q1_incidence[-1], Q2=self.Q2_incidence[-1],
                  obs_species_count=self.observed_species_count[-1]))

        # if self.hill == "q1" or self.hill == "all":
        #     self.D1_sample_incidence.append(
        #         hill_q(1, self.reference_sample_incidence, self.total_number_species_incidences))
        #     self.D1_estimated_incidence.append(
        #         hill_q_asymptotic(1, self.reference_sample_incidence, self.number_observations_incidence,
        #                           abundance=False,
        #                           Q1=self.Q1_incidence[-1], Q2=self.Q2_incidence[-1],
        #                           obs_species_count=self.observed_species_count[-1]))
        # if self.hill == "q2" or self.hill == "all":
        #     self.D2_sample_incidence.append(
        #         hill_q(2, self.reference_sample_incidence, self.total_number_species_incidences))
        #     self.D2_estimated_incidence.append(
        #         hill_q_asymptotic(2, self.reference_sample_incidence, self.number_observations_incidence,
        #                           abundance=False,
        #                           Q1=self.Q1_incidence[-1], Q2=self.Q2_incidence[-1],
        #                           obs_species_count=self.observed_species_count[-1]))

        self.completeness_incidence.append(
            completeness(self.reference_sample_incidence, obs_species_count=self.observed_species_count[-1],
                         s_P=self.chao2_incidence[-1]))
        self.coverage_incidence.append(
            coverage(self.number_observations_incidence, self.reference_sample_incidence, Q1=self.Q1_incidence[-1],
                     Q2=self.Q2_incidence[-1], Y=self.total_number_species_incidences))
        self.l80_incidence.append(
            sampling_effort_incidence(.80, self.reference_sample_incidence, self.number_observations_incidence,
                                      comp=self.completeness_incidence[-1], Q1=self.Q1_incidence[-1],
                                      Q2=self.Q2_incidence[-1]))
        self.l90_incidence.append(
            sampling_effort_incidence(.90, self.reference_sample_incidence, self.number_observations_incidence,
                                      comp=self.completeness_incidence[-1], Q1=self.Q1_incidence[-1],
                                      Q2=self.Q2_incidence[-1]))
        self.l95_incidence.append(
            sampling_effort_incidence(.95, self.reference_sample_incidence, self.number_observations_incidence,
                                      comp=self.completeness_incidence[-1], Q1=self.Q1_incidence[-1],
                                      Q2=self.Q2_incidence[-1]))
        self.l99_incidence.append(
            sampling_effort_incidence(.99, self.reference_sample_incidence, self.number_observations_incidence,
                                      comp=self.completeness_incidence[-1], Q1=self.Q1_incidence[-1],
                                      Q2=self.Q2_incidence[-1]))
