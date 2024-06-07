import dataclasses

from pm4py.objects.log.obj import EventLog

import src.completeness.species_estimator as species_estimator
import src.completeness.species_retrieval as species_retrieval
from functools import partial
import pm4py
import pandas as pd



def profile_log(log_path: str, name: str) -> pd.DataFrame:
    log: EventLog = pm4py.read_xes(log_path, return_legacy_log_object=True)
    estimators = \
        {
            "1-gram": species_estimator.SpeciesEstimator(partial(species_retrieval.retrieve_species_n_gram, n=1),
                                                         quantify_all=True),
            "2-gram": species_estimator.SpeciesEstimator(partial(species_retrieval.retrieve_species_n_gram, n=2),
                                                         quantify_all=True),
            "3-gram": species_estimator.SpeciesEstimator(partial(species_retrieval.retrieve_species_n_gram, n=3),
                                                         quantify_all=True),
            "4-gram": species_estimator.SpeciesEstimator(partial(species_retrieval.retrieve_species_n_gram, n=4),
                                                         quantify_all=True),
            "5-gram": species_estimator.SpeciesEstimator(partial(species_retrieval.retrieve_species_n_gram, n=5),
                                                         quantify_all=True),
            "trace_variants": species_estimator.SpeciesEstimator(
                species_retrieval.retrieve_species_trace_variant, quantify_all=True),
            "est_act_1": species_estimator.SpeciesEstimator(
                partial(species_retrieval.retrieve_timed_activity, interval_size=1), quantify_all=True),
            "est_act_5": species_estimator.SpeciesEstimator(
                partial(species_retrieval.retrieve_timed_activity, interval_size=5), quantify_all=True),
            "est_act_30": species_estimator.SpeciesEstimator(
                partial(species_retrieval.retrieve_timed_activity, interval_size=30), quantify_all=True),
            "est_act_exp": species_estimator.SpeciesEstimator(
                partial(species_retrieval.retrieve_timed_activity_exponential), quantify_all=True)
        }

    metrics_stats = {}
    print("Profiling log")
    for est_id, est in estimators.items():
        print(name, est_id)
        est.profile_log(log)
        metrics_stats[est_id] = {}

    for est_id, est in estimators.items():
        print()
        print(name, est_id)
        for metric in species_estimator.METRICS:
            metrics_stats[est_id][metric] = getattr(est, metric)[-1]
            print(metric + ": " + str(getattr(est, metric)))

        # plot_rank_abundance(est, str(name) + " - " + str(est_id))
        print("Total Abundances: " + str(est.total_number_species_abundances))
        print("Total Incidences: " + str(est.total_number_species_incidences))
        print("Degree of Aggregation: " + str(est.degree_spatial_aggregation))
        print("Observations Incidence: " + str(est.number_observations_incidence))
        print("Species Counts: ")
        [print(x, end=" ") for x in est.reference_sample_incidence.values()]
        print()
    df_stats = pd.DataFrame.from_dict(metrics_stats)
    return df_stats


