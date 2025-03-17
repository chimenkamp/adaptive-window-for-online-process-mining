import dataclasses
from functools import partial
from typing import Callable, List

import pandas as pd
from special4pm.estimation import SpeciesEstimator
from special4pm.estimation.metrics import coverage
from special4pm.species import (
    retrieve_species_n_gram,
    retrieve_species_trace_variant,
    retrieve_timed_activity,
    retrieve_timed_activity_exponential
)

SpeciesDef = Callable[[pd.DataFrame], pd.DataFrame]
SpeciesRetrivalDef = tuple[str, SpeciesDef]


@dataclasses.dataclass
class SpeciesRetrievalRepository:
    retrieve_species_1_gram: SpeciesRetrivalDef = ("1-gram", partial(retrieve_species_n_gram, n=1))
    retrieve_species_2_gram: SpeciesRetrivalDef = ("2-gram", partial(retrieve_species_n_gram, n=2))
    retrieve_species_3_gram: SpeciesRetrivalDef = ("3-gram", partial(retrieve_species_n_gram, n=3))
    retrieve_species_4_gram: SpeciesRetrivalDef = ("4-gram", partial(retrieve_species_n_gram, n=4))
    retrieve_species_5_gram: SpeciesRetrivalDef = ("5-gram", partial(retrieve_species_n_gram, n=5))

    trace_variants: SpeciesRetrivalDef = ("trace_variant", retrieve_species_trace_variant)

    retrieve_timed_activity: SpeciesRetrivalDef = ("timed_activity", partial(retrieve_timed_activity, interval_size=2))
    retrieve_timed_activity_exponential: SpeciesRetrivalDef \
        = ("timed_activity_exponential", retrieve_timed_activity_exponential)

    def get_all(self) -> List[SpeciesRetrivalDef]:
        """
        Retrieve all species retrieval definitions.

        :return: List of all SpeciesRetrivalDef objects from the class instance.
        """
        return [getattr(self, field.name) for field in dataclasses.fields(self)]
