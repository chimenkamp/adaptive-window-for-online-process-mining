from typing import List, Tuple, Dict, Any
from pm4py.objects.conversion.dfg import converter as dfg_mining
import pandas as pd
import pm4py
from pm4py.algo.evaluation.generalization import algorithm as generalization_evaluator
from pm4py.algo.evaluation.simplicity import algorithm as simplicity_evaluator
from src.utils.methods import calculate_conformance_metrics, ConformanceMetrics

Event = Tuple[str, str, str]  # (ci, ai, ti)


class OnlineHM:
    def __init__(self, max_qa: int, max_qc: int, max_qr: int):
        self.max_qa = max_qa
        self.max_qc = max_qc
        self.max_qr = max_qr
        self.qa: List[Tuple[str, int]] = []
        self.qc: Dict[str, str] = {}
        self.qr: List[Tuple[str, str, int]] = []

    def observe(self, e: Event, index: int) -> None:
        if not self.analyze(e):
            return

        ci, ai, ti = e

        # Update QA
        if not any(a == ai for a, w in self.qa):
            if len(self.qa) == self.max_qa:
                self.qa.pop()  # Remove last entry if QA is full
            self.qa.insert(0, (ai, 0))
        else:
            w = self.get(self.qa, ai)
            self.qa.insert(0, (ai, w))
        self.qa = self.update_weights(self.qa, 1)

        # Update QC and QR
        if ci in self.qc:
            a = self.qc.pop(ci)
            if not any(as_ == a and af == ai for as_, af, u in self.qr):
                if len(self.qr) == self.max_qr:
                    self.qr.pop()  # Remove last entry if QR is full
                self.qr.insert(0, (a, ai, 0))
            else:
                u = self.get_relation(self.qr, (a, ai))
                self.qr.insert(0, (a, ai, u))
            self.qr = self.update_relation_weights(self.qr, 1)
        else:
            if len(self.qc) == self.max_qc:
                oldest_key = next(iter(self.qc))
                self.qc.pop(oldest_key)
            self.qc[ci] = ai

        # Generate model
        if self.qa and self.qr:
            self.generate_model(self.qa, self.qr, index)

    def analyze(self, e: Event) -> bool:
        # Simulate analyzing if the event should be used
        return True

    def get(self, queue: List[Tuple[str, int]], key: str) -> int:
        for item in queue:
            if item[0] == key:
                queue.remove(item)
                return item[1]
        return 0

    def get_relation(self, queue: List[Tuple[str, str, int]], key: Tuple[str, str]) -> int:
        for item in queue:
            if (item[0], item[1]) == key:
                queue.remove(item)
                return item[2]
        return 0

    def update_weights(self, queue: List[Tuple[str, int]], increment: int) -> List[Tuple[str, int]]:
        if queue:
            first_item = queue[0]
            updated_item = (first_item[0], first_item[1] + increment)
            queue[0] = updated_item
        return queue

    def update_relation_weights(self, queue: List[Tuple[str, str, int]], increment: int) -> List[Tuple[str, str, int]]:
        if queue:
            first_item = queue[0]
            updated_item = (first_item[0], first_item[1], first_item[2] + increment)
            queue[0] = updated_item
        return queue

    def generate_model(self, qa: List[Tuple[str, int]], qr: List[Tuple[str, str, int]], i: int):
        # print("Generating model with QA:", qa)
        # print("Generating model with QR:", qr)
        pass

    def get_dfg(self) -> Dict[Tuple[str, str], int]:
        dfg_activities: dict[tuple[str, str], int] = {}

        for (a, d, w) in self.qr:
            dfg_activities[(a, d)] = int(w)

        start: dict[tuple[str, str], int] = {(self.qr[0][0], self.qr[0][1]): int(self.qr[0][2])}
        end: dict[tuple[str, str], int] = {(self.qr[-1][0], self.qr[-1][1]): int(self.qr[-1][2])}

        return dfg_activities

    def visualize(self):
        pm4py.view_dfg(self.get_dfg(), None, None)


BPI_C_2012: pd.DataFrame = pd.read_feather("/Users/christianimenkamp/Documents/Data-Repository/Community/Road-Traffic-Fine-Management-Process/Road_Traffic_Fine_Management_Process.feather")[:15000]






online_hm = OnlineHM(max_qa=1000, max_qc=1000, max_qr=1000)

for i, row in BPI_C_2012.iterrows():
    online_hm.observe((row['case:concept:name'], row['concept:name'], row['time:timestamp']), i)

# online_hm.visualize()
net, im, fm = dfg_mining.apply(online_hm.get_dfg())
pm4py.view_petri_net(net, im, fm)


fitness: dict[str, float] = pm4py.fitness_token_based_replay(BPI_C_2012, net, im, fm)
precision: float = pm4py.precision_token_based_replay(BPI_C_2012, net, im, fm)
gen: float = generalization_evaluator.apply(BPI_C_2012, net, im, fm)
simp: float = simplicity_evaluator.apply(net)


cm = ConformanceMetrics(fitness["log_fitness"], precision, gen, simp)
print(cm)
