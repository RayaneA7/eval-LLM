import evaluate
import numpy as np
from .base import BaseScoreEvaluator


class MeteorScoreEvaluator(BaseScoreEvaluator):
    fields = ["meteor"]

    def __init__(self):
        super().__init__("meteor")
        self.scorer = evaluate.load("meteor", keep_in_memory=True)

    def score(self, dataset):
        results = {}
        results.update(
            self.scorer.compute(
                predictions=dataset["predictions"], references=dataset["references"]
            )
        )
        return results
