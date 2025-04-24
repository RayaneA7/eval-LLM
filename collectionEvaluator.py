from .base import BaseScoreEvaluator
from .rouge import RougeScoreEvaluator
from .sacrebleu import BleuScoreEvaluator
from .meteor import MeteorScoreEvaluator
from .bertscore import BertScoreEvaluator
from .utils import get_scorers


class CollectionScoreEvaluator(BaseScoreEvaluator):
    def __init__(self, metrics_names, **kwargs):
        super().__init__("Collection")
        self.scorers = get_scorers(metrics_names, **kwargs)

    def score(self, dataset):
        scores = {}
        for scorer in self.scorers:
            scores.update(scorer.score(dataset))
        return scores

    def compute_mean(self, scores):
        mean_scores = {}
        for scorer in self.scorers:
            mean_scores.update(scorer.compute_mean(scores))
        return mean_scores
