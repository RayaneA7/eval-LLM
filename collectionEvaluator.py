from .base import BaseScoreEvaluator
from .rouge import RougeScoreEvaluator
from .sacrebleu import BleuScoreEvaluator
from .meteor import MeteorScoreEvaluator
from .bertscore import BertScoreEvaluator


class CollectionScoreEvaluator(BaseScoreEvaluator):
    def __init__(self, metrics_names, **kwargs):
        super().__init__("Collection")
        self.scorers = self.get_scorers(metrics_names, **kwargs)

    def get_scorers(self, metrics_names, **kwargs):
        scorers = []
        for metric_name in metrics_names:
            if metric_name == "rouge":
                scorers.append(RougeScoreEvaluator(**kwargs))
            elif metric_name == "bleu":
                scorers.append(BleuScoreEvaluator(**kwargs))
            elif metric_name == "meteor":
                scorers.append(MeteorScoreEvaluator(**kwargs))
            elif metric_name == "bertscore":
                scorers.append(BertScoreEvaluator(**kwargs))
            else:
                raise ValueError(f"Unknown metric: {metric_name}")
        return scorers

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
