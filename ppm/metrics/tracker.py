class MetricsTracker:
    def __init__(self, targets_metrics):
        self.metrics = {
            target: {metric: [] for metric in metrics}
            for target, metrics in targets_metrics.items()
        }
        self.best_metrics = {
            target: {
                metric: float("inf") if metric == "loss" else 0.0 for metric in metrics
            }
            for target, metrics in targets_metrics.items()
        }

    def update(self, target, **kwargs):
        for metric, value in kwargs.items():
            self.metrics[target][metric].append(value)
            if metric == "loss" and value < self.best_metrics[target][metric]:
                self.best_metrics[target][metric] = value
            if metric == "acc" and value > self.best_metrics[target][metric]:
                self.best_metrics[target][metric] = value

    def latest(self):
        current = {
            f"{target}_{metric}": values[-1]
            for target, metrics in self.metrics.items()
            for metric, values in metrics.items()
        }
        best = {
            f"best_{target}_{metric}": self.best_metrics[target][metric]
            for target, metrics in self.metrics.items()
            for metric in metrics
        }
        return {**current, **best}

    def history(self):
        return self.metrics

    def list_metrics(self):
        return [
            f"{target}_{metric}"
            for target, metrics in self.metrics.items()
            for metric, values in metrics.items()
        ]
