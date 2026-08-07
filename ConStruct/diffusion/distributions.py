import torch


class DistributionNodes:
    def __init__(self, histogram):
        """ Compute the distribution of the number of nodes in the dataset, and sample from this distribution.
            historgram: dict. The keys are num_nodes, the values are counts
        """

        if type(histogram) == dict:
            max_n_nodes = max(histogram.keys())
            prob = torch.zeros(max_n_nodes + 1)
            for num_nodes, count in histogram.items():
                prob[num_nodes] = count
        else:
            prob = histogram

        self.prob = prob / prob.sum()
        self.m = torch.distributions.Categorical(probs=prob)

    def sample_n(self, n_samples, device, min_nodes=None):
        if min_nodes is None:
            distribution = self.m
        else:
            min_nodes = int(min_nodes)
            probabilities = self.prob.clone()
            probabilities[:min_nodes] = 0
            if probabilities.sum() <= 0:
                raise ValueError(
                    f"The node-count distribution has no support for n >= {min_nodes}."
                )
            distribution = torch.distributions.Categorical(
                probs=probabilities / probabilities.sum()
            )
        idx = distribution.sample((n_samples,))
        return idx.to(device)

    def log_prob(self, batch_n_nodes):
        assert len(batch_n_nodes.size()) == 1
        p = self.prob.to(batch_n_nodes.device)

        probas = p[batch_n_nodes]
        log_p = torch.log(probas + 1e-30)
        return log_p
