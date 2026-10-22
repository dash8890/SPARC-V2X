import math
from scipy.stats import poisson_binom

class LocalGreedyController:
    """Strong deployable model-based local controller."""
    def __init__(self, lambda_l: float = 0.1, lambda_b: float = 0.01):
        self.lambda_l = lambda_l
        self.lambda_b = lambda_b
        
    def select_action(self, obs, k: int, p_hats: list, b_per_worker: float) -> int | None:
        """
        p_hats: predicted timely-return probabilities for feasible helpers, sorted descending.
        """
        max_utility = -float('inf')
        best_n = k
        
        H = len(p_hats)
        if H < k:
            return None  # Explicit infeasibility, not a worker count.
            
        for n in range(k, H + 1):
            # Probability of >= k successes among n workers using Poisson-Binomial
            # p_hats is sorted, so we take the top n
            probs = p_hats[:n]
            
            # Simple approximation or exact poisson binomial tail
            # For speed in simulation, can use normal approximation if n is large,
            # but Poisson-Binomial is exact.
            # SciPy poisson_binom.sf(k-1, probs) gives P(X >= k)
            p_rec = poisson_binom.sf(k - 1, probs)
            
            cost = self.lambda_b * (n * b_per_worker)
            # lambda_l * expected latency could be added here
            
            utility = p_rec - cost

            # p_rec is at most one. Once even that optimistic utility cannot
            # beat the incumbent, every larger n is provably dominated because
            # the bandwidth cost is monotone increasing.
            if n > k and 1.0 - cost <= max_utility:
                break
            
            if utility > max_utility:
                max_utility = utility
                best_n = n
                
        return best_n
