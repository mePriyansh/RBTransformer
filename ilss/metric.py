import numpy as np

class ILSS:
    """
    ILSS: Inverse Logarithmic Stability Score.

    This metric quantifies the stability and reliability of a model across multiple cross-validation folds
    by jointly considering both its mean accuracy (μ) and the variance (σ) of its performance.

    ILSS penalizes high variance and rewards consistently high-performing models by computing:

        ILSS = (sqrt(μ) / 10) * min( -log10(σ / (1 + λ * μ^2.5) + ε), 10 )

    Where:
        - μ is the mean accuracy across folds (in percent),
        - σ is the standard deviation across folds,
        - λ is a scaling factor (default: 71),
        - ε is a small constant to avoid log(0) (default: 1e-10),
        - The min(..., 10) caps the stability score at 10 to prevent runaway values at low σ.

    The final ILSS score is bounded between 0 and 10, where:
        - 0 implies poor stability (low accuracy or high variance),
        - 10 implies excellent stability (high accuracy and minimal variance).

    Higher ILSS scores indicate greater model stability and consistency across folds.
    """

    def __init__(self, lambda_: float = 71, epsilon: float = 1e-10):
        """
        Initialize the ILSS metric.

        Args:
            lambda_ (float): Scaling factor to modulate accuracy influence.
            epsilon (float): Small constant for numerical stability.
        """
        self.lambda_ = lambda_
        self.epsilon = epsilon

    def compute(self, mean_accuracy: float, std_dev: float) -> float:
        """
        Compute the ILSS value based on mean accuracy and standard deviation.

        Args:
            mean_accuracy (float): Mean validation accuracy (in percent).
            std_dev (float): Standard deviation of accuracy across folds.

        Returns:
            float: ILSS score, bounded between 0 and 10.
        """
        numerator = std_dev
        denominator = 1 + self.lambda_ * (mean_accuracy**2.5)
        log_term = np.log10((numerator / denominator) + self.epsilon)
        ilss_uncapped = -log_term
        ilss_capped = min(ilss_uncapped, 10)
        return np.sqrt(mean_accuracy) / 10 * ilss_capped


if __name__ == "__main__":
    # List of (mean_accuracy, std_dev) across n-folds
    results = [
        (99.74, 0.06),
        (98.25, 2.85),
        (94.72, 2.56),
        (97.11, 2.01),
        (98.97, 0.49),
        (84.54, 10.18),
    ]

    ilss = ILSS()

    print(f"{'Index':<6} {'Mean Acc':>10} {'Std Dev':>10} {'ILSS Score':>12}")
    print("-" * 42)
    for i, (mu, sigma) in enumerate(results, 1):
        score = ilss.compute_ilss(mean_accuracy=mu, std_dev=sigma)
        print(f"{i:<6} {mu:>10.2f} {sigma:>10.2f} {score:>12.4f}")
