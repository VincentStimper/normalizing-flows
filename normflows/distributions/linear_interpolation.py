class LinearInterpolation:
    """
    Linear interpolation of two distributions in the log space
    """

    def __init__(self, dist1, dist2, alpha):
        """Constructor

        Interpolation parameter alpha:

        ```
        log_p = alpha * log_p_1 + (1 - alpha) * log_p_2
        ```

        Args:
          dist1: First distribution
          dist2: Second distribution
          alpha: Interpolation parameter
        """
        self.alpha = alpha
        self.dist1 = dist1
        self.dist2 = dist2

    def log_prob(self, z):
        return self.alpha * self.dist1.log_prob(z) + (
            1 - self.alpha
        ) * self.dist2.log_prob(z)
