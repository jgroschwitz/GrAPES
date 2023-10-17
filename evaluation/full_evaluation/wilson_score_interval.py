from statsmodels.stats.proportion import proportion_confint


def wilson_score_interval(positive, total, alpha=0.05):
    """
    Calculates the Wilson score interval for a proportion. This is a binomial proportion confidence interval.
    :param positive: The number of positive outcomes
    :param total: The total number of outcomes
    :param alpha: The significance level (default 0.05)
    :return: A tuple containing the lower and upper bound of the confidence interval
    """
    return proportion_confint(positive, total, alpha=alpha, method="wilson")


def main():
    print(wilson_score_interval(10, 50))


if __name__ == "__main__":
    main()
