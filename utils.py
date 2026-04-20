import numpy as np

def feature_engineering(p1970, p2020):
    change = p2020 - p1970
    growth_ratio = p2020 / (p1970 + 1)
    log_change = np.log1p(p2020) - np.log1p(p1970)

    return np.array([[p1970, p2020, change, growth_ratio, log_change]])
