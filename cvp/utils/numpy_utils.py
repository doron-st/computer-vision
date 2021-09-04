import numpy as np


def safe_add(a, b, max_values):
    diff_a_max = max_values - a
    masked_b = b.copy()
    np.putmask(masked_b, b > diff_a_max, diff_a_max)
    return a + masked_b


def safe_subtract(a, b):
    masked_a = a.copy()
    np.putmask(masked_a, b > a, b)
    return masked_a - b