import numpy as np


def IT21ta_ZH04_S9_Aufg2(A, A_tilde, b, b_tilde):
    A = np.array(A, dtype=np.float64)
    A_tilde = np.array(A_tilde, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    b_tilde = np.array(b_tilde, dtype=np.float64)
    cond_A = np.linalg.cond(A, np.inf)
    rel_error_A = np.linalg.norm(A - A_tilde, np.inf) / np.linalg.norm(A, np.inf)
    rel_error_b = np.linalg.norm(b - b_tilde, np.inf) / np.linalg.norm(b, np.inf)

    x = np.linalg.solve(A, b)
    x_tilde = np.linalg.solve(A_tilde, b_tilde)
    dx_max = cond_A / (1 - cond_A * rel_error_A) * (rel_error_A + rel_error_b)
    dx_obs = np.linalg.norm(x - x_tilde, np.inf) / np.linalg.norm(x, np.inf)
    return x, x_tilde, dx_max, dx_obs
