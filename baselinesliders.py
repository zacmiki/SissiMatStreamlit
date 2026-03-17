import streamlit as st
import numpy as np
from numpy.linalg import norm
from scipy import sparse


# -------------------------------------------------------------
def baseline_arPLS(y, ratio=1e-6, lam=1e03, niter=30, full_output=False):
    """
    Adaptive Robust Penalized Least Squares (arPLS) baseline estimation.

    Args:
        y (np.ndarray): Input signal
        ratio (float): Convergence criterion
        lam (float): Smoothness parameter
        niter (int): Maximum number of iterations
        full_output (bool): Return additional information if True

    Returns:
        np.ndarray or tuple: Estimated baseline or (baseline, residual, info)
    """
    L = len(y)

    diag = np.ones(L - 2)
    D = sparse.spdiags([diag, -2 * diag, diag], [0, -1, -2], L, L - 2)

    H = lam * D.dot(D.T)  # Smoothness matrix

    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)

    crit = 1
    count = 0

    while crit > ratio:
        z = sparse.linalg.spsolve(W + H, W * y)
        d = y - z
        dn = d[d < 0]

        m = np.mean(dn)
        s = np.std(dn)

        w_new = 1 / (1 + np.exp(2 * (d - (2 * s - m)) / s))

        crit = norm(w_new - w) / norm(w)

        w = w_new
        W.setdiag(w)  # Update diagonal values

        count += 1

        if count > niter:
            st.warning("Maximum number of iterations exceeded")
            break

    if full_output:
        info = {"num_iter": count, "stop_criterion": crit}
        return z, d, info
    else:
        return z
