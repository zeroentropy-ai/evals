import numpy as np
from numpy.typing import NDArray

# Type alias for numpy float arrays
npf = NDArray[np.float64]


def elos_loss(w: npf, elos: npf) -> float:
    N = len(elos)
    elos_col = elos.reshape(-1, 1)
    elos_row = elos.reshape(1, -1)

    # Stable computation of log(exp(elos_i) + exp(elos_j))
    max_elos = np.maximum(elos_col, elos_row)
    log_pairwise_sums = max_elos + np.log(
        np.exp(elos_col - max_elos) + np.exp(elos_row - max_elos)
    )

    # Calculate elos_i - log(exp(elos_i) + exp(elos_j))
    log_diff = np.broadcast_to(elos_col, (N, N)) - log_pairwise_sums

    # We want to maximize the loglikelihood of the observed w with respect to elos
    loglikelihood = float(np.sum(w * log_diff))

    # Return the loss that we're trying to minimize
    return -loglikelihood


def calculate_elos(
    w: npf,
    *,
    # How close we must be to the log-likelihood loss
    epsilon: float = 1e-4,
    # If you have ELOs calculated from a similar W, then it will converge faster by initializing to the same ELOs
    initial_elos: npf | None = None,
    # Max iters before giving up
    max_iters: int = 1000,
) -> tuple[npf, list[float]]:
    # https://hackmd.io/@-Gjw1zWMSH6lMPRlziQFEw/B15B4Rsleg

    N = len(w)
    elos = initial_elos.copy() if initial_elos is not None else np.zeros(N)

    losses: list[float] = []
    for _iter in range(max_iters):
        # Create all pairwise differences elo_j - elo_i in a matrix
        # outer(ones, elos) - outer(elos, ones)
        D: npf = elos.reshape(1, N) - elos.reshape(N, 1)  # Shape: (N, N)
        # Calculate sigmoid matrix
        S: npf = 1.0 / (1.0 + np.exp(-D))  # S[i,j] = sigmoid(elo_j - elo_i)

        # Calculate the update terms
        numerator: npf = np.sum(w * S, axis=1)  # Shape: (N,)
        denominator: npf = np.sum(w.T * S.T, axis=1)  # Shape: (N,)
        # Apply update rule, using decreasing learning rate.
        learning_rate = float((1.0 + _iter) ** (-0.125))
        elos += (np.log(numerator) - np.log(denominator)) * learning_rate
        elos -= np.mean(elos)

        # Calculate loss for this iteration
        loss = elos_loss(w, elos)
        losses.append(loss)
        if len(losses) > 2 and abs(losses[-2] - losses[-1]) < epsilon:
            break
    if abs(losses[-2] - losses[-1]) > epsilon:
        print(f"ERROR! Not within epsilon after {len(losses)} iterations!")

    return elos, losses


