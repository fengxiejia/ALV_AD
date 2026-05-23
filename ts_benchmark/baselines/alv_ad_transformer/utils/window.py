import numpy as np


def window_scores_to_sequence(window_scores: np.ndarray) -> np.ndarray:
    return np.asarray(window_scores).reshape(-1).astype(np.float32)
