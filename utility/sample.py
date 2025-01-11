import pandas as pd
import numpy as np


def sample(data_directory: str, sample_size: int) -> np.array:
    data = pd.read_csv(data_directory, header = None)
    samples = data.sample(sample_size)[0]
    return samples.to_numpy()