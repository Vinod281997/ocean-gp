"""Loading and basic processing of the ocean flow data"""

import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt


grid_size = 3.0 # grid spacing in km

def load_data(data_dir, n_frames = 100):
    """Load the ocean flow field data from the csv files.
    
    Args:
        data_dir: str or Path
            The directory containing the files 1u.csv, 1v.csv, 2u.csv, 2v.csv, ..., n_frames u.csv, n_frames v.csv
     n_frames: int
            number of time snapshots to load.
    
    Returns:
        u, v: np.ndarray
            Horizontal and vertical velocity components, each of shape  n_frames, ny, nx).
            
    """

    data_dir = Path(data_dir)
    u = np.stack([pd.read_csv(data_dir/f"{i+1}u.csv", header=None).values 
                  for i in range(n_frames)])
    
    v = np.stack([pd.read_csv(data_dir/f"{i+1}v.csv", header=None).values
                  for i in range(n_frames)])
    
    return u, v

def compute_speed(u, v):
    """Magnitude of the velocity field."""

    return np.sqrt(u**2 + v**2)


def load_mask(data_dir):
    """Load the land/water mask. Returns a boolean array, True on land.

    Args:
        data_dir : str or Path
            Directory containing mask.csv (0 = land, 1 = water in the source file).

    Returns:
        np.ndarray
            Boolean array of shape (ny, nx), True where there is land.
    """
    mask = pd.read_csv(Path(data_dir) / "mask.csv", header=None).values
    return mask == 0


def magma_dark():
    """Magma colormap with its darkest 18% trimmed.

    Returns:
        matplotlib.colors.Colormap
    """
    base = plt.cm.magma
    return LinearSegmentedColormap.from_list("magma_dark",
                                             base(np.linspace(0.18, 1.0, 256)))