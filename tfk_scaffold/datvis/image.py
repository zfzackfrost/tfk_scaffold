from .common import plot_function
from typing import Optional, Union, List

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

import pandas as pd
import numpy as np

@plot_function
def plot_image(
    data: Union[pd.DataFrame, pd.Series, np.ndarray, List[Union[int, float]]],
    fig: Optional[Figure],
    ax: Optional[Axes],
    *args,
    usecolumn: Optional[str] = None,
    **kwargs
):
    if fig is None:
        return
    if ax is None:
        return
    
    d = data
    if isinstance(d, pd.DataFrame):
        d = d[usecolumn]
    ax.imshow(d, cmap=plt.cm.binary)