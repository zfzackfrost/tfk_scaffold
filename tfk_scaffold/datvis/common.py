from typing import Protocol, Optional, Tuple, Any, cast, Union, List

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

import pandas as pd
import numpy as np


class PlotFunctionT(Protocol):
    def __call__(
        self,
        data: Union[pd.DataFrame, pd.Series, np.ndarray, List[Union[int,
                                                                    float]]],
        fig: Optional[Figure],
        ax: Optional[Axes],
        *args,
        usecolumn: Optional[str] = None,
        **kwargs
    ):
        pass


def plot_function(fn: PlotFunctionT):
    def inner_function(
        data: Union[pd.DataFrame, pd.Series, np.ndarray, List[Union[int,
                                                                    float]]],
        fig: Optional[Figure],
        ax: Optional[Axes],
        *args,
        subplot: Tuple[int, int, int] = (1, 1, 1),
        usecolumn: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        **kwargs
    ):
        if fig is None:
            fig = plt.figure()
        if ax is None:
            ax = cast(Any, plt.subplot(*subplot))
        cast(Any, fig).set_tight_layout(True)
        fn(data, fig, ax, *args, usecolumn=usecolumn, **kwargs)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)

    return inner_function
