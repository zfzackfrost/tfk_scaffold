# For environment variables
import os

# For building CLI
import click

# Make TensorFlow silent
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib
import sys

from typing import cast, Any

@click.group()
@click.option('-m', '--model', type=str, required=True, metavar='MODELNAME')
@click.pass_context
def cli(ctx, model):
    # ensure that ctx.obj exists and is a dict (in case `cli()` is called
    # by means other than the `if` block below)
    ctx.ensure_object(dict)

    ctx.obj['MODELNAME'] = model

@cli.command(name="info")
def info_cmd():
    # Using casts since the typechecker does not recognize
    # the __version__ variable in all of the modules, even
    # though it is defined.
    pyver = (sys.version_info.major, sys.version_info.minor, sys.version_info.micro)
    pyver = (str(x) for x in pyver)
    pyver = ".".join(pyver)
    print(f"Python Version: {pyver}")
    print(f"TensorFlow Version: {cast(Any, tf).__version__}")
    print(f"NumPy Version: {cast(Any, np).__version__}")
    print(f"Pandas Version: {cast(Any, pd).__version__}")
    print(f"Matplotlib Version: {cast(Any, matplotlib).__version__}")


@cli.group(name='data')
def data_gr():
    """Dataset operations for the model MODELNAME."""
    pass

@data_gr.command(name='cache')
def cache_cmd():
    """Cache a dataset for the model MODELNAME."""
    print("Fetch Data")

@cli.group(name="model")
def model_gr():
    """Operate on the model MODELNAME."""

@model_gr.command(name='train')
@click.pass_context
def train_cmd(ctx):
    print(f"Training: {ctx.obj['MODELNAME']}")

@model_gr.command(name='test')
@click.pass_context
def test_cmd(ctx):
    print(f"Testing: {ctx.obj['MODELNAME']}")


@cli.group(name='plot')
@click.option('-d', '--dataset', type=str, required=True, help="Name of the dataset to plot.")
@click.pass_context
def plot_gr(ctx, dataset):
    """Plotting and data visualization for the model MODELNAME."""
    ctx.obj['DATASET'] = dataset

@plot_gr.command(name='pair')
@click.pass_context
def pairplot_cmd(ctx):
    """Displays multiple graphs in a grid; Every dataset feature against every other."""
    print(f"Pair Plot: {ctx.obj['DATASET']}")

if __name__ == '__main__':
    cli(obj={})