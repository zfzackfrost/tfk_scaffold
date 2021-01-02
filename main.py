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

from tfk_scaffold import *


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
    pyver = (
        sys.version_info.major, sys.version_info.minor, sys.version_info.micro
    )
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
@click.option(
    '-p',
    '--path',
    type=str,
    required=True,
    help="Path or URL to fetch the dataset from"
)
@click.option(
    '-f',
    '--format',
    type=click.Choice(['csv', 'mnist']),
    required=True,
    help="The format of the input dataset"
)
@click.option(
    '-o',
    '--output',
    type=str,
    required=True,
    help="Path to store the cached dataset at"
)
@click.option(
    '-z',
    '--compression',
    type=click.Choice(['infer', 'gzip', 'bz2', 'zip', 'xz', 'off']),
    default='infer',
    help="Compression method for the cached dataset"
)
@click.pass_context
def cache_cmd(ctx, path, format, output, compression):
    """Cache a dataset for the model MODELNAME."""
    mname = ctx.obj['MODELNAME']
    m = ModelRegistry.get(mname)
    fmt = DataFormat.from_str(format)
    if fmt is None:
        raise NotImplementedError()
    dataset = m.fetch_data(path, fmt)
    if compression == 'off':
        compression = None
    dataset.save(output, compression)


def validate_seed(ctx, param, value):
    try:
        seed = int(value.strip())
    except ValueError:
        raise click.BadParameter('`seed` must be convertable to `int`')
    if seed <= 0:
        raise click.BadParameter('`seed` must be >= 1')
    return seed


@data_gr.command(name='sample')
@click.option(
    '-i',
    '--input',
    type=str,
    required=True,
    help="Path to the cached dataset"
)
@click.option(
    '-s',
    '--subset',
    type=click.Choice(['train', 'test', 'raw']),
    default='raw',
    help="The sub-dataset to sample from"
)
@click.option(
    '-n', '--count', type=int, default=5, help="Number of rows to sample"
)
@click.option(
    '--seed',
    callback=validate_seed,
    default='10',
    help="The random seed to use."
)
def sample_cmd(
    input,
    subset,
    count,
    seed,
):
    ds = DataSet()
    ds.load(input)
    df = getattr(ds, subset)
    df = df.sample(n=count, random_state=seed)
    print(df)


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


@model_gr.command(name='predict')
@click.pass_context
def predict_cmd(ctx):
    print(f"Predicting: {ctx.obj['MODELNAME']}")


@cli.group(name='plot')
@click.option(
    '-d',
    '--dataset',
    type=str,
    required=True,
    help="Name of the dataset to plot."
)
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