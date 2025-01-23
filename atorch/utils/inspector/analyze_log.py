import argparse
import re
import time
from functools import partial
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

""" Analyze log codes taken from Megatron inspector with modifition. Only works for transformers model."""


def get_step(s):
    match = re.search(r"step(\d+)\,", s)
    if match:
        step = match.group(1)
        return int(step)
    else:
        return None


def get_layer_index(s):
    match = re.search(r"layers\.(\d+)", s)
    if match:
        step = match.group(1)
        return int(step)
    else:
        return None


def get_layer_name(s):
    match = re.search(r"(mlp|self_attention|self_attn)\.((\w|_)+)", s)
    if match:
        step = match.group(2)
        return step
    else:
        return None


def get_rank(filename):
    match = re.search(r"Rank (\d+)", filename)
    if match:
        rank = match.group(1)
        return int(rank)
    else:
        return None


def get_tensor(s):
    match = re.search(r"(fc1|fc2|proj|qkv)\.((\w|_)+)\.step", s)
    if match:
        tensor = match.group(2)
        return tensor
    else:
        return None


def get_value(s, key):
    pattern = key + r": ([\d.e+-]+)"
    match = re.search(pattern, s)
    if match:
        value = match.group(1)
        return float(value)


def extract(line):
    # Get Rank
    rank = get_rank(line)
    if rank is None:
        print("Can't extract Rank from line: {line}")

    # Get step
    step = get_step(line)
    if step is None:
        print("Can't extract step from line: {line}")

    # 3. Get layer index and name
    # exception: step0.model.decoder.final_layernorm, step0.model.output_layer
    if re.search("embedding", line):
        layer_index = 0
        layer_name = "embedding"
    elif re.search("final_layernorm", line):
        layer_index = 99
        layer_name = "final_LN"
    elif re.search("output_layer", line):
        layer_index = 99
        layer_name = "output_layer"
    else:
        # 3. get layer index
        layer_index = get_layer_index(line)
        assert layer_index is not None, f"layer index Error in line: {line}"

        # 4. get layer name
        layer_name = get_layer_name(line)
        assert layer_name is not None, f"layer name Error in line: {line}"

    # Get Tensor Name: fwd_x, fwd_w, bwd_dy
    tensor = get_tensor(line)

    # Get metrics
    mse = get_value(line, "mse")
    cos = get_value(line, "cos")
    underflows = get_value(line, "underflows\(\%\)")  # noqa: W605
    overflows = get_value(line, "overflows\(\%\)")  # noqa: W605
    cs = get_value(line, "amax scale")  # current scale (from amax)
    ds = get_value(line, "meta scale")  # delayed scale
    amin = get_value(line, "amin")
    amax = get_value(line, "amax")

    return rank, step, layer_index, layer_name, tensor, mse, cos, underflows, overflows, cs, ds, amin, amax


def process(filename, save_df_file=False):
    """
    Process log file into a DataFrame(df)
    """
    # Line example:
    # 3: [Rank 3] decoder.layers.11.mlp.linear_fc2.bwd_dy.step8712, \
    # amin: 0.000e+00, amax: 2.772e-06, amax scale: 2.067e+10, \
    # meta scale: 2.863e+09, cos: 1.000, mse: 9.216e-18, \
    # underflows(%): 0.0, overflows(%): 0.0
    # 6: [Rank 6] decoder.layers.11.mlp.linear_fc2.bwd_dy.step8712, \
    # amin: 0.000e+00, amax: 1.676e-06, amax scale: 3.436e+10, \
    # meta scale: 3.511e+09, cos: 1.000, mse: 7.535e-18, \
    # underflows(%): 0.0, overflows(%): 0.0
    gradnorm_pattern = "Rank.*step"

    # Read file, save desired lines
    lines = list()
    with open(filename, "r") as f:
        for line in f:
            if re.search(gradnorm_pattern, line):
                lines.append(line.strip())
    print(f"Find number of lines: {len(lines)}")

    # Create DataFrame
    # extract: step, layer, name, value
    table = list()
    for line in lines:
        row = extract(line)
        table.append(row)

    columns = [
        "Rank",
        "Step",
        "LayerIndex",
        "LayerName",
        "Tensor",
        "MSE",
        "Cosine",
        "Underflows",
        "Overflows",
        "CurrentScale",
        "DelayScale",
        "AMin",
        "AMax",
    ]
    df = pd.DataFrame(table, columns=columns)

    if save_df_file:
        print(f"Save this dataframe into a pickle format gz file: {args.filename}.gz")
        df.to_pickle(args.filename + ".gz")

    return df


metrics_in_logscale = ("MSE", "AMax", "DelayScale", "ScaleRatio(Current_div_Delay)")


def remove_outliers(df, col, threshold=10):
    """
    Drop outlier by z score: https://saturncloud.io/blog/how-to-detect-and-exclude-outliers-in-a-pandas-dataframe/
    col: the column to compute z score
    """
    z = np.abs(stats.zscore(df[col]))
    outliers = df[z > threshold]
    df = df.drop(outliers.index)
    return df


def plot_task(m, t_df, max_z, filename):
    outfile = f"{filename}.{m}"
    y_logscale = True if m in metrics_in_logscale else False
    print(f"Plot metric: {m}, using y logscale: {y_logscale}")

    #   m_df = remove_outliers(t_df, m, max_z)
    plot(t_df, m, outfile, y_logscale)


def filter_metrics(met, df):
    """filter out metrics without valid data"""
    new_met = []
    for m in met:
        values = df[m].unique().tolist()
        if len(values) > 1 or (len(values) == 1 and values[0] is not None and not pd.isna(values[0])):
            new_met.append(m)
    return new_met


def plot(df, m, filename, y_logscale=False):
    """
    Plot each metrics against step, with all layerindex on the same plot
    Different Tensor and LayerName on different subfigure
    """
    title = " ".join(filename.split(".")[-2:])
    print(f"Title: {title}")
    tic = time.perf_counter()
    g = sns.relplot(
        data=df,
        x="Step",
        y=m,
        hue="LayerIndex",
        #     style="LayerIndex",
        palette="deep",
        col="LayerName",
        col_wrap=2,
        legend="full",
        kind="line",
        errorbar=None,
        height=4,
        aspect=2,
    )
    if y_logscale:  # set y axis log-scale
        g.set(yscale="log")
    g.fig.suptitle(title)
    toc = time.perf_counter()
    print(f"Plotted figure in {toc-tic:0.3f} s")

    print(f"Save figure to file: {filename}.png")
    plt.savefig(f"{filename}.png", dpi=400)


def analyze_log(args, df=None):
    if df is None:
        assert args.filename.endswith(".gz"), "Assume a dataframe saved in a gz file!"
        # Load df from .gz file
        print(f"Read df from gz file: {args.filename}")
        df = pd.read_pickle(args.filename)

    metrics = ["MSE", "Cosine", "Underflows", "Overflows", "AMax", "DelayScale"]
    tensors = ("fwd_x", "fwd_w", "bwd_dy")
    max_z = 10  # Max z score to identify outlier

    # Select steps
    if args.max_steps > 0:
        print(f"Select data Step < {args.max_steps}")
        df = df[df.Step < args.max_steps]

    # Select ranks
    print(f"Select data on Rank: {args.rank}")
    df = df[df.Rank == args.rank]

    # Add new columns: ScaleRatio(Current/Delay)
    df["ScaleRatio(Current_div_Delay)"] = df.CurrentScale / df.DelayScale
    metrics.append("ScaleRatio(Current_div_Delay)")

    # Define the custom order, is it needed?
    """
    custom_order = ['linear_qkv',
                    'linear_proj',
                    'linear_fc1',
                    'linear_fc2']
    # Create a categorical column with the custom order
    df['layer_ordered'] = pd.Categorical(
        df['LayerName'], categories=custom_order, ordered=True)
    df = df.sort_values(by='layer_ordered')
    """

    # Parallel plotting metrics by multi-processing
    sns.set_theme()
    with Pool(len(metrics)) as p:
        for tensor in tensors:
            print(f"Plot tensor: {tensor}")
            t_df = df[df.Tensor == tensor]
            valid_metrics = filter_metrics(metrics, t_df)

            p.map(partial(plot_task, t_df=t_df, max_z=max_z, filename=f"{args.filename}.{tensor}"), valid_metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a log file and/or Plot a dataframe file")
    parser.add_argument("-f", "--filename", type=str, help="Log file or Dataframe file")
    parser.add_argument("-s", "--max_steps", type=int, default=-1, help="Max Steps")
    parser.add_argument("-r", "--rank", type=int, default=0, help="Max Steps")
    parser.add_argument("--process_log", default=False, action="store_true", help="process log file into data frame")
    parser.add_argument("--save_df_file", default=False, action="store_true", help="save processed dataframe into file")
    args = parser.parse_args()

    df = None
    if args.process_log:
        # Process log into a df
        df = process(args.filename, save_df_file=args.save_df_file)

        print("Finished creating dataframe from log file." "\nShow head of this dataframe:")
        print(df.head())

    analyze_log(args, df=df)
