import os

import numpy as np
import pandas as pd

# import amean, gmean, prod, wsum, wprod
import plotly.express as px
import streamlit as st
from streamlit_plotly_events import plotly_events

from molscore.gui.utils import utils
from molscore.utils import (
    aggregation_functions,
    transformation_functions,
)


# @st.cache
def plotly_parallel_plot(mdf):
    """Draw parallel plots based on a melted df with columns 'x_var' and 'value'."""
    # Draw scatter
    fig = px.scatter(
        data_frame=mdf,
        x="x_var",
        y="value",
        color="run",
        hover_data=["step", "batch_idx"],
        template="plotly_white",
    )
    # Add lines
    for run in mdf["run"].unique():
        for idx in mdf.loc[mdf.run == run]["idx"].unique():
            ttdf = mdf.loc[(mdf["run"] == run) & (mdf["idx"] == idx), :]
            fig.add_traces(
                list(
                    px.line(
                        data_frame=ttdf, x="x_var", y="value", template="plotly_white"
                    ).select_traces()
                )
            )
    fig.update_traces(line=dict(color="Black", width=0.25))
    fig.update_layout(title=None, xaxis_title="Parameters")
    return fig


# @st.cache(suppress_st_warning=True)
def plotly_mpo_events(df, x_variables, step=None, k=None):
    """Draw parallel plot and return selection"""
    # Melt
    mdf = df.melt(
        id_vars=["run", "idx", "step", "batch_idx"],
        var_name="x_var",
        value_vars=x_variables,
        value_name="value",
    )
    # Draw figure
    fig = plotly_parallel_plot(mdf)
    selection = plotly_events(fig, click_event=False, select_event=True)
    # Map selection to melted df
    selection = [
        mdf[mdf.run == mdf.run.unique()[sel["curveNumber"]]].index[sel["pointNumber"]]
        for sel in selection
    ]
    # Map melted df to main df
    selection = [
        int(df.loc[(df.run == r) & (df["idx"] == idx)].index[0])
        for i, (r, idx) in mdf.loc[selection, ["run", "idx"]].iterrows()
    ]

    return selection


def maxminnorm(series, invert):
    data = series.to_numpy()
    if invert:
        data = data * -1
    data_norm = transformation_functions.norm(x=data, objective="maximize", max=max(data), min=min(data))
    series_norm = pd.Series(data=data_norm.flatten(), name=series.name)
    return series_norm


# @st.experimental_memo
def calculate_aggregate(df, x_variables, x_orders, x_weights, agg_method):
    agg_method = getattr(aggregation_functions, agg_method)
    # Re-normalize data
    top_df = df.loc[:, x_variables].apply(
        lambda x: maxminnorm(x, x_orders[x_variables.index(x.name)]), axis=0
    )
    # Calculate geometric mean
    return (
        top_df.fillna(1e6)
        .apply(lambda x: agg_method(x, w=x_weights), axis=1, raw=True)
        .tolist()
    )


def mpo_plot(main_df, SS, dock_path=False):
    """Show parallel plots of Top-K molecules"""

    # ----- MPO @ step -----
    st.subheader("Per step MPO")
    st.text(
        "Select multiple paramerers to co-plot (only showing valid, unique molecules)"
    )
    x_variables = st.multiselect(
        "x-axis", [c for c in main_df.columns.tolist() if c not in SS.exclude_params]
    )
    if len(main_df.step.unique()) == 1:
        step_idx = 1
    else:
        step_idx = st.slider(
            "Step",
            min_value=int(main_df.step.min()),
            max_value=int(main_df.step.max()),
            value=int(main_df.step.max()),
        )
    tdf = main_df.loc[(main_df.valid == "true") & (main_df.unique == True), :].copy()

    # Plot graph
    selection = plotly_mpo_events(tdf[tdf.step == step_idx], x_variables, step=step_idx)
    # Plot mols
    with st.expander(label="Display selected molecules"):
        utils.display_selected_data(
            y=x_variables,
            main_df=main_df,
            key="StepMPO",
            dock_path=dock_path,
            selection=selection,
            viewer=None,
            pymol=SS.pymol,
        )

    # ----- MPO & Top-K -----
    if len(x_variables) > 0:
        st.subheader("Top-K MPO")
        st.markdown("**Description**")
        st.markdown(
            "Select Top-K molecules according to scoring function variables returned. Variables are maxmin normalized again in-case of any moving goal post normalisation during generation. If lower is better, reverse the variable in options below."
        )
        st.markdown(
            "Additionally select how you'd like to aggregate selected variables"
        )
        # Get input options
        with st.expander(label="Options"):
            k = int(st.number_input(label="Top k", value=10))
            # Set aggregation method
            agg_method = st.selectbox(
                label="Aggregation method",
                options=["amean", "gmean", "prod", "wsum", "wprod"],
                index=1,
            )
            # Get x orders & weights
            x_orders = []
            x_weights = []
            for x in x_variables:
                st.write(x)
                col1, col2 = st.columns(2)
                x_orders.append(
                    col1.selectbox(
                        label="Reverse",
                        options=[True, False],
                        index=1,
                        key=f"{x}_order",
                    )
                )
                x_weights.append(
                    col2.number_input(label="Weight", value=1.0, key=f"{x}_weight")
                )
            x_weights = np.asarray(x_weights)

        tdf["topk_agg"] = calculate_aggregate(
            tdf, x_variables, x_orders, x_weights, agg_method
        )

        # Plot graph
        topk_df = tdf.sort_values("topk_agg", ascending=False).iloc[:k]
        _ = plotly_mpo_events(topk_df.iloc[: min(k, 100)], x_variables)
        selection = list(topk_df.index)

        # Plot mols
        utils.display_selected_data(
            y=x_variables,
            main_df=main_df,
            key="TopKMPO",
            dock_path=dock_path,
            selection=selection,
            pymol=SS.pymol,
        )

        # ----- Export data -----
        with st.expander("Export top-K data"):
            out_file = st.text_input(label="File name", key="topk_csv_output")
            out_file = os.path.abspath(f"{out_file}.csv")
            st.write(out_file)
            if st.button(label="Save", key="export_topk_csv"):
                topk_df.to_csv(out_file, index=False)
                st.write("Saved!")

        # ----- Add option to save sdf -----
        if dock_path:
            with st.expander(label="Export top-K molecules"):
                # User input
                out_file = st.text_input(label="File name", key="topk_sdf_output")
                out_file = os.path.abspath(f"{out_file}.sdf")
                st.write(out_file)
                if st.button(label="Save", key="export_topk_sdf"):
                    file_paths, mol_names = utils.find_sdfs(selection, main_df)
                    utils.save_sdf(
                        mol_paths=file_paths, mol_names=mol_names, out_file=out_file
                    )
                    st.write("Saved!")
