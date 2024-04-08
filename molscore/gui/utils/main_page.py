import os

import streamlit as st
from streamlit_plotly_events import plotly_events

from molscore.gui.utils import utils


def single_plot(main_df, SS, dock_path=None):
    """The streamlit monitors main page"""

    # ----- Show central plot -----
    col1, col2, col3, col4, col5 = st.columns(5)
    x_axis = col1.selectbox("Plot x-axis", ["step", "index"], index=0)
    y_axis = col2.selectbox(
        "Plot y-axis",
        [c for c in main_df.columns.tolist() if c not in SS.exclude_params],
        index=7,
    )
    valid_only = col3.checkbox(label="Valid only")
    unique_only = col3.checkbox(label="Unique only")
    trendline = col4.selectbox(
        "Trendline", [None, "median", "mean", "max", "min"], index=1
    )
    col5.write("")
    col5.write("")  # Hacky vertical fill
    trendline_only = col5.checkbox(label="Trendline only")

    tdf = main_df
    if valid_only:
        tdf = tdf.loc[tdf.valid == "true", :]
    if unique_only:
        tdf = tdf.loc[tdf.unique == True, :]

    fig = utils.plotly_plot(
        y_axis, tdf, x=x_axis, trendline=trendline, trendline_only=trendline_only
    )
    selection = plotly_events(fig, click_event=False, select_event=True)
    selection = [
        int(
            tdf[tdf.run == tdf.run.unique()[sel["curveNumber"] // 2]].index[
                sel["pointNumber"]
            ]
        )
        for sel in selection
    ]

    # ----- Show selected data -----
    st.subheader("Selected structures")
    utils.display_selected_data(
        y=y_axis,
        main_df=main_df,
        key="main",
        dock_path=dock_path,
        selection=selection,
        viewer=None,
        pymol=SS.pymol,
    )

    # ----- Add option to save sdf -----
    if dock_path and (selection is not None):
        with st.expander(label="Export selected molecules"):
            # User input
            out_file = st.text_input(label="File name")
            out_file = os.path.abspath(f"{out_file}.sdf")
            st.write(out_file)
            if st.button(label="Save", key="save_all_selected"):
                file_paths, mol_names = utils.find_sdfs(selection, main_df)
                utils.save_sdf(
                    mol_paths=file_paths, mol_names=mol_names, out_file=out_file
                )
                st.write("Saved!")
