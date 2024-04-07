from itertools import cycle

import streamlit as st

from molscore.gui.utils import utils


def multi_plot(main_df, SS):
    """Show multiple plots"""

    col1, col2, col3, col4, col5 = st.columns([1, 3, 1, 1, 1])
    x_axis = col1.selectbox("Plot x-axis", ["step", "index"], index=0)
    y_variables = col2.multiselect(
        "y-axis",
        [c for c in main_df.columns.tolist() if c not in SS.exclude_params],
        default=["valid", "unique", "occurrences"],
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

    for y, col in zip(y_variables, cycle(st.columns(3))):
        sub_fig = utils.plotly_plot(
            y,
            tdf,
            x=x_axis,
            size=(250, 200),
            trendline=trendline,
            trendline_only=trendline_only,
        )
        col.plotly_chart(sub_fig)
