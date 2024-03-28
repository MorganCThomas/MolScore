from itertools import cycle

import streamlit as st
from molscore.gui.utils import utils


def multi_plot(main_df, SS):
    """ Show multiple plots """

    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
    y_variables = col1.multiselect('y-axis', [c for c in main_df.columns.tolist() if c not in SS.exclude_params], default=['valid', 'unique', 'occurrences'])
    valid_only = col2.checkbox(label='Valid only')
    unique_only = col2.checkbox(label='Unique only')
    trendline = col3.selectbox('Trendline', [None, 'median', 'mean', 'max', 'min'], index=1)
    col4.write("") ; col4.write("") # Hacky vertical fill
    trendline_only = col4.checkbox(label='Trendline only')

    tdf = main_df
    if valid_only:
        tdf = tdf.loc[tdf.valid == 'true', :]
    if unique_only:
        tdf = tdf.loc[tdf.unique == True, :]

    for y, col in zip(y_variables, cycle(st.columns(3))):
        sub_fig = utils.plotly_plot(y, tdf, size=(250, 200), trendline=trendline, trendline_only=trendline_only)
        col.plotly_chart(sub_fig)
