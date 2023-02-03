from itertools import cycle

try:
    from bokeh.plotting import gridplot
except ImportError:
    pass

import streamlit as st
from molscore.gui.utils import utils


def multi_plot(main_df, SS, dock_path=None, plotting='plotly'):
    """ Show multiple plots """

    y_variables = st.multiselect('y-axis', main_df.columns.tolist(), default=['valid', 'unique', 'occurrences'])

    if plotting == 'plotly':
        for y, col in zip(y_variables, cycle(st.columns(3, gap='large'))):
            sub_fig = utils.plotly_plot(y, main_df, size=(250, 200))
            col.plotly_chart(sub_fig)
    
    elif plotting == 'bokeh':
        plots = []
        for y in y_variables:
            p = utils.bokeh_plot(y, main_df, size=(500, 300))
            plots.append(p)
        grid = gridplot(plots, ncols=3)
        st.bokeh_chart(grid)
    
    else:
        raise ValueError("Unrecognized plotting library, should be plotly or bokeh")
