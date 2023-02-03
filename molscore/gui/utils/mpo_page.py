import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import streamlit as st
from molscore.gui.utils import utils
from molscore.utils.aggregation_functions import amean, gmean, prod, wsum, wprod

import plotly.express as px

try:
    from bokeh.plotting import figure
    from bokeh.models import ColumnDataSource, BoxSelectTool
except ImportError:
    pass

def plotly_parallel_plot(mdf):
    """ Draw parallel plots based on a melted df with columns 'x_var' and 'value'. """
     # Draw scatter
    fig = px.scatter(data_frame=mdf, x='x_var', y='value', width=1000, height=500, hover_data=['step', 'batch_idx'], template='plotly_white')
    # Add lines
    for idx in mdf['index'].unique():
        ttdf = mdf.loc[mdf['index'] == idx, :]
        fig.add_traces(list(px.line(data_frame=ttdf, x='x_var', y='value', width=1000, height=500, template='plotly_white').select_traces()))
    fig.update_traces(line=dict(color="Black", width=0.5))
    fig.update_layout(
        title=None,
        xaxis_title=None
        )
    return fig

def plotly_mpo_events(df, x_variables, step=None, k=None):
    """ Draw parallel plot and return selection """
    if step:
        # Subset
        df = df.loc[df.step == step, x_variables + ['step', 'batch_idx']]
        # Melt
        mdf = df.reset_index(drop=False).melt(id_vars=['index', 'step', 'batch_idx'], var_name='x_var', value_vars=x_variables, value_name='value')
        # Draw figure
        fig = plotly_parallel_plot(mdf)
        selection = utils.plotly_events(fig, click_event=False, select_event=True)
        selection = list(mdf.loc[[i['pointIndex'] for i in selection], 'index'].unique())
    
    if k:
        if k > 100: st.write("Warning: Limiting display to first 100")
        # Subset top-k unique
        df = df.iloc[:k, :]
        # Melt
        mdf = df.reset_index(drop=False).melt(id_vars=['index', 'step', 'batch_idx'], var_name='x_var', value_vars=x_variables, value_name='value')
        # Draw figure
        fig = plotly_parallel_plot(mdf)
        selection = utils.plotly_events(fig, click_event=False, select_event=True)
        selection = list(mdf.loc[[i['pointIndex'] for i in selection], 'index'].unique())

    return selection

def mpo_plot(main_df, SS, dock_path=None, plotting='plotly'):
    """ Show parallel plots of Top-K molecules"""

    # ----- MPO @ step ----- 
    st.subheader('Per step MPO')
    x_variables = st.multiselect('x-axis', main_df.columns.tolist())
    step_idx = st.slider('Step', min_value=int(main_df.step.min()), max_value=int(main_df.step.max()),
                            value=int(main_df.step.max()))

    if plotting == 'plotly':
        selection = plotly_mpo_events(main_df, x_variables, step=step_idx)

    elif plotting == 'bokeh':
        # TODO move this to utils function
        p = figure(
            plot_width=1000, plot_height=500, x_range=x_variables,
            tooltips=
            """
            <div>
            @img{safe}
            Step_batch_idx: @ids<br>
            </div>
            """
            )
        p.add_tools(BoxSelectTool())
        # TODO figure out indices selection
        for i, r in main_df.loc[main_df.step == step_idx, :].iterrows():
            data = dict(x=x_variables,
                        y=r[x_variables].values,
                        ids=[f"{r['step']}_{r['batch_idx']}"]*len(x_variables),
                        img=[mol2svg(Chem.MolFromSmiles(r['smiles']))]*len(x_variables))
            source = ColumnDataSource(data)
            p.circle(x='x', y='y', source=source)
            p.line(x='x', y='y', source=source)
        selection = utils.streamlit_bokeh_events(bokeh_plot=p, events="BOX_SELECT", key="mpo",
                                                 refresh_on_update=True, override_height=None, debounce_time=0)
    
    else:
        raise ValueError("Unrecognized plotting library, should be plotly or bokeh")

    # ----- Top-K MPO -----
    if len(x_variables) > 0:
        st.subheader('Top-K MPO')
        st.markdown("**Description**")
        st.markdown("Select Top-K molecules according to scoring function variables returned. Variables are maxmin normalized again in-case of any moving goal post normalisation during generation. If lower is better, reverse the variable in options below.")
        st.markdown("Additionally select how you'd like to aggregate selected variables")
        with st.expander(label='Options'):
            k = int(st.number_input(label='Top k', value=10))
            # Set aggregation method
            agg_method = st.selectbox(label='Aggregation method', options=[m.__name__ for m in [amean, gmean, prod, wsum, wprod]], index=1)
            agg_method = [agg for agg in [amean, gmean, prod, wsum, wprod] if agg_method == agg.__name__][0]
            # Get x orders & weights
            x_orders = []
            x_weights = []
            for x in x_variables:
                st.write(x)
                col1, col2 = st.columns(2, gap='large')
                x_orders.append(col1.selectbox(label=f'Reverse',
                                             options=[True, False], index=1, key=f'{x}_order'))
                x_weights.append(col2.number_input(label=f'Weight', value=1.0, key=f'{x}_weight'))
            x_weights = np.asarray(x_weights)

        def maxminnorm(series, invert):
            series = series * (-1 if invert else 1)
            data = series.to_numpy().reshape(-1, 1)
            data_norm = MinMaxScaler().fit_transform(data)
            series_norm = pd.Series(data=data_norm.flatten(), name=series.name)
            return series_norm
        
        # Re-normalize data
        top_df = main_df.loc[:, x_variables].apply(lambda x: maxminnorm(x, x_orders[x_variables.index(x.name)]), axis=0)
        # Calculate geometric mean
        top_df['topk_agg'] = top_df.fillna(1e6).apply(lambda x: agg_method(x, w=x_weights), axis=1, raw=True)
        top_df = pd.concat([main_df.loc[:, ['step', 'batch_idx', 'smiles', 'unique']], top_df], axis=1)
        # Subset unique and sort
        top_df = top_df.sort_values(by='topk_agg', ascending=False)

        if plotting == 'plotly':
            _ = plotly_mpo_events(top_df.loc[top_df.unique == True, :], x_variables, k=min(k, 100))
            selection = top_df.loc[top_df.unique == True, :].iloc[:min(k, 100)].index

        elif plotting == 'bokeh':
            # TODO move to external function
            if k > 100: st.write("Warning: Limiting display to first 100")
            p2 = figure(plot_width=1000, plot_height=500, x_range=x_variables,
                        tooltips=
                        """
                        <div>
                        @img{safe}
                        Step_batch_idx: @ids<br>
                        </div>
                        """)
            for i, r in top_df[:100].iterrows():
                data = dict(x=x_variables,
                            y=r[x_variables].values,
                            ids=[f"{r['step']}_{r['batch_idx']}"]*len(x_variables),
                            img=[mol2svg(Chem.MolFromSmiles(r['smiles']))]*len(x_variables))
                source2 = ColumnDataSource(data)
                p2.circle(x='x', y='y', source=source2)
                p2.line(x='x', y='y', source=source2)
            selection = streamlit_bokeh_events(bokeh_plot=p2, events="BOX_SELECT", key="mpo2",
                                                refresh_on_update=True, override_height=None, debounce_time=0)
            selection = {"BOX_SELECT": {"data": top_df.index.to_list()}}
        else:
            raise ValueError("Unrecognized plotting library, should be plotly or bokeh")

        # ----- Plot mols -----
        utils.display_selected_data('topk_agg', top_df, dock_path=dock_path,
                                    selection=selection,
                                    viewer=None, pymol=SS.pymol)

        # ----- Add option to save sdf -----
        # TODO specify molscore ouput path as suggest path, so that it can be saved anywhere, or two output paths
        if dock_path is not None:
            with st.expander(label='Save top k (in molscore output directory)'):
                # User input
                out_file = st.text_input(label='File name')
                st.write(f'File name: {out_file}.sdf')
                save_top_k = st.button(label='Save', key='save_top_k_button')
                if save_top_k:
                    file_paths, mol_names = find_sdfs(top_df.index.to_list(), main_df, dock_path, gz_only=True)
                    save_sdf(mol_paths=file_paths,
                                mol_names=mol_names,
                                out_name=out_file)