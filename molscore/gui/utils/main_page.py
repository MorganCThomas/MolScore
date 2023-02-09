import os
import streamlit as st
from molscore.gui.utils import utils


def single_plot(main_df, SS, dock_path=None):
    """ The streamlit monitors main page """

    # ----- Show central plot -----
    col1, col2, col3 = st.columns(3)
    x_axis = col1.selectbox('Plot x-axis', ['step', 'index'], index=0)
    y_axis = col2.selectbox('Plot y-axis', main_df.columns.tolist(), index=7)
    valid_only = col3.checkbox(label='Valid only')
    unique_only = col3.checkbox(label='Unique only')

    tdf = main_df
    if valid_only:
        tdf = main_df.loc[main_df.valid == 'true', :]
    if unique_only:
        tdf = main_df.loc[main_df.unique == True, :]

    fig = utils.plotly_plot(y_axis, tdf, x=x_axis)
    selection = utils.plotly_events(fig, click_event=False, select_event=True)
    selection = [int(tdf[tdf.run == tdf.run.unique()[sel['curveNumber']//2]].index[sel['pointNumber']]) for sel in selection]
  
    # ----- Show selected data -----
    st.subheader('Selected structures')
    utils.display_selected_data(
        y=y_axis,
        main_df=main_df,
        key='main',
        dock_path=dock_path,
        selection=selection,
        viewer=None,
        pymol=SS.pymol
        )
    
    # ----- Add option to save sdf -----
    if (dock_path is not None) and (selection is not None):
        with st.expander(label='Export selected molecules'):
            # User input
            out_file = st.text_input(label='File name')
            out_file = os.path.abspath(f'{out_file}.sdf')
            st.write(out_file)
            if st.button(label='Save', key='save_all_selected'):
                file_paths, mol_names = utils.find_sdfs(selection, main_df, dock_path)
                utils.save_sdf(
                    mol_paths=file_paths,
                    mol_names=mol_names,
                    out_file=out_file
                    )
                st.write('Saved!')
