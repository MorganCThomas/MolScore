import sys
import os
import sys

import streamlit as st

from molscore.gui.utils.pymol_wrapper import PyMol
from molscore.gui.utils import single_plot, multi_plot, mpo_plot, scaffold_plot


SS = st.session_state

# ----- Setup page -----
st.set_page_config(
     page_title='Streamlit monitor',
     layout="wide",
     initial_sidebar_state="expanded",
)


def main():
    # ----- Universal ----
    # TODO if sys argv[1] not provided, load file selection
    it_path = os.path.join(os.path.abspath(sys.argv[1]), 'iterations')
    main_df, it_files = load_iterations(it_path)
    dock_path = check_dock_paths(os.path.abspath(sys.argv[1]))

    st.sidebar.title('MolScore')
    # TODO button to add more runs
    st.sidebar.header(f"Run: {main_df['model'].values[0]}-{main_df['task'].values[0]}")
    if st.sidebar.button('Update'):
        main_df, it_files = update_files(it_path, it_files, main_df)

    # Radio buttons for navigation
    nav = st.sidebar.radio(label='Navigation', options=['Main', 'Multi-plot', 'MPO', 'Scaffold memory'])

    # Setup PyMol
    if 'pymol' not in SS:
        SS.pymol=None
    if 'PYMOL_PATH' in os.environ:
        if SS.pymol is None:
            pymol = PyMol(pymol_path=os.environ['PYMOL_PATH'])
            SS.pymol = pymol
        pymol = SS.pymol
    else:
        print('Export path to PyMol as \'PYMOL_PATH\' to enable sending 3D molecules directly to PyMol')
        pymol = None

    # ----- Main page -----
    if nav == 'Main':
        single_plot(main_df=main_df, SS=SS, dock_path=dock_path)

    # ----- Multi-plot -----
    if nav == 'Multi-plot':
        multi_plot(main_df=main_df, SS=SS, dock_path=dock_path)        

    # ----- MPO Graph -----
    if nav == 'MPO':
        mpo_plot(main_df=main_df, SS=SS, dock_path=dock_path)

    # ----- Scaffold memory page ----
    if nav == 'Scaffold memory':
        memory_path = os.path.join(os.path.abspath(sys.argv[1]), 'scaffold_memory.csv')
        if os.path.exists(memory_path):
            scaffold_plot(main_df=main_df, SS=SS, memory_path=memory_path, dock_path=dock_path)
        else:
            raise FileNotFoundError("It seems the scaffold memory diversity filter wasn't used for this run")
        
        
    exit = st.sidebar.button('Exit')
    if exit:
        if SS.pymol is not None:
            SS.pymol.close()
        os._exit(0)


if __name__ == '__main__':
    main()
