import streamlit as st
from molscore.gui.utils import utils


def single_plot(main_df, SS, dock_path=None, plotting='plotly', show_3D=False):
    """ The streamlit monitors main page """

    # Initialize mviewer
    if (dock_path is not None) and show_3D:
        from molscore.gui.utils.py3Dmol_viewer import MetaViewer
        mviewer = MetaViewer()
    else:
        mviewer=None

    # ----- Show central plot -----
    col1, col2 = st.columns(2, gap='large')
    with col1:
        x_axis = st.selectbox('Plot x-axis', ['step', 'index'], index=0)
    with col2:
        y_axis = st.selectbox('Plot y-axis', main_df.columns.tolist(), index=6)

    if plotting == 'plotly':
        fig = utils.plotly_plot(y_axis, main_df, x=x_axis)
        selection = utils.plotly_events(fig, click_event=False, select_event=True)
        selection = [i['pointNumber'] for i in selection]
    elif plotting == 'bokeh':
        p = bokeh_plot(y_axis, main_df)
        st.bokeh_chart(p)
        selection = streamlit_bokeh_events(
                bokeh_plot=p,
                events="BOX_SELECT",
                key="main",
                refresh_on_update=True,
                override_height=None,
                debounce_time=0)
        selection = selection['BOX_SELECT']['data']
    else:
        raise ValueError("Unrecognized plotting library, should be plotly or bokeh")

    # ----- Show selected data -----
    st.subheader('Selected structures')
    utils.display_selected_data(
        y=y_axis,
        main_df=main_df,
        dock_path=dock_path,
        selection=selection,
        viewer=None,
        pymol=SS.pymol
        )
    
    # ----- Add option to save sdf -----
    if (dock_path is not None) and (selection is not None):
        with st.expander(label='Save selected'):
            # User input
            out_file = st.text_input(label='File name')
            st.write(f'File name: {out_file}.sdf')
            save_all_selected = st.button(label='Save', key='save_all_selected')
            if save_all_selected:
                file_paths, mol_names = find_sdfs(selection, main_df, dock_path)
                save_sdf(
                    mol_paths=file_paths,
                    mol_names=mol_names,
                    out_name=out_file
                    )

    # ----- Show 3D poses (no longer supported in python=3.10) -----
    if show_3D:
        st.subheader('Selected 3D poses')

        # ---- User options -----
        col1, col2 = st.columns(2)
        if 'main_structure_path' not in SS: SS.main_structure_path = './'
        if 'main_ref_path' not in SS: SS.main_ref_path = './'
        #input_structure_SS = get(key='input_structure', input_path='./', ref_path='./')
        SS.main_structure_path = st_file_selector(
            label='Input structure',
            st_placeholder=col1.empty(),
            path=SS.main_structure_path,
            key='main_structure'
            )
        input_structure = SS.main_structure_path
        col1.write(f"Selected: {input_structure}")
        mviewer.add_receptor(path=input_structure)

        SS.main_ref_path = st_file_selector(
            label='Reference ligand',
            st_placeholder=col2.empty(),
            path=SS.main_ref_path,
            key='main_ref'
            )
        ref_path = SS.main_ref_path
        col2.write(f"Selected: {ref_path}")

        col1, col2, col3 = st.columns(3)
        surface = col1.selectbox(label='Surface', options=[None, 'VDW', 'MS', 'SAS', 'SES'])
        mviewer.add_surface(surface)
        show_residue_labels = col2.selectbox(label='Label residues', options=[True, False], index=1)
        if show_residue_labels:
            mviewer.label_receptor()
        show_residues = col3.multiselect(label='Show residues', options=mviewer.get_residues())
        _ = [mviewer.show_residue(r) for r in show_residues]

        mviewer.add_ligand(path=ref_path, color='orange')

        mviewer.render2st()
