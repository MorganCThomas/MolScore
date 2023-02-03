import numpy as np
import pandas as pd
from itertools import cycle

import streamlit as st
from molscore.gui.utils import utils

import plotly.express as px

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource

from rdkit.Chem import AllChem as Chem


def scaffold_plot(main_df, SS, memory_path, dock_path=None, plotting='plotly', show_3D=False):

        # Initialize mviewer
        if (dock_path is not None) and show_3D:
            from molscore.gui.utils.py3Dmol_viewer import MetaViewer
            mviewer = MetaViewer()
        else:
            mviewer=None

        # ---- Process data ----
        memory = pd.read_csv(memory_path)
        memory_list = []
        max_score = 0.0
        max_size = 1
        max_step = 0.0
        # Change format into a list of records per cluster
        for i in memory.to_records():
            if i['Cluster'] == len(memory_list):
                memory_list.append({'centroid': i['Scaffold'],
                                    'members': [i['SMILES']],
                                    'score': [i['total_score']],
                                    'step': [i['step']]})
            else:
                memory_list[i['Cluster']]['members'] += [i['SMILES']]
                if len(memory_list[i['Cluster']]['members']) > max_size:
                    max_size = len(memory_list[i['Cluster']]['members'])

                memory_list[i['Cluster']]['score'] += [i['total_score']]
                if np.mean(memory_list[i['Cluster']]['score']) > max_score:
                    max_score = float(np.mean(memory_list[i['Cluster']]['score']))

                memory_list[i['Cluster']]['step'] += [i['step']]
                if np.mean(memory_list[i['Cluster']]['step']) > max_step:
                    max_step = float(np.mean(memory_list[i['Cluster']]['step']))

        # Filter options
        Score_filter = st.sidebar.slider(label='Score', min_value=0.0, max_value=max_score, value=(0.0, max_score))
        Size_filter = st.sidebar.slider(label='Size', min_value=0, max_value=max_size, value=(0, max_size))
        Step_filter = st.sidebar.slider(label='Step', min_value=0.0, max_value=max_step, value=(0.0, max_step))

        # Sort options
        sort_key = st.sidebar.selectbox(label='Sort by', options=['Score', 'Size', 'Step'], index=1)
        sort_order = st.sidebar.selectbox(label='Descending', options=[True, False], index=0)

        # Filter
        memory_list = [i for i in memory_list
                        if (Score_filter[0] <= np.mean(i['score']) <= Score_filter[1]) and
                        (Size_filter[0] <= len(i['members']) <= Size_filter[1]) and
                        (Step_filter[0] <= np.mean(i['step']) <= Step_filter[1])]

        # Sort
        if sort_key == 'Score':
            memory_list.sort(key=lambda x: np.mean(x['score']), reverse=sort_order)

        if sort_key == 'Size':
            memory_list.sort(key=lambda x: len(x['members']), reverse=sort_order)

        if sort_key == 'Step':
            memory_list.sort(key=lambda x: np.mean(x['step']), reverse=sort_order)

        # ---- Bar graph ----
        if plotting == 'plotly':
            col1, col2 = st.columns([3, 1], gap='large')
            with col1:
                st.subheader('Clusters histogram (show centroid on click)')
                selection = utils.plotly_events(px.bar(x=[i for i in range(len(memory_list))], y=[len(c['members']) for c in memory_list], width=1500, height=300, template='plotly_white'))
            with col2:
                if selection:
                    selection = [i['x'] for i in selection][0] # Click event so only one selection
                    st.subheader('Selected cluster centroid')
                    cluster = memory_list[selection]
                    st.image(utils.mol2png(Chem.MolFromSmiles(cluster['centroid'])))
                    st.text(
                        f"Cluster size: {len(cluster['members'])}\n"
                        f"Mean score: {np.mean(cluster['score']):.02f}\n"
                        f"Mean step: {np.mean(cluster['step']):.02f}"
                        )

        elif plotting == 'bokeh':
            hist = figure(plot_width=1000, plot_height=400, tooltips="""
            <div>
            @img{safe}
            </div>
            """)
            hist_data = dict(
                x=[i for i in range(len(memory_list))],
                top=[len(c['members']) for c in memory_list],
                img=[utils.mol2svg(Chem.MolFromSmiles(m['centroid']))
                        if isinstance(m['centroid'], str) else utils.mol2svg(Chem.MolFromSmiles(''))
                        for m in memory_list]
            )
            hist_source = ColumnDataSource(hist_data)
            hist.vbar(x='x',
                        width=0.5, bottom=0,
                        top='top',
                        source=hist_source)
            st.bokeh_chart(hist)
        
        else:
            raise ValueError("Unrecognized plotting library, should be plotly or bokeh")


        # ----- Plot first 20 centroids -----
        st.subheader('Cluster centroids')
        st.write('This may be the Bemis-Murcko scaffold depending on diversity filter used')
        show_no = st.number_input(label='Number of clusters to show', value=20, step=5)

        for i, (cluster, column) in enumerate(zip(memory_list[:show_no], cycle(st.columns(5)))):
            # Show image
            column.image(utils.mol2png(Chem.MolFromSmiles(cluster['centroid'] if isinstance(cluster['centroid'], str) else '')))
            column.text(f"Cluster size: {len(cluster['members'])}\n"
                        f"Mean score: {np.mean(cluster['score']):.02f}\n"
                        f"Mean step: {np.mean(cluster['step']):.02f}")
            # Set expansion
            if 'expand' not in SS: SS.expand=None
            expand = column.button(label='Expand', key=f"{cluster['centroid']}_expand")
            collapse = column.button(label='Collapse', key=f"{cluster['centroid']}_collapse")
            if collapse:
                SS.expand = None
            if expand or SS.expand == i:
                SS.expand = i
                with st.container():
                    st.subheader('Cluster members')
                    for j, (m, column2) in enumerate(zip(cluster['members'], cycle(st.columns(5)))):
                        column2.image(utils.mol2png(Chem.MolFromSmiles(m if isinstance(m, str) else '')))
                        column2.text(f"Score: {cluster['score'][j]:.02f}\n"
                                        f"Step: {cluster['step'][j]}")
        
                        # Send mol to pymol TODO

                        # Send mol to mviewer
                        if mviewer is not None: # TODO send2pymol instead
                            show_3D = column2.button('Show 3D', key=f'{j}_{m}')
                            if show_3D:
                                match_idx = main_df.index[main_df.smiles == m].tolist()
                                paths, names = find_sdfs(match_idx, main_df, dock_path)
                                mviewer.add_ligand(path=paths[0])

                    # Send all to pymol TODO

                    # Send mol to viewer
                    if mviewer is not None:
                        show_all_3D = st.button('Show all 3D', key='Memory_all')
                        if show_all_3D:
                            match_idx = main_df.index[main_df.smiles.isin(cluster['members'])].tolist()
                            paths, names = utils.find_sdfs(match_idx, main_df, dock_path)
                            for p in paths:
                                mviewer.add_ligand(path=p)
                    break

        # ----- Option to save sdf -----
        if not expand:
            with st.expander(label='Save selected'):
                out_file = st.text_input(label='File name')
                st.write(f'File name: {out_file}.sdf')
                save_all_clusters = st.button(label='Save', key='save_all_clusters')
                if save_all_clusters:
                    file_paths = []
                    mol_names = []
                    for cluster in memory_list[:show_no]:
                        match_idx = main_df.index[main_df.smiles.isin(cluster['members'])].tolist()
                        paths, names = find_sdfs(match_idx, main_df, dock_path)
                        file_paths += paths
                        names = [f'ClusterCentroid: {names[0].split(":")[1]} - {n}' for n in names]
                        mol_names += names
                    save_sdf(mol_paths=file_paths,
                                mol_names=mol_names,
                                out_name=out_file)

        # ----- Show 3D poses -----
        if show_3D:
            st.subheader('Selected 3D poses')
        
            # ---- User options -----
            # TODO convert to function? Update get to ss?
            col1, col2 = st.columns(2)
            input_structure_ss = get(key='input_structure', input_path='./', ref_path='./')
            input_structure_ss.input_path = st_file_selector(label='Input structure',
                                                            st_placeholder=col1.empty(),
                                                            path=input_structure_ss.input_path,
                                                            key='input_structure')
            input_structure = input_structure_ss.input_path
            col1.write(f"Selected: {input_structure}")
            mviewer.add_receptor(path=input_structure)

            input_structure_ss.ref_path = st_file_selector(label='Reference ligand',
                                                        st_placeholder=col2.empty(),
                                                        path=input_structure_ss.ref_path,
                                                        key='reference_ligand')
            ref_path = input_structure_ss.ref_path
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
