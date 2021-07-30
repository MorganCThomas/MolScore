import sys
import os
import gzip
import pandas as pd
import numpy as np
from itertools import cycle, chain
from glob import glob
import matplotlib.colors as mcolors

import streamlit as st
from streamlit_bokeh_events import streamlit_bokeh_events
from molscore.utils.streamlit.SessionState import get

import py3Dmol
import parmed

from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource, CustomJS, BoxSelectTool

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import MolsToGridImage, MolDraw2DSVG, MolDraw2DCairo
import base64
from io import BytesIO

# ----- Setup page -----
st.set_page_config(
     page_title='Streamlit monitor',
     layout="wide",
     initial_sidebar_state="expanded",
)


# ----- Load in iterations files -----
#@st.cache(allow_output_mutation=True)
def load_iterations(it_path):
    it_path = os.path.join(os.path.abspath(sys.argv[1]), 'iterations')
    it_files = glob(os.path.join(it_path, '*.csv'))
    main_df = pd.DataFrame()
    if len(it_files) > 0:
        it_files = sorted(it_files)
        for f in it_files:
            main_df = main_df.append(pd.read_csv(f, index_col=0, dtype={'valid': object,
                                                                        'unique': object,
                                                                        'passed_diversity_filter': object}))
    return main_df, it_files


# ----- Load in dock path if available -----
@st.cache
def check_dock_paths(path):
    subdirectories = [x for x in os.walk(os.path.abspath(path))][0][1]
    try:
        # Find dock path
        dock_sub = [d for d in subdirectories if ('Dock' in d) or ('ROCS' in d)][0]
        dock_path = os.path.join(os.path.abspath(sys.argv[1]), dock_sub)
    except (KeyError, IndexError):
        dock_path = None
    return dock_path


def update_files(path, files, df):
    # Check for new files
    check_files = glob(os.path.join(path, '*.csv'))
    if len(check_files) > 0:
        check_files = sorted(check_files)
        new_files = [f for f in check_files if (f not in files)]
        if len(new_files) > 0:
            # Append new files to df and files list
            for new_file in new_files:
                it_df = pd.read_csv(new_file, index_col=0, dtype={'valid': object, 'unique': object})
                df = df.append(it_df)
                files += [new_file]
            return df, files
        else:
            return df, files


def st_file_selector(st_placeholder, key, path='.', label='Please, select a file/folder...'):
    # get base path (directory)
    base_path = '.' if path is None or path is '' else path
    base_path = base_path if os.path.isdir(
        base_path) else os.path.dirname(base_path)
    base_path = '.' if base_path is None or base_path is '' else base_path
    # list files in base path directory
    files = os.listdir(base_path)
    #if base_path is not '.':
    files.insert(0, '..')
    files.insert(0, '.')
    selected_file = st_placeholder.selectbox(
        label=label, options=files, key=key)
    selected_path = os.path.normpath(os.path.join(base_path, selected_file))
    if selected_file is '.':
        return selected_path
    if os.path.isdir(selected_path):
        selected_path = st_file_selector(st_placeholder=st_placeholder,
                                         path=selected_path, label=label, key=key)
    return os.path.abspath(selected_path)


def mol2svg(mol):
    AllChem.Compute2DCoords(mol)
    try:
        Chem.Kekulize(mol)
    except:
        pass
    d2d = MolDraw2DSVG(200, 200)
    d2d.DrawMolecule(mol)
    d2d.FinishDrawing()
    return d2d.GetDrawingText().replace('svg:', '')


def mol2png(mol):
    AllChem.Compute2DCoords(mol)
    try:
        Chem.Kekulize(mol)
    except:
        pass
    d2d = MolDraw2DCairo(200, 200)
    d2d.DrawMolecule(mol)
    d2d.FinishDrawing()
    return d2d.GetDrawingText()


def bokeh_plot(y, main_df, *args):
    TOOLTIPS = """
    <div>
    Step_batch_idx: @ids<br>
    </div>
    """
    # @img{safe}

    if (y == 'valid') or (y == 'unique') or (y == 'passes_diversity_filter'):
        p = figure(plot_width=1000, plot_height=500)
        steps = main_df.step.unique().tolist()
        ratios = main_df.groupby('step')[y].apply(lambda x: (x == 'true').mean()).tolist()
        p.line(x=steps, y=ratios)

    else:
        data = dict(
            x=main_df.step.tolist(),
            y=main_df[y].tolist(),
            y_mean=main_df[y].rolling(window=100).mean(),
            y_median=main_df[y].rolling(window=100).median(),
            ids=(main_df.step.map(str) + "_" + main_df.batch_idx.map(str)).tolist(),
            # img=[mol2svg(m) if m else None for m in main_df.mol]
        )
        source = ColumnDataSource(data)

        # Required for callback
        source.selected.js_on_change(
            "indices",
            CustomJS(
                args=dict(source=source),
                code="""
                document.dispatchEvent(
                    new CustomEvent("BOX_SELECT", {detail: {data: source.selected.indices}})
                )
                """
            )
        )

        p = figure(plot_width=1000, plot_height=500, tooltips=TOOLTIPS)
        p.add_tools(BoxSelectTool())
        p.circle(x='x', y='y', size=8, source=source)
        p.line(x='x', y='y_mean',
               line_color='blue', legend_label='mean', source=source)
        p.line(x='x', y='y_median',
               line_color='red', legend_label='median', source=source)

    p.xaxis[0].axis_label = 'Step'
    p.yaxis[0].axis_label = y

    return p


def display_selected_data(y, selection=None):
    max_structs = 24
    structs_per_row = 4
    empty_plot = "data:image/gif;base64,R0lGODlhAQABAAAAACwAAAAAAQABAAA="
    if selection is None:
        return empty_plot
    else:
        match_idx = selection['BOX_SELECT']['data']
        st.write(main_df.iloc[match_idx])
        smis = main_df.loc[match_idx, 'smiles'].tolist()
        mols = [Chem.MolFromSmiles(smi) for smi in smis]
        name_list = list(main_df.iloc[match_idx][y])
        batch_list = [f"{step}_{batch_idx}" for step, batch_idx in main_df.loc[match_idx, ['step', 'batch_idx']].values]
        name_list = [f"{x:.02f}" if isinstance(x, float) else f"{x}" for x in name_list]
        legends = [f"{idx}\n{y}: {name}" for idx, name in zip(batch_list, name_list)]
        img = MolsToGridImage(mols[0:max_structs], molsPerRow=structs_per_row, legends=legends[0:max_structs],
                              subImgSize=(300, 300))
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        encoded_image = base64.b64encode(buffered.getvalue())
        src_str = 'data:image/png;base64,{}'.format(encoded_image.decode())
        return src_str


def display_selected_data2(y, main_df, dock_path=None, selection=None, viewer=None):

    if selection is None:
        return
    else:
        match_idx = selection['BOX_SELECT']['data']
        st.write(main_df.iloc[match_idx])
        smis = main_df.loc[match_idx, 'smiles'].tolist()
        mols = [Chem.MolFromSmiles(smi) for smi in smis]
        name_list = list(main_df.iloc[match_idx][y])
        batch_list = [f"step: {step}\nbatch_index: {batch_idx}" for step, batch_idx in main_df.loc[match_idx, ['step', 'batch_idx']].values]
        name_list = [f"{x:.02f}" if isinstance(x, float) else f"{x}" for x in name_list]
        legends = [f"{idx}\n{y}: {name}" for idx, name in zip(batch_list, name_list)]
        for mol, midx, legend, col in zip(mols, match_idx, legends, cycle(st.beta_columns(5))):
            col.image(mol2png(mol))
            col.text(legend)
            if (dock_path is not None) and (viewer is not None):
                show_3D = col.button(label='Show 3D', key=f'{legend}_3D_button')

                if show_3D:
                    # Grab best variants
                    file_paths, _ = find_sdfs([midx], main_df, dock_path)
                    viewer.add_ligand(path=file_paths[0])

        show_all_3D = st.button('Show all 3D')
        if show_all_3D:
            file_paths, _ = find_sdfs(match_idx)
            for p in file_paths:
                viewer.add_ligand(path=p)
    return


def find_sdfs(match_idxs, main_df, dock_path=None):
    # Drop duplicate smiles
    sel_smiles = main_df.loc[match_idxs, 'smiles'].drop_duplicates().tolist()
    # Find first (potentially non-matching idx of first recorded unique smiles)
    first_idxs = []
    for smi in sel_smiles:
        first_idx = main_df.loc[
            main_df['smiles'] == smi,
            ['step', 'batch_idx']].drop_duplicates().to_records(index=False)
        first_idxs.append(first_idx[0])

    # List names of matching index (drop duplicate smiles in selection)
    idx_names = main_df.loc[
        match_idxs,
        ['smiles', 'step', 'batch_idx']].drop_duplicates(subset=['smiles']).to_records(index=False)

    file_paths = [glob(os.path.join(dock_path, str(s), f'{s}_{b}*sdf*'))[0] for s, b in first_idxs]

    return file_paths, [f'Mol: {s}_{b}' for _, s, b in idx_names]


def save_sdf(mol_paths, mol_names, out_name=''):
    # Setup writer
    out_file = os.path.join(os.path.abspath(sys.argv[1]), f'{out_name}.sdf')
    writer = AllChem.SDWriter(out_file)

    for path, name in zip(mol_paths, mol_names):
        if ('.sdfgz' in path) or ('.sdf.gz' in path):
            with gzip.open(path) as rf:
                suppl = Chem.ForwardSDMolSupplier(rf, removeHs=False)
                mol = suppl.__next__() # Grab first mol
                mol.SetProp('_Name', name)
                writer.write(mol)
        elif '.sdf' in path:
            with open(path) as rf:
                suppl = Chem.ForwardSDMolSupplier(rf, removeHs=False)
                mol = suppl.__next__()
                mol.SetProp('_Name', name)
                writer.write(mol)
    writer.flush()
    writer.close()
    #st.write(f'Saved to: {out_file}')
    return


class MetaViewer(py3Dmol.view):
    def __init__(self):
        super().__init__(width=1000, height=800)
        self.removeAllModels()
        self.setBackgroundColor(0x000000)

        # Set simple parameters
        self.n_ligands = 0
        self.rec_model = None
        self.rec_path = None
        self.colors = list(mcolors.CSS4_COLORS.keys())

    def add_receptor(self, path):
        if '.pdb' in path:
            self.rec_path = path
            self.addModel(open(path, 'r').read(), 'pdb')
            self.rec_model = self.getModel()
            self.rec_model.setStyle({}, {'cartoon': {'color': '#44546A'}})

    def add_ligand(self, path, keepHs=False, color=None):
        if color is None:
            color = self.colors[self.n_ligands]
        if ('.sdfgz' in path) or ('.sdf.gz' in path):
            with gzip.open(path) as rf:
                suppl = Chem.ForwardSDMolSupplier(rf, removeHs=False)
                for i, m in enumerate(suppl):
                    if i == 0:
                        self.addModel(Chem.MolToMolBlock(m), 'mol', {'keepH': keepHs})
                        model = self.getModel()
                        model.setStyle({}, {'stick': {"colorscheme": f'{color}Carbon'}})  # #ED7D31 orange
                        self.n_ligands += 1
        elif '.sdf' in path:
            self.addModel(open(path, 'r').read(), 'sdf', {'keepH': keepHs})
            model = self.getModel()
            model.setStyle({}, {'stick': {"colorscheme": f'{color}Carbon'}})
            self.n_ligands += 1
        elif '.pdb' in path:
            self.addModel(path, 'pdb')
            model = self.getModel()
            model.setStyle({}, {'stick': {"colorscheme": f'{color}Carbon'}})
            self.n_ligands += 1
        else:
            st.write('Unknown format, ligand must be sdf or pdb')

    def add_surface(self, surface):
        rec = {'resn': ["AQD", "UNL"], 'invert': 1}
        if surface is not None:
            if surface == 'VDW':
                self.addSurface(py3Dmol.VDW, {'opacity': 0.75, 'color': 'white'}, rec)
            if surface == 'MS':
                self.addSurface(py3Dmol.MS, {'opacity': 0.75, 'color': 'white'}, rec)
            if surface == 'SAS':
                self.addSurface(py3Dmol.SAS, {'opacity': 0.75, 'color': 'white'}, rec)
            if surface == 'SES':
                self.addSurface(py3Dmol.SES, {'opacity': 0.75, 'color': 'white'}, rec)

    def label_receptor(self):
        structure = parmed.read_PDB(self.rec_path)
        self.addResLabels({'resi': [n.number for n in structure.residues]},
                          {'font': 'Arial', 'fontColor': 'white', 'showBackground': 'false',
                           'fontSize': 10})

    def get_residues(self):
        if self.rec_path is not None:
            structure = parmed.read_PDB(self.rec_path)
            return [r.number for r in structure.residues]
        else:
            return []

    def show_residue(self, number):
        self.rec_model.setStyle({'resi': number}, {'stick': {'colorscheme': 'darkslategrayCarbon'}})

    def render2st(self):
        # ----- render -----
        self.zoomTo({'model': -1})  # Zoom to last model
        #self.zoomTo()
        self.render()

        t = self.js()
        f = open('viz.html', 'w')
        f.write(t.startjs)
        f.write(t.endjs)
        f.close()

        st.components.v1.html(open('viz.html', 'r').read(), width=1200, height=800)


def main():
    # ----- Universal ----
    it_path = os.path.join(os.path.abspath(sys.argv[1]), 'iterations')
    main_df, it_files = load_iterations(it_path)
    dock_path = check_dock_paths(os.path.abspath(sys.argv[1]))

    st.sidebar.title('MolScore')
    st.sidebar.header(f"Run: {main_df['model'].values[0]}-{main_df['task'].values[0]}")
    if st.sidebar.button('Update'):
        main_df, it_files = update_files(it_path, it_files, main_df)

    # Radio buttons for navigation
    nav = st.sidebar.radio(label='Navigation', options=['Main', 'Scaffold memory'])

    # ----- Main page -----
    if nav == 'Main':
        y_axis = st.sidebar.selectbox('Plot y-axis', main_df.columns.tolist(), index=6)
        p = bokeh_plot(y_axis, main_df)
        selection = streamlit_bokeh_events(
                bokeh_plot=p,
                events="BOX_SELECT",
                key="main",
                refresh_on_update=True,
                override_height=None,
                debounce_time=0)

        mviewer = MetaViewer()

        # ----- Show selected data -----
        st.subheader('Selected structures')
        display_selected_data2(y=y_axis, main_df=main_df, dock_path=dock_path, selection=selection, viewer=mviewer)
        # ----- Add option to save sdf -----
        if (dock_path is not None) and (selection is not None):
            with st.beta_expander(label='Save selected'):
                # User input
                out_file = st.text_input(label='File name')
                st.write(f'File name: {out_file}.sdf')
                save_all_selected = st.button(label='Save', key='save_all_selected')
                if save_all_selected:
                    file_paths, mol_names = find_sdfs(selection['BOX_SELECT']['data'], main_df, dock_path)
                    save_sdf(mol_paths=file_paths,
                             mol_names=mol_names,
                             out_name=out_file)

        # ----- Show 3D poses -----
        st.subheader('Selected 3D poses')
        # ---- User options -----
        col1, col2 = st.beta_columns(2)
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

        col1, col2, col3 = st.beta_columns(3)
        surface = col1.selectbox(label='Surface', options=[None, 'VDW', 'MS', 'SAS', 'SES'])
        mviewer.add_surface(surface)
        show_residue_labels = col2.selectbox(label='Label residues', options=[True, False], index=1)
        if show_residue_labels:
            mviewer.label_receptor()
        show_residues = col3.multiselect(label='Show residues', options=mviewer.get_residues())
        _ = [mviewer.show_residue(r) for r in show_residues]

        mviewer.add_ligand(path=ref_path, color='orange')

        mviewer.render2st()

    # ----- Scaffold memory page ----
    if nav == 'Scaffold memory':
        memory_path = os.path.join(os.path.abspath(sys.argv[1]), 'scaffold_memory.csv')
        if os.path.exists(memory_path):
            if dock_path is not None:
                mviewer = MetaViewer()

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

            # Plot figure
            hist = figure(plot_width=1000, plot_height=400, tooltips="""
            <div>
            @img{safe}
            </div>
            """)
            hist_data = dict(
                x=[i for i in range(len(memory_list))],
                top=[len(c['members']) for c in memory_list],
                img=[mol2svg(Chem.MolFromSmiles(m['centroid']))
                     if isinstance(m['centroid'], str) else mol2svg(Chem.MolFromSmiles(''))
                     for m in memory_list]
            )
            hist_source = ColumnDataSource(hist_data)
            hist.vbar(x='x',
                      width=0.5, bottom=0,
                      top='top',
                      source=hist_source)
            st.bokeh_chart(hist)

            # ----- Plot first 20 centroids -----
            st.subheader('Cluster centroids (may be a scaffold)')
            show_no = st.number_input(label='Number to show', value=20, step=5)
            for i, (cluster, column) in enumerate(zip(memory_list[:show_no], cycle(st.beta_columns(5)))):
                column.image(mol2png(Chem.MolFromSmiles(cluster['centroid'] if isinstance(cluster['centroid'], str) else '')))
                column.text(f"Cluster size: {len(cluster['members'])}\n"
                            f"Mean score: {np.mean(cluster['score']):.02f}\n"
                            f"Mean step: {np.mean(cluster['step']):.02f}")
                memory_ss = get(key='memory', expand=None)
                expand = column.button(label='Expand', key=cluster['centroid'] + '_expand')
                collapse = column.button(label='Collapse', key=cluster['centroid'] + '_collapse')
                if collapse:
                    memory_ss.expand = None
                if expand or memory_ss.expand == i:
                    memory_ss.expand = i
                    with st.beta_container():
                        st.subheader('Cluster members')
                        for j, (m, column2) in enumerate(zip(cluster['members'], cycle(st.beta_columns(5)))):
                            column2.image(mol2png(Chem.MolFromSmiles(m if isinstance(m, str) else '')))
                            column2.text(f"Score: {cluster['score'][j]:.02f}\n"
                                         f"Step: {cluster['step'][j]}")
                            if dock_path is not None:
                                show_3D = column2.button('Show 3D', key=f'{j}_{m}')
                                if show_3D:
                                    match_idx = main_df.index[main_df.smiles == m].tolist()
                                    paths, names = find_sdfs(match_idx, main_df, dock_path)
                                    mviewer.add_ligand(path=paths[0])

                        if dock_path is not None:
                            show_all_3D = st.button('Show all 3D', key='Memory_all')
                            if show_all_3D:
                                match_idx = main_df.index[main_df.smiles.isin(cluster['members'])].tolist()
                                paths, names = find_sdfs(match_idx, main_df, dock_path)
                                for p in paths:
                                    mviewer.add_ligand(path=p)
                        break
            # ----- Option to save sdf -----
            if not expand:
                with st.beta_expander(label='Save selected'):
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
            if dock_path is not None:
                st.subheader('Selected 3D poses')
                # ---- User options -----
                col1, col2 = st.beta_columns(2)
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

                col1, col2, col3 = st.beta_columns(3)
                surface = col1.selectbox(label='Surface', options=[None, 'VDW', 'MS', 'SAS', 'SES'])
                mviewer.add_surface(surface)
                show_residue_labels = col2.selectbox(label='Label residues', options=[True, False], index=1)
                if show_residue_labels:
                    mviewer.label_receptor()
                show_residues = col3.multiselect(label='Show residues', options=mviewer.get_residues())
                _ = [mviewer.show_residue(r) for r in show_residues]

                mviewer.add_ligand(path=ref_path, color='orange')

                mviewer.render2st()

    exit = st.sidebar.button('Exit')
    if exit:
        os._exit(0)


if __name__ == '__main__':
    main()
