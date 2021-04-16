import sys
import os
import gzip
import pandas as pd
import numpy as np
from itertools import cycle
from glob import glob
import matplotlib.colors as mcolors

import streamlit as st
from streamlit_bokeh_events import streamlit_bokeh_events
from molscore.utils.streamlit.SessionState import get

import py3Dmol

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
it_path = os.path.join(os.path.abspath(sys.argv[1]), 'iterations')
it_files = glob(os.path.join(it_path, '*.csv'))
main_df = pd.DataFrame()
if len(it_files) > 0:
    it_files = sorted(it_files)
    for f in it_files:
        main_df = main_df.append(pd.read_csv(f, index_col=0, dtype={'valid': object,
                                                                    'unique': object,
                                                                    'passed_diversity_filter': object}))


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
                #it_df['mol'] = [Chem.MolFromSmiles(s) if Chem.MolFromSmiles(s) else None for s in main_df.smiles]
                #_ = [AllChem.Compute2DCoords(m) for m in it_df.mol if m]
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


def bokeh_plot(y, *args):
    # Bring in global values
    global main_df

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


def add_ligand(viewer, path, keepHs=False, colorscheme="orangeCarbon"):
    if ('.sdfgz' in path) or ('.sdf.gz' in path):
        with gzip.open(path) as rf:
            suppl = Chem.ForwardSDMolSupplier(rf, removeHs=False)
            for i, m in enumerate(suppl):
                if i == 0:
                    viewer.addModel(Chem.MolToMolBlock(m), 'mol', {'keepH': keepHs})
                    model = viewer.getModel()
                    model.setStyle({}, {'stick': {"colorscheme": colorscheme}})  # #ED7D31 orange
    elif '.sdf' in path:
        viewer.addModel(open(path, 'r').read(), 'sdf', {'keepH': keepHs})
        model = viewer.getModel()
        model.setStyle({}, {'stick': {"colorscheme": colorscheme}})
    elif '.pdb' in path:
        viewer.addModel(path, 'pdb')
        model = viewer.getModel()
        model.setStyle({}, {'stick': {"colorscheme": colorscheme}})
    else:
        st.write('Unknown format, ligand must be sdf or pdb')
        pass


def main():
    # ----- Universal ----
    global main_df
    global it_path
    global it_files
    st.sidebar.title('MolScore')
    st.sidebar.header(f"Run: {main_df['model'].values[0]}-{main_df['task'].values[0]}")
    if st.sidebar.button('Update'):
        main_df, it_files = update_files(it_path, it_files, main_df)

    # Radio buttons for navigation
    nav = st.sidebar.radio(label='Navigation', options=['Main', 'Scaffold memory', '3D'])

    # ----- Main page -----
    if nav == 'Main':
        y_axis = st.sidebar.selectbox('Plot y-axis', main_df.columns.tolist(), index=6)
        p = bokeh_plot(y_axis)
        selection = streamlit_bokeh_events(
                bokeh_plot=p,
                events="BOX_SELECT",
                key="main",
                refresh_on_update=True,
                override_height=None,
                debounce_time=0)

        st.subheader('Selected structures')
        st.image(display_selected_data(y=y_axis, selection=selection))

    # ----- Scaffold memory page ----
    if nav == 'Scaffold memory':
        memory_path = os.path.join(os.path.abspath(sys.argv[1]), 'scaffold_memory.csv')
        if os.path.exists(memory_path):
            memory = pd.read_csv(memory_path)
            memory_list = []
            max_score = 0
            max_size = 0
            max_step = 0
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
            sort_key = st.sidebar.selectbox(label='Sort by', options=['Score', 'Size', 'Step'])
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
                img=[mol2svg(Chem.MolFromSmiles(m['centroid'])) for m in memory_list]
            )
            hist_source = ColumnDataSource(hist_data)
            hist.vbar(x='x',
                      width=0.5, bottom=0,
                      top='top',
                      source=hist_source)
            st.bokeh_chart(hist)

            # Plot first 20 centroids
            st.subheader('Cluster centroids (may be a scaffold)')
            show_no = st.number_input(label='Number to show', value=20, step=5)
            for cluster, column in zip(memory_list[:show_no], cycle(st.beta_columns(5))):
                column.image(mol2png(Chem.MolFromSmiles(cluster['centroid'])))
                column.text(f"Cluster size: {len(cluster['members'])}")
                column.text(f"Mean score: {np.mean(cluster['score']):.02f}")
                column.text(f"Mean step: {np.mean(cluster['step']):.02f}")
                expand = column.button(label='Expand', key=cluster['centroid'] + '_expand')
                if expand:
                    with st.beta_container():
                        st.subheader('Cluster members')
                        collapse = st.button(label='Collapse', key=cluster['centroid'] + '_collapse')
                        if collapse:
                            expand = False
                        for i, (m, column2) in enumerate(zip(cluster['members'], cycle(st.beta_columns(5)))):
                            column2.image(mol2png(Chem.MolFromSmiles(m)))
                            column2.text(f"Score: {cluster['score'][i]:.02f}")
                            column2.text(f"Step: {cluster['step'][i]}")
                        break

    # ----- 3D docked pose -----
    if nav == '3D':
        input_structure_ss = get(key='input_structure', input_path='./', ref_path='./')
        input_structure_ss.input_path = st_file_selector(label='Input structure',
                                                         st_placeholder=st.empty(),
                                                         path=input_structure_ss.input_path,
                                                         key='input_structure')
        input_structure = input_structure_ss.input_path
        st.write(f"Selected: {input_structure}")

        input_structure_ss.ref_path = st_file_selector(label='Reference ligand',
                                                       st_placeholder=st.empty(),
                                                       path=input_structure_ss.ref_path,
                                                       key='reference_ligand')
        ref_path = input_structure_ss.ref_path
        st.write(f"Selected: {ref_path}")

        if (os.path.exists(input_structure)) and ('.pdb' in os.path.basename(input_structure)):
            # ---- Show plot to allow selection ----
            y_axis = st.sidebar.selectbox('Plot y-axis', main_df.columns.tolist(), index=6)
            p = bokeh_plot(y_axis)
            selection = streamlit_bokeh_events(
                bokeh_plot=p,
                events="BOX_SELECT",
                key="main",
                refresh_on_update=True,
                override_height=None,
                debounce_time=0)
            if selection is not None:
                match_idx = selection['BOX_SELECT']['data']
                st.write(main_df.iloc[match_idx])

            # ---- User options -----
            st.subheader('User options')
            surface = st.selectbox(label='Surface', options=[None, 'VDW', 'MS', 'SAS', 'SES'])

            # ---- Set up viewer -----
            viewer = py3Dmol.view(width=1000, height=800)
            viewer.removeAllModels()
            viewer.setBackgroundColor(0x000000)
            #viewer.setCameraParameters({'fov': '50', 'z': 150});

            # ---- receptor ----
            viewer.addModel(open(input_structure, 'r').read(), 'pdb')
            rec_model = viewer.getModel()
            rec = {'resn': ["AQD", "UNL"], 'invert': 1}
            rec_model.setStyle({}, {'cartoon': {'color': '#44546A'}})
            rec_model.addResLabels({'res': [i for i in range(100, 110)]}, {'fontcolor': 'white'})
            if surface is not None:
                if surface == 'VDW':
                    viewer.addSurface(py3Dmol.VDW, {'opacity': 0.75, 'color': 'white'}, rec)
                if surface == 'MS':
                    viewer.addSurface(py3Dmol.MS, {'opacity': 0.75, 'color': 'white'}, rec)
                if surface == 'SAS':
                    viewer.addSurface(py3Dmol.SAS, {'opacity': 0.75, 'color': 'white'}, rec)
                if surface == 'SES':
                    viewer.addSurface(py3Dmol.SES, {'opacity': 0.75, 'color': 'white'}, rec)

            # ----- ref ligand -----
            add_ligand(viewer=viewer, path=ref_path)

            # ----- Add selected -----
            if selection is not None:
                match_idx = selection['BOX_SELECT']['data']
                subdirectories = [x for x in os.walk(os.path.abspath(sys.argv[1]))][0][1]
                try:
                    # Find dock path
                    dock_sub = [d for d in subdirectories if 'Dock' in d][0]
                    dock_path = os.path.join(os.path.abspath(sys.argv[1]), dock_sub)
                    # Grab best variants
                    bv_col = [c for c in main_df.columns if 'best_variant' in c][0]
                    best_variants = main_df.loc[match_idx, bv_col].tolist()
                    # For each best variant ..
                    for bv, c in zip(best_variants, cycle(mcolors.CSS4_COLORS.keys())):
                        step, bidx = bv.split('_')
                        bv_qpath = os.path.join(dock_path, step, f'{bv}*')
                        bv_path = glob(bv_qpath)[0]  # Return first (should be only) hit
                        add_ligand(viewer, path=bv_path, colorscheme=f"{c}Carbon")

                except IndexError:
                    st.write('No subdirectory containing \'Dock\' found')

            # ----- render -----
            viewer.zoomTo({'model': 1})  # Zoom to last model
            viewer.render()

            t = viewer.js()
            f = open('viz.html', 'w')
            f.write(t.startjs)
            f.write(t.endjs)
            f.close()

            st.components.v1.html(open('viz.html', 'r').read(), width=1200, height=800)
        else:
            st.write('No receptor file found')


if __name__ == '__main__':
    main()
