import sys
import os
import gzip
import tempfile
import pandas as pd
import numpy as np
from itertools import cycle, chain
from glob import glob
import matplotlib.colors as mcolors
from scipy.stats import gmean as geometricmean
from sklearn.preprocessing import MinMaxScaler

import streamlit as st
from streamlit_bokeh_events import streamlit_bokeh_events
from streamlit_plotly_events import plotly_events

from molscore.gui.utils.pymol_wrapper import PyMol

import plotly.express as px
from plotly.subplots import make_subplots

try:
    from bokeh.plotting import figure, show, output_file, gridplot
    from bokeh.models import ColumnDataSource, CustomJS, BoxSelectTool
except ImportError:
    pass

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import MolsToGridImage, MolDraw2DSVG, MolDraw2DCairo
import base64
from io import BytesIO

# ----- Load in iterations files -----
#@st.cache(allow_output_mutation=True)
def load_iterations(it_path):
    it_path = os.path.join(os.path.abspath(sys.argv[1]), 'iterations')
    it_files = glob(os.path.join(it_path, '*.csv'))
    main_df = pd.DataFrame()
    if len(it_files) > 0:
        it_files = sorted(it_files)
        for f in it_files:
            main_df = pd.concat([main_df, pd.read_csv(f, index_col=0, dtype={'valid': object})], axis=0)
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


def st_file_selector(st_placeholder, key, path='.', label='Please, select a file/folder...', counter=1):
    """
    Code for a file selector widget which remembers where you are...
    """
    # get base path (directory)
    base_path = '.' if (path is None) or (path == '') else path
    base_path = base_path if os.path.isdir(
        base_path) else os.path.dirname(base_path)
    base_path = '.' if (base_path is None) or (base_path == '') else base_path
    # list files in base path directory
    files = os.listdir(base_path)
    files.insert(0, '..')
    files.insert(0, '.')
    selected_file = st_placeholder.selectbox(
        label=label, options=files, key=key)
    selected_path = os.path.normpath(os.path.join(base_path, selected_file))
    if selected_file == '.':
        return selected_path
    if os.path.isdir(selected_path):
        # ----
        counter += 1
        key = key + str(counter)
        # ----
        selected_path = st_file_selector(st_placeholder=st_placeholder,
                                            path=selected_path, label=label,
                                            key=key, counter=counter)
    return os.path.abspath(selected_path)


def mol2svg(mol):
    try:
        AllChem.Compute2DCoords(mol)
        try:
            Chem.Kekulize(mol)
        except:
            pass
    except:
        mol = Chem.MolFromSmiles('')
        AllChem.Compute2DCoords(mol)
    d2d = MolDraw2DSVG(200, 200)
    d2d.DrawMolecule(mol)
    d2d.FinishDrawing()
    return d2d.GetDrawingText().replace('svg:', '')


def mol2png(mol):
    try:
        AllChem.Compute2DCoords(mol)
        try:
            Chem.Kekulize(mol)
        except:
            pass
    except:
        mol = Chem.MolFromSmiles('')
        AllChem.Compute2DCoords(mol)
    d2d = MolDraw2DCairo(200, 200)
    d2d.DrawMolecule(mol)
    d2d.FinishDrawing()
    return d2d.GetDrawingText()


def bokeh_plot(y, main_df, size=(1000, 500), *args):
    TOOLTIPS = """
    <div>
    Step_batch_idx: @ids<br>
    </div>
    """
    # @img{safe}

    if y == 'valid':
        p = figure(plot_width=size[0], plot_height=size[1])
        steps = main_df.step.unique().tolist()
        ratios = main_df.groupby('step')[y].apply(lambda x: (x == 'true').mean()).tolist()
        p.line(x=steps, y=ratios)

    elif (y == 'unique') or (y == 'passes_diversity_filter'):
        p = figure(plot_width=size[0], plot_height=size[1])
        steps = main_df.step.unique().tolist()
        ratios = main_df.groupby('step')[y].mean().tolist()
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

        p = figure(plot_width=size[0], plot_height=size[1], tooltips=TOOLTIPS)
        p.add_tools(BoxSelectTool())
        p.circle(x='x', y='y', size=8, source=source)
        p.line(x='x', y='y_mean',
                line_color='blue', legend_label='mean', source=source)
        p.line(x='x', y='y_median',
                line_color='red', legend_label='median', source=source)

    p.xaxis[0].axis_label = 'Step'
    p.yaxis[0].axis_label = y

    return p

@st.cache
def plotly_plot(y, main_df, size=(1000, 500), x='step'):
    # If validity show line plot
    if y == 'valid':
        steps = main_df.step.unique().tolist()
        ratios = main_df.groupby('step')[y].apply(lambda x: (x == 'true').mean()).tolist()
        fig = px.line(x=steps, y=ratios, range_y=(0, 1), template='plotly_white')
    elif (y == 'unique') or (y == 'passes_diversity_filter'):
        steps = main_df.step.unique().tolist()
        ratios = main_df.groupby('step')[y].mean().tolist()
        fig = px.line(x=steps, y=ratios, range_y=(0, 1), range_x=(0, None), template='plotly_white')
    else:
        x == 'index'
        main_df = main_df.reset_index(drop=False)
        fig = px.scatter(data_frame=main_df, x=x, y=y, trendline='rolling', trendline_options=dict(function='median', window=100), trendline_color_override='black', template='plotly_white')
        #fig.update_traces(hovertemplate=None)
        #fig.update_layout(hovermode="x unified")
    return fig


def display_selected_data_old(y, selection=None):
    raise DeprecationWarning
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


def display_selected_data(y, main_df, dock_path=None, selection=None, viewer=None, pymol=None):

    if selection is None:
        return
    else:
        match_idx = selection
        if len(match_idx) > 100:
            st.write("Warning: Limiting display to first 100")
            match_idx = match_idx[:100]
        # Subset df
        st.write(main_df.iloc[match_idx])
        smis = main_df.loc[match_idx, 'smiles'].tolist()
        mols = [Chem.MolFromSmiles(smi) for smi in smis]
        name_list = list(main_df.iloc[match_idx][y])
        batch_list = [f"step: {step}\nbatch_index: {batch_idx}" for step, batch_idx in main_df.loc[match_idx, ['step', 'batch_idx']].values]
        name_list = [f"{x:.02f}" if isinstance(x, float) else f"{x}" for x in name_list]
        legends = [f"{idx}\n{y}: {name}" for idx, name in zip(batch_list, name_list)]
        # Plot molecule graphs in columns
        for mol, midx, legend, col in zip(mols, match_idx, legends, cycle(st.columns(5))):
            col.image(mol2png(mol))
            col.text(legend)
            if dock_path is not None:
                if viewer is not None:
                    show_3D = col.button(label='Show 3D', key=f'{legend}_3D_button')
                    if show_3D:
                        # Grab best variants
                        file_paths, _ = find_sdfs([midx], main_df, dock_path)
                        viewer.add_ligand(path=file_paths[0])
                
                if pymol is not None:
                    show_pymol = col.button(label='send2pymol', key=f'{legend}_pymol_button')
                    if show_pymol:
                        file_paths, _ = find_sdfs([midx], main_df, dock_path)
                        idx = '-'.join(legend.split("\n")[:2]).replace(' ', '')
                        file_path = file_paths[0]
                        if '.sdfgz' in file_path:
                            new_file_path = os.path.join(os.path.dirname(file_path), os.path.basename(file_path).split(".")[0] + '.sdf.gz')
                            os.system(f'cp {file_path} {new_file_path} && gunzip -f {new_file_path}')
                            file_path = os.path.join(os.path.dirname(file_path), os.path.basename(file_path).split(".")[0] + '.sdf')
                        if '.sdf.gz' in file_path:
                            os.system(f'gunzip -f {file_path}')
                            file_path = os.path.join(os.path.dirname(file_path), os.path.basename(file_path).split(".")[0] + '.sdf')
                        pymol(f'load {file_path}, {idx}')
                
        if (dock_path is not None) and (viewer is not None):
            show_all_3D = st.button('Show all 3D')
            if show_all_3D:
                file_paths, _ = find_sdfs(match_idx, main_df, dock_path)
                for p in file_paths:
                    viewer.add_ligand(path=p)
    return


def find_sdfs(match_idxs, main_df, dock_path, gz_only=False):
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

    if gz_only:
        file_paths = [glob(os.path.join(dock_path, str(s), f'{s}_{b}-*sdfgz'))[0] for s, b in first_idxs]
    else:
        file_paths = [glob(os.path.join(dock_path, str(s), f'{s}_{b}-*sdf*'))[0] for s, b in first_idxs]

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