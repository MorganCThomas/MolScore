import sys
import os
import pandas as pd
from glob import glob

import streamlit as st
from streamlit_bokeh_events import streamlit_bokeh_events

from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource, CustomJS, BoxSelectTool

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import MolsToGridImage, MolDraw2DSVG
import base64
from io import BytesIO

# Load in iterations files
it_path = os.path.join(os.path.abspath(sys.argv[1]), 'iterations')
it_files = glob(os.path.join(it_path, '*.csv'))
main_df = pd.DataFrame()
if len(it_files) > 0:
    it_files = sorted(it_files)
    for f in it_files:
        main_df = main_df.append(pd.read_csv(f, index_col=0, dtype={'valid': object, 'unique': object}))
    main_df['mol'] = [Chem.MolFromSmiles(s) if Chem.MolFromSmiles(s) else None for s in main_df.smiles]
    _ = [AllChem.Compute2DCoords(m) for m in main_df.mol if m]


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
                it_df['mol'] = [Chem.MolFromSmiles(s) if Chem.MolFromSmiles(s) else None for s in main_df.smiles]
                _ = [AllChem.Compute2DCoords(m) for m in it_df.mol if m]
                df = df.append(it_df)
                files += [new_file]
            return df, files
        else:
            return df, files

st.title('MolScore')
st.header(f"Run: {main_df['model'].values[0]}-{main_df['task'].values[0]}")

if st.button('Update'):
    main_df, it_files = update_files(it_path, it_files, main_df)

y_axis = st.selectbox('y-axis', main_df.columns.tolist(), index=6)


def bokeh_plot(y, *args):
    # Bring in global values
    global main_df

    TOOLTIPS = """
    <div>
    Step_batch_idx: @ids<br>
    @img{safe}
    </div>
    """

    def mol2svg(mol):
        try:
            Chem.Kekulize(mol)
        except:
            pass
        d2d = MolDraw2DSVG(200, 200)
        d2d.DrawMolecule(mol)
        d2d.FinishDrawing()
        return d2d.GetDrawingText().replace('svg:', '')

    if (y == 'valid') or (y == 'unique') or (y == 'passes_diversity_filter'):
        p = figure(plot_width=600, plot_height=300)
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
            img=[mol2svg(m) if m else None for m in main_df.mol]
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

        p = figure(plot_width=600, plot_height=300, tooltips=TOOLTIPS)
        p.add_tools(BoxSelectTool())
        p.circle(x='x', y='y', size=8, source=source)
        p.line(x='x', y='y_mean',
               line_color='blue', legend_label='mean', source=source)
        p.line(x='x', y='y_median',
               line_color='red', legend_label='median', source=source)
               
    p.xaxis[0].axis_label = 'Step'
    p.yaxis[0].axis_label = y

    return p


p = bokeh_plot(y_axis)

selection = streamlit_bokeh_events(
        bokeh_plot=p,
        events="BOX_SELECT",
        key="main",
        refresh_on_update=True,
        override_height=None,
        debounce_time=0)

st.subheader('Selected structures')


def display_selected_data(y, selection=None):
    max_structs = 24
    structs_per_row = 4
    empty_plot = "data:image/gif;base64,R0lGODlhAQABAAAAACwAAAAAAQABAAA="
    if selection is None:
        return empty_plot
    else:
        match_idx = selection['BOX_SELECT']['data']
        st.write(main_df.iloc[match_idx])
        mols = main_df.loc[match_idx, 'mol'].tolist()
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


st.image(display_selected_data(y=y_axis, selection=selection))
