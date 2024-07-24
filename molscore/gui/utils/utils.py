import gzip
import os
import re
import warnings
from glob import glob
from itertools import cycle
from tempfile import NamedTemporaryFile
from time import sleep
from typing import Union

import pandas as pd
import plotly.express as px
import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import MolDraw2DCairo, MolDraw2DSVG

from molscore.scoring_functions.utils import get_mol

warnings.simplefilter(action="ignore", category=FutureWarning)


# ----- Load & update data -----
def load(input_dir: os.PathLike, latest_idx: int = 0):
    """
    For an input directory load iteration files (if not then load score)
    """
    it_path = os.path.join(input_dir, "iterations")
    scores_path = os.path.join(input_dir, "scores.csv")

    if os.path.exists(scores_path):
        df = pd.read_csv(scores_path, index_col=0, dtype={"valid": object})
    
    elif os.path.exists(it_path):
        df = pd.DataFrame()
        it_files = sorted(glob(os.path.join(it_path, "*.csv")))
        if len(it_files) > latest_idx:
            for f in it_files[latest_idx:]:
                df = pd.concat(
                    [df, pd.read_csv(f, index_col=0, dtype={"valid": object})], axis=0
                )
            latest_idx = len(it_files)

    else:
        raise FileNotFoundError(
            f"Could not find iterations directory or scores.csv for {os.path.basename(input_dir)}"
        )

    # Add run and dock_path as columns
    df["run"] = [os.path.basename(input_dir)] * len(df)
    df["dock_path"] = [check_dock_paths(input_dir)] * len(df)

    return df, latest_idx


def update(SS):
    """
    Update data for current inputs, mutates dataframe in session
    """
    for i, (curr_dir, curr_idx) in enumerate(zip(SS.input_dirs, SS.input_latest)):
        # Load new iterations
        df, new_idx = load(input_dir=curr_dir, latest_idx=curr_idx)
        # Update latest idx
        SS.input_latest[i] = new_idx
        # First update mol index
        df.reset_index(inplace=True)
        df.rename(
            columns={"index": "idx"}, inplace=True
        )  # Index should carry between iteration as index col is read in
        # Rename run 
        run_name = df.run.unique()[0]
        df.loc[df.run == run_name, "run"] = SS.rename_map[run_name]
        # Add to main_df
        SS.main_df = pd.concat([SS.main_df, df], axis=0, ignore_index=True)
        # Sort
        SS.main_df.sort_values(by=["run", "idx"])


def rename_runs(SS):
    assert len(set(list(SS.rename_map.values()))) == len(
        set(list(SS.rename_map.values()))
    ), "New names must be unique"
    for k, v in SS.rename_map.items():
        SS.main_df.loc[SS.main_df.run == k, "run"] = v


def delete_run(SS, run):
    input_dir = SS.main_df.loc[SS.main_df.run == run, "input_dir"].unique()[0]
    SS.main_df = SS.main_df.loc[SS.main_df.run != run]
    # Remove input_dir and latest
    SS.input_latest.pop(SS.input_dirs.index(input_dir))
    SS.input_dirs.remove(input_dir)
    # Correct for empty
    if len(SS.main_df) == 0:
        SS.main_df = None


def check_dock_paths(input_path):
    subdirectories = [x for x in os.walk(os.path.abspath(input_path))][0][1]
    dock_paths = []
    for subd in subdirectories:
        if re.search("Dock|ROCS|Align3D", subd):
            dock_paths.append(os.path.join(input_path, subd))
    if dock_paths:
        return dock_paths


# ----- Plotting molecules -----
def mol2svg(mol):
    mol = get_mol(mol)
    try:
        AllChem.Compute2DCoords(mol)
        try:
            Chem.Kekulize(mol)
        except Exception:
            pass
    except Exception:
        mol = Chem.MolFromSmiles("")
        AllChem.Compute2DCoords(mol)
    d2d = MolDraw2DSVG(200, 200)
    d2d.DrawMolecule(mol)
    d2d.FinishDrawing()
    return d2d.GetDrawingText().replace("svg:", "")


def mol2png(mol):
    mol = get_mol(mol)
    try:
        AllChem.Compute2DCoords(mol)
        try:
            Chem.Kekulize(mol)
        except Exception:
            pass
    except Exception:
        mol = Chem.MolFromSmiles("")
        AllChem.Compute2DCoords(mol)
    d2d = MolDraw2DCairo(200, 200)
    d2d.DrawMolecule(mol)
    d2d.FinishDrawing()
    return d2d.GetDrawingText()


def display_selected_data(
    main_df,
    key,
    y: Union[str, list, None] = None,
    selection=None,
    dock_path=False,
    viewer=None,
    pymol=None,
):
    if isinstance(y, str):
        y = [y]
    if selection is None:
        return
    else:
        match_idx = list(set(selection))
        if len(match_idx) > 100:
            st.write("Warning: Limiting display to first 100")
            match_idx = match_idx[:100]
        # Subset df
        st.write(main_df.iloc[match_idx])
        smis = main_df.iloc[match_idx]["smiles"].tolist()
        leg_cols = (
            ["run", "step", "batch_idx"] + y if y else ["run", "step", "batch_idx"]
        )
        tdf = main_df.iloc[match_idx][leg_cols]
        legends = [
            "\n".join(
                [
                    f"{k}: {v:.02f}" if isinstance(v, float) else f"{k}: {v}"
                    for k, v in rec.items()
                ]
            )
            for rec in tdf.to_dict("records")
        ]
        # Plot molecule graphs in columns
        for smi, midx, legend, col in zip(
            smis, match_idx, legends, cycle(st.columns(5))
        ):
            col.image(mol2png(smi))
            col.text(legend)
            if dock_path:
                if viewer is not None:
                    show_3D = col.button(
                        label="Show3D", key=f"{key} {legend} 3D_button"
                    )
                    if show_3D:
                        # Grab best variants
                        file_paths, _ = find_sdfs([midx], main_df)
                        for p in file_paths:
                            viewer.add_ligand(path=p)

                if pymol is not None:
                    if col.button(
                        label="Send2PyMol", key=f"{key} {legend} pymol_button"
                    ):
                        file_paths, names = find_sdfs([midx], main_df)
                        for p, n in zip(file_paths, names):
                            send2pymol(name=n, path=p, pymol=pymol)

        if dock_path:
            if viewer is not None:
                if st.button("ShowAll3D", key=f"All3D_{key}"):
                    paths, names = find_sdfs(match_idx, main_df)
                    for p in paths:
                        viewer.add_ligand(path=p)
            if pymol is not None:
                if st.button("SendAll2Pymol", key=f"AllPymol_{key}"):
                    paths, names = find_sdfs(match_idx, main_df)
                    for p, n in zip(paths, names):
                        send2pymol(name=n, path=p, pymol=pymol)

    return


# ---- Exporting -----
def send2pymol(name, path, pymol, col=None):
    if path is None:
        return
    # Load molecule
    if path.endswith("gz"):
        with gzip.open(path) as f:
            mol = next(Chem.ForwardSDMolSupplier(f, removeHs=False))
    else:
        mol = next(Chem.ForwardSDMolSupplier(path, removeHs=False))
    # Save it to tempfile
    if mol:
        with NamedTemporaryFile(mode="w+t", suffix=".sdf") as tfile:
            writer = AllChem.SDWriter(tfile.name)
            writer.write(mol)
            writer.flush()
            writer.close()
            pymol(f"load {tfile.name}, {name}")
            sleep(0.1)  # Give PyMol one hot 100th
    else:
        if col is not None:
            col.write("RDKit error parsing molecule")


def _find_sdf(query_dir, step, batch_idx):
    if query_dir is None:
        return
    step = int(step)
    batch_idx = int(batch_idx)
    # Search for an sdf file
    possible_files = glob(
        os.path.join(query_dir, str(step), f"{step}_{batch_idx}-*.sd*")
    )
    # Try without variant
    if len(possible_files) == 0:
        possible_files = glob(
            os.path.join(query_dir, str(step), f"{step}_{batch_idx}*.sd*")
        )
    # Try another subdirectory
    if len(possible_files) == 0:
        possible_files = glob(
            os.path.join(query_dir, str(step), f"{step}_{batch_idx}*", "*.sd*")
        )
    # Return first match (should be only match)
    if len(possible_files) > 1:
        # Likely different formats
        if any([f.endswith(".sdf") for f in possible_files]):
            return [f for f in possible_files if f.endswith(".sdf")][0]
        else:
            print(f"Ambiguous file {possible_files}")
            return possible_files[0]
    if len(possible_files) == 1:
        return possible_files[0]


def find_sdfs(match_idxs, main_df):
    # Drop duplicate smiles per run
    sel_smiles = (
        main_df.loc[match_idxs, ["run", "smiles"]]
        .drop_duplicates()
        .to_records(index=False)
    )

    # List names of matching index (drop duplicate smiles in selection)
    idx_names = (
        main_df.loc[match_idxs, ["run", "smiles", "step", "batch_idx"]]
        .drop_duplicates(subset=["run", "smiles"])
        .to_records(index=False)
    )

    # Find first (potentially non-matching idx of first recorded unique smiles)
    first_idxs = []
    for run, smi in sel_smiles:
        first_idx = (
            main_df.loc[
                (main_df.run == run) & (main_df["smiles"] == smi),
                ["run", "step", "batch_idx", "dock_path"],
            ]
            .drop_duplicates(subset=["run", "step", "batch_idx"])
            .to_records(index=False)
        )
        first_idxs.append(first_idx[0])

    # Get file paths
    file_paths = []
    names = []
    for (_, s, bi, dps), (_, _, og_s, og_b) in zip(first_idxs, idx_names):
        for dp in dps:
            file_paths.append(_find_sdf(query_dir=dp, step=s, batch_idx=bi))
            dock_path_prefix = os.path.basename(dp).split("_")[0]
            names.append(f"Mol: {og_s}_{og_b} - {dock_path_prefix}")
    return file_paths, names


def save_sdf(mol_paths, mol_names, out_file):
    # Setup writer
    writer = AllChem.SDWriter(out_file)
    for path, name in zip(mol_paths, mol_names):
        if path:
            if path.endswith("gz"):
                with gzip.open(path) as rf:
                    suppl = Chem.ForwardSDMolSupplier(
                        rf, removeHs=False, sanitize=False
                    )
                    mol = suppl.__next__()
                    if mol:
                        mol.SetProp("_Name", name)
                        writer.write(mol)
            elif (".sdf" in path) or (".sd" in path):
                suppl = Chem.ForwardSDMolSupplier(path, removeHs=False, sanitize=False)
                mol = suppl.__next__()
                if mol:
                    mol.SetProp("_Name", name)
                    writer.write(mol)
        else:
            st.write(f"No path found for {name}")
    writer.flush()
    writer.close()


# ----- Plotting -----
def plotly_plot(
    y, main_df, size=(1000, 500), x="step", trendline="median", trendline_only=False
):
    if y == "valid":
        if x == "index":
            x = "idx"
            main_df = main_df.copy()
            main_df.idx = (main_df.idx // 100) * 100
        tdf = (
            main_df.groupby(["run", x])[y]
            .agg(lambda x: (x == "true").mean())
            .reset_index()
        )
        fig = px.line(
            data_frame=tdf,
            x=x,
            y=y,
            range_y=(0, 1),
            color="run",
            template="plotly_white",
        )
        fig.update_layout(xaxis_title=x, yaxis_title=y)

    elif (y == "unique") or (y == "passes_diversity_filter"):
        if x == "index":
            x = "idx"
            main_df = main_df.copy()
            main_df.idx = (main_df.idx // 100) * 100
        tdf = main_df.groupby(["run", x])[y].mean().reset_index()
        fig = px.line(
            data_frame=tdf,
            x=x,
            y=y,
            range_y=(0, 1),
            color="run",
            template="plotly_white",
        )
        fig.update_layout(xaxis_title=x, yaxis_title=y)
    else:
        if x == "index":
            x = "idx"
        if trendline:
            n_runs = len(main_df.run.unique())
            if n_runs > 1:
                trendline_color = False
            else:
                trendline_color = "black"
            if trendline in ["max", "min"]:
                fig = px.scatter(
                    data_frame=main_df,
                    x=x,
                    y=y,
                    color="run",
                    hover_data=["run", "step", "batch_idx", y],
                    trendline="expanding",
                    trendline_options=dict(function=trendline),
                    trendline_color_override=trendline_color,
                    opacity=0.4,
                    template="plotly_white",
                )
            else:
                fig = px.scatter(
                    data_frame=main_df,
                    x=x,
                    y=y,
                    color="run",
                    hover_data=["run", "step", "batch_idx", y],
                    trendline="rolling",
                    trendline_options=dict(function=trendline, window=100),
                    trendline_color_override=trendline_color,
                    opacity=0.4,
                    template="plotly_white",
                )

            if trendline_only:
                fig.data = [t for t in fig.data if t.mode == "lines"]
                fig.update_traces(
                    showlegend=True
                )  # trendlines have showlegend=False by default

        else:
            fig = px.scatter(
                data_frame=main_df,
                x=x,
                y=y,
                color="run",
                hover_data=["run", "step", "batch_idx", y],
                opacity=0.4,
                template="plotly_white",
            )

    return fig


try:
    import streamlit_bokeh_events
    from bokeh.models import BoxSelectTool, ColumnDataSource, CustomJS
    from bokeh.plotting import figure, gridplot

    def bokeh_plot(y, main_df, size=(1000, 500), *args):
        TOOLTIPS = """
        <div>
        Step_batch_idx: @ids<br>
        </div>
        """
        # @img{safe}

        if y == "valid":
            p = figure(plot_width=size[0], plot_height=size[1])
            steps = main_df.step.unique().tolist()
            ratios = (
                main_df.groupby("step")[y]
                .apply(lambda x: (x == "true").mean())
                .tolist()
            )
            p.line(x=steps, y=ratios)

        elif (y == "unique") or (y == "passes_diversity_filter"):
            p = figure(plot_width=size[0], plot_height=size[1])
            steps = main_df.step.unique().tolist()
            ratios = main_df.groupby("step")[y].mean().tolist()
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
                    """,
                ),
            )

            p = figure(plot_width=size[0], plot_height=size[1], tooltips=TOOLTIPS)
            p.add_tools(BoxSelectTool())
            p.circle(x="x", y="y", size=8, source=source)
            p.line(
                x="x", y="y_mean", line_color="blue", legend_label="mean", source=source
            )
            p.line(
                x="x",
                y="y_median",
                line_color="red",
                legend_label="median",
                source=source,
            )

        p.xaxis[0].axis_label = "Step"
        p.yaxis[0].axis_label = y

        return p

    def bokeh_plot_events(df, y_axis):
        p = bokeh_plot(y_axis, df)
        st.bokeh_chart(p)
        selection = streamlit_bokeh_events(
            bokeh_plot=p,
            events="BOX_SELECT",
            key="main",
            refresh_on_update=True,
            override_height=None,
            debounce_time=0,
        )
        selection = selection["BOX_SELECT"]["data"]  # Probably incorrect
        return selection

    def multi_bokeh_plot(df, y_variables):
        plots = []
        for y in y_variables:
            p = bokeh_plot(y, df, size=(500, 300))
            plots.append(p)
        grid = gridplot(plots, ncols=3)
        return grid

    def bokeh_mpo_events(df, x_variables, step=None, k=None):
        if step:
            df = df.loc[df.step == step, :]
        if k:
            if k > 100:
                st.write("Warning: Limiting display to first 100")
            # Subset top-k unique
            df = df.iloc[:k, :]

        p = figure(
            plot_width=1000,
            plot_height=500,
            x_range=x_variables,
            tooltips="""
            <div>
            @img{safe}
            Step_batch_idx: @ids<br>
            </div>
            """,
        )
        p.add_tools(BoxSelectTool())
        for i, r in df.iterrows():
            data = dict(
                x=x_variables,
                y=r[x_variables].values,
                ids=[f"{r['step']}_{r['batch_idx']}"] * len(x_variables),
                img=[mol2svg(r["smiles"])] * len(x_variables),
            )
            source = ColumnDataSource(data)
            p.circle(x="x", y="y", source=source)
            p.line(x="x", y="y", source=source)
        selection = streamlit_bokeh_events(
            bokeh_plot=p,
            events="BOX_SELECT",
            key="mpo",
            refresh_on_update=True,
            override_height=None,
            debounce_time=0,
        )
        selection = {"BOX_SELECT": {"data": df.index.to_list()}}  # Probably incorrect
        return selection

    def scaffold_bokeh_plot(memory_list):
        hist = figure(
            plot_width=1000,
            plot_height=400,
            tooltips="""
            <div>
            @img{safe}
            </div>
            """,
        )
        hist_data = dict(
            x=[i for i in range(len(memory_list))],
            top=[len(c["members"]) for c in memory_list],
            img=[
                mol2svg(Chem.MolFromSmiles(m["centroid"]))
                if isinstance(m["centroid"], str)
                else mol2svg(Chem.MolFromSmiles(""))
                for m in memory_list
            ],
        )
        hist_source = ColumnDataSource(hist_data)
        hist.vbar(x="x", width=0.5, bottom=0, top="top", source=hist_source)
        return hist


except ImportError:
    pass
