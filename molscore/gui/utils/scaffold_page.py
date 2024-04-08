import os
from itertools import cycle

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from rdkit.Chem import AllChem as Chem
from streamlit_plotly_events import plotly_events

from molscore.gui.utils import utils


def scaffold_plot(main_df, SS, dock_path=None):
    # ----- If multiple input dirs, select one -----
    if len(SS.input_dirs) > 1:
        input_dir = st.selectbox(
            label="Multiple input directories detected, please select which to investigate ...",
            options=SS.input_dirs,
        )
    else:
        input_dir = SS.input_dirs[0]

    memory_path = os.path.join(input_dir, "scaffold_memory.csv")
    if not os.path.exists(memory_path):
        raise FileNotFoundError(
            "It seems the scaffold memory diversity filter wasn't used for this run"
        )

    # ----- Process data -----
    memory = pd.read_csv(memory_path)
    memory_list = []
    max_score = 0.0
    max_size = 1
    max_step = 0.0
    # Change format into a list of records per cluster
    for i in memory.to_records():
        if i["Cluster"] == len(memory_list):
            memory_list.append(
                {
                    "centroid": i["Scaffold"],
                    "members": [i["SMILES"]],
                    "score": [i["total_score"]],
                    "step": [i["step"]],
                }
            )
        else:
            memory_list[i["Cluster"]]["members"] += [i["SMILES"]]
            if len(memory_list[i["Cluster"]]["members"]) > max_size:
                max_size = len(memory_list[i["Cluster"]]["members"])

            memory_list[i["Cluster"]]["score"] += [i["total_score"]]
            if np.mean(memory_list[i["Cluster"]]["score"]) > max_score:
                max_score = float(np.mean(memory_list[i["Cluster"]]["score"]))

            memory_list[i["Cluster"]]["step"] += [i["step"]]
            if np.mean(memory_list[i["Cluster"]]["step"]) > max_step:
                max_step = float(np.mean(memory_list[i["Cluster"]]["step"]))

    # Filter options
    Score_filter = st.sidebar.slider(
        label="Score", min_value=0.0, max_value=max_score, value=(0.0, max_score)
    )
    Size_filter = st.sidebar.slider(
        label="Size", min_value=0, max_value=max_size, value=(0, max_size)
    )
    Step_filter = st.sidebar.slider(
        label="Step", min_value=0.0, max_value=max_step, value=(0.0, max_step)
    )

    # Sort options
    sort_key = st.sidebar.selectbox(
        label="Sort by", options=["Score", "Size", "Step"], index=1
    )
    sort_order = st.sidebar.selectbox(
        label="Descending", options=[True, False], index=0
    )

    # Filter
    memory_list = [
        i
        for i in memory_list
        if (Score_filter[0] <= np.mean(i["score"]) <= Score_filter[1])
        and (Size_filter[0] <= len(i["members"]) <= Size_filter[1])
        and (Step_filter[0] <= np.mean(i["step"]) <= Step_filter[1])
    ]

    # Sort
    if sort_key == "Score":
        memory_list.sort(key=lambda x: np.mean(x["score"]), reverse=sort_order)

    if sort_key == "Size":
        memory_list.sort(key=lambda x: len(x["members"]), reverse=sort_order)

    if sort_key == "Step":
        memory_list.sort(key=lambda x: np.mean(x["step"]), reverse=sort_order)

    # ---- Bar graph ----
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("Clusters histogram (click to show centroid)")
        selection = plotly_events(
            px.bar(
                x=[i for i in range(len(memory_list))],
                y=[len(c["members"]) for c in memory_list],
                width=1500,
                height=300,
                template="plotly_white",
            )
        )
    with col2:
        if selection:
            selection = [i["x"] for i in selection][
                0
            ]  # Click event so only one selection
            st.subheader("Selected cluster centroid")
            cluster = memory_list[selection]
            st.image(utils.mol2png(Chem.MolFromSmiles(cluster["centroid"])))
            st.text(
                f"Cluster size: {len(cluster['members'])}\n"
                f"Mean score: {np.mean(cluster['score']):.02f}\n"
                f"Mean step: {np.mean(cluster['step']):.02f}"
            )

    # ----- Plot first 20 centroids -----
    st.subheader("Cluster centroids")
    st.write("This may be the Bemis-Murcko scaffold depending on diversity filter used")
    show_no = st.number_input(label="Number of clusters to show", value=20, step=5)

    for i, (cluster, column) in enumerate(
        zip(memory_list[:show_no], cycle(st.columns(5)))
    ):
        # Show image
        column.image(
            utils.mol2png(
                Chem.MolFromSmiles(
                    cluster["centroid"] if isinstance(cluster["centroid"], str) else ""
                )
            )
        )
        column.text(
            f"Cluster size: {len(cluster['members'])}\n"
            f"Mean score: {np.mean(cluster['score']):.02f}\n"
            f"Mean step: {np.mean(cluster['step']):.02f}"
        )
        # Set expansion
        if "expand" not in SS:
            SS.expand = None
        expand = column.button(label="Expand", key=f"{cluster['centroid']}_expand")
        collapse = column.button(
            label="Collapse", key=f"{cluster['centroid']}_collapse"
        )
        if collapse:
            SS.expand = None
        if expand or SS.expand == i:
            SS.expand = i
            with st.container():
                st.subheader("Cluster members")
                for j, (m, column2) in enumerate(
                    zip(cluster["members"], cycle(st.columns(5)))
                ):
                    column2.image(utils.mol2png(m))
                    column2.text(
                        f"Score: {cluster['score'][j]:.02f}\n"
                        f"Step: {cluster['step'][j]}"
                    )

                    # Send mol to pymol
                    if SS.pymol is not None:
                        show_pymol = column2.button(
                            label="Send2PyMol", key=f"{m}_pymol_button"
                        )
                        if show_pymol:
                            match_idx = (
                                main_df.loc[
                                    (main_df.run == os.path.basename(input_dir))
                                    & (main_df.smiles == m)
                                ]
                                .drop_duplicates()
                                .index[0]
                            )
                            file_paths, names = utils.find_sdfs([match_idx], main_df)
                            utils.send2pymol(
                                name=names[0], path=file_paths[0], pymol=SS.pymol
                            )

                # Send all to pymol
                if SS.pymol is not None:
                    if st.button(
                        "SendAll2Pymol", key=f"AllPyMol_{os.path.basename(input_dir)}"
                    ):
                        match_idxs = list(
                            main_df.loc[
                                (main_df.run == os.path.basename(input_dir))
                                & (main_df.smiles.isin(cluster["members"]))
                            ]
                            .drop_duplicates()
                            .index
                        )
                        paths, names = utils.find_sdfs(match_idxs, main_df)
                        for p, n in zip(paths, names):
                            utils.send2pymol(name=n, path=p, pymol=SS.pymol)
                break

    # ----- Option to save sdf -----
    if not expand:
        with st.expander(label="Export selected clusters"):
            out_file = st.text_input(label="File name")
            out_file = os.path.abspath(f"{out_file}.sdf")
            st.write(out_file)
            if st.button(label="Save", key="save_all_clusters"):
                file_paths = []
                mol_names = []
                for cluster in memory_list[:show_no]:
                    match_idx = list(
                        main_df.loc[
                            (main_df.run == os.path.basename(input_dir))
                            & (main_df.smiles.isin(cluster["members"]))
                        ]
                        .drop_duplicates()
                        .index
                    )
                    paths, names = utils.find_sdfs(match_idx, main_df)
                    file_paths += paths
                    mol_names += [
                        f'ClusterCentroid: {names[0].split(":")[1]} - {n}'
                        for n in names
                    ]
                utils.save_sdf(
                    mol_paths=file_paths, mol_names=mol_names, out_file=out_file
                )
