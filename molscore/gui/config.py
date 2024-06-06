import ast
import inspect
import json
import os
import re

import streamlit as st

# from molscore.gui.utils.file_picker import st_file_selector
import molscore.scaffold_memory as scaffold_memory
import molscore.scoring_functions as scoring_functions
from molscore import utils

# Set session variables, persistent on re-runs
ss = st.session_state
if "configs" not in ss:
    ss.input_path = "configs"
if "n_sf" not in ss:
    ss.n_sf = 0
if "n_sp" not in ss:
    ss.n_sp = 0
if "pidgin_docstring" not in ss:
    ss.pidgin_docstring = False


# ----- Functions -----
def st_file_selector(
    st_placeholder, key, path=".", label="Please, select a file/folder...", counter=1
):
    """
    Code for a file selector widget which remembers where you are...
    """
    # get base path (directory)
    base_path = "." if (path is None) or (path == "") else path
    base_path = base_path if os.path.isdir(base_path) else os.path.dirname(base_path)
    base_path = "." if (base_path is None) or (base_path == "") else base_path
    # list files in base path directory
    files = os.listdir(base_path)
    files.insert(0, "..")
    files.insert(0, ".")
    selected_file = st_placeholder.selectbox(label=label, options=files, key=key)
    selected_path = os.path.normpath(os.path.join(base_path, selected_file))
    if selected_file == ".":
        return selected_path
    if os.path.isdir(selected_path):
        # ----
        counter += 1
        key = key + str(counter)
        # ----
        selected_path = st_file_selector(
            st_placeholder=st_placeholder,
            path=selected_path,
            label=label,
            key=key,
            counter=counter,
        )
    return os.path.abspath(selected_path)


def type2widget(ptype, label, key, default=None, options=None):
    """
    Infer widget based on parameter datatype
    :param ptype: Parameter data type
    :param label: Description for the widget to write
    :param key: Unique key for widget
    :param default: Default parameter value
    :param options: If present use selectbox
    :return:
    """
    if options is not None:
        widget = st.selectbox(
            label=label,
            options=options,
            index=options.index(default) if default in options else 0,
            key=key,
        )
    else:
        if ptype == str:
            widget = st.text_input(
                label=label, value=default if default is not None else "", key=key
            )
        if ptype in [list, dict]:
            widget = st.text_area(
                label=label, value=default if default is not None else "", key=key
            )
        if ptype == int:
            widget = st.number_input(
                label=label, value=default if default is not None else 0, key=key
            )
            widget = int(widget)
        if ptype == float:
            widget = st.number_input(
                label=label, value=default if default is not None else 0.0, key=key
            )
            widget = float(widget)
        if ptype == bool:
            widget = st.selectbox(
                label=label,
                options=[True, False],
                index=[True, False].index(default) if default in [True, False] else 0,
                key=key,
            )
        if ptype == type(None):
            st.write(
                "WARNING: default is None and no type annotation, using text_input"
            )
            widget = st.text_input(label=label, key=key)
        if ptype == os.PathLike:
            if default is not None and os.path.exists(default):
                session_key = f"{key}_input_path"
                ss[session_key] = default

            else:
                session_key = f"{key}_input_path"
                ss[session_key] = "configs"
            ss[session_key] = st_file_selector(
                label=label, path=ss[session_key], st_placeholder=st.empty(), key=key
            )
            st.write(f"Selected: {ss[session_key]}")
            widget = ss[session_key]
    return widget


def parseobject(obj, exceptions=[]):
    """
    Parse an object (class or function) to identify parameters and annotation including docstring (must be reST/PyCharm style).
    Additionally any list enclosed in square brackets will be interpretted as options.
    :param obj: The class or function to be parsed
    :param exceptions: A list of parameters to exclude in parsing [example1, example2]
    :return: Object, A dictionary of parameters including name, type, default, description and options
    """
    exceptions += exceptions + ["kwargs"]

    # Find type of object
    if inspect.isclass(obj):
        docs = inspect.getdoc(obj.__init__)
    else:
        docs = inspect.getdoc(obj)

    # Remove description
    docs = docs.strip().replace("\n", "").split(":")
    _ = docs.pop(0)

    # Inspect parameter info
    sig = inspect.signature(obj).parameters
    params = {}
    for p in sig:
        if p in exceptions:
            continue
        params[p] = {
            "name": p,
            "type": sig[p].annotation if sig[p].annotation != inspect._empty else None,
            "default": sig[p].default if sig[p].default != inspect._empty else None,
            "optional": True
            if (sig[p].default is None) and (sig[p].default != inspect._empty)
            else False,
            "description": None,
            "options": None,
        }

    # Add docstring annotation for parameters
    for d in range(0, len(docs), 2):
        if docs[d].split(" ")[0] != "param":
            continue
        p = docs[d].split(" ")[1]
        if p in exceptions:
            continue

        p_description = docs[d + 1].strip()
        p_options = re.search("\[(.*?)\]", docs[d + 1].strip())
        if p_options is not None:
            p_description = p_description.replace(p_options.group(), "").strip()
            p_options = p_options.group().strip("[]").split(", ")

        try:
            params[p]["description"] = p_description
            params[p]["options"] = p_options
        except KeyError as e:
            print(f"Parameter-docstring mismatch: {e}")
            pass

    return obj, params


def object2dictionary(obj, key_i=0, exceptions=[]):
    """
    Take an object (function or class.__init__) and use the st interface to return
     a dictionary of parameters and arguments - depends on correctly annotated format.
    :param obj: Class or function
    :param key_i: Key identifier
    :param exceptions: Values we don't want user input for
    :return: dict
    """
    result_dict = {}
    obj, params = parseobject(obj, exceptions)

    for p, pinfo in params.items():
        label = f"{p}: {pinfo['description']}"

        # Check to see if it's optional
        if pinfo["optional"]:
            add_optional = st.checkbox(
                label=f"Specificy {p} [optional]",
                value=False,
                key=f"{key_i}: {obj.__name__}_{p}_optional",
            )

        if not pinfo["optional"] or add_optional:
            # No default
            if pinfo["default"] is None:
                # If no type either, print warning and use text input
                if pinfo["type"] is None:
                    st.write(
                        f"WARNING: {p} is has no default or type annotation, using text_input"
                    )
                    result_dict[p] = st.text_input(
                        label=label, key=f"{key_i}: {obj.__name__}_{p}"
                    )

                else:
                    # Check to see if ptype is union i.e., multiple types possible
                    ptype_args = getattr(pinfo["type"], "__args__", None)
                    if ptype_args is not None:
                        pinfo["type"] = st.selectbox(
                            label=f"{p}: input type",
                            options=ptype_args,
                            index=0,
                            key=f"{key_i}: {obj.__name__}_{p}_type",
                        )
                    result_dict[p] = type2widget(
                        pinfo["type"],
                        key=f"{key_i}: {obj.__name__}_{p}",
                        label=label,
                        options=pinfo["options"],
                    )
                    if pinfo["type"] == int:
                        result_dict[p] = int(result_dict[p])

            else:
                # Use type annotation if present
                if pinfo["type"] is not None:
                    with st.expander(f"{p}={pinfo['default']}"):
                        # Check to see if ptype is union i.e., multiple types possible
                        ptype_args = getattr(pinfo["type"], "__args__", None)
                        if ptype_args is not None:
                            pinfo["type"] = st.selectbox(
                                label=f"{p}: input type",
                                options=ptype_args,
                                index=0,
                                key=f"{key_i}: {obj.__name__}_{p}_type",
                            )
                        result_dict[p] = type2widget(
                            pinfo["type"],
                            key=f"{key_i}: {obj.__name__}_{p}",
                            default=pinfo["default"],
                            label=label,
                            options=pinfo["options"],
                        )
                        # if pinfo['type'] == int: result_dict[p] = int(result_dict[p])

                # Otherwise use type of default
                else:
                    with st.expander(f"{p}={pinfo['default']}"):
                        result_dict[p] = type2widget(
                            type(pinfo["default"]),
                            key=f"{key_i}: {obj.__name__}_{p}",
                            default=pinfo["default"],
                            label=label,
                            options=pinfo["options"],
                        )

            # If list convert correctly
            if pinfo["type"] == list:
                result_dict[p] = (
                    result_dict[p].replace(",", "").replace("\n", " ").split(" ")
                )
                # Check if empty and handle properly
                if (
                    result_dict[p] == [""]
                    or result_dict[p] == ["[]"]
                    or result_dict[p] == ["", ""]
                ):
                    result_dict[p] = []
            if pinfo["type"] == dict:
                try:
                    result_dict[p] = ast.literal_eval(result_dict[p])
                except (SyntaxError, ValueError):
                    st.write(
                        'Please input as if typing in python, e.g., {0: "A string", 1: [4, 5], 2: ["A list of strings",]}'
                    )
                    pass  # raise e

    return result_dict


def getsfconfig(key_i, tab):
    """
    Get configs for scoring functions
    :param key_i: Key identifier for widgets
    :return: dict
    """
    sf_config = {}
    # Do it in the tab
    with tab:
        # Choose scoring functions
        sf_config["name"] = st.selectbox(
            label="Type of scoring function",
            options=[
                s.__name__
                for s in scoring_functions.all_scoring_functions
                if s.__name__
                not in ["TanimotoSimilarity", "RDKitDescriptors", "SKLearnModel"]
            ],  # Remove these as only present for backwards compatability
            index=0,
            key=f"{key_i}: sf_name",
            help="Select which type of scoring function to run from the dropdown list.",
        )
        sf_config["run"] = True
        # Get (class/function) from name ...
        sf_obj = [
            s
            for s in scoring_functions.all_scoring_functions
            if s.__name__ == sf_config["name"]
        ][0]
        # Write doc of class (not instance)
        if (sf_config["name"] == "PIDGIN") and not ss.pidgin_docstring:
            st.write("Populating options, please wait a minute or two ...")
            sf_obj.set_docstring()  # Populate PIDGIN docstring which takes a few seconds
            ss.pidgin_docstring = True
        sf_doc = inspect.getdoc(sf_obj)
        st.markdown("**Description**")
        if sf_doc is not None:
            st.write(sf_doc)
        else:
            st.write("No documentation")
        st.markdown("**Parameters**")
        sf_config["parameters"] = object2dictionary(sf_obj, key_i=key_i)
        st.markdown("**Config format**")
        with st.expander(label="Check parsing"):
            st.write(sf_config)
    return sf_config


def getspconfig(options, key_i, tab):
    """
    Get configs for scoring parameters
    :param options: List of options taken from selected scoring functions
    :param key_i: Key identifier
    :return: dict
    """
    # TODO undo col and change parameters for range/min/max to only those available
    global ss
    sp_config = {}
    # Do it within a tab
    with tab:
        sp_config["name"] = st.selectbox(
            label="Scoring parameter",
            options=options,
            index=0,
            key=f"{key_i}: sp_name",
            help="Select which scoring parameter to include in the final molecule score/reward from the dropdown list.",
        )
        if sp_config["name"] is None:
            return
        # Get filter
        sp_config["filter"] = st.checkbox(
            label="Filter",
            value=False,
            key=f"{key_i}: sp_filter",
            help="Select whether this parameter is a filter (to multiply the final aggregated score)",
        )
        # Get weight
        with st.expander(label="Weight (only applicable if using wsum or wprod)"):
            sp_config["weight"] = st.number_input(
                label="weight",
                value=1.0,
                key=f"{key_i}: sp_weight",
                help="These weights are normalized by the sum of all weights (i.e., any positive value can be used).",
            )
        # Get (class/function) for transformation/modification from name and print doc ...
        sp_config["modifier"] = st.selectbox(
            label="Transformation function",
            options=[m.__name__ for m in utils.all_score_modifiers],
            index=0,
            key=f"{key_i}: sp_modifier",
            help="Select the transformation function used to transform the parameter value to between 0 and 1 from the dropdown list.",
        )
        smod_obj = [
            m for m in utils.all_score_modifiers if m.__name__ == sp_config["modifier"]
        ][0]
        # Write doc for func
        smod_doc = inspect.getdoc(smod_obj).split(":")[0]
        st.markdown("**Description**")
        if smod_doc is not None:
            st.write(smod_doc)
        else:
            st.write("No documentation")

        # If norm, optional specification of max/min
        st.markdown("**Parameters**")
        if smod_obj.__name__ == "norm":
            col1, col2 = st.columns(2)
            # Buttons
            if f"maxmin{key_i}" not in ss:
                ss[f"maxmin{key_i}"] = False
            if col1.button(
                label="Specify max/min",
                key=f"{key_i}: maxmin",
                help="Parameter value will be maxmin normalized between these values",
            ):
                ss[f"maxmin{key_i}"] = True
            if col2.button(
                label="Don't specify max/min",
                key=f"{key_i}: nomaxmin",
                help="Pameter value will be dynamically maxmin normalized based on the maximum and minimum observed values so far (i.e., moving goal post).",
            ):
                ss[f"maxmin{key_i}"] = False
            # Grab parameters and plot
            if ss[f"maxmin{key_i}"]:
                sp_config["parameters"] = object2dictionary(
                    smod_obj, key_i=key_i, exceptions=["x"]
                )
            else:
                sp_config["parameters"] = object2dictionary(
                    smod_obj, key_i=key_i, exceptions=["x", "max", "min"]
                )

        # Otherwise just do it
        else:
            sp_config["parameters"] = object2dictionary(
                smod_obj, key_i=key_i, exceptions=["x"]
            )

        col1, col2, col3 = st.columns([1, 1, 1])
        try:
            with col1:
                st.write("Example value transformation shown here (if relevant):")
            with col2:
                st.write(
                    utils.transformation_functions.plot_mod(
                        smod_obj, sp_config["parameters"]
                    )
                )
        except Exception:
            pass

        st.markdown("**Config format**")
        with st.expander(label="Check parsing"):
            st.write(sp_config)
    return sp_config


# ----- Start App -----
st.title("MolScore Configuration Writer")
st.markdown(
    """
    This interface can be used to help write the configuration file describing how to score molecules required by MolScore.
    Below each section, you can preview the resulting JSON for that section. Widgets and descriptions for functions and
    function parameters are automatically extracted from documentation and typing.
    """
)
config = dict()


# ------ Basic information ------
st.markdown("#")  # Add spacing
st.subheader("Run parameters")
config["task"] = (
    st.text_input(
        label="Task name (for file naming)",
        value="QED",
        help="This should succinctly and uniquely name your objective.",
        key="task",
    )
    .strip()
    .replace(" ", "_")
)

config["output_dir"] = st.text_input(
    label="Output directory",
    value="./",
    help="Output directory to where MolScore will save another directory containing the run output that will be automatically named.",
    key="output_dir",
)

absolute_output_path = st.checkbox(
    label="Absolute path",
    help="Whether this is an absolute file path or not. If unchecked, this is assumed to be a relative path based depending on where you run generative model script.",
)
if absolute_output_path:
    config["output_dir"] = os.path.abspath(config["output_dir"])
    st.write(f"Selected: {config['output_dir']}")

config["load_from_previous"] = st.checkbox(
    label="Continue from previous directory",
    help="Continue scoring from a previous run and automatically run from the more recent step, generative model iterations may need to be modified.",
)
if config["load_from_previous"]:
    col1, col2 = st.columns([1, 9])
    with col2:
        ss.previous_dir = "configs"
        ss.previous_dir = st_file_selector(
            label="Select a previously used folder",
            st_placeholder=st.empty(),
            path=ss.previous_dir,
            key="previous_dir_selector",
        )
        config["previous_dir"] = ss.previous_dir
        st.write(f"Selected: {config['previous_dir']}")

# ------ Logging ------
config["logging"] = st.checkbox(
    label="Verbose logging",
    help="Set logging level to info and additionally channel logging to the terminal.",
    key="logging",
)

# ------ App monitor ------
config["monitor_app"] = st.checkbox(
    label="Run live monitor app",
    help="Automatically run the monitor gui as a subprocess after the first iteration is scored. This can also be run manually at any time.",
    key="monitor_app",
)

# ------ Termination criteria ------
st.markdown("#")  # Add spacing
st.subheader("Termination criteria")
# Budget
if st.checkbox(
    label="Specify budget",
    value=False,
    help="Set a budget number of molecules for optimization.",
    key="budget_optional",
):
    config["budget"] = st.number_input(
        label="Budget",
        value=10000,
        min_value=1,
        help="The number of molecules to score before MolScore.finished=True",
        key="budget_value",
    )
# Termination threshold
if st.checkbox(
    label="Specify termination threshold",
    value=False,
    help="Set the threshold value for the final score before MolScore.finished=True",
    key="termination_threshold_optional",
):
    config["termination_threshold"] = st.number_input(
        label="Termination threshold",
        value=0.5,
        min_value=0.0,
        max_value=1.0,
        help="The final score threshold reached before MolScore.finished=True.",
        key="termination_threshold_value",
    )
# Termination patience
if st.checkbox(
    label="Specify termination patience",
    value=False,
    help="Set a period to wait after the threshold is reached, or to wait while no improvement is made before MolScore.finished=True",
    key="termination_patience_optional",
):
    config["termination_patience"] = st.number_input(
        label="Termination patience",
        value=5,
        min_value=1,
        help="The number of iterations to wait after the threshold, or with no improvement before MolScore.finished=True.",
        key="termination_patience_value",
    )
# Termination exit
config["termination_exit"] = st.checkbox(
    label="Exit on termination [WARNING]",
    value=False,
    help="If true, after appropriate termination MolScore will exit the program for you with sys.exit(). Therefore any code after MolScore.finished=True will not be run. THIS IS INCOMPATIBLE WITH CURRICULUM LEARNING.",
    key="termination_exit",
)

with st.expander(label="Check parsing"):
    st.write(config)

# ----- Scoring functions ------
st.markdown("#")  # Add spacing
st.subheader("Scoring functions")
st.markdown(
    """
    Select and run one or more scoring functions found in the `all_scoring_functions` list specified in `molscore/scoring_functions/__init__.py`.
    Multiple scoring functions can be run without using all of them for scoring molecules
    (e.g., it may be useful to always run MolecularDescriptors to record property changes).
    """
)
# Buttons to add and delete scoring function (i.e. append no. of scoring functions to Session State)
col1, col2 = st.columns(2)
with col1:
    if st.button(
        label="Add scoring function",
        help="Increases the number of scoring functions run, will add a new tab below.",
    ):
        ss.n_sf += 1
with col2:
    if st.button(
        label="Delete scoring function",
        help="Decreases the number of scoring functions run, will remove the most recent tab below.",
    ):
        ss.n_sf -= 1
if ss.n_sf > 0:
    # Get user input parameters
    sf_tabs = st.tabs([f"SF{i+1}" for i in range(ss.n_sf)])
    config["scoring_functions"] = [
        getsfconfig(i, tab=t) for i, t in zip(range(ss.n_sf), sf_tabs)
    ]
else:
    config["scoring_functions"] = []


# ----- Scoring transformations -----
st.markdown("#")  # Add spacing
st.subheader("Score transformation")
st.markdown(
    """
    Determine which returned metrics/parameters should be included in the molecules final score, and how to transform each value to between 0 and 1.
    Available return metrics/parameters shown are determined by the combination of scoring function `prefix` specified for each scoring function (see above)
    and `return_metrics` attribute
    of scoring functions.
    """
)
config["scoring"] = {}

# Buttons to add and delete scoring function (i.e. append no. of scoring functions to Session State)
col1, col2 = st.columns(2)
with col1:
    if st.button(
        label="Add scoring parameter",
        help="Increase the number of scoring parameters, will add a new tab below.",
    ):
        ss.n_sp += 1
with col2:
    if st.button(
        label="Delete scoring parameter",
        help="Decrease the number of scoring parameters, will remove the most recent tab below.",
    ):
        ss.n_sp -= 1
if ss.n_sp > 0:
    sp_tabs = st.tabs([f"SF{i+1}" for i in range(ss.n_sp)])
    # Get user input parameters if scoring functions have been defined
    smetric_options = ["valid_score"]
    for sf in config["scoring_functions"]:
        sf_name = sf["name"]
        sf_prefix = sf["parameters"]["prefix"]
        sf_obj = [
            sf
            for sf in scoring_functions.all_scoring_functions
            if sf.__name__ == sf_name
        ][0]
        try:
            sf_metrics = sf_obj.return_metrics
            _ = [
                smetric_options.append(f"{sf_prefix.strip().replace(' ', '_')}_{m}")
                for m in sf_metrics
            ]
        except AttributeError:
            st.write(f"WARNING: No return metrics found for {sf_name}")
            continue
    # Get parameter inputs
    config["scoring"]["metrics"] = [
        getspconfig(options=smetric_options, key_i=i, tab=t)
        for i, t in zip(range(ss.n_sp), sp_tabs)
    ]
else:
    config["scoring"]["metrics"] = []

# ----- Score aggregation -----
st.markdown("#")  # Add spacing
st.subheader("Score aggregation")
st.markdown(
    """
    Specify how to aggregate multiple score parameters into a single numerical value between 0 and 1. If only using a single scoring parameter use `single`.
    For any more than one scoring parameter, select any other aggregation function.
    """
)

config["scoring"]["method"] = st.radio(
    label="Aggregation function",
    options=[m.__name__ for m in utils.all_score_methods],
    index=0,
    key="Scoring method",
)

# Get (class/function) from name and print doc ...
sm_obj = [
    s for s in utils.all_score_methods if s.__name__ == config["scoring"]["method"]
][0]
sm_doc = inspect.getdoc(sm_obj)
st.markdown("**Description**")
if sm_doc is not None:
    st.markdown(sm_doc.split(":")[0])
else:
    st.markdown("No documentation")


# ----- Diversity filters -----
st.markdown("#")  # Add spacing
st.subheader("Diversity filter")
config["diversity_filter"] = {}
config["diversity_filter"]["run"] = st.checkbox(
    label="Run diversity filter",
    help="Apply a diversity filter penalizing non-diverse molecules. Click to expand further options.",
    key="run_DF",
)
if config["diversity_filter"]["run"]:
    config["diversity_filter"]["name"] = st.radio(
        label="Type of diversity filter",
        options=["Unique", "Occurrence"]
        + [s.__name__ for s in scaffold_memory.all_scaffold_filters],
        key="DF_name",
        index=0,
    )
    if config["diversity_filter"]["name"] == "Unique":
        st.markdown("**Description**")
        st.write("Penalize non-unique molecules by assigning a score of 0")
        st.markdown("**Parameters**")
        config["diversity_filter"]["parameters"] = {}
    elif config["diversity_filter"]["name"] == "Occurrence":
        st.markdown("**Description**")
        st.write("Penalize non-unique molecules based on the number of occurrences")
        st.markdown("**Parameters**")
        config["diversity_filter"]["parameters"] = {
            "tolerance": st.number_input(
                label="Number of duplicates allowed" " before penalization",
                min_value=0,
                value=5,
            ),
            "buffer": st.number_input(
                label="Number of linear penalization's"
                " until a reward of 0 is returned",
                min_value=0,
                value=5,
            ),
        }
    else:  # 'Memory-assisted' types:
        st.markdown("**Description**")
        st.write(
            "Use as dynamic memory: see https://jcheminf.biomedcentral.com/articles/10.1186/s13321-020-00473-0"
        )
        # Get (class/function) from name ...
        dv_obj = [
            s
            for s in scaffold_memory.all_scaffold_filters
            if s.__name__ == config["diversity_filter"]["name"]
        ][0]
        st.markdown("**Parameters**")
        config["diversity_filter"]["parameters"] = object2dictionary(dv_obj)

    st.markdown("**Config format**")
    with st.expander(label="Check parsing"):
        st.write(config["diversity_filter"])


# ----- Output -----
st.markdown("#")  # Add spacing
st.subheader("Output json")
with st.expander(label="Show"):
    st.write(config)
out_file = os.path.abspath(
    st.text_input(label="Output file", value=f'{config["task"]}.json')
)
st.write(f"Selected: {out_file}")
col1, col2 = st.columns(2)
with col1:
    if st.button(label="Save"):
        if not os.path.exists(os.path.dirname(out_file)):
            os.makedirs(os.path.dirname(out_file))
            st.write("Creating directory")
        with open(out_file, "w") as f:
            json.dump(config, f, indent=2)
            st.write("File saved")
with col2:
    if st.button(label="Exit"):
        os._exit(0)

# ----- Navigation Sidebar -----
st.sidebar.header("Load from previous")
ss.input_config = st_file_selector(
    st_placeholder=st.sidebar.empty(),
    path="./",
    label="Please select a config file",
    key="init_input",
)


def load_previous_meta():
    """
    Set widget values from previous config
    """
    with open(ss.input_config, "rt") as f:
        pconfig = json.load(f)
        # Update meta
        ss.task = pconfig["task"]
        ss.output_dir = pconfig["output_dir"]
        if "previous_dir" in pconfig:
            ss.previous_dir_selector = pconfig["previous_dir"]
        ss.logging = pconfig["logging"]
        ss.monitor_app = pconfig["monitor_app"]
        # Update SFs
        ss.n_sf = len(pconfig["scoring_functions"])
        # Update SPs
        ss.n_sp = len(pconfig["scoring"]["metrics"])
        # Update method
        ss["Scoring method"] = pconfig["scoring"]["method"]
        # Update DF
        if pconfig.get("diversity_filter"):
            if pconfig["diversity_filter"].get("run"):
                ss["run_DF"] = pconfig["diversity_filter"]["run"]
            else:
                ss["run_DF"] = False


def load_previous_widgets():
    with open(ss.input_config, "rt") as f:
        pconfig = json.load(f)
        # Update SFs
        for i, sf in enumerate(pconfig["scoring_functions"]):
            ss[f"{i}: sf_name"] = sf["name"]
            for pk, pv in sf["parameters"].items():
                # Add optional widgets
                if f'{i}: {sf["name"]}_{pk}' not in ss:
                    ss[f'{i}: {sf["name"]}_{pk}_optional'] = True
                    continue
                # Update parameter type
                ptype = type(pv)
                try:
                    ss[f'{i}: {sf["name"]}_{pk}_type'] = ptype
                    continue
                except KeyError:
                    pass
            # Always update prefix (too avoid bug if not in optional selections for scoring parameter)
            ss[f'{i}: {sf["name"]}_prefix'] = sf["parameters"]["prefix"]

        # Update SPs
        for i, sp in enumerate(pconfig["scoring"]["metrics"]):
            ss[f"{i}: sp_name"] = sp["name"]
            ss[f"{i}: sp_weight"] = sp["weight"]
            ss[f"{i}: sp_modifier"] = sp["modifier"]
            if (sp["modifier"] == "norm") and any(
                [k in sp["parameters"] for k in ["max", "min"]]
            ):
                ss[f"maxmin{i}"] = True
        # Update DFs
        try:
            ss["DF_name"] = pconfig["diversity_filter"]["name"]
        except KeyError:
            pass


def load_previous_params():
    with open(ss.input_config, "rt") as f:
        pconfig = json.load(f)
        # Update SFs
        for i, sf in enumerate(pconfig["scoring_functions"]):
            for pk, pv in sf["parameters"].items():
                try:
                    if isinstance(pv, list):
                        ss[f'{i}: {sf["name"]}_{pk}'] = (
                            str(pv).strip("[]").replace("'", "")
                        )
                    else:
                        ss[f'{i}: {sf["name"]}_{pk}'] = pv
                except Exception:
                    continue
        # Update SPs
        for i, sp in enumerate(pconfig["scoring"]["metrics"]):
            for pk, pv in sp["parameters"].items():
                try:
                    ss[f'{i}: {sp["modifier"]}_{pk}'] = pv
                except Exception:
                    continue
        # Update DFs
        try:
            for pk, pv in pconfig["diversity_filter"]["parameters"].items():
                ss[f'{i}: {pconfig["diversity_filter"]["name"]}_{pk}'] = pv
        except KeyError:
            pass


st.sidebar.markdown(
    "Attempt to be load in three consecutive stages due to interactive nature of widgets and callbacks (second button may be needed twice)"
)
col1, col2, col3 = st.sidebar.columns(3)
col1.button("Load meta", on_click=load_previous_meta, help="Must load me first")
col2.button("Load widgets", on_click=load_previous_widgets, help="Must load me second")
col3.button("Load params", on_click=load_previous_params, help="Must load me third")

st.sidebar.header("Navigate")
st.sidebar.markdown("[Run parameters](#run-parameters)")
st.sidebar.markdown("[Termination criteria](#termination-criteria)")
st.sidebar.markdown("[Scoring functions](#scoring-functions)")
st.sidebar.markdown("[Score transformation](#score-transformation)")
st.sidebar.markdown("[Score aggregation](#score-aggregation)")
st.sidebar.markdown("[Diversity filter](#diversity-filter)")
st.sidebar.markdown("[Save](#output-json)")
if st.sidebar.button("Exit", key="sidebar_exit"):
    os._exit(0)
