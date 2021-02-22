import sys
import os
import pandas as pd
from glob import glob

import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from rdkit import Chem
from rdkit.Chem.Draw import MolsToGridImage
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


app = dash.Dash(__name__)

app.layout = html.Div([
    html.Div([
        html.H1("MolScore", style={'color': 'white'}),
        html.H4(f"Run: {main_df['model'].values[0]}-{main_df['task'].values[0]}", style={'color': 'white'})
    ], style={'backgroundColor': '#44546A'}),
    html.Div([
        html.Button('update', id='update_button', n_clicks=0),
        dcc.Graph(id='live-update-graph')
    ]),
    html.Div([
        html.Div('Select y-axis:', style={'display': 'inline-block'}),
        html.Div(
            dcc.Dropdown(id='y-column',
                         value='valid',
                         options=[{'label': key, 'value': key} for key in main_df.columns.tolist()]),
            style={'width': '50%', 'display': 'inline-block'}),
    ]),
    html.Div(
        html.Div('Selected 2D structures'),
        style={'padding-top': '1cm'}),
    html.Div(
        html.Img(id='structure-image', style={"width": "100%", "border": "2px black solid"})
    ),
    html.Div("MolScore table", style={'padding-top': '1cm'}),
    dash_table.DataTable(
        id='table',
        columns=[{"name": i, "id": i} for i in main_df.columns],
        data=main_df.to_dict('records'),
        page_size=10,
        style_table={'overflowX': 'auto'},
        editable=False,
        filter_action="native",
        sort_action="native",
        sort_mode="multi",
        page_action="native",
        page_current=0
    ),
    # Hidden dev for empty output hack
    html.Div(id='hidden-div', style={'display': 'none'})
])


@app.callback(Output('hidden-div', 'children'),
              Input('update_button', 'n_clicks'))
def update_data(n_clicks=0):  # Bad practice to modify global variable, but okay for single user use, not production
    global main_df
    global it_files
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'update_button' in changed_id:
        main_df, it_files = update_files(it_path, it_files, main_df)
    return None


# Multiple components can update everytime interval gets fired.
@app.callback(Output('live-update-graph', 'figure'),
              [Input('y-column', 'value'),
               Input('hidden-div', 'children')])
def update_graph_live(y='valid', *args):
    # Bring in global values
    global main_df

    # Plotly go
    if (y == 'valid') or (y == 'unique') or (y == 'passes_diversity_filter'):
        steps = main_df.step.unique().tolist()
        ratios = main_df.groupby('step')[y].apply(lambda x: (x == 'true').mean()).tolist()
        fig = go.Figure()
        fig.add_trace(
            go.Scattergl(
                x=steps,
                y=ratios,
                mode='lines',
                name='Molecules',
            )
        )
    else:
        fig = go.Figure()
        fig.add_trace(
            go.Scattergl(
                x=main_df.step.tolist(),
                y=main_df[y].tolist(),
                mode='markers',
                name='Molecules',
                text=main_df.smiles.tolist()
            )
        )
        fig.add_trace(
            go.Scattergl(
                x=main_df.step.tolist(),
                y=main_df[y].rolling(window=100).median().tolist(),
                name='Rolling Median (window=100)'
            )
        )
        fig.add_trace(
            go.Scattergl(
                x=main_df.step.tolist(),
                y=main_df[y].rolling(window=100).mean().tolist(),
                name='Rolling Mean (window=100)',
                line={'color': 'blue'}
            )
        )

    fig.update_layout(yaxis={'title': y},
                      xaxis={'title': 'Steps'})
    return fig


@app.callback(
    Output('structure-image', 'src'),
    [Input('live-update-graph', 'selectedData'),
     Input('y-column', 'value')])
def display_selected_data(selectedData, y):
    max_structs = 24
    structs_per_row = 6
    empty_plot = "data:image/gif;base64,R0lGODlhAQABAAAAACwAAAAAAQABAAA="
    if selectedData:
        if len(selectedData['points']) == 0:
            return empty_plot
        match_idx = [x['pointIndex'] for x in selectedData['points']]
        smiles_list = [Chem.MolFromSmiles(x) for x in list(main_df.iloc[match_idx].smiles)]
        name_list = list(main_df.iloc[match_idx][y])
        batch_list = [f"{step}_{batch_idx}" for step, batch_idx in main_df.loc[match_idx, ['step', 'batch_idx']].values]
        name_list = [f"{x:.02f}" if isinstance(x, float) else f"{x}" for x in name_list]
        legends = [f"{idx}\n{y}: {name}" for idx, name in zip(batch_list, name_list)]
        img = MolsToGridImage(smiles_list[0:max_structs], molsPerRow=structs_per_row, legends=legends,
                              subImgSize=(300, 300))
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        encoded_image = base64.b64encode(buffered.getvalue())
        src_str = 'data:image/png;base64,{}'.format(encoded_image.decode())
    else:
        return empty_plot
    return src_str


@app.callback(
    [Output('table', 'columns'),
     Output('table', 'data')],
    Input('hidden-div', 'children')
)
def update_table(*args):
    global main_df
    return [{"name": i, "id": i} for i in main_df.columns], main_df.to_dict('records')


if __name__ == '__main__':
    app.run_server(debug=True)

