""" Dashboard for visualizing experiments.

Note:
    Under mantaince, need refactor.

Execution::

    $ python dash.py

"""

import os
import sys
import plotly
import pandas
import datetime
import urllib.request, json

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css', 'assets/modify.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

def get_first_dir():
    base = '../data/'
    in_data = os.listdir(base)
    for dir in in_data:
        if os.path.isdir(base + dir):
            return base+dir
    return []


hidden_style = dict()
@app.callback(Output('content_div', 'style'),
              [Input('dropdown', 'value')])
def check_hidden(value):
    if value == []:
        return dict(display='none', width='100%')
    else:
        return dict(width='100%')


td_list_columns = []
td_list_values = []
for td in range(20):
    td_list_columns.append(html.Th(id='config_col-' + str(td)))
    td_list_values.append(html.Td(id='config_val-' + str(td)))


app.layout = html.Div([

    html.Div([
        html.Div(
            html.Img(src='assets/resam.svg', width='60%'),
            style={"width":"40%", "float": "left"}
        ),
        html.Div([
            html.H6("Selecione o Experimento:"),
            dcc.Dropdown(
                id='dropdown',
                options=[],
                value=get_first_dir(),
                optionHeight=60
            )
        ], style={"width":"60%", "float": "right"})
    ], style={"height":"100pt"}),

    html.Div([
        dcc.Interval(
            id='interval',
            interval=10000,
            n_intervals=0
        ),
        html.Div([
            html.Div(
                html.Table([
                    html.Tr(td_list_columns[:10]),
                    html.Tr(td_list_values[:10])
                ], style={"width":"100%"}),
            style={"width":"100%", "height":"100pt", "display": "inline-block"}),
            html.Div(
                html.Table([
                    html.Tr(td_list_columns[10:]),
                    html.Tr(td_list_values[10:])
                ], style={"width":"100%"}),
            style={"width":"100%", "height":"100pt", "display": "inline-block"})
        ]),

        html.Div([
            html.Div(
                dcc.Graph(
                    id='ep_ret', style={'width': '800'}
                ), style={'display': 'inline-block'}
            ),
            html.Div(
                dcc.Graph(
                    id='buysellhold', style={'width': '800'}
                ), style={'display': 'inline-block'}
            )
        ]),

        html.Div([
            html.Div(
                dcc.Graph(
                    id='testprofit', style={'width': '800'}
                ), style={'display': 'inline-block'}
            ),
            html.Div(
                dcc.Graph(
                    id='profit', style={'width': '800'}
                ), style={'display': 'inline-block'}
            )
        ]),

        html.Div([
            html.Div(
                dcc.Graph(
                    id='qval', style={'width': '800'}
                ), style={'display': 'inline-block'}
            ),
            html.Div(
                dcc.Graph(
                    id='loss', style={'width': '800'}
                ), style={'display': 'inline-block'}
            )
        ])
    ],id='content_div')

])

def graph_testprofit(data):
    fig = plotly.tools.make_subplots(rows=1, cols=1)
    fig['layout']['margin'] = {
        'l': 30, 'r': 10, 'b': 30, 't': 50
    }
    fig['layout']['legend'] = {'x': 1, 'y': 1, 'xanchor': 'right'}
    fig['layout']['height'] = 350
    fig['layout']['title'] = "Lucro Médio em Teste"
    fig['layout']['yaxis']['title'] = "Reais"
    fig['layout']['xaxis']['title'] = "Épocas"

    fig.append_trace({
        'x': data['Epoch'],
        'y': data['TestEpProfit'],
        'name': 'TestEpProfit',
        'mode': 'lines+markers',
        'type': 'scatter',
        'marker': {
            'color': 'rgba(53, 165, 176, 1.0)'
        }
    }, 1, 1)

    return fig


def graph_profit(data):
    fig = plotly.tools.make_subplots(rows=1, cols=1)
    fig['layout']['margin'] = {
        'l': 30, 'r': 10, 'b': 30, 't': 50
    }
    fig['layout']['legend'] = {'x': 1, 'y': 1, 'xanchor': 'right'}
    fig['layout']['height'] = 350
    fig['layout']['title'] = "Lucro em Treino"
    fig['layout']['yaxis']['title'] = "Reais"
    fig['layout']['xaxis']['title'] = "Épocas"

    fig.append_trace({
        'name': 'Mínimo',
        'x': data['Epoch'],
        'y': data['MinEpProfit'],
        'marker': {
            'color': 'rgba(66, 195, 207,1.0)'
        }
    }, 1, 1)

    fig.append_trace({
        'name': 'Máximo',
        'x': data['Epoch'],
        'y': data['MaxEpProfit'],
        'fill':'tonexty',
        'fillcolor': 'rgba(91, 201, 212, 0.5)',
        'marker': {
            'color': 'rgba(66, 195, 207, 1.0)'
        }
    }, 1, 1)

    fig.append_trace({
        'x': data['Epoch'],
        'y': data['AverageEpProfit'],
        'name': 'Médio',
        'mode': 'lines+markers',
        'marker': {
            'color': 'rgba(66, 195, 207,1.0)'
        },
        'type': 'scatter'
    }, 1, 1)

    return fig

def graph_ep_ret(data):
    fig = plotly.tools.make_subplots(rows=1, cols=1)
    fig['layout']['margin'] = {
        'l': 30, 'r': 10, 'b': 30, 't': 50
    }
    fig['layout']['legend'] = {'x': 1, 'y': 1, 'xanchor': 'right'}
    fig['layout']['height'] = 350
    fig['layout']['title'] = "Retorno Médio"
    fig['layout']['yaxis']['title'] = "Valor do Retorno"
    fig['layout']['xaxis']['title'] = "Épocas"

    fig.append_trace({
        'x': data['Epoch'],
        'y': data['AverageEpRet'],
        'name': 'Treino',
        'mode': 'lines+markers',
        'type': 'scatter',
        'marker': {
            'color': 'rgba(91, 201, 212,1.0)'
        }
    }, 1, 1)
    fig.append_trace({
        'x': data['Epoch'],
        'y': data['AverageTestEpRet'],
        'name': 'Teste',
        'mode': 'lines+markers',
        'type': 'scatter',
        'marker': {
            'color': 'rgba(53, 165, 176, 1.0)'
        }
    }, 1, 1)

    return fig

def graph_loss(data):
    fig = plotly.tools.make_subplots(rows=1, cols=1)
    fig['layout']['margin'] = {
        'l': 30, 'r': 10, 'b': 30, 't': 50
    }
    fig['layout']['legend'] = {'x': 1, 'y': 1, 'xanchor': 'right'}
    fig['layout']['height'] = 350
    fig['layout']['title'] = "Perdas"
    fig['layout']['yaxis']['title'] = "Valor de Perda"
    fig['layout']['xaxis']['title'] = "Épocas"

    fig.append_trace({
        'x': data['Epoch'],
        'y': data['LossQ'],
        'name': 'Perda Q',
        'mode': 'lines+markers',
        'type': 'scatter',
        'marker': {
            'color': 'rgba(250,110,102,1.0)'
        }
    }, 1, 1)
    fig.append_trace({
        'x': data['Epoch'],
        'y': data['LossPi'],
        'name': 'Perda Pi',
        'mode': 'lines+markers',
        'type': 'scatter',
        'marker': {
            'color': 'rgba(188, 136, 227,1.0)'
        }
    }, 1, 1)

    return fig

def graph_qval(data):
    fig = plotly.tools.make_subplots(rows=1, cols=1)
    fig['layout']['margin'] = {
        'l': 30, 'r': 10, 'b': 30, 't': 50
    }
    fig['layout']['legend'] = {'x': 1, 'y': 1, 'xanchor': 'right'}
    fig['layout']['height'] = 350
    fig['layout']['title'] = "Q-Valor"
    fig['layout']['yaxis']['title'] = "Q-Valor"
    fig['layout']['xaxis']['title'] = "Épocas"

    fig.append_trace({
        'name': 'Mínimo',
        'x': data['Epoch'],
        'y': data['MinQVals'],
        'marker': {
            'color': 'rgba(93,122,252,1.0)'
        }
    }, 1, 1)

    fig.append_trace({
        'name': 'Máximo',
        'x': data['Epoch'],
        'y': data['MaxQVals'],
        'fill':'tonexty',
        'fillcolor': 'rgba(143,163,255,0.5)',
        'marker': {
            'color': 'rgba(93,122,252,1.0)'
        }
    }, 1, 1)

    fig.append_trace({
        'x': data['Epoch'],
        'y': data['AverageQVals'],
        'name': 'Médio',
        'mode': 'lines+markers',
        'marker': {
            'color': 'rgba(93,122,252,1.0)'
        },
        'type': 'scatter'
    }, 1, 1)

    return fig

def graph_buysellhold(data):
    fig = plotly.tools.make_subplots(rows=1, cols=1)
    fig['layout']['margin'] = {
        'l': 30, 'r': 10, 'b': 30, 't': 50
    }
    fig['layout']['legend'] = {'x': 1, 'y': 1, 'xanchor': 'right'}
    fig['layout']['height'] = 350
    fig['layout']['title'] = "Ações Tomadas"
    fig['layout']['yaxis']['title'] = "Quantidade"
    fig['layout']['xaxis']['title'] = "Épocas"

    fig.append_trace({
        'name': 'Compra',
        'x': data['Epoch'],
        'y': data['Buy'],
        'marker': {
            'color': 'rgba(185, 115, 235, 1.0)'
        },
        'type': 'scatter',
        'mode': 'lines+markers'
    }, 1, 1)

    fig.append_trace({
        'name': 'Venda',
        'x': data['Epoch'],
        'y': data['Sell'],
        'marker': {
            'color': 'rgba(235, 115, 159, 1.0)'
        },
        'type': 'scatter',
        'mode': 'lines+markers'
    }, 1, 1)

    fig.append_trace({
        'name': 'Espera',
        'x': data['Epoch'],
        'y': data['Hold'],
        'mode': 'lines+markers',
        'marker': {
            'color': 'rgba(189, 189, 189, 1.0)'
        },
        'type': 'scatter',
        'mode': 'lines+markers'
    }, 1, 1)

    return fig

@app.callback([Output('dropdown', 'options')],
              [Input('interval', 'n_intervals')])
def get_data_dirs(n):
    base = '../data/'
    in_data = os.listdir(base)
    dir_list = []

    for root, _, files in os.walk(base):
        if len(root.split('/')) == 5:
            label = root.split('/')
            label = f"{label[2]}/{label[4]}"
            dir_list.append({'label':label, 'value':root})

    return [dir_list]


output_columns = []
output_values = []
for output in range(20):
    output_columns.append(Output('config_col-' + str(output), 'children'))
    output_values.append(Output('config_val-' + str(output), 'children'))


@app.callback(output_columns,
              [Input('dropdown', 'value')])
def config_list_columns(value):

    columns = ["Stocks and Year", "Dinheiro Inicial", "Taxas", "Lotes", "Obs Preço", "Recompensa",
               "Seed" ,"Épocas", "Passos por Época",
               "Start Steps", "Update After", "Update Every", "Batch Size",
               "Gamma", "Learning Rate Q", "Learning Rate Pi", "Polyak", "Replay Buffer Size",
               "Act Noise", "Hidden Layers"]

    return columns

@app.callback(output_values,
              [Input('dropdown', 'value')])
def config_list_values(value):
    for root, _, files in os.walk(value):
        for file in files:
            if file.endswith(".json"):
                config_file = os.path.join(root, file)

    with open(config_file) as json_file:
        data = json.load(json_file)

    if len(value.split('/')) == 5:
        path = value+'/../../config.json'
    else:
        path = value+'/config.json'

    print(path)
    with open(path) as json_file:
        data2 = json.load(json_file)
    print(data2)

    try:
        hidden_layers = data['ac_kwargs']['hidden_sizes']
    except:
        hidden_layers = '(256,256)'

    values = [value.split('/')[2], str(data2['s_money']), str(data2['taxes']),
            str(data2['allotment']), str(data2['price_obs']), str(data2['reward']),
            str(data["seed"]), str(data["epochs"]),
            str(data["steps_per_epoch"]), str(data["start_steps"]), str(data["update_after"]),
            str(data["update_every"]), str(data['batch_size']), str(data['gamma']),
            str(data['q_lr']), str(data["pi_lr"]), str(data["polyak"]),
            str(data["replay_size"]), str(data["act_noise"]), hidden_layers]

    return values


@app.callback([Output('profit', 'figure'), Output('testprofit', 'figure'), Output('loss', 'figure'),
               Output('ep_ret', 'figure'), Output('qval', 'figure'), Output('buysellhold', 'figure')],
              [Input('interval', 'n_intervals'),
               Input('dropdown', 'value')])
def get_data(n, value):
    progress = []
    result = {}
    for root, _, files in os.walk(value):
        for file in files:
            if file.endswith(".txt"):
                progress.append(os.path.join(root, file))

    last_train = progress[-1]

    try:
        columns = open(last_train).readlines()[0].split('\t')

        for i in columns:
            result[i] = []

        for line in open(last_train).readlines()[1:]:
            for i, value in enumerate(line.split('\t')):
                result[columns[i]].append(float(value))

        df_data = pandas.DataFrame(result)

        return graph_profit(df_data), graph_testprofit(df_data), graph_loss(df_data), graph_ep_ret(df_data), graph_qval(df_data), graph_buysellhold(df_data)
    except:
        fig = plotly.tools.make_subplots(rows=1, cols=1)
        return fig, fig, fig, fig, fig, fig



if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0')
