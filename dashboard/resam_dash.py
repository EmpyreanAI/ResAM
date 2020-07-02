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
from dash.dependencies import Input, Output

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div([
    html.Img(src='assets/resam.svg', width='30%'),
    
    html.Div([
        html.Div(
            
        ),
        html.Div(
            html.Ul(
                # html.Li("Khalil Gay")
            )
        )
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
    ]),

    dcc.Interval(
        id='interval',
        interval=10000,
        n_intervals=0
    )
])


def graph_testprofit(data):
    fig = plotly.tools.make_subplots(rows=1, cols=1)
    fig['layout']['margin'] = {
        'l': 30, 'r': 10, 'b': 30, 't': 50
    }
    fig['layout']['legend'] = {'x': 1, 'y': 1, 'xanchor': 'right'}
    fig['layout']['height'] = 350
    fig['layout']['title'] = "Lucro em Teste"
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
    fig['layout']['title'] = "Retorno"
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


@app.callback([Output('profit', 'figure'), Output('testprofit', 'figure'), Output('loss', 'figure'),
               Output('ep_ret', 'figure'), Output('qval', 'figure'), Output('buysellhold', 'figure')],
              [Input('interval', 'n_intervals')])
def get_data(n):
    progress = []
    result = {}
    for root, _, files in os.walk(os.path.join("../", 'data')):
        for file in files:
            if file.endswith(".txt"):
                progress.append(os.path.join(root, file))

    last_train = progress[-1]

    columns = open(last_train).readlines()[0].split('\t')

    for i in columns:
        result[i] = []

    for line in open(last_train).readlines()[1:]:
        for i, value in enumerate(line.split('\t')):
            result[columns[i]].append(float(value))
    
    df_data = pandas.DataFrame(result)

    return graph_profit(df_data), graph_testprofit(df_data), graph_loss(df_data), graph_ep_ret(df_data), graph_qval(df_data), graph_buysellhold(df_data)


if __name__ == '__main__':
    app.run_server(debug=True)