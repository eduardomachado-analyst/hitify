import os
import pathlib
import pickle
import re
import numpy as np

import dash
import dash_core_components as dcc
import dash_html_components as html
import spotipy
from dash.dependencies import Input, Output, State
import dash_daq as daq
import pandas as pd
from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyClientCredentials

app = dash.Dash(
    __name__
)

app.title = 'HITIFY'

server = app.server
app.config["suppress_callback_exceptions"] = True

APP_PATH = str(pathlib.Path(__file__).parent.resolve())
APP_PATH = str(pathlib.Path(__file__).parent.resolve())

load_dotenv('.env')

mean_hit_df = pd.read_csv('data/mean_hit.csv', index_col=0)

client_id = os.getenv('SPOTIFY_ID')
client_secret = os.getenv('SPOTIFY_SECRET')
redirect_uri = os.getenv('SPOTIFY_REDIRECT_URI')


def build_banner():
    return html.Div(
        id="banner",
        className="banner",
        children=[
            html.Div(
                id="banner-text",
                children=[
                    html.H5("HIT SONG PREDICTION"),
                    html.H6("Predicting Hit Songs With Spotify "),
                ],
            ),
            html.Div(
                id="banner-logo",
                children=[
                    html.Img(id="logo", src=app.get_asset_url("Spotify-Logotipo.png")),
                ],
            ),
        ],
    )


def build_tabs():
    return html.Div(
        id="tabs",
        className="tabs",
        children=[
            dcc.Tabs(
                id="app-tabs",
                value="tab1",
                className="custom-tabs",
                children=[
                    dcc.Tab(
                        id="Specs-tab",
                        label="About",
                        value="tab1",
                        className="custom-tab",
                        selected_className="custom-tab--selected",
                    ),
                    dcc.Tab(
                        id="Control-chart-tab",
                        label="HITIFIER",
                        value="tab2",
                        className="custom-tab",
                        selected_className="custom-tab--selected",
                    ),
                ],
            )
        ],
    )


def build_tab_1():
    return [
        html.Div(
            id="set-specs-intro-container",
            className='row',
            children=html.P(
                """The aim of this project is to use machine learning techniques to predict the likelihood of a song becoming a hit.
                The TOP HITS playlists created by Spotify from 2000 to 2020 were considered. 
                About 35000 hits and non hits had their features extracted through the Spotify Web API. 
                The LightGBM classifier model used was able to predict a hit with 73% accuracy in the test data.
                """
            ),
        ),
    ]


def generate_section_banner(title):
    return html.Div(className="section-banner", children=title)


def build_table():
    return html.Div(
        id="top-section-container",
        className="container",
        children=[
            html.Div(
                id="data-set-head",
                className="set-specs-intro-container",
                children=[
                    generate_section_banner("Features description"),
                    html.Div(
                        id="metric-div",
                        children=[
                            html.Table(make_dash_table(pd.read_csv('data/variables.csv')))
                        ],
                    ),
                ],
            ),

        ],
    )


def build_tab_2(text=None):
    return html.Div(
        id="container",
        children=[
            html.Div(
                id="title-card",
                children=[
                    html.P("Insert the Spotify URI song")
                ],
            ),
            html.Div(
                children=[
                    html.Img(id="uri", src=app.get_asset_url("uri_song.png"))
                ],
            ),
            html.Div(
                id="container",
                children=[dcc.Input(id="input_uri", type="text", placeholder="Spotify URI", value=text),
                          html.Button("Let's go!", id='submit-val', n_clicks=0)],
            ),

        ],
    )


def connect_spotify(uri):
    # scope = "user-library-read playlist-read-private user-read-playback-state user-modify-playback-state"

    # token = SpotifyOAuth(scope=scope,
    #                      client_id=client_id,
    #                      client_secret=client_secret,
    #                      redirect_uri=redirect_uri)

    # sp = spotipy.Spotify(auth_manager=token)

    client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

    uri_feat = uri.split(":")[-1]

    feat_df = pd.DataFrame(sp.audio_features(uri_feat))

    data_audio = sp.audio_analysis(uri_feat)
    duration = data_audio['track']['duration']

    columns = ['bars', 'beats', 'sections', 'segments', 'tatums']
    for col in columns:
        feat_df[col] = np.divide(len([x['duration'] for x in data_audio[col]]), duration)

    feat_df.drop(feat_df.select_dtypes('object'), axis=1, inplace=True)

    feat_df['labels'] = 0

    return feat_df


def predict_result(uri):
    if uri:
        df = connect_spotify(uri)

        danceability = df['danceability'][0] / (2 * mean_hit_df.loc['danceability'][0]) * 10
        loudness = df['loudness'][0] / (2 * mean_hit_df.loc['loudness'][0]) * 10
        speechiness = df['speechiness'][0] / (2 * mean_hit_df.loc['speechiness'][0]) * 10
        instrumentalness = df['instrumentalness'][0] / (2 * mean_hit_df.loc['instrumentalness'][0]) * 10
        valence = df['valence'][0] / (2 * mean_hit_df.loc['valence'][0]) * 10
        sections = df['sections'][0] / (2 * mean_hit_df.loc['sections'][0]) * 10

        pickle_in = open('data/model.pkl', 'rb')
        model = pickle.load(pickle_in)

        prediction = model.predict_proba(df)[:, 1][0] * 100

        return int(prediction), danceability, loudness, speechiness, instrumentalness, valence, sections
    return 0, 0, 0, 0, 0, 0, 0


def build_result(uri, n_clicks=0):
    rsl, danceability, loudness, speechiness, instrumentalness, valence, sections = 0, 0, 0, 0, 0, 0, 0
    if n_clicks > 0:
        rsl, danceability, loudness, speechiness, instrumentalness, valence, sections = predict_result(uri)
    return html.Div(
        id='container',
        children=[
            html.Div(
                daq.LEDDisplay(
                    id="operator-led",
                    value=rsl,
                    color="#ff8300",
                    backgroundColor="#090909",
                    size=50,
                )
            ),
            html.Div(className='column',
                     children=[
                         daq.Gauge(
                             showCurrentValue=True,
                             units="Hit mean",
                             value=danceability,
                             label='danceability',
                             max=10,
                             min=0,
                             color={"gradient": False,
                                    "ranges": {"#1b2039": [0, 4], "green": [4, 6], "#1b2031": [6, 10]}},
                         ),
                         daq.Gauge(
                             showCurrentValue=True,
                             units="Hit mean",
                             value=loudness,
                             label='loudness',
                             max=10,
                             min=0,
                             color={"gradient": False,
                                    "ranges": {"#1b2039": [0, 4], "green": [4, 6], "#1b2031": [6, 10]}},
                         ),
                     ],
                     ),

            html.Div(className='column',
                     children=[
                         daq.Gauge(
                             showCurrentValue=True,
                             units="Hit mean",
                             value=speechiness,
                             label='speechiness',
                             max=10,
                             min=0,
                             color={"gradient": False,
                                    "ranges": {"#1b2039": [0, 4], "green": [4, 6], "#1b2031": [6, 10]}},
                         ),
                         daq.Gauge(
                             showCurrentValue=True,
                             units="Hit mean",
                             value=instrumentalness,
                             label='instrumentalness',
                             max=10,
                             min=0,
                             color={"gradient": False,
                                    "ranges": {"#1b2039": [0, 4], "green": [4, 6], "#1b2031": [6, 10]}},
                         )
                     ],
                     ),

            html.Div(className='column',
                     children=[
                         daq.Gauge(
                             showCurrentValue=True,
                             units="Hit mean",
                             value=valence,
                             label='valence',
                             max=10,
                             min=0,
                             color={"gradient": False,
                                    "ranges": {"#1b2039": [0, 4], "green": [4, 6], "#1b2031": [6, 10]}},
                         ),
                         daq.Gauge(
                             showCurrentValue=True,
                             units="Hit mean",
                             value=sections,
                             label='sections',
                             max=10,
                             min=0,
                             color={"gradient": False,
                                    "ranges": {"#1b2039": [0, 4], "green": [4, 6], "#1b2031": [6, 10]}},
                         ),
                     ],
                     ),

        ],
    )


def make_dash_table(df):
    """ Return a dash definition of an HTML table for a Pandas dataframe """
    table = []
    for index, row in df.iterrows():
        html_row = []
        for i in range(len(row)):
            html_row.append(html.Td([row[i]]))
        table.append(html.Tr(html_row))
    return table


app.layout = html.Div(
    id="big-app-container",
    children=[
        build_banner(),
        build_tabs(),
        html.Div(
            id="app-container",
        ),
    ],
)


@app.callback(
    [Output("app-content-tab2", "children")],
    [Input('submit-val', 'n_clicks'), Input("input_uri", "value")],
)
def render_tab_content(n_clicks, text):
    if n_clicks > 0 and text:
        return (
            html.Div(
                id="app-content-tab2",
                children=[
                    html.Div(
                        id='row',
                        children=[
                            html.Div(
                                className='six columns',
                                children=[build_tab_2()]
                            ),
                            html.Div(
                                className='six columns',
                                children=build_result(text, n_clicks)
                            ),
                        ],
                    ),
                ],
            ),
        )
    else:
        return (
            html.Div(
                id="app-content-tab2",
                children=[
                    html.Div(
                        id='row',
                        children=[
                            html.Div(
                                className='five columns',
                                children=[build_tab_2(text)]
                            ),
                            html.Div(
                                className='seven columns',
                                children=build_result(text)
                            ),
                        ],
                    ),
                ],
            ),
        )


@app.callback(
    [Output("app-container", "children")],
    [Input("app-tabs", "value")],
)
def render_tab_content(tab_switch):
    if tab_switch == "tab1":
        return (
            html.Div(
                id="app-content-tab1",
                children=[
                    html.Div(
                        children=[
                            html.Div(
                                id="graphs-container",
                                children=build_tab_1(),
                            ),
                        ],
                    ),
                    html.Div(
                        children=[
                            html.Div(
                                id="graphs-container",
                                children=build_table(),
                            ),
                        ],
                    ),
                ],
            ),
        )
    return (
        html.Div(
            id="app-content-tab2",
            children=[
                html.Div(
                    id='row',
                    children=[
                        html.Div(
                            className='six columns',
                            children=[build_tab_2()]
                        ),
                        # html.Div(
                        #     className='six columns',
                        #     children=build_result()
                        # ),
                    ],
                ),
            ],
        ),
    )


# Running the server
if __name__ == "__main__":
    app.run_server(debug=True, port=8050)
