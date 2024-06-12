import dash
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import acorr_ljungbox
from prettytable import PrettyTable
from Toolbox import *
from dash import dash_table
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import numpy as np
import pandas as pd
import plotly.express as px
from statsmodels.tsa.seasonal import STL
import statsmodels.tsa.holtwinters as ets
from numpy import linalg as la
from sklearn.preprocessing import StandardScaler
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import warnings

warnings.filterwarnings('ignore')


# loading the datset
df = pd.read_csv('north.csv')
df1 = pd.read_csv('south.csv')
print(df.head(), df1.head(), df.info(), df1.info())
print('*' * 100)

df.columns = df.columns.str.strip()
df1.columns = df1.columns.str.strip()
print(df.columns, df.info(), df1.columns, df1.info())

df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str) + '-' + df['Day'].astype(str))
df.rename(columns={'index': 'Identity'}, inplace=True)
df1['Date'] = pd.to_datetime(df1['Year'].astype(str) + '-' + df1['Month'].astype(str) + '-' + df1['Day'].astype(str))
df1.rename(columns={'index': 'Identity'}, inplace=True)
print(df.head(), df1.head(), df.info(), df1.info())

print('*' * 100)

my_app = dash.Dash('My App', external_stylesheets= [dbc.themes.BOOTSTRAP])
server = my_app.server

Sidebar_style = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

content_style = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

sidebar = html.Div([
    html.H2(f"Menu", className="display-15"),
    html.Hr(),
    dbc.Nav(
        [
            dbc.NavLink('Information about the Daily sea ice extent', href = "/", active= "exact"),
            dbc.NavLink('Data, Data and all about data', href="/page-one", active="exact"),
            dbc.NavLink('Data preprocessing and stationarity', href="/page-two", active="exact"),
            dbc.NavLink('Time series decomposition and Holt-Winters method', href="/page-three", active="exact"),
            dbc.NavLink('Feature selection and collinearity', href="/page-four", active="exact"),
            dbc.NavLink('Base Model', href="/page-five", active="exact"),
            dbc.NavLink('Multiple Linear Regression', href="/page-six", active="exact" ),
            dbc.NavLink('ARMA/ARIMA', href="/page-seven", active="exact" ),
            dbc.NavLink('SARIMA', href="/page-eight", active="exact" ),
            dbc.NavLink('Conclusion', href="/page-nine", active="exact" ),
        ],
        vertical= True,
        pills= True,
    ),
],
style= Sidebar_style,
)

content = html.Div(id="page-content", children=[], style = content_style)
image_path ='assets/Image.jpeg'
image1_path = 'assets/Image1.jpeg'
image2_path = 'assets/articnews.png'
image3_path = 'assets/sc.png'
image4_path = 'assets/antarticmap.png'
image5_path = 'assets/Ant.png'
image6_path = 'assets/both.png'
image7_path = 'assets/MLR.png'
image8_path = 'assets/MLR2.png'
image9_path = 'assets/Gpac_N.png'
image10_path = 'assets/Gpac_S.png'
image11_path = 'assets/conclusion1.png'


dropdown_menu = dcc.Dropdown(
    id='dropdown',
    options=[
        {'label': 'Arctic', 'value': 'Arctic'},
        {'label': 'Antarctic', 'value': 'Antarctic'}
    ],
    value='Arctic'  # Set a default value
)

train,test = train_test_split(df,test_size=0.2,shuffle=False)
X = df[['Identity', 'Year', 'Month', 'Day', 'Missing 10^6 sq km']]
y = df['Extent 10^6 sq km']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

train1,test1 = train_test_split(df1,test_size=0.2,shuffle=False)
X1 = df1[['Identity', 'Year', 'Month', 'Day', 'Missing 10^6 sq km']]
y1 = df1['Extent 10^6 sq km']
x1_train, x1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, shuffle=False)

#northern
#create a new table
table = PrettyTable()
table.field_names = [
    "Methods&Models", "MSE TRAIN", "MSE TEST", "RMSE TRAIN", "RMSE TEST",
    "Q-VALUE", "MEAN Of Residual", "MEAN Of Forecasted",
    "VARIANCE Of Residual", "VARIANCE Of Forecasted"
]
table.add_row(["Holt-Winter's Method", 0.00475, 116.701, 0.06896, 10.8028, 2932.549, -7.70385, 9.9234, 0.00475, 18.2271])
table.add_row(["Average Method", 10.3631, 13.2878, 3.2191, 3.6452, 2935.3854, -0.4710, -1.1179, 10.1412,  12.0379])
table.add_row(["Naive Method", 0.02712, 12.6711, 0.1646, 3.5596, 2936.3971, -5.3782e-05, 0.7957, 0.02712, 12.0379])
table.add_row(["Drift Method", 0.0088,12.8449, 0.0939, 3.5839, 2936.4296, -0.00160, 0.87804, 0.0088, 12.0739])
table.add_row(["Simple Exponential smoothing Method", 13.8441, 12.5993, 3.7207, 3.5495, 2936.3955, 1.8674, 0.7492, 10.3590, 12.0379])
table.add_row(["Multiple Linear Regression", 5.8112, 7.0924, 2.4106, 2.6631, 2886.5243, 0.01312, -1.1087, 7.0922, 5.8112])
table.add_row(["ARMA (1, 1)", 0.0073, 11.9849, 0.0859, 3.4619, 2936.1499, -0.00014, -0.24470, 0.00739, 11.9250])
table.add_row(["SARIMA(1,0,1)(1,0,365)", 0.0025, 23.7334, 0.0505, 4.8717, 2935.7328, 0.0009, -1.7973, 0.0025, 20.5028])
data1 = {}
for col in table.field_names:
    data1[col] = [row[table.field_names.index(col)] for row in table._rows]
data1 = pd.DataFrame(data1)
table_html = table.get_html_string()

#antartic
table2 = PrettyTable()
# Define columns
table2.field_names = ["Methods&Models", "MSE TRAIN", "MSE TEST", "RMSE TRAIN", "RMSE TEST", "Q-VALUE", "MEAN Of Residual", "MEAN Of Forecasted", "VARIANCE Of Residual", "VARIANCE Of Forecasted"]
# Add rows
table2.add_row(["Holt-Winter's Method", 0.0117, 381.3051, 0.1083, 19.5270, 2933.6204, 8.9988e-06, -17.9126, 0.01173, 60.4415])
table2.add_row(["Average Method", 10.7033, 12.8211, 3.2715, 3.7176, 2935.3854, -0.01840, -1.3353, 10.7030,  12.0379])
table2.add_row(["Naive Method", 0.0822, 61.5345, 0.2867, 7.8443, 2939.6603, -0.00010, -5.3030, 0.0822, 33.4122])
table2.add_row(["Drift Method", 0.02536,59.8427, 0.1592, 7.7358, 2939.4564, -0.2122, -5.8747, 75.5931, 12.1087])
table2.add_row(["Simple Exponential smoothing Method", 52.7551, 60.6333, 7.2632, 7.7867, 2936.6617, -4.6082, -5.2173, 31.5277, 33.4122])
table2.add_row(["Multiple Linear Regression", 13.1136, 16.1189, 3.6212, 4.0148, 2830.5515, -1.1665, 3.5159e-12, 14.7580, 13.113])
table2.add_row(["ARMA (1, 0)", 75.52, 29.86, 8.690, 5.46, 2936.5473, -0.217, -4.0324, 75.477, 13.6023])
table2.add_row(["SARIMA(1,0,1)(0,0,365)", 0.0049, 56.639, 0.0705, 7.5259, 2932.8995, 0.0009, -0.2232, 0.0049, 0.004971])
data2 ={}
for col in table2.field_names:
    data2[col] = [row[table.field_names.index(col)] for row in table2._rows]
data2 = pd.DataFrame(data2)
table_html = table2.get_html_string()

my_app.layout = html.Div(
    [
        dcc.Location(id="url"), sidebar, content
    ]
)

@my_app.callback(Output("page-content", "children"),
                 [Input("url", "pathname"),
                  ])
def render_page_content(pathname):
    if pathname == f"/":
        return html.Div(
            children= [
                html.H1("Daily Sea Ice Extent in Hemispheres", className='display-4', style={'font-weight': 'bold'}),
                html.H2("About the climate changes", style={"font-weight": "bold"})
,
            html.P("Presently, one of the most widely discussed and recognized subjects globally is climate change. From elementary school textbooks to daily newspapers, climate change has emerged as a prominent issue in the 21st century."),
            html.P("According to NASA's publication on climate change, 'The impacts of human-induced global warming are currently observable, irreversible for the current generation, and will exacerbate with ongoing human emissions of greenhouse gases into the atmosphere.' Observable consequences, as forecasted by scientists, include the depletion of sea ice, the thawing of glaciers and ice sheets, rising sea levels, and heightened occurrences of severe heatwaves."),
            html.P("Projections from the scientific community indicate a sustained rise in global temperatures due to anthropogenic greenhouse gas emissions. Furthermore, the anticipated escalation and intensification of severe weather events contribute to an increased risk of substantial damage. The realization of forecasted consequences, such as diminishing sea ice, accelerated sea level rise, and prolonged, more severe heatwaves, underscores the urgency of addressing global climate change."),
            html.Img(src=image_path, style={"width": '100%','height':'100%'}),
            html.Hr(),
            html.H2("Research on sea ice extent in hemispheres", style= {"font-weight": "bold"}),
            html.P("Sea ice also plays a fundamental role in polar ecosystems. When the ice melts in the summer, it releases nutrients into the water, stimulating the growth of phytoplankton, the center of the marine food web. As the ice melts, it exposes ocean water to sunlight, spurring photosynthesis in phytoplankton. When ice freezes, the underlying water gets saltier and sinks, mixing the water column and bringing nutrients to the surface. The ice itself is habitat for animals such as seals, Arctic foxes, polar bears, and penguins."),
            html.Img(src=image1_path, style={"width": '100%', 'height': '100%'}),
            html.Br(),
            html.Br(),
            html.H2("Artic sea Ice ", className = "display-20"),
            html.P("Global air temperature records date back to the 1880s and can offer a stand-in (proxy) for Arctic sea ice conditions; but such temperature records were initially collected at just 11 locations. Russia’s Arctic and Antarctic Research Institute has compiled ice charts since 1933.Today, scientists studying Arctic sea ice trends can rely on a fairly comprehensive record dating back to 1953. They use a combination of satellite records, shipping records, and ice charts from several countries."),
            html.P("Arctic sea ice occupies an ocean basin mostly enclosed by land. Because there is no landmass at the North Pole, sea ice extends all the way to the pole, making the ice subject to the most extreme oscillations between wintertime darkness and summertime sunlight. Likewise, because the ocean basin is surrounded by land, ice has less freedom of movement to drift into lower latitudes and melt. Sea ice also forms in areas south of the Arctic Ocean in winter, including the Sea of Okhotsk, the Bering Sea, Baffin Bay, Hudson Bay, the Greenland Sea, and the Labrador Sea"),
            html.P("According to the research by NASA earth observatory, In March and September each year, Arctic sea ice typically reaches its maximum and minimum extents, respectively, historically spanning approximately 14-16 million square kilometers in late winter to about 7 million square kilometers each September. Recent years, however, have seen significantly lower figures. The dominant cause of atmospheric variability in the North Pole region over years to decades is the Arctic Oscillation (AO), which influences air mass shifts between polar and mid-latitude regions, impacting the strength of prevailing westerly winds and storm tracks. During the 'positive' AO phase, winds intensify, enlarging leads in the ice pack, while during 'negative' phases, winds weaken, affecting the movement of multiyear ice. Despite this relationship, recent years have shown a weakened correlation between the AO and summer sea ice extents, with factors beyond the AO exerting influence."),
            html.Img(src = image3_path, style={"width": '100%', 'height': '100%'}),
            html.P("Since 1979, the monthly September ice extent has declined 13.4 percent per decade relative to the average from 1981 to 2010. (NASA Earth Observatory graph by Joshua Stevens, based on data from the National Snow and Ice Data Center"),
            html.Br(),
            html.Br(),
            html.P("So, On March 14, 2024, Arctic sea ice likely reached its maximum extent for the year, at 15.01 million square kilometers (5.80 million square miles), the fourteenth lowest extent in the satellite record. This year’s maximum extent is 640,000 square kilometers (247,000 square miles) below the 1981 to 2010 average maximum of 15.65 million square kilometers (6.04 million square miles) and 600,000 square kilometers (232,000 square miles) above the lowest maximum of 14.41 million square kilometers (5.56 million square miles) set on March 7, 2017. The date of the maximum this year, March 14, was two days later than the 1981 to 2010 average date of March 12."),
            html.Img(src=image2_path, style={"width": '100%', 'height': '100%'}),
            html.P(dcc.Markdown("For reference, check this website for more information [https://nsidc.org/arcticseaicenews/], [https://earthobservatory.nasa.gov/features/SeaIce](https://community.plotly.com/)")),
            html.H2("Antartic sea Ice ", className="display-20"),
            html.P("The Antarctic is in some ways the opposite of the Arctic. The Arctic is an ocean basin surrounded by land, with the sea ice corralled in the coldest, darkest part of the Northern Hemisphere. The Antarctic is a continent surrounded by ocean. Whereas Northern Hemisphere sea ice can extend from the North Pole to a latitude of 45°N (along the northeast coasts of Asia and North America), most of the ice is found above 70°N. Southern Hemisphere sea ice does not get that close to the South Pole; it fringes the continent and reaches to 55°S latitude at its greatest extent."),
            html.P("Because of this geography, Antarctic sea ice coverage is larger than the Arctic’s in winter, but smaller in the summer. Total Antarctic sea ice peaks in September—the end of Southern Hemisphere winter—historically rising to an extent of roughly 17-20 million square kilometers (about 6.6-7.7 million square miles). Ice extent reaches its minimum in February, when it dips to roughly 3-4 million square kilometers (about 1.2-1.5 million square miles)"),
            html.P("To comprehend Antarctic sea ice dynamics, researchers categorize the ice pack into five sectors: the Weddell Sea, Indian Ocean, western Pacific Ocean, Ross Sea, and Bellingshausen and Amundsen seas. Variability across these sectors is influenced by geographic and climatic diversity, making it challenging to generalize the impact of climate patterns on the entire Southern Hemisphere ice pack. Atmospheric oscillations, particularly the Antarctic Oscillation, drive fluctuations in sea ice extent, altering wind patterns and temperature distribution. These oscillations can cause complex responses across different sectors, with positive phases strengthening prevailing westerly winds and negative phases leading to shifts in ice distribution."),
            html.Img(src=image4_path, style={"width": '75%', 'height': '50%'}),
            html.Br(),
            html.Br(),
            html.P("Antarctic sea ice extent exhibits notable variability, with fluctuations observed both annually and across sectors. While overall annual extent has shown a slight increase since 1979, fluctuations are evident, with some sectors experiencing decreases, particularly the Bellingshausen and Amundsen seas near the Antarctic Peninsula. This regional variation complicates predictions of future Antarctic ice trends in the face of ongoing global warming. Despite these complexities, long-term declines are expected as temperatures rise, albeit at a slower pace compared to the Arctic region."),
            html.Img(src=image5_path, style={"width": '75%', 'height': '100%'}),
            html.Br(),
            html.Br(),
            html.P("The increase in Antarctic sea ice, while intriguing, raises questions amidst global warming trends. The smaller magnitude of this increase, coupled with uncertainties, contrasts starkly with the pronounced decline in Arctic sea ice. Additionally, the already extensive summer melt of Antarctic ice lessens the potential impact of further reductions, unlike the dramatic consequences a similar scenario would pose in the Arctic. Understanding these dynamics requires careful consideration of various factors, including atmospheric changes, ocean circulation patterns, and the interplay between sea ice and ice shelves, with implications for global sea level rise and climate stability."),
            html.P(dcc.Markdown("For further reference, check this website for more information [https://www.climate.gov/news-features/event-tracker/2023-antarctic-sea-ice-winter-maximum-lowest-record-wide-margin], [https://earthobservatory.nasa.gov/features/SeaIce#:~:text=an%20important%20factor.-,Antarctic%20Sea%20Ice,-The%20Antarctic%20is]"),
                   ),])
    elif pathname == f"/page-one":
        return html.Div(
            [
                html.H3("Talking about the dataset and references",  style= {"font-weight": "bold"}),
                html.P("This project is a complied work of two separate datasets called the ""Northern Hemisphere"" and the ""Southern Hemisphere"" which typically means the artic hemisphere and the antartic hemisphere."),
                html.P("Sourced from the National Snow & Ice Data Center (NSIDC), this dataset emerges as a valuable repository for comprehending and scrutinizing global climate patterns, particularly in examining the repercussions of climate change on polar regions."),
                html.P("The Daily Sea Ice Extent dataset furnishes daily assessments of sea ice extent in both the Northern and Southern Hemispheres, measured in million square kilometers. It comprises two distinct files: N_seaice_extent_daily_v3.0.csv and S_seaice_extent_daily_v3.0.csv, containing data for the Northern Hemisphere and Southern Hemisphere, respectively."),
                html.P("These files encompass records spanning various years, identified by the Year column. The assessments are logged on a daily basis, enabling the tracking of alterations in sea ice coverage across time."),
                html.P("The dataset has 14691 data points (ranging between the time period of 1978-10-26 to 2023 -07-23) with 7 columns."),
                html.Img(src=image6_path, style = {'width': '100%', 'height': '100'}),
                html.P("The First Image shows the information about the raw data of Northern Hemisphere whereas the second image shows the information about the Southern Hemisphere"),
                html.Br(),
                html.Hr(),
                html.P("Select the hemisphere you would like to see the time series plot"),
                html.Br(),
                dropdown_menu,  # Include the dropdown menu here
                html.Br(),
                dcc.Graph(id='line', style={'width': '800px', 'height': '400px', 'margin': '0 auto'}),
                html.Br(),
                html.H2("References"),
                html.P(dcc.Markdown("1. The background suspects a publication where the author used Recurrent neural network models in order to do the daily scale prediction of artic sea ice Concentration. [https://www.mdpi.com/2077-1312/11/12/2319]")),
                html.P(dcc.Markdown("2. A publication on communications earth and environment has stated that the Antarctic sea ice coverage has remained at, or near, record low values during 2023. [https://www.nature.com/articles/s43247-023-00961-9]")),
                html.P(dcc.Markdown("3. Monthly Climate Timeseries: Northern Hemisphere Sea ice extent/area. [https://psl.noaa.gov/data/timeseries/monthly/NHICE/]"))
            ]
        )
    elif pathname == f"/page-two":
        return html.Div([
            html.H3("Data Preprocessing and Stationarity",  style= {"font-weight": "bold"}),
            html.H3("Preprocessing"),
            html.P('Choose an option:'),
            dcc.Dropdown(
                id='dropdown',
                options=[
                    {'label': 'Artic', 'value': 'artic'},
                    {'label': 'Antartic', 'value': 'antartic'}
                ],
                placeholder="select one",
                value='preprocess',
                style= {"width": "40%"},
            ),
            html.Plaintext(id = "preprocess", style={'backgroundColor': 'white', 'color': 'black', 'font-size': '15px', 'textAlign': 'left'}),
            html.Hr(),
            html.H3('Stationarity analysis', style= {'font-weight': 'bold'}),
            html.P('Northern hemisphere(Artic) analysis for stationarity:', style= {'font-weight': 'bold'}, className="display-20"),
            html.Hr(),
            #artic analysis
            dcc.RadioItems(id='radioitems1',
            options=[ {'label': 'Raw data of Northern hemisphere', 'value': 'raw-data'},
                      {'label': 'First Order Differencing', 'value': '1st-order'},
                      {'label': 'Second Order Differencing', 'value': '2nd-order'},
            ], inline = True, value='raw-data', style={'backgroundColor': 'white'}, inputStyle={"margin-right": "10px"}),
            html.Br(),
            html.P("Augmented Dickey-Fuller test (Adf) test statistics:", style={'backgroundColor': 'white', 'font-weight': 'bold'}),
            html.Div(id="testout", style={'backgroundColor': 'white', 'color': 'black', 'font-size': '15px', 'textAlign': 'left'}),
            html.Br(),
            html.P("Kwiatkowski-Phillips-Schmidt-Shin test (Kpss) test statistics:",
                   style={'backgroundColor': 'white', 'font-weight': 'bold'}),
            html.Div(id="testout1", style={'backgroundColor': 'white', 'color': 'black', 'font-size': '15px', 'textAlign': 'left'}),
            html.Br(),
            dcc.Graph(id="testout2"),
            html.Br(),
            dcc.Graph(id="testout3"),
            html.Br(),
            dcc.Graph(id="testout4"),
            html.Br(),
            dcc.Graph(id="testout5"),
            html.Br(),
            html.Hr(),
            html.P("The result: ", style={'font-weight': 'bold'}),
            html.Div(id="testout6", style = {'backgroundColor': 'white', 'color': 'black', 'font-size': '15px', 'textAlign': 'left'}),
            html.Hr(),
            html.P('Southern hemisphere(Antartic) analysis for stationarity:', style={'font-weight': 'bold'}, className="display-20"),
            html.Hr(),
            # amtartic analysis
            dcc.RadioItems(id='radioitems2',
                           options=[{'label': 'Raw data of Southern hemisphere', 'value': 'raw-data1'},
                                    {'label': 'First Order Differencing', 'value': '1st-order-ant'},
                                    {'label': 'Second Order Differencing', 'value': '2nd-order-ant'},
                                    ], inline= True, value='raw-data1', style={'backgroundColor': 'white'},inputStyle={"margin-right": "10px"} ),
            html.Br(),
            html.P("Augmented Dickey-Fuller test (Adf) test statistics:",
                   style={'backgroundColor': 'white', 'font-weight': 'bold'}),
            html.Div(id="testant",
                     style={'backgroundColor': 'white', 'color': 'black', 'font-size': '15px', 'textAlign': 'left'}),
            html.Br(),
            html.P("Kwiatkowski-Phillips-Schmidt-Shin test (Kpss) test statistics:",
                   style={'backgroundColor': 'white', 'font-weight': 'bold'}),
            html.Div(id="testant1",
                     style={'backgroundColor': 'white', 'color': 'black', 'font-size': '15px', 'textAlign': 'left'}),
            html.Br(),
            dcc.Graph(id="testant2"),
            html.Br(),
            dcc.Graph(id="testant3"),
            html.Br(),
            dcc.Graph(id="testant4"),
            html.Br(),
            dcc.Graph(id="testant5"),
            html.Br(),
            html.Hr(),
            html.P("The result: ", style={'font-weight': 'bold'}),
            html.Div(id="testant6",
                     style={'backgroundColor': 'white', 'color': 'black', 'font-size': '15px', 'textAlign': 'left'}),

        ]
        )
    elif pathname == f"/page-three":
        return html.Div([
            html.H3("Decomposition and Holt-winter's method",  style= {"font-weight": "bold", "text-align": "center"}),
            html.H3('Seasonal and Trend decomposition using Loess', style= {"text-align": "center"}),
            html.P('STL (Seasonal and Trend decomposition using Loess) decomposition is a technique used to decompose a time series into three components: seasonal, trend, and remainder (residual). This decomposition is helpful in understanding the underlying patterns and structures within a time series data.', style= {"text-align": "center"}),
            html.Hr(),
            html.P('Choose a hemisphere to see the STL decompositions:', style= {"text-align": "center"}),
            dcc.Dropdown(
                id='dropdown_stl',
                options=[
                    {'label': 'Artic', 'value': 'artic'},
                    {'label': 'Antartic', 'value': 'antartic'}
                ],
                placeholder="select one",
                value='artic',
                style={"width": "40%", 'margin': 'auto'},
            ),
            html.Br(),
            html.Br(),
            html.Hr(),
            dcc.Graph(id = 'Iterations', style={'width': '800px', 'height': '400px', 'margin': '0 auto'}),
            html.Br(),
            html.Br(),
            dcc.Graph(id='TRS', style={'width': '800px', 'height': '400px', 'margin': '0 auto'}),
            html.Br(),
            html.Hr(),
            html.Div(id = 'strength',style={'backgroundColor': 'white', 'color': 'black', 'font-size': '15px', 'textAlign': 'center'}),
            html.Br(),
            html.Div(id = 'Trend', style={'backgroundColor': 'white', 'color': 'black', 'font-size': '15px', 'textAlign': 'center'}),
            html.Hr(),
            html.Br(),
            dcc.Graph(id='seasonally adjusted', style={'width': '800px', 'height': '400px', 'margin': '0 auto'}),
            html.Br(),
            html.Br(),
            dcc.Graph(id='detrended', style={'width': '800px', 'height': '400px', 'margin': '0 auto'}),
            html.Br(),
            html.Br(),
            html.Hr(),
            html.H3('Holt-winters Method', style= {"text-align": "center"}),
            html.P("Holt-Winters method, named after its developers Peter Winters and Charles Holt, is a time series forecasting technique that extends simple exponential smoothing to capture seasonality and trends in data. It involves modeling the level, trend, and seasonality components to provide accurate predictions for future values in time series data.", style= {"text-align": "center"}),
            html.Br(),
            html.P('Select an option: ', style= {"text-align": "center"}),
            dcc.Dropdown(
                id='dropdown_hwt',
                options=[
                    {'label': 'Artic', 'value': 'artic'},
                    {'label': 'Antartic', 'value': 'antartic'}
                ],
                placeholder="select one",
                value='artic',
                style={"width": "40%", 'margin': 'auto'},
            ),
            html.Br(),
            html.Br(),
            html.Hr(),
            dcc.Graph(id='HW-model', style={'width': '800px', 'height': '400px', 'margin': '0 auto'}),
            html.Br(),
            dcc.Graph(id='Acf-res', style={'width': '800px', 'height': '400px', 'margin': '0 auto'}),
            html.Br(),
            dcc.Graph(id='Acf-fore', style={'width': '800px', 'height': '400px', 'margin': '0 auto'}),
            html.Br(),
            html.Hr(),
            html.Div(id='perf-hw',
                     style={'backgroundColor': 'white', 'color': 'black', 'font-size': '15px', 'textAlign': 'center'}),
            html.Hr()
        ])
    elif pathname == f"/page-four":
        return html.Div([
            html.H3("Feature selection and collinearity",  style= {"font-weight": "bold", "text-align": "left"}),
            html.H3("Feature selection", style= {"text-align": "left"}),
            html.P('Select an option: ', style={"text-align": "left"}),
            dcc.Dropdown(
                id='dropdown_fc',
                options=[
                    {'label': 'Artic', 'value': 'artic'},
                    {'label': 'Antartic', 'value': 'antartic'}
                ],
                placeholder="select one",
                value='artic',
                style={"width": "70%"},
            ),
            html.Br(),
            html.Br(),
            html.Hr(),
            html.Div(id='svdc',
                     style={'backgroundColor': 'white', 'color': 'black', 'font-size': '15px', 'textAlign': 'left'}),
            html.Hr(),
            html.Div(id='output-original-model'),
            html.Br(),
            html.Br(),
            html.P("*Click the 'Run Stepwise Regression' button in order to see the Model after the backward stepwise regression.*"),
            html.Button('Run Stepwise Regression', id='button', style={'textAlign': 'center'}),
            html.Br(),
            html.Br(),
            html.Div(id='output-container-button-clicked'),
            html.H3("Collinearity removing processes"),
            html.Div(id = 'removal', style={'backgroundColor': 'white', 'color': 'black', 'font-size': '15px', 'textAlign': 'left'}),

        ]
        )
    elif pathname == f"/page-five":
        return html.Div([
            html.H3("Base models",  style= {"font-weight": "bold"}),
            html.Hr(),
            html.Br(),
            html.P("There are four types of base models: "),
            html.P("Base models, also known as baseline models, are simple models used as references for evaluating the performance of more complex models in machine learning and statistical analysis. They serve as benchmarks against which the performance of sophisticated models can be compared."),
            html.P(" * Average method "),
            html.P(" * Drift Method "),
            html.P(" * Naive Method "),
            html.P(" * Simple Exponential Smoothing Method "),
            html.Br(),
            html.H3("Average Method:"), html.P("In this type of base model, predictions are made based on the average or base value of the target variable across the entire dataset. For regression problems, the base model might predict the mean or median of the target variable. For classification problems, the base model might predict the majority class for all instances."),
            html.Br(),
            html.H3("Drift Method:"), html.P("A drift model predicts the future value of a variable based on the current value and a drift factor, often represented as a constant. This type of model assumes that the variable will change by a fixed amount over each time step. Drift models are commonly used in time series analysis and forecasting."),
            html.Br(),
            html.H3("Naive Method:"), html.P("The naive model makes predictions based on very simple assumptions or heuristics. In classification, a naive model might assign probabilities to classes based on their frequency in the training data, without considering any relationships between features. In regression, a naive model might assume a simple linear relationship between the input features and the target variable."),
            html.Br(),
            html.H3("Simple Exponential Smoothing Method: "), html.P("SES is a time series forecasting method that uses a weighted average of past observations, with exponentially decreasing weights as observations become older. It is particularly useful for data with a trend and seasonal patterns. SES assigns different weights to past observations, giving more weight to recent observations and less weight to older ones."),
            html.Hr(),
            html.P("Choose the hemisphere:"),
            dcc.Dropdown(
                id='dropdown_bm',
                options=[
                    {'label': 'Artic', 'value': 'artic'},
                    {'label': 'Antartic', 'value': 'antartic'}
                ],
                placeholder="select one",
                value='artic',
                style={"width": "100%"},
            ),
            html.Br(),
            html.P("Choose the Model: "),
            dcc.Dropdown(
                id='dropdown_model',
                options=[
                    {'label': 'Average Model', 'value': 'average'},
                    {'label': 'Drift Model', 'value': 'drift'},
                    {'label': 'Naive Model', 'value': 'naive'},
                    {'label': 'Simple Exponential Smoothing Model', 'value': 'ses'},
                ],
                clearable= True,
                placeholder="select the model",
                value='average',
                style={"width": "100%"},
            ),
            dcc.Graph(id='model'),
            html.Br(),
            dcc.Graph(id='testvpred'),
            html.Br(),
            dcc.Graph(id='res-plot'),
            html.Br(),
            dcc.Graph(id='fore-plot'),
            html.Br(),
            html.Hr(),
            html.Div(id ="perf-model", style={'backgroundColor': 'white', 'color': 'black', 'font-size': '15px', 'textAlign': 'left'}),
            html.Hr(),
        ]
        )
    elif pathname == f"/page-six":
        return html.Div([
            html.H3("Multiple Linear Regression",  style= {"font-weight": "bold"}),
            html.P("A multiple linear regression model is a statistical method used to analyze the relationship between multiple independent variables (also called predictors, features, or regressors) and a single dependent variable (also called the target or response variable). It extends the concept of simple linear regression, where only one independent variable is used to predict the dependent variable, to situations where there are multiple predictors."),
            html.P("The general form of a multiple linear regression model with p predictors is:"),
            html.Img(src=image7_path, style={"width": '65%', 'height': '70%'}),
            html.P("The goal of multiple linear regression is to estimate the coefficients that minimize the sum of squared errors between the observed and predicted values of the dependent variable."),
            html.Img(src=image8_path, style={"width": '65%', 'height': '70%'}),
            html.Br(),
            html.Hr(),
            html.P("Select the hemisphere:"),
            dcc.Dropdown(
                id='dropdown_mlr',
                options=[
                    {'label': 'Artic', 'value': 'artic'},
                    {'label': 'Antartic', 'value': 'antartic'}
                ],
                placeholder="select one",
                value='artic',
                style={"width": "100%"},
            ),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Hr(),
            html.P("F-Test:"),
            html.Div( id = 'f-value'),
            html.Br(),
            html.Div(id = 'p1-value'),
            html.Br(),
            html.Hr(),
            html.Div(id='mlr-model'),
            html.Br(),
            html.Hr(),
            dcc.Graph("mlr-model1"),
            dcc.Graph("res-mlr"),
            dcc.Graph("fore-mlr"),
            html.Br(),
            html.Hr(),
            html.Div(id='perf-mlr'),
            html.Br(),
            html.Hr(),

        ]
        )
    elif pathname == f"/page-seven":
        return html.Div([
            html.H3("ARMA/ARIMA",  style= {"font-weight": "bold"}),
            html.P("Select the hemisphere:"),
            dcc.Dropdown(
                id='dropdown_arsa',
                options=[
                    {'label': 'Artic', 'value': 'artic'},
                    {'label': 'Antartic', 'value': 'antartic'}
                ],
                placeholder="select one",
                value='artic',
                style={"width": "100%"},
            ),
            html.Br(),
            dcc.Graph(id = 'acf-order'),
            html.Br(),
            html.Div(id = 'Image', style={"width": '70%', 'height': '70%'} ),
            html.Br(),
            html.Hr(),
            html.Div(id='estimated1'),
            html.Br(),
            html.Div(id='model_sum'),
            html.Br(),
            html.Hr(),
            html.Div(id = 'para-div'),
            html.Hr(),
            html.Br(),
            dcc.Graph(id='acf-arma-res'),
            html.Br(),
            dcc.Graph(id='acf-arma-fore'),
            html.Br(),
            dcc.Graph(id='acf-arma11'),
            html.Br(),
            dcc.Graph(id='acf-testvfore'),
            html.Br(),
            dcc.Graph(id='acf-trainvspred'),
            html.Br(),
            html.Hr(),
            html.Div(id='arma-performance'),
            html.Br(),
            html.Hr(),
            html.Div(id='arma-chi'),
            html.Br(),
            html.Hr(),
            html.Div(id='arma-confi'),
            html.Hr(),
            html.Div(id='arma-zeropoles'),
            html.Hr(),
            html.Div(id='cov-mat'),
            html.Hr(),
            html.Div(id='std-err'),
            html.Hr(),
            html.Div(id='err-std'),
            html.Br(),
        ])
    elif pathname == f"/page-eight":
        return html.Div([
            html.H3("SARIMA and Forecasting",  style= {"font-weight": "bold"}),
            html.P("Select the hemisphere:"),
            dcc.Dropdown(
                id='dropdown_sarima',
                options=[
                    {'label': 'Artic', 'value': 'artic'},
                    {'label': 'Antartic', 'value': 'antartic'}
                ],
                placeholder="select one",
                value='artic',
                style={"width": "100%"},
            ),
            html.Br(),
            html.Hr(),
            html.Div(id = "Model-fit"),
            html.Br(),
            html.Hr(),
            dcc.Graph(id ="Acf-res-sarima"),
            html.Br(),
            html.Hr(),
            dcc.Graph(id ="Acf-fore-sarima"),
            html.Br(),
            html.Hr(),
            html.Div(id ="Perf-sarima"),
            html.Br(),
            html.Hr(),
            dcc.Graph(id ="model-sarima"),
            html.Br(),
            html.Hr(),
            dcc.Graph(id ="testvpred-sarima"),
            html.Br(),
            html.Hr(),
            dcc.Graph(id ="trainvpred-sarima"),
            html.Br(),
            html.Hr(),
            html.Br(),
            html.H3("Finding the best Model for Forecasting of Northern Hemisphere"),
            html.Hr(),
            dash_table.DataTable(
                data= data1.to_dict('records'),
                columns=[{'id': c, 'name': c} for c in data1.columns],
                page_size=10
            ),
            html.Br(),
            html.P('Considering the metrics carefully,\n'
                   'The mean squared error of Sarima model is 0.0025 for train set which is less when compared to other models\n'
                   'The mean sqaured error of Multi linear regression is 7.0924 in test set which is less when compared to other models\n'
                   'The root mean squared error of sarima model is 0.0505 for train set which is less when compared to other models\n'
                   'The root mean sqaured error of Multi linear regression is 2.4106 in test set which is less when compared to other models\n'
                   'Since, both the models work best on the train and test, I am considering the all criterias such as the acf plot, model capturing the test and prediction\n'
                   'While considering all the criteria, it is evident that the sarima model outperforms among all the models.\n'),
            html.H3("Finding the best Model for Forecasting of Southern Hemisphere"),
            html.Hr(),
            dash_table.DataTable(
                data= data2.to_dict('records'),
                columns=[{'id': c, 'name': c} for c in data2.columns],
                page_size=10
            ),
            html.Br(),
            html.P('Considering the metrics carefully,\n'
                   'The mean squared error of Sarima model is 0.0049 for train set which is less when compared to other models.\n'
                   'The mean sqaured error of Multi linear regression is 16.1189 in test set which is less when compared to other models.\n'
                   'The root mean squared error of sarima model is 0.0705 for train set which is less when compared to other models.\n'
                   'The root mean sqaured error of Multi linear regression is 4.0148 in test set which is less when compared to other models.\n'
                   'Since, both the models work best on the train and test, I am considering the all criterias such as the acf plot, model capturing the test and prediction.\n'
                   'While considering all the criteria, it is evident that the sarima model outperforms among all the models.\n'),

            html.H3("Forecasting the Selected Model"),
            html.P("Select the hemisphere:"),
            dcc.Dropdown(
                id='dropdown_sarimap',
                options=[
                    {'label': 'Artic', 'value': 'artic'},
                    {'label': 'Antartic', 'value': 'antartic'}
                ],
                placeholder="select one",
                value='artic',
                style={"width": "100%"},
            ),
            html.Br(),
            dcc.Graph(id = 'testvsprediction')
        ]
        )

    elif pathname == f"/page-nine":
        return html.Div([
            html.H3("Conclusion",  style= {"font-weight": "bold"}),
            html.Hr(),
            html.Br(),
            html.P("The preservation and accurate prediction of sea ice extent hold profound significance for understanding and mitigating the impacts of climate change. Sea ice serves as a vital component of Earth's climate system, influencing oceanic and atmospheric circulation patterns, regulating global temperatures, and providing crucial habitat for diverse ecosystems and species. Moreover, sea ice extent serves as a key indicator of climate variability and long-term climate trends. Consequently, reliable forecasts of sea ice extent are essential for informing climate policy, facilitating sustainable resource management, and safeguarding vulnerable ecosystems and communities."),
            html.Br(),
            html.Img(src = image11_path, style={"width": '100%', 'height': '100%'}),
            html.Br(),
            html.P("After comprehensive evaluation, the SARIMA model emerged as the most robust predictor of sea ice extent in the hemispheres. Its selection was informed by meticulous analysis of its performance on both training and test datasets. Notably, in comparison to other models, including the Multilinear Regression model, SARIMA demonstrated superior accuracy in forecasting sea ice extent."),
            html.Br(),
            html.P("The SARIMA model, configured with a seasonality of 365 and further disaggregated into 12 segments, exhibited exceptional predictive capability on both datasets. Its forecasts closely aligned with the observed sea ice extent, as corroborated by data from the National Snow and Ice Data Center's Arctic Sea Ice Graph."),
            html.Br(),
            html.P("Moving forward, the development of an interactive user dashboard represents a significant enhancement, offering stakeholders access to comprehensive information on dataset attributes and model performance. Furthermore, the potential to automate model updates in response to evolving sea ice extent data stands as a noteworthy avenue for future development, promising heightened precision in forecasting future sea ice extents."),
            html.H3("Author Information"),
            html.P("Publishsed By Pon swarnalaya Ravichandran"),
            html.P("Contact : ponswarnalaya.r@gmail.com"),
        ]
        )


@my_app.callback(
    Output('line', 'figure'),
    [Input('dropdown', 'value')]  # Assuming you have a dropdown for selecting data
)
def update_time_series_graph(input1):
    if input1 == 'Arctic':

        # Create time series line plot for the Arctic dataset using Plotly Express
        fig1 = px.line(df, x='Date', y='Extent 10^6 sq km', title='Arctic Time Series Graph')

        return fig1
    elif input1 == 'Antarctic':

        # Create time series line plot for the Antarctic dataset using Plotly Express
        fig = px.line(df1, x='Date', y='Extent 10^6 sq km', title='Antarctic Time Series Graph')

        return fig

@my_app.callback(
 Output('preprocess', 'children'),
    [Input('dropdown', 'value')]
)

def update_tab_two(input1):
    if input1 == 'artic':
        d = df.isnull().sum()
        df.dropna(inplace=True)
        return f"{d}\nDataset doesn't have missing values"

    elif input1 == "antartic":
        d1 = df1.isnull().sum()
        df1.dropna(inplace=True)
        return f"{d1}\nDataset doesn't have missing values"


@my_app.callback(
        Output('testout', 'children'),
               Output('testout1', 'children'),
               Output('testout2', 'figure'),
               Output('testout3', 'figure'),
               Output('testout4', 'figure'),
               Output('testout5', 'figure'),
               Output('testout6', 'children'),
               [Input('radioitems1', 'value')
                ]
)

def test_drop(inp):
    if inp == "raw-data":
        ad = ADF_cal(df["Extent 10^6 sq km"])
        kp = kpss_test(df["Extent 10^6 sq km"])

        #rolling mean and variance
        x1 = df['Extent 10^6 sq km']
        y1 = df['Date']
        rolling_mean = []
        rolling_variance = []
        for i in range(1, len(x1) + 1):
            result = np.mean(x1[:i])
            result_variance = np.var(x1[:i])
            rolling_mean.append(result)
            rolling_variance.append(result_variance)

        # Create a Plotly figure for rolling mean
        fig_mean = go.Figure()
        fig_mean.add_trace(go.Scatter(x=y1, y=rolling_mean, mode='lines', marker=dict(color='yellow')))
        fig_mean.update_layout(title="Plot of Rolling Mean", xaxis_title="Samples", yaxis_title="Rolling Mean")

        # Create a Plotly figure for rolling variance
        fig_variance = go.Figure()
        fig_variance.add_trace(go.Scatter(x=y1, y=rolling_variance, mode='lines', marker=dict(color='purple')))
        fig_variance.update_layout(title="Plot of Rolling Variance", xaxis_title="Samples",
                                   yaxis_title="Rolling Variance")


        #acf
        data = df['Extent 10^6 sq km']
        max_lag = 50
        acf_values = sm.tsa.acf(data, nlags=max_lag)
        lags = np.arange(0, max_lag + 1)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=lags, y=acf_values, mode='markers', marker=dict(color='blue'), name='ACF'))
        fig.add_trace(go.Scatter(x=-1*lags, y=acf_values, mode='markers', marker=dict(color='blue'), name='ACF'))
        # Add stems
        for lag, acf_value in zip(lags, acf_values):
            fig.add_trace(go.Scatter(x=[lag,lag], y=[0,acf_value], mode='lines', line=dict(color='blue', width=1)))
        for lag, acf_value in zip(-1*lags, acf_values):
            fig.add_trace(go.Scatter(x=[lag,lag], y=[0,acf_value], mode='lines', line=dict(color='blue', width=1)))
        # Update layout
        fig.update_layout(
            title="Autocorrelation Function (ACF) Stem Plot",
            xaxis_title="Lags",
            yaxis_title="Autocorrelation",
            plot_bgcolor='rgba(0,0,0,0)'
        )

        #acf/pacf
        y = df['Extent 10^6 sq km']
        lags = 400
        acf = sm.tsa.stattools.acf(y, nlags=lags)
        pacf = sm.tsa.stattools.pacf(y, nlags=lags)
        trace_acf = go.Bar(x=np.arange(1, lags + 1), y=acf[1:])
        trace_pacf = go.Bar(x=np.arange(1, lags + 1), y=pacf[1:])
        fig1 = go.Figure(data=[trace_acf, trace_pacf])
        fig1.update_layout(
            title=f'ACF/PACF of the raw data: {lags}',
            xaxis_title='Lags',
            yaxis_title='Autocorrelation',
            barmode='group',
            plot_bgcolor='rgba(0,0,0,0)'
        )

        #result
        resulttt = "The ADF statistic (0.0000) and the KPSS test result (0.01000) provide conflicting indications regarding stationarity. However, further examination using alternative stationarity checks such as rolling mean, rolling variance, and ACF/PACF suggests that the data is not stationary. The rolling mean and variance suggests that the data is not stable and doesn't have a flat point at 0. The acf plot of raw data shows that the data doesn't have white noise. The acf/pacf shows that the data is seasonal and converges over time. Therefore, it was reasonable to proceed with First Order Differencing."

        return ad, kp, fig_mean, fig_variance, fig, fig1, resulttt

    elif inp == "1st-order":
        df.set_index('Date', inplace=True)
        df.reset_index(inplace=True)

        #diffferncing-one
        diff1 = difference(df['Extent 10^6 sq km'], interval=365)
        diff_df = pd.DataFrame(diff1, index=df.index[365:])

        #adf&kpss
        adfff = ADF_cal(diff_df)
        kp_1 = kpss_test(diff_df)

        # rolling mean and variance
        x11 = diff_df[0]
        y11 = diff_df.index
        rolling_mean1 = []
        rolling_variance1 = []
        for i in range(1, len(x11) + 1):
            result1 = np.mean(x11[:i])
            result_variance1 = np.var(x11[:i])
            rolling_mean1.append(result1)
            rolling_variance1.append(result_variance1)

        # Create a Plotly figure for rolling mean
        fig_mean1 = go.Figure()
        fig_mean1.add_trace(go.Scatter(x=y11, y=rolling_mean1, mode='lines', marker=dict(color='yellow')))
        fig_mean1.update_layout(title="Plot of Rolling Mean", xaxis_title="Samples", yaxis_title="Rolling Mean")

        # Create a Plotly figure for rolling variance
        fig_variance1 = go.Figure()
        fig_variance1.add_trace(go.Scatter(x=y11, y=rolling_variance1, mode='lines', marker=dict(color='purple')))
        fig_variance1.update_layout(title="Plot of Rolling Variance", xaxis_title="Samples",
                                   yaxis_title="Rolling Variance")

        # acf
        data1 = diff1
        max_lag = 50
        acf_values1 = sm.tsa.acf(data1, nlags=max_lag)
        lags = np.arange(0, max_lag + 1)
        fig_11 = go.Figure()
        fig_11.add_trace(go.Scatter(x=lags, y=acf_values1, mode='markers', marker=dict(color='blue'), name='ACF'))
        fig_11.add_trace(go.Scatter(x=-1 * lags, y=acf_values1, mode='markers', marker=dict(color='blue'), name='ACF'))
        # Add stems
        for lag, acf_value in zip(lags, acf_values1):
            fig_11.add_trace(go.Scatter(x=[lag, lag], y=[0, acf_value], mode='lines', line=dict(color='blue', width=1)))
        for lag, acf_value in zip(-1 * lags, acf_values1):
            fig_11.add_trace(go.Scatter(x=[lag, lag], y=[0, acf_value], mode='lines', line=dict(color='blue', width=1)))
        # Update layout
        fig_11.update_layout(
            title="Autocorrelation Function (ACF) Stem Plot",
            xaxis_title="Lags",
            yaxis_title="Autocorrelation",
            plot_bgcolor='rgba(0,0,0,0)'
        )

        # acf/pacf
        y_1 = diff1
        lags = 400
        acf = sm.tsa.stattools.acf(y_1, nlags=lags)
        pacf = sm.tsa.stattools.pacf(y_1, nlags=lags)
        trace_acf = go.Bar(x=np.arange(1, lags + 1), y=acf[1:])
        trace_pacf = go.Bar(x=np.arange(1, lags + 1), y=pacf[1:])
        fig1_1 = go.Figure(data=[trace_acf, trace_pacf])
        fig1_1.update_layout(
            title=f'ACF/PACF of the raw data: {lags}',
            xaxis_title='Lags',
            yaxis_title='Autocorrelation',
            barmode='group',
            plot_bgcolor='rgba(0,0,0,0)'
        )

        #result
        resultt = "The ADF test indicates strong evidence against non-stationarity, with the test statistic significantly lower than critical values and a very small p-value. Conversely, the KPSS test yields conflicting results, with its statistic below critical values but a p-value above the typical threshold. Despite this, further tests are needed for a conclusive assessment. Additionally, the plots of rolling mean and variance suggest data instability, while the ACF plot indicates the absence of white noise and the ACF/PACF plot reveals a seasonal pattern at 365 intervals."
        return adfff, kp_1, fig_mean1, fig_variance1, fig_11, fig1_1, resultt

    elif inp == "2nd-order":
        # diff2
        # diffferncing-one
        diff1 = difference(df['Extent 10^6 sq km'], interval=365)
        diff_df = pd.DataFrame(diff1, index=df.index[365:])
        diff2 = difference(diff1, 1)
        diff_df1 = pd.DataFrame(diff1, df.index[365:])  # Adjust the index range
        diff_df2 = pd.DataFrame(diff2, df.index[366:])

        diff_df21 = pd.DataFrame(diff2, index=df.index[366:],
                                 columns=['Extent 10^6 sq km'])  # Adjust column name if needed

        #adf
        # adf&kpss
        adf_12 = ADF_cal(diff_df21)
        kp_12 = kpss_test(diff_df21)

        #rolling mean and variance
        x12 = diff_df21['Extent 10^6 sq km']
        y12 = diff_df21.index
        rolling_mean2 = []
        rolling_variance2 = []
        for i in range(1, len(x12) + 1):
            result1 = np.mean(x12[:i])
            result_variance2 = np.var(x12[:i])
            rolling_mean2.append(result1)
            rolling_variance2.append(result_variance2)

        # Create a Plotly figure for rolling mean
        fig_mean2 = go.Figure()
        fig_mean2.add_trace(go.Scatter(x=y12, y=rolling_mean2, mode='lines', marker=dict(color='yellow')))
        fig_mean2.update_layout(title="Plot of Rolling Mean", xaxis_title="Samples", yaxis_title="Rolling Mean")

        # Create a Plotly figure for rolling variance
        fig_variance2 = go.Figure()
        fig_variance2.add_trace(go.Scatter(x=y12, y=rolling_variance2, mode='lines', marker=dict(color='purple')))
        fig_variance2.update_layout(title="Plot of Rolling Variance", xaxis_title="Samples",
                                    yaxis_title="Rolling Variance")

        #acf

        data2 = diff2
        max_lag = 50
        acf_values12 = sm.tsa.acf(data2, nlags=max_lag)
        lags = np.arange(0, max_lag + 1)
        fig_12 = go.Figure()
        fig_12.add_trace(go.Scatter(x=lags, y=acf_values12, mode='markers', marker=dict(color='blue'), name='ACF'))
        fig_12.add_trace(go.Scatter(x=-1 * lags, y=acf_values12, mode='markers', marker=dict(color='blue'), name='ACF'))
        # Add stems
        for lag, acf_value in zip(lags, acf_values12):
            fig_12.add_trace(go.Scatter(x=[lag, lag], y=[0, acf_value], mode='lines', line=dict(color='blue', width=1)))
        for lag, acf_value in zip(-1 * lags, acf_values12):
            fig_12.add_trace(go.Scatter(x=[lag, lag], y=[0, acf_value], mode='lines', line=dict(color='blue', width=1)))
        # Update layout
        fig_12.update_layout(
            title="Autocorrelation Function (ACF) Stem Plot",
            xaxis_title="Lags",
            yaxis_title="Autocorrelation",
            plot_bgcolor='rgba(0,0,0,0)'
        )

        #acf and pacf
        # acf/pacf
        y_12 = diff2
        lags = 400
        acf = sm.tsa.stattools.acf(y_12, nlags=lags)
        pacf = sm.tsa.stattools.pacf(y_12, nlags=lags)
        trace_acf2 = go.Bar(x=np.arange(1, lags + 1), y=acf[1:])
        trace_pacf2 = go.Bar(x=np.arange(1, lags + 1), y=pacf[1:])
        fig1_12 = go.Figure(data=[trace_acf2, trace_pacf2])
        fig1_12.update_layout(
            title=f'ACF/PACF of the raw data: {lags}',
            xaxis_title='Lags',
            yaxis_title='Autocorrelation',
            barmode='group',
            plot_bgcolor='rgba(0,0,0,0)'
        )

        #final result
        result = "The ADF test's p-value of 0.000000 strongly indicates non-stationarity, whereas the KPSS test's p-value of 0.100000 suggests potential trend or level stationarity. However, considering the rolling mean and variance, the data appears stable with a consistent pattern over time. Additionally, the presence of white noise in the acf plot and the observed seasonal cycle in the acf/pacf plot further support the notion of stationarity, albeit with seasonal variations."
        return adf_12, kp_12, fig_mean2, fig_variance2, fig_12, fig1_12, result

@my_app.callback(
        Output('testant', 'children'),
               Output('testant1', 'children'),
               Output('testant2', 'figure'),
               Output('testant3', 'figure'),
               Output('testant4', 'figure'),
               Output('testant5', 'figure'),
               Output('testant6', 'children'),
               [Input('radioitems2', 'value')
                ]
)
def drop_test_2(input3):
    if input3 == "raw-data1":
        ad_ant = ADF_cal(df1["Extent 10^6 sq km"])
        kp_ant = kpss_test(df1["Extent 10^6 sq km"])

        #rolling mean and variance
        x1_ant = df1['Extent 10^6 sq km']
        y1_ant = df1['Date']
        rolling_mean_ant = []
        rolling_variance_ant = []
        for i in range(1, len(x1_ant) + 1):
            result_ant = np.mean(x1_ant[:i])
            result_variance_ant = np.var(x1_ant[:i])
            rolling_mean_ant.append(result_ant)
            rolling_variance_ant.append(result_variance_ant)

        # Create a Plotly figure for rolling mean
        fig_mean_ant = go.Figure()
        fig_mean_ant.add_trace(go.Scatter(x=y1_ant, y=rolling_mean_ant, mode='lines', marker=dict(color='yellow')))
        fig_mean_ant.update_layout(title="Plot of Rolling Mean", xaxis_title="Samples", yaxis_title="Rolling Mean")

        # Create a Plotly figure for rolling variance
        fig_variance_ant = go.Figure()
        fig_variance_ant.add_trace(go.Scatter(x=y1_ant, y=rolling_variance_ant, mode='lines', marker=dict(color='purple')))
        fig_variance_ant.update_layout(title="Plot of Rolling Variance", xaxis_title="Samples",
                                   yaxis_title="Rolling Variance")

        # acf
        data_ant = df1['Extent 10^6 sq km']
        max_lag = 50
        acf_values_ant = sm.tsa.acf(data_ant, nlags=max_lag)
        lags = np.arange(0, max_lag + 1)
        fig_ant = go.Figure()
        fig_ant.add_trace(go.Scatter(x=lags, y=acf_values_ant, mode='markers', marker=dict(color='blue'), name='ACF'))
        fig_ant.add_trace(go.Scatter(x=-1 * lags, y=acf_values_ant, mode='markers', marker=dict(color='blue'), name='ACF'))
        # Add stems
        for lag, acf_value in zip(lags, acf_values_ant):
            fig_ant.add_trace(go.Scatter(x=[lag, lag], y=[0, acf_value], mode='lines', line=dict(color='blue', width=1)))
        for lag, acf_value in zip(-1 * lags, acf_values_ant):
            fig_ant.add_trace(go.Scatter(x=[lag, lag], y=[0, acf_value], mode='lines', line=dict(color='blue', width=1)))
        # Update layout
        fig_ant.update_layout(
            title="Autocorrelation Function (ACF) Stem Plot",
            xaxis_title="Lags",
            yaxis_title="Autocorrelation",
            plot_bgcolor='rgba(0,0,0,0)'
        )

        # acf/pacf
        y_ant = df1['Extent 10^6 sq km']
        lags = 400
        acf_ant = sm.tsa.stattools.acf(y_ant, nlags=lags)
        pacf_ant = sm.tsa.stattools.pacf(y_ant, nlags=lags)
        trace_acf = go.Bar(x=np.arange(1, lags + 1), y=acf_ant[1:])
        trace_pacf = go.Bar(x=np.arange(1, lags + 1), y=pacf_ant[1:])
        fig1_ant = go.Figure(data=[trace_acf, trace_pacf])
        fig1_ant.update_layout(
            title=f'ACF/PACF of the raw data: {lags}',
            xaxis_title='Lags',
            yaxis_title='Autocorrelation',
            barmode='group',
            plot_bgcolor='rgba(0,0,0,0)'
        )

        # result
        resulttt_ant = "The ADF test statistic (-17.807437) is significantly lower than the critical values at all common significance levels, with a p-value (0.000000) indicating strong evidence against non-stationarity. However, the KPSS test statistic (0.020546) is below the critical values but has a p-value (0.100000) exceeding the typical threshold of 0.05, suggesting potential trend or level stationarity. The rolling mean and variance plot reveals fluctuating data without a stable point, indicating non-stationarity. Additionally, the ACF plot exhibits non-white noise behavior, and the ACF/PACF plots show convergence over time. These findings collectively imply the data lacks stationarity, warranting to proceed further with first order differencing"

        return ad_ant, kp_ant, fig_mean_ant, fig_variance_ant, fig_ant, fig1_ant, resulttt_ant

    elif input3 == "1st-order-ant":
        df1.set_index('Date', inplace=True)
        df1.reset_index(inplace=True)

        # diffferncing-one
        diff_ant1 = difference(df1['Extent 10^6 sq km'], interval=365)
        diff_df1 = pd.DataFrame(diff_ant1, index=df.index[365:])

        # adf&kpss
        adf_ant1 = ADF_cal(diff_df1)
        kp_ant1 = kpss_test(diff_df1)

        # rolling mean and variance
        x11_ant = diff_df1[0]
        y11_ant = diff_df1.index
        rolling_mean1_ant = []
        rolling_variance1_ant = []
        for i in range(1, len(x11_ant) + 1):
            result1_ant = np.mean(x11_ant[:i])
            result_variance1_ant = np.var(x11_ant[:i])
            rolling_mean1_ant.append(result1_ant)
            rolling_variance1_ant.append(result_variance1_ant)

        # Create a Plotly figure for rolling mean
        fig_mean1_ant = go.Figure()
        fig_mean1_ant.add_trace(go.Scatter(x=y11_ant, y=rolling_mean1_ant, mode='lines', marker=dict(color='yellow')))
        fig_mean1_ant.update_layout(title="Plot of Rolling Mean", xaxis_title="Samples", yaxis_title="Rolling Mean")

        # Create a Plotly figure for rolling variance
        fig_variance1_ant = go.Figure()
        fig_variance1_ant.add_trace(go.Scatter(x=y11_ant, y=rolling_variance1_ant, mode='lines', marker=dict(color='purple')))
        fig_variance1_ant.update_layout(title="Plot of Rolling Variance", xaxis_title="Samples",
                                    yaxis_title="Rolling Variance")

        # acf
        data1_ant = diff_ant1
        max_lag = 50
        acf_values1_ant = sm.tsa.acf(data1_ant, nlags=max_lag)
        lags = np.arange(0, max_lag + 1)
        fig_11_ant = go.Figure()
        fig_11_ant.add_trace(go.Scatter(x=lags, y=acf_values1_ant, mode='markers', marker=dict(color='blue'), name='ACF'))
        fig_11_ant.add_trace(go.Scatter(x=-1 * lags, y=acf_values1_ant, mode='markers', marker=dict(color='blue'), name='ACF'))
        # Add stems
        for lag, acf_value in zip(lags, acf_values1_ant):
            fig_11_ant.add_trace(go.Scatter(x=[lag, lag], y=[0, acf_value], mode='lines', line=dict(color='blue', width=1)))
        for lag, acf_value in zip(-1 * lags, acf_values1_ant):
            fig_11_ant.add_trace(go.Scatter(x=[lag, lag], y=[0, acf_value], mode='lines', line=dict(color='blue', width=1)))
        # Update layout
        fig_11_ant.update_layout(
            title="Autocorrelation Function (ACF) Stem Plot",
            xaxis_title="Lags",
            yaxis_title="Autocorrelation",
            plot_bgcolor='rgba(0,0,0,0)'
        )

        # acf/pacf
        y_1_ant = diff_ant1
        lags = 400
        acf_ant_1 = sm.tsa.stattools.acf(y_1_ant, nlags=lags)
        pacf_ant_1 = sm.tsa.stattools.pacf(y_1_ant, nlags=lags)
        trace_acf_ant_1 = go.Bar(x=np.arange(1, lags + 1), y=acf_ant_1[1:])
        trace_pacf_ant_1 = go.Bar(x=np.arange(1, lags + 1), y=pacf_ant_1[1:])
        fig1_1_ant = go.Figure(data=[trace_acf_ant_1, trace_pacf_ant_1])
        fig1_1_ant.update_layout(
            title=f'ACF/PACF of the raw data: {lags}',
            xaxis_title='Lags',
            yaxis_title='Autocorrelation',
            barmode='group',
            plot_bgcolor='rgba(0,0,0,0)'
        )

        # result
        resultt_ant = "The ADF test statistic (-14.846734) shows strong evidence against non-stationarity, with a very small corresponding p-value (0.000000), while the KPSS test yields conflicting results, indicating potential trend or level stationarity with a p-value (0.100000) exceeding 0.05. Additional stationarity tests may be needed for a conclusive assessment. The rolling mean and variance plots suggest that the data is not stationarity, as they lack a flat point and exhibit instability with fluctuations. Moreover, the ACF plot does not show white noise, and the ACF/PACF plots indicate a seasonal cycle at 365 with convergence over time. Hence, proceeding with the second-order differencing"
        return adf_ant1, kp_ant1, fig_mean1_ant, fig_variance1_ant, fig_11_ant, fig1_1_ant, resultt_ant

    elif input3 == '2nd-order-ant':
        # diff2
        # diffferncing-one
        diff_ant1 = difference(df1['Extent 10^6 sq km'], interval=365)
        diff_df_ant = pd.DataFrame(diff_ant1, index=df1.index[365:])
        diff_ant2 = difference(diff_ant1, 1)
        diff_df1_ant = pd.DataFrame(diff_ant1, df1.index[365:])  # Adjust the index range
        diff_df2_ant = pd.DataFrame(diff_ant2, df1.index[366:])

        diff_df21_ant = pd.DataFrame(diff_ant2, index=df.index[366:],
                                 columns=['Extent 10^6 sq km'])  # Adjust column name if needed

        # adf
        # adf&kpss
        adf_12_ant = ADF_cal(diff_df21_ant)
        kp_12_ant = kpss_test(diff_df21_ant)

        #rolling mean and variance
        # rolling mean and variance
        x12_ant = diff_df21_ant['Extent 10^6 sq km']
        y12_ant = diff_df21_ant.index
        rolling_mean2_ant = []
        rolling_variance2_ant = []
        for i in range(1, len(x12_ant) + 1):
            result2_ant = np.mean(x12_ant[:i])
            result_variance2_ant = np.var(x12_ant[:i])
            rolling_mean2_ant.append(result2_ant)
            rolling_variance2_ant.append(result_variance2_ant)

        # Create a Plotly figure for rolling mean
        fig_mean2_ant = go.Figure()
        fig_mean2_ant.add_trace(go.Scatter(x=y12_ant, y=rolling_mean2_ant, mode='lines', marker=dict(color='yellow')))
        fig_mean2_ant.update_layout(title="Plot of Rolling Mean", xaxis_title="Samples", yaxis_title="Rolling Mean")

        # Create a Plotly figure for rolling variance
        fig_variance2_ant = go.Figure()
        fig_variance2_ant.add_trace(go.Scatter(x=y12_ant, y=rolling_variance2_ant, mode='lines', marker=dict(color='purple')))
        fig_variance2_ant.update_layout(title="Plot of Rolling Variance", xaxis_title="Samples",
                                    yaxis_title="Rolling Variance")

        # acf

        data2_ant = diff_ant2
        max_lag = 50
        acf_values12_ant = sm.tsa.acf(data2_ant, nlags=max_lag)
        lags = np.arange(0, max_lag + 1)
        fig_12_ant = go.Figure()
        fig_12_ant.add_trace(go.Scatter(x=lags, y=acf_values12_ant, mode='markers', marker=dict(color='blue'), name='ACF'))
        fig_12_ant.add_trace(go.Scatter(x=-1 * lags, y=acf_values12_ant, mode='markers', marker=dict(color='blue'), name='ACF'))
        # Add stems
        for lag, acf_value in zip(lags, acf_values12_ant):
            fig_12_ant.add_trace(go.Scatter(x=[lag, lag], y=[0, acf_value], mode='lines', line=dict(color='blue', width=1)))
        for lag, acf_value in zip(-1 * lags, acf_values12_ant):
            fig_12_ant.add_trace(go.Scatter(x=[lag, lag], y=[0, acf_value], mode='lines', line=dict(color='blue', width=1)))
        # Update layout
        fig_12_ant.update_layout(
            title="Autocorrelation Function (ACF) Stem Plot",
            xaxis_title="Lags",
            yaxis_title="Autocorrelation",
            plot_bgcolor='rgba(0,0,0,0)'
        )

        # acf and pacf
        # acf/pacf
        y_12_ant = diff_ant2
        lags = 400
        acf2_ant = sm.tsa.stattools.acf(y_12_ant, nlags=lags)
        pacf2_ant = sm.tsa.stattools.pacf(y_12_ant, nlags=lags)
        trace_acf2_ant = go.Bar(x=np.arange(1, lags + 1), y=acf2_ant[1:])
        trace_pacf2_ant = go.Bar(x=np.arange(1, lags + 1), y=pacf2_ant[1:])
        fig1_12_ant = go.Figure(data=[trace_acf2_ant, trace_pacf2_ant])
        fig1_12_ant.update_layout(
            title=f'ACF/PACF of the raw data: {lags}',
            xaxis_title='Lags',
            yaxis_title='Autocorrelation',
            barmode='group',
            plot_bgcolor='rgba(0,0,0,0)'
        )

        # final result
        result_ant = "The ADF test indicates strong evidence against non-stationarity, with a test statistic significantly lower than critical values and a p-value of 0.000000. However, the KPSS test yields conflicting results, suggesting potential trend or level stationarity. Additional analysis, including further stationarity tests, may be needed for a definitive conclusion. The rolling mean and variance support stationarity, showing stability and a flat point at 0. Additionally, the ACF plot exhibits white noise, while the ACF/PACF plots reveal clear seasonality every 365 cycles and convergence over time."

        return adf_12_ant, kp_12_ant, fig_mean2_ant, fig_variance2_ant,fig_12_ant, fig1_12_ant, result_ant

@my_app.callback(
        Output('Iterations', 'figure'),
               Output('TRS', 'figure'),
               Output('strength', 'children'),
               Output('Trend', 'children'),
               Output('seasonally adjusted', 'figure'),
               Output('detrended', 'figure'),
               [Input('dropdown_stl', 'value')
                ]
)

def update_stl_decomposition(input4):
    if input4 == 'artic':
        ind = pd.date_range('1978-10-26', periods=len(df), freq='1D')  # Assuming data is recorded every 1 hour

        Extent = pd.Series(np.array(df['Extent 10^6 sq km']), index=ind)
        STL_1 = STL(Extent)
        res_1 = STL_1.fit()
        # Create traces for the trend, seasonal, and residuals
        trace_trend = go.Scatter(x=ind, y=res_1.trend, mode='lines', name='Trend')
        trace_seasonal = go.Scatter(x=ind, y=res_1.seasonal, mode='lines', name='Seasonal')
        trace_residuals = go.Scatter(x=ind, y=res_1.resid, mode='lines', name='Residuals')

        # Create the plotly figure
        fig = go.Figure(data=[trace_trend, trace_seasonal, trace_residuals])

        # Update layout
        fig.update_layout(
            title="STL Decomposition",
            xaxis_title="Iterations",
            template="plotly_dark"
        )
        T = res_1.trend
        S = res_1.seasonal
        R = res_1.resid

        fig1 = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=("Trend", "Residual", "Seasonal"))
        fig1.add_trace(go.Scatter(x=ind, y=T, mode='lines', name='Trend'), row=1, col=1)
        fig1.add_trace(go.Scatter(x=ind, y=R, mode='lines', name='Residual'), row=2, col=1)
        fig1.add_trace(go.Scatter(x=ind, y=S, mode='lines', name='Seasonal'), row=3, col=1)
        fig1.update_layout(
            xaxis_title="Iterations",
            yaxis_title="STL",
            title="Trend, Seasonality, and Residuals",
            template="plotly_dark",
            showlegend=False
        )

        # Strength of trend
        var = 1 - (np.var(R) / np.var(T + R))
        Ft = f"Strength of trend: {max([0, var])}"
        #print("Strength of trend:", Ft)

        # Strength of seasonality
        var1 = 1 - (np.var(R) / np.var(S + R))
        Fs = f"Strength of seasonality: {max([0, var1])}"
        #print("Strength of seasonality:", Fs)

        # Seasonally adjusted data
        seasonally_adj = Extent - S
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=ind, y=df['Extent 10^6 sq km'], mode='lines', name='Original'))
        fig2.add_trace(go.Scatter(x=ind, y=seasonally_adj, mode='lines', name='Adjusted'))
        fig2.update_layout(
            xaxis_title="Time",
            yaxis_title="Extent 10^6 sq km",
            title="Seasonally Adjusted vs. Original",
            template="plotly_dark"
        )

        # Detrended data
        detrended = Extent - T
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=ind, y=df['Extent 10^6 sq km'], mode='lines', name='Original'))
        fig3.add_trace(go.Scatter(x=ind, y=detrended, mode='lines', name='Detrended'))
        fig3.update_layout(
            xaxis_title="Time",
            yaxis_title="Extent 10^6 sq km",
            title="Detrended vs. Original Data",
            template="plotly_dark"
        )

        return fig, fig1, Fs, Ft, fig2, fig3

    elif input4 == "antartic":
        ind1 = pd.date_range('1978-10-26', periods=len(df), freq='1D')  # Assuming data is recorded every 1 hour

        Extent2 = pd.Series(np.array(df1['Extent 10^6 sq km']), index=ind1)
        STL_2 = STL(Extent2)
        res_2 = STL_2.fit()
        # Create traces for the trend, seasonal, and residuals
        trace_trend2 = go.Scatter(x=ind1, y=res_2.trend, mode='lines', name='Trend')
        trace_seasonal2 = go.Scatter(x=ind1, y=res_2.seasonal, mode='lines', name='Seasonal')
        trace_residuals2 = go.Scatter(x=ind1, y=res_2.resid, mode='lines', name='Residuals')

        # Create the plotly figure
        fig_ant_st1 = go.Figure(data=[trace_trend2, trace_seasonal2, trace_residuals2])

        # Update layout
        fig_ant_st1.update_layout(
            title="STL Decomposition",
            xaxis_title="Iterations",
            template="plotly_dark"
        )
        T2 = res_2.trend
        S2 = res_2.seasonal
        R2 = res_2.resid

        fig1_ant_st = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=("Trend", "Residual", "Seasonal"))
        fig1_ant_st.add_trace(go.Scatter(x=ind1, y=T2, mode='lines', name='Trend'), row=1, col=1)
        fig1_ant_st.add_trace(go.Scatter(x=ind1, y=R2, mode='lines', name='Residual'), row=2, col=1)
        fig1_ant_st.add_trace(go.Scatter(x=ind1, y=S2, mode='lines', name='Seasonal'), row=3, col=1)
        fig1_ant_st.update_layout(
            xaxis_title="Iterations",
            yaxis_title="STL",
            title="Trend, Seasonality, and Residuals",
            template="plotly_dark",
            showlegend=False
        )

        # Strength of trend
        var_ant1 = 1 - (np.var(R2) / np.var(T2 + R2))
        Ft2 = f"Strength of trend: {max([0, var_ant1])}"
        # print("Strength of trend:", Ft)

        # Strength of seasonality
        var1_ant2 = 1 - (np.var(R2) / np.var(S2 + R2))
        Fs2 = f"Strength of seasonality: {max([0, var1_ant2])}"
        # print("Strength of seasonality:", Fs)

        # Seasonally adjusted data
        seasonally_adj_2 = Extent2 - S2
        fig2_ant_st = go.Figure()
        fig2_ant_st.add_trace(go.Scatter(x=ind1, y=df1['Extent 10^6 sq km'], mode='lines', name='Original'))
        fig2_ant_st.add_trace(go.Scatter(x=ind1, y=seasonally_adj_2, mode='lines', name='Adjusted'))
        fig2_ant_st.update_layout(
            xaxis_title="Time",
            yaxis_title="Extent 10^6 sq km",
            title="Seasonally Adjusted vs. Original",
            template="plotly_dark"
        )

        # Detrended data
        detrended2 = Extent2 - T2
        fig3_ant_st = go.Figure()
        fig3_ant_st.add_trace(go.Scatter(x=ind1, y=df1['Extent 10^6 sq km'], mode='lines', name='Original'))
        fig3_ant_st.add_trace(go.Scatter(x=ind1, y=detrended2, mode='lines', name='Detrended'))
        fig3_ant_st.update_layout(
            xaxis_title="Time",
            yaxis_title="Extent 10^6 sq km",
            title="Detrended vs. Original Data",
            template="plotly_dark"
        )

        return fig_ant_st1, fig1_ant_st, Ft2, Fs2, fig2_ant_st, fig3_ant_st

@my_app.callback(
        Output('HW-model', 'figure'),
               Output('Acf-res', 'figure'),
               Output('Acf-fore', 'figure'),
               Output('perf-hw', 'children'),
               [Input('dropdown_hwt', 'value')
                ]
)

def Holt_winter_model(input5):
    if input5 == 'artic':

        holtw1 = ets.ExponentialSmoothing(y_train, damped_trend=True, trend='add', seasonal='add', seasonal_periods=365)
        holtw = holtw1.fit()

        # train
        holtw_pred_train = holtw.predict(start=y_train.index[0], end=y_train.index[-1])
        holtw_df_train = pd.DataFrame(holtw_pred_train, columns=['Extent 10^6 sq km'], index=y_train.index)

        ##test
        holtw_pred_test = holtw.predict(start=y_test.index[0], end=y_test.index[-1])
        holtw_df_test = pd.DataFrame(holtw_pred_test, columns=['Extent 10^6 sq km'], index=y_test.index)

        # plot of hwmodel
        fig_hw = make_subplots(rows=1, cols=1)
        fig_hw.add_trace(
            go.Scatter(x=y_train.index, y=y_train, mode='lines', name='Train'),
        )
        fig_hw.add_trace(
            go.Scatter(x=y_test.index, y=y_test, mode='lines', name='Test'),
        )
        fig_hw.add_trace(
            go.Scatter(x=holtw_df_test.index, y=holtw_df_test['Extent 10^6 sq km'], mode='lines',
                       name='Holt-Winters Prediction (Test)'),
        )
        fig_hw.update_layout(
            xaxis_title="Time",
            yaxis_title="Extent",
            title="Holts Winter Model",
            height=600,
            width=1000,
        )

        #plot of residual error
        HW_reserror=y_train-holtw_df_train['Extent 10^6 sq km']

        data1_hwres = HW_reserror.values
        max_lag = 60
        acf_values1_hw = sm.tsa.acf(data1_hwres, nlags=max_lag)
        lags = np.arange(0, max_lag + 1)
        fig_1_hwres = go.Figure()
        fig_1_hwres.add_trace(
            go.Scatter(x=lags, y=acf_values1_hw, mode='markers', marker=dict(color='blue'), name='ACF'))
        fig_1_hwres.add_trace(
            go.Scatter(x=-1 * lags, y=acf_values1_hw, mode='markers', marker=dict(color='blue'), name='ACF'))
        # Add stems
        for lag, acf_value in zip(lags, acf_values1_hw):
            fig_1_hwres.add_trace(
                go.Scatter(x=[lag, lag], y=[0, acf_value], mode='lines', line=dict(color='blue', width=1)))
        for lag, acf_value in zip(-1 * lags, acf_values1_hw):
            fig_1_hwres.add_trace(
                go.Scatter(x=[lag, lag], y=[0, acf_value], mode='lines', line=dict(color='blue', width=1)))
        # Update layout
        fig_1_hwres.update_layout(
            title="Autocorrelation Function (ACF) Stem Plot of Residual error",
            xaxis_title="Lags",
            yaxis_title="Autocorrelation",
            plot_bgcolor='rgba(0,0,0,0)'
        )

        # Forecast error
        HW_foerror=y_test-holtw_df_test['Extent 10^6 sq km']

        data1_hwfore = HW_foerror.values
        max_lag = 60
        acf_values1_hwfore = sm.tsa.acf(data1_hwfore, nlags=max_lag)
        lags = np.arange(0, max_lag + 1)
        fig_1_hwfore = go.Figure()
        fig_1_hwfore.add_trace(
            go.Scatter(x=lags, y=acf_values1_hwfore, mode='markers', marker=dict(color='blue'), name='ACF'))
        fig_1_hwfore.add_trace(
            go.Scatter(x=-1 * lags, y=acf_values1_hwfore, mode='markers', marker=dict(color='blue'), name='ACF'))
        # Add stems
        for lag, acf_value in zip(lags, acf_values1_hwfore):
            fig_1_hwfore.add_trace(
                go.Scatter(x=[lag, lag], y=[0, acf_value], mode='lines', line=dict(color='blue', width=1)))
        for lag, acf_value in zip(-1 * lags, acf_values1_hwfore):
            fig_1_hwfore.add_trace(
                go.Scatter(x=[lag, lag], y=[0, acf_value], mode='lines', line=dict(color='blue', width=1)))
        # Update layout
        fig_1_hwfore.update_layout(
            title="Autocorrelation Function (ACF) Stem Plot of forecasted error",
            xaxis_title="Lags",
            yaxis_title="Autocorrelation",
            plot_bgcolor='rgba(0,0,0,0)'
        )

        # MSE
        HW_train_mse=mean_squared_error(y_train,holtw_df_train[['Extent 10^6 sq km']])
        HW_test_mse=mean_squared_error(y_test,holtw_df_test['Extent 10^6 sq km'])
        mse = f"The Mean sqaured error of Holts Winter method on train data: : {HW_train_mse} and the Mean Squared Error of MSE of Holts Winter method on test data is : {HW_test_mse}."

        # RMSE
        HW_train_rmse=mean_squared_error(y_train,holtw_df_train['Extent 10^6 sq km'],squared=False)
        HW_test_rmse=mean_squared_error(y_test,holtw_df_test['Extent 10^6 sq km'],squared=False)
        rmse = f'RMSE of Holts Winter method on train data: {HW_train_rmse} and RMSE of Holts Winter method on test data: {HW_test_rmse}.'

        #Q-value
        hotl_q_t=sm.stats.acorr_ljungbox(HW_reserror, lags=5,return_df=True)
        lbvalue=sm.stats.acorr_ljungbox(HW_foerror,lags=5,return_df=True)
        q_Value_hw = f'Q-value (residual): {hotl_q_t} and Q-value (Forecast):\n {lbvalue}'

        # Error mean and variance
        emv = (f'Holts winter: Mean of residual error is {np.mean(HW_reserror)} and Forecast error is : {np.mean(HW_foerror)}.\n'
               f' Holts winter: Variance of residual error is : {np.var(HW_reserror)} and Forecast error is {np.var(HW_foerror)} ')

        # Create HTML components for model performance metrics
        mse_html_hw = html.Pre(mse)
        rmse_html_hw = html.Pre(rmse)
        q_html_hw = html.Pre(q_Value_hw)
        emv_html_hw = html.Pre(emv)

        # Combine theccomponents into a single HTML div
        performance_div_hw = html.Div([
            html.H4('Model Performance Metrics:'),
            html.Div(mse_html_hw),
            html.Div(rmse_html_hw),
            html.Div(q_html_hw),
            html.Div(emv_html_hw),
        ])
        return fig_hw, fig_1_hwres, fig_1_hwfore, performance_div_hw

    if input5 == 'antartic':

        holtw1_ant = ets.ExponentialSmoothing(y1_train, damped_trend=True, trend='add', seasonal='add', seasonal_periods=365)
        holtw_ant = holtw1_ant.fit()

        # train
        holtw_pred_train_ant = holtw_ant.predict(start=y1_train.index[0], end=y1_train.index[-1])
        holtw_df1_train = pd.DataFrame(holtw_pred_train_ant, columns=['Extent 10^6 sq km'], index=y1_train.index)

        ##test
        holtw_pred_test_ant = holtw_ant.predict(start=y1_test.index[0], end=y1_test.index[-1])
        holtw_df1_test = pd.DataFrame(holtw_pred_test_ant, columns=['Extent 10^6 sq km'], index=y1_test.index)

        # plot of hwmodel
        fig_hw_ant = make_subplots(rows=1, cols=1)
        fig_hw_ant.add_trace(
            go.Scatter(x=y1_train.index, y=y1_train, mode='lines', name='Train'),
        )
        fig_hw_ant.add_trace(
            go.Scatter(x=y1_test.index, y=y1_test, mode='lines', name='Test'),
        )
        fig_hw_ant.add_trace(
            go.Scatter(x=holtw_df1_test.index, y=holtw_df1_test['Extent 10^6 sq km'], mode='lines',
                       name='Holt-Winters Prediction (Test)'),
        )
        fig_hw_ant.update_layout(
            xaxis_title="Time",
            yaxis_title="Extent",
            title="Holts Winter Model",
            height=600,
            width=1000,
        )

        # plot of residual error
        HW_reserror_ant = y1_train - holtw_df1_train['Extent 10^6 sq km']

        data2_hwres = HW_reserror_ant.values
        max_lag = 60
        acf_values2_hw = sm.tsa.acf(data2_hwres, nlags=max_lag)
        lags = np.arange(0, max_lag + 1)
        fig_2_hwres = go.Figure()
        fig_2_hwres.add_trace(
            go.Scatter(x=lags, y=acf_values2_hw, mode='markers', marker=dict(color='blue'), name='ACF'))
        fig_2_hwres.add_trace(
            go.Scatter(x=-1 * lags, y=acf_values2_hw, mode='markers', marker=dict(color='blue'), name='ACF'))
        # Add stems
        for lag, acf_value in zip(lags, acf_values2_hw):
            fig_2_hwres.add_trace(
                go.Scatter(x=[lag, lag], y=[0, acf_value], mode='lines', line=dict(color='blue', width=1)))
        for lag, acf_value in zip(-1 * lags, acf_values2_hw):
            fig_2_hwres.add_trace(
                go.Scatter(x=[lag, lag], y=[0, acf_value], mode='lines', line=dict(color='blue', width=1)))
        # Update layout
        fig_2_hwres.update_layout(
            title="Autocorrelation Function (ACF) Stem Plot of Residual error",
            xaxis_title="Lags",
            yaxis_title="Autocorrelation",
            plot_bgcolor='rgba(0,0,0,0)'
        )

        # Forecast error
        HW_foerror_ant = y1_test - holtw_df1_test['Extent 10^6 sq km']

        data2_hwfore = HW_foerror_ant.values
        max_lag = 60
        acf_values2_hwfore = sm.tsa.acf(data2_hwfore, nlags=max_lag)
        lags = np.arange(0, max_lag + 1)
        fig_2_hwfore = go.Figure()
        fig_2_hwfore.add_trace(
            go.Scatter(x=lags, y=acf_values2_hwfore, mode='markers', marker=dict(color='blue'), name='ACF'))
        fig_2_hwfore.add_trace(
            go.Scatter(x=-1 * lags, y=acf_values2_hwfore, mode='markers', marker=dict(color='blue'), name='ACF'))
        # Add stems
        for lag, acf_value in zip(lags, acf_values2_hwfore):
            fig_2_hwfore.add_trace(
                go.Scatter(x=[lag, lag], y=[0, acf_value], mode='lines', line=dict(color='blue', width=1)))
        for lag, acf_value in zip(-1 * lags, acf_values2_hwfore):
            fig_2_hwfore.add_trace(
                go.Scatter(x=[lag, lag], y=[0, acf_value], mode='lines', line=dict(color='blue', width=1)))
        # Update layout
        fig_2_hwfore.update_layout(
            title="Autocorrelation Function (ACF) Stem Plot of forecasted error",
            xaxis_title="Lags",
            yaxis_title="Autocorrelation",
            plot_bgcolor='rgba(0,0,0,0)'
        )

        # MSE
        HW_train1_mse = mean_squared_error(y1_train, holtw_df1_train[['Extent 10^6 sq km']])
        HW_test1_mse = mean_squared_error(y1_test, holtw_df1_test['Extent 10^6 sq km'])
        mse1 = f"The Mean sqaured error of Holts Winter method on train data: : {HW_train1_mse} and the Mean Squared Error of MSE of Holts Winter method on test data is : {HW_test1_mse}."

        # RMSE
        HW_train1_rmse = mean_squared_error(y1_train, holtw_df1_train['Extent 10^6 sq km'], squared=False)
        HW_test1_rmse = mean_squared_error(y1_test, holtw_df1_test['Extent 10^6 sq km'], squared=False)
        rmse1 = f'RMSE of Holts Winter method on train data: {HW_train1_rmse} and RMSE of Holts Winter method on test data: {HW_test1_rmse}.'

        # Q-value
        hotl_q_t1 = sm.stats.acorr_ljungbox(HW_reserror_ant, lags=5, return_df=True)
        lbvalue1 = sm.stats.acorr_ljungbox(HW_foerror_ant, lags=5, return_df=True)
        q_Value_hw1 = f'Q-value (residual): {hotl_q_t1} and Q-value (Forecast):\n {lbvalue1}'

        # Error mean and variance
        emv1 = (f'Holts winter: Mean of residual error is {np.mean(HW_reserror_ant)} and Forecast error is : {np.mean(HW_foerror_ant)}.\n',
                f'Holts winter: Variance of residual error is : {np.var(HW_reserror_ant)} and Forecast error is {np.var(HW_foerror_ant)} ')

        # Create HTML components for model performance metrics
        mse_html_hw1 = html.Pre(mse1)
        rmse_html_hw1 = html.Pre(rmse1)
        q_html_hw1 = html.Pre(q_Value_hw1)
        emv_html_hw1 = html.Pre(emv1)

        # Combine theccomponents into a single HTML div
        performance_div_hw1 = html.Div([
            html.H4('Model Performance Metrics:'),
            html.Div(mse_html_hw1),
            html.Div(rmse_html_hw1),
            html.Div(q_html_hw1),
            html.Div(emv_html_hw1),
        ])
        return fig_hw_ant, fig_2_hwres, fig_2_hwfore, performance_div_hw1

@my_app.callback(
               Output('svdc', 'children'),
               Output('output-original-model', 'children'),
               Output('output-container-button-clicked', 'children'),
               Output('removal', 'children'),
               [Input('dropdown_fc', 'value'),
                Input('button', 'n_clicks')
                ]
)

def sin_cond(input6, n_clicks):
    if input6 == 'artic':
        X_mat = x_train.values
        Y = y_train.values
        X_svd = sm.add_constant(X_mat)
        H = np.matmul(X_svd.T, X_svd)
        s, d, v = np.linalg.svd(H.astype(float))
        #sngular
        singular = f'Singular value: {d}'

        #condition
        conditional = f'Conditional value: {la.cond(X_svd)}'

        # Create HTML components for model performance metrics
        sing_html = html.Pre(singular)
        cond_html = html.Pre(conditional)

        # Combine theccomponents into a single HTML div
        performance_div_sc = html.Div([
            html.H4('Singular value decomposition and Conditional value decomposition'),
            html.Div(sing_html),
            html.Br(),
            html.Div(cond_html),
        ])

        #original model
        original_model = sm.OLS(y_train, x_train).fit()

        # Dropping 'Day' column
        x_train_stepwise = x_train.drop(['Day', 'Missing 10^6 sq km'], axis=1)

        # Fitting original model
        original_model_summary = html.Div([
            html.H4('Original Model:'),
            html.Pre(original_model.summary().as_text()),
            html.Br(),
            html.P("The missing extent variable has Nan values, because this column doesn't have any data recorded yet so it has 0.0 as the values which is consider as NaN values. Hence dropping this column.")
        ])

        # vif
        vif_data = pd.DataFrame()
        vif_data["Variable"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        vif_before = f'The VIF before the collinearity removing process is {vif_data}'

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(x_train)
        X_test_scaled = scaler.transform(x_test)

        # Ridge Regression with Cross-Validated Grid Search
        ridge = Ridge()
        param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
        grid_search = GridSearchCV(ridge, param_grid, scoring='neg_mean_squared_error', cv=5)
        grid_search.fit(X_train_scaled, y_train)

        # Best hyperparameters
        best_alpha = grid_search.best_params_['alpha']

        # Ridge Regression with optimal alpha
        ridge_optimal = Ridge(alpha=best_alpha)
        ridge_optimal.fit(X_train_scaled, y_train)

        # Predictions on train and test sets
        y_train_pred = ridge_optimal.predict(X_train_scaled)
        y_test_pred = ridge_optimal.predict(X_test_scaled)

        # Evaluate MSE
        mse_train = mean_squared_error(y_train, y_train_pred)
        mse_test = mean_squared_error(y_test, y_test_pred)

        # VIF Calculation
        vif = pd.DataFrame()
        vif["Variable"] = X.columns
        vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

        re = f'MSE of Ridge regression on train data: {mse_train}\n, MSE of Ridge regression on test data: {mse_test}\n, Optimal alpha:, {best_alpha}\n, VIF values:\n, {vif}. '
        # Standardize the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # pca
        pca = PCA(n_components='mle', svd_solver='full')
        pca.fit(X)
        x_pca = pca.transform(X)
        pc = f'Explained variance ratio: Original Feature space vs.Reduced Feature space\n{pca.explained_variance_ratio_}'
        ev = pca.explained_variance_ratio_
        cv = np.cumsum(ev) * 100
        num_com_95 = np.argmax(cv >= 95) + 1
        num_com_to_remove = pca.n_components_ - num_com_95
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_explained_variance = explained_variance_ratio.cumsum()
        num_components = np.argmax(cumulative_explained_variance >= 0.95) + 1

        # Calculate VIF for the new features after PCA
        vif_data_1 = pd.DataFrame()
        vif_data_1["Variable"] = range(1, x_pca.shape[1] + 1)  # Use range as index for VIF calculation
        vif_data_1["VIF"] = [variance_inflation_factor(x_pca, i) for i in range(x_pca.shape[1])]
        VF = f'VIF values after PCA:\n {vif_data_1}'

        # Create HTML components for model performance metrics
        Vif_html_before = html.Pre(vif_before)
        Re_html = html.Pre(re)
        Pc_html = html.Pre(pc)
        Vf_html = html.Pre(VF)

        # Combine theccomponents into a single HTML div
        performance_div_sc_Df = html.Div([
            html.H4('Collinearity Removal Metrics'),
            html.Div(Vif_html_before),
            html.Div(Re_html),
            html.Div(Pc_html),
            html.Div(Vf_html),
        ])


        if n_clicks is None:
            return performance_div_sc, original_model_summary, '' , performance_div_sc_Df

         # Fitting OLS model after stepwise regression
        model = sm.OLS(y_train, x_train_stepwise).fit()

        # Dropping columns with p-value greater than 0.05
        while model.pvalues.max() > 0.05:
                    col_to_drop = model.pvalues.idxmax()
                    x_train_stepwise.drop([col_to_drop], axis=1, inplace=True)
                    model = sm.OLS(y_train, x_train_stepwise).fit()

        # Stepwise model summary
        stepwise_model_summary = html.Div([
                    html.H4('Stepwise Regression Model After Feature Selection:'),
                    html.Pre(model.summary().as_text()),
                    html.P("Now, the columns which was against the criteria were removed and the final model is shown with the features.")
                ])

        #dropping variables in test
        x_test.drop(['Day', 'Missing 10^6 sq km'], axis=1)


        return performance_div_sc,original_model_summary, stepwise_model_summary, performance_div_sc

    elif input6 == 'antartic':
        X1_mat = x1_train.values
        Y1 = y1_train.values
        X1_svd = sm.add_constant(X1_mat)
        H1 = np.matmul(X1_svd.T, X1_svd)
        s1, d1, v1 = np.linalg.svd(H1.astype(float))
        #sngular
        singular1 = f'Singular value: {d1}'

        #condition
        conditional1 = f'Conditional value: {la.cond(X1_svd)}'

        # Create HTML components for model performance metrics
        sing1_html = html.Pre(singular1)
        cond1_html = html.Pre(conditional1)

        # Combine theccomponents into a single HTML div
        performance_div_sc1 = html.Div([
            html.H4('Singular value decomposition and Conditional value decomposition'),
            html.Div(sing1_html),
            html.Br(),
            html.Div(cond1_html),
        ])

        # original model
        original_model1 = sm.OLS(y1_train, x1_train).fit()

        # Dropping 'Day' column
        x1_train_stepwise = x1_train.drop(['Day', 'Missing 10^6 sq km'], axis=1)

        # Fitting original model
        original_model_summary1 = html.Div([
            html.H4('Original Model:'),
            html.Pre(original_model1.summary().as_text()),
            html.Br(),
            html.P(
                "The missing extent variable has Nan values, because this column doesn't have any data recorded yet so it has 0.0 as the values which is consider as NaN values. Hence dropping this column.")
        ])

        # vif
        vif1_data = pd.DataFrame()
        vif1_data["Variable"] = X.columns
        vif1_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        vif1_before = f'The VIF before the collinearity removing process is {vif1_data}'

        scaler = StandardScaler()
        X1_train_scaled = scaler.fit_transform(x1_train)
        X1_test_scaled = scaler.transform(x1_test)

        # Ridge Regression with Cross-Validated Grid Search
        ridge = Ridge()
        param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
        grid_search1 = GridSearchCV(ridge, param_grid, scoring='neg_mean_squared_error', cv=5)
        grid_search1.fit(X1_train_scaled, y_train)

        # Best hyperparameters
        best1_alpha = grid_search1.best_params_['alpha']

        # Ridge Regression with optimal alpha
        ridge_optimal = Ridge(alpha=best1_alpha)
        ridge_optimal.fit(X1_train_scaled, y_train)

        # Predictions on train and test sets
        y1_train_pred = ridge_optimal.predict(X1_train_scaled)
        y1_test_pred = ridge_optimal.predict(X1_test_scaled)

        # Evaluate MSE
        mse1_train = mean_squared_error(y1_train, y1_train_pred)
        mse1_test = mean_squared_error(y1_test, y1_test_pred)

        # VIF Calculation
        vif_12 = pd.DataFrame()
        vif_12["Variable"] = X1.columns
        vif_12["VIF"] = [variance_inflation_factor(X1.values, i) for i in range(X1.shape[1])]

        re1 = f'MSE of Ridge regression on train data: {mse1_train}\n, MSE of Ridge regression on test data: {mse1_test}\n, Optimal alpha:, {best1_alpha}\n, VIF values:\n, {vif_12}. '
        # Standardize the features
        scaler = StandardScaler()
        X1_scaled = scaler.fit_transform(X1)

        # pca
        pca1 = PCA(n_components='mle', svd_solver='full')
        pca1.fit(X1)
        x1_pca = pca1.transform(X1)
        pc1 = f'Explained variance ratio: Original Feature space vs.Reduced Feature space\n{pca1.explained_variance_ratio_}'
        ev1 = pca1.explained_variance_ratio_
        cv1 = np.cumsum(ev1) * 100
        num1_com_95 = np.argmax(cv1 >= 95) + 1
        num1_com_to_remove = pca1.n_components_ - num1_com_95
        explained_variance_ratio = pca1.explained_variance_ratio_
        cumulative_explained_variance = explained_variance_ratio.cumsum()
        num_components = np.argmax(cumulative_explained_variance >= 0.95) + 1

        # Calculate VIF for the new features after PCA
        vif1_data_1 = pd.DataFrame()
        vif1_data_1["Variable"] = range(1, x1_pca.shape[1] + 1)  # Use range as index for VIF calculation
        vif1_data_1["VIF"] = [variance_inflation_factor(x1_pca, i) for i in range(x1_pca.shape[1])]
        VF1 = f'VIF values after PCA:\n {vif1_data_1}'

        # Create HTML components for model performance metrics
        Vif_html_before_df1 = html.Pre(vif1_before)
        Re_html_df1 = html.Pre(re1)
        Pc_html_df1 = html.Pre(pc1)
        Vf_html_df1 = html.Pre(VF1)

        # Combine theccomponents into a single HTML div
        performance_div_sc_df1 = html.Div([
            html.H4('Collinearity Removal Metrics'),
            html.Div(Vif_html_before_df1),
            html.Div(Re_html_df1),
            html.Div(Pc_html_df1),
            html.Div(Vf_html_df1),
        ])

        if n_clicks is None:
                return performance_div_sc1, original_model_summary1, '', performance_div_sc_df1

         # Fitting OLS model after stepwise regression
        model1 = sm.OLS(y1_train, x1_train_stepwise).fit()

        # Dropping columns with p-value greater than 0.05
        while model1.pvalues.max() > 0.05:
                    col_to_drop = model1.pvalues.idxmax()
                    x1_train_stepwise.drop([col_to_drop], axis=1, inplace=True)
                    model1 = sm.OLS(y1_train, x1_train_stepwise).fit()

        # Stepwise model summary
        stepwise_model_summary1 = html.Div([
                    html.H4('Stepwise Regression Model After Feature Selection:'),
                    html.Pre(model1.summary().as_text()),
                    html.P("Now, the columns which was against the criteria were removed and the final model is shown with the features.")
                ])
        x1_test.drop(['Day', 'Missing 10^6 sq km'], axis=1)



        return performance_div_sc1, original_model_summary1, stepwise_model_summary1, performance_div_sc_df1

@my_app.callback(
               Output('model', 'figure'),
               Output('testvpred', 'figure'),
               Output('res-plot', 'figure'),
               Output('fore-plot', 'figure'),
               Output('perf-model', 'children'),
               [Input('dropdown_bm', 'value'),
                Input('dropdown_model', 'value'),
                ]
)

def base_models(input8, input9):
        if input8 == 'artic':
            if input9 == 'average':
                train_pred_avg=avg_one(y_train)
                test_pred_avg=avg_hstep(y_train,y_test)

                #avg-model
                fig_avg = go.Figure()
                fig_avg.add_trace(go.Scatter(x=y_train.index, y=y_train, mode='lines', name='Train'))
                fig_avg.add_trace(go.Scatter(x=y_test.index, y=y_test, mode='lines', name='Test'))
                fig_avg.add_trace(go.Scatter(x=y_test.index, y=test_pred_avg, mode='lines', name='Predicted'))
                fig_avg.update_layout(
                    xaxis_title='Time',
                    yaxis_title='Extent 10^6 sq km',
                    title='Average method predictions',
                    xaxis=dict(tickangle=-90),
                    legend=dict(x=0, y=1, traceorder="normal", font=dict(family="sans-serif", size=12, color="black"))
                )

                #testvpred
                fig_tp = go.Figure()
                fig_tp.add_trace(go.Scatter(x=y_test.index, y=y_test, mode='lines', name='Test'))
                fig_tp.add_trace(go.Scatter(x=y_test.index, y=test_pred_avg, mode='lines', name='Forecasted'))
                fig_tp.update_layout(
                    xaxis_title='Time',
                    yaxis_title='Extent 10^6 sq km',
                    title='Average method Forecast',
                    xaxis=dict(tickangle=-90),
                    legend=dict(x=0, y=1, traceorder="normal", font=dict(family="sans-serif", size=12, color="black"))
                )

                #residual and forecast error
                avg_res=y_train-train_pred_avg
                avg_fore=y_test-test_pred_avg

                #acfres
                data1_avgres = avg_res.values[1:]
                max_lag = 60
                acf_values1_avg = sm.tsa.acf(data1_avgres, nlags=max_lag)
                lags = np.arange(0, max_lag + 1)
                fig_1_avg = go.Figure()
                fig_1_avg.add_trace(
                    go.Scatter(x=lags, y=acf_values1_avg, mode='markers', marker=dict(color='blue'), name='ACF'))
                fig_1_avg.add_trace(
                    go.Scatter(x=-1 * lags, y=acf_values1_avg, mode='markers', marker=dict(color='blue'), name='ACF'))
                # Add stems
                for lag, acf_value in zip(lags, acf_values1_avg):
                    fig_1_avg.add_trace(
                        go.Scatter(x=[lag, lag], y=[0, acf_value], mode='lines', line=dict(color='blue', width=1)))
                for lag, acf_value in zip(-1 * lags, acf_values1_avg):
                    fig_1_avg.add_trace(
                        go.Scatter(x=[lag, lag], y=[0, acf_value], mode='lines', line=dict(color='blue', width=1)))
                # Update layout
                fig_1_avg.update_layout(
                    title="Autocorrelation Function (ACF) Stem Plot of Residual error",
                    xaxis_title="Lags",
                    yaxis_title="Autocorrelation",
                    plot_bgcolor='rgba(0,0,0,0)'
                )

                #acffore
                data2_avg_fore = avg_fore.values
                max_lag = 60
                acf_values2_avg_fore = sm.tsa.acf(data2_avg_fore, nlags=max_lag)
                lags = np.arange(0, max_lag + 1)
                fig_2_avg_fore = go.Figure()
                fig_2_avg_fore.add_trace(
                    go.Scatter(x=lags, y=acf_values2_avg_fore, mode='markers', marker=dict(color='blue'), name='ACF'))
                fig_2_avg_fore.add_trace(
                    go.Scatter(x=-1 * lags, y=acf_values2_avg_fore, mode='markers', marker=dict(color='blue'),
                               name='ACF'))
                # Add stems
                for lag, acf_value in zip(lags, acf_values2_avg_fore):
                    fig_2_avg_fore.add_trace(
                        go.Scatter(x=[lag, lag], y=[0, acf_value], mode='lines', line=dict(color='blue', width=1)))
                for lag, acf_value in zip(-1 * lags, acf_values2_avg_fore):
                    fig_2_avg_fore.add_trace(
                        go.Scatter(x=[lag, lag], y=[0, acf_value], mode='lines', line=dict(color='blue', width=1)))
                # Update layout
                fig_2_avg_fore.update_layout(
                    title="Autocorrelation Function (ACF) Stem Plot of forecasted error",
                    xaxis_title="Lags",
                    yaxis_title="Autocorrelation",
                    plot_bgcolor='rgba(0,0,0,0)'
                )

                # MSE
                avg_train_mse=mean_squared_error(y_train[1:],train_pred_avg[1:])
                avg_test_mse=mean_squared_error(y_test,test_pred_avg)
                mse_avg = f"The Mean sqaured error of Average method on train data: : {avg_train_mse} and the Mean Squared Error of MSE of Average method on test data is : {avg_test_mse}."

                #RMSE
                avg_train_rmse= mean_squared_error(y_train[1:],train_pred_avg[1:],squared=False)
                avg_test_rmse= mean_squared_error(y_test,test_pred_avg,squared=False)
                rmse_avg = f'RMSE of Average method on train data: {avg_train_rmse} and RMSE of Average method on test data: {avg_test_rmse}.'

                #Q-value
                q_avg_train=sm.stats.acorr_ljungbox(avg_res.values[1:], lags=5, boxpierce=True,return_df=True)
                q_avgtest=sm.stats.acorr_ljungbox(avg_fore.values[1:],lags=5,boxpierce=True,return_df=True)
                q_Value_avg = f'Q-value (residual): {q_avg_train} and Q-value (Forecast):\n {q_avgtest}'

                # Error mean and variance
                emv_avg = (f'Average Method: Mean of residual error is {np.mean(avg_res)} and Forecast error is : {np.mean(avg_fore)}.\n',
                           f' Average Method: Variance of residual error is : {np.var(avg_res)} and Forecast error is {np.var(avg_fore)} ')

                # Create HTML components for model performance metrics
                mse_html_avg = html.Pre(mse_avg)
                rmse_html_avg = html.Pre(rmse_avg)
                q_html_avg = html.Pre(q_Value_avg)
                emv_html_avg = html.Pre(emv_avg)

                # Combine theccomponents into a single HTML div
                performance_div_avg = html.Div([
                    html.H4('Model Performance Metrics:'),
                    html.Div(mse_html_avg),
                    html.Div(rmse_html_avg),
                    html.Div(q_html_avg),
                    html.Div(emv_html_avg),
                ])
                return fig_avg, fig_tp, fig_1_avg, fig_2_avg_fore, performance_div_avg

            elif input9 == 'drift':
                train_drift = []
                value = 0
                for i in range(len(y_train)):
                    if i > 1:
                        slope_val = (y_train[i - 1]-y_train[0]) / (i-1)
                        y_predict = (slope_val * i) + y_train[0]
                        train_drift.append(y_predict)
                    else:
                        continue

                test_drift= []
                for h in range(len(y_test)):
                    slope_val = (y_train.values[-1] - y_train.values[0] ) /( len(y_train) - 1 )
                    y_predict= y_train.values[-1] + ((h +1) * slope_val)
                    test_drift.append(y_predict)

                # drift-model
                fig_drift = go.Figure()
                fig_drift.add_trace(go.Scatter(x=y_train.index, y=y_train, mode='lines', name='Train'))
                fig_drift.add_trace(go.Scatter(x=y_test.index, y=y_test, mode='lines', name='Test'))
                fig_drift.add_trace(go.Scatter(x=y_test.index, y=test_drift, mode='lines', name='Predicted'))
                fig_drift.update_layout(
                    xaxis_title='Time',
                    yaxis_title='Extent 10^6 sq km',
                    title='Drift method predictions',
                    xaxis=dict(tickangle=-90),
                    legend=dict(x=0, y=1, traceorder="normal",
                                font=dict(family="sans-serif", size=12, color="black"))
                )

                # testvpred-drift
                fig_tp1 = go.Figure()
                fig_tp1.add_trace(go.Scatter(x=y_test.index, y=y_test, mode='lines', name='Test'))
                fig_tp1.add_trace(go.Scatter(x=y_test.index, y=test_drift, mode='lines', name='Forecasted'))
                fig_tp1.update_layout(
                        xaxis_title='Time',
                        yaxis_title='Extent 10^6 sq km',
                        title='Drift method Forecast',
                        xaxis=dict(tickangle=-90),
                        legend=dict(x=0, y=1, traceorder="normal",
                                    font=dict(family="sans-serif", size=12, color="black"))
                )

                # #residual and forecast error
                drift_res = y_train[2:]-train_drift
                drift_fore = y_test-test_drift

                # acfres
                data1_drift_res = drift_res.values[1:]
                max_lag = 60
                acf_values1_drift = sm.tsa.acf(data1_drift_res, nlags=max_lag)
                lags = np.arange(0, max_lag + 1)
                fig_1_drift = go.Figure()
                fig_1_drift.add_trace(
                    go.Scatter(x=lags, y=acf_values1_drift, mode='markers', marker=dict(color='blue'), name='ACF'))
                fig_1_drift.add_trace(
                    go.Scatter(x=-1 * lags, y=acf_values1_drift, mode='markers', marker=dict(color='blue'),
                                   name='ACF'))
                # Add stems
                for lag, acf_value in zip(lags, acf_values1_drift):
                    fig_1_drift.add_trace(
                        go.Scatter(x=[lag, lag], y=[0, acf_value], mode='lines', line=dict(color='blue', width=1)))
                for lag, acf_value in zip(-1 * lags, acf_values1_drift):
                    fig_1_drift.add_trace(
                         go.Scatter(x=[lag, lag], y=[0, acf_value], mode='lines', line=dict(color='blue', width=1)))
                # Update layout
                fig_1_drift.update_layout(
                        title="Autocorrelation Function (ACF) Stem Plot of Residual error",
                        xaxis_title="Lags",
                        yaxis_title="Autocorrelation",
                        plot_bgcolor='rgba(0,0,0,0)'
                )

                # acffore
                data1_drift_fore = drift_res.values
                max_lag = 60
                acf_values1_drift_fore = sm.tsa.acf(data1_drift_fore , nlags=max_lag)
                lags = np.arange(0, max_lag + 1)
                fig_1_drift_fore = go.Figure()
                fig_1_drift_fore.add_trace(
                    go.Scatter(x=lags, y=acf_values1_drift_fore, mode='markers', marker=dict(color='blue'),
                                   name='ACF'))
                fig_1_drift_fore.add_trace(
                    go.Scatter(x=-1 * lags, y=acf_values1_drift_fore, mode='markers', marker=dict(color='blue'),
                                   name='ACF'))
                # Add stems
                for lag, acf_value in zip(lags, acf_values1_drift_fore):
                    fig_1_drift_fore.add_trace(
                            go.Scatter(x=[lag, lag], y=[0, acf_value], mode='lines', line=dict(color='blue', width=1)))
                for lag, acf_value in zip(-1 * lags, acf_values1_drift_fore):
                    fig_1_drift_fore.add_trace(
                            go.Scatter(x=[lag, lag], y=[0, acf_value], mode='lines', line=dict(color='blue', width=1)))
                    # Update layout
                    fig_1_drift_fore.update_layout(
                        title="Autocorrelation Function (ACF) Stem Plot of forecasted error",
                        xaxis_title="Lags",
                        yaxis_title="Autocorrelation",
                        plot_bgcolor='rgba(0,0,0,0)'
                    )

                #Model performance on train and test data

                #MSE
                drift_train_mse=mean_squared_error(y_train[2:],train_drift)
                drift_test_mse=mean_squared_error(y_test,test_drift)
                mse_drift = f"The Mean squared error of Drift Method on train data: : {drift_train_mse} and the Mean Squared Error of MSE of Drift Method on test data is : {drift_test_mse}."

                # RMSE
                drift_train_rmse = mean_squared_error(y_train[2:], train_drift, squared=False)
                drift_test_rmse = mean_squared_error(y_test, test_drift, squared=False)
                rmse_drift = f'RMSE of Drift Method on train data: {drift_train_rmse} and RMSE of Drift Method on test data: {drift_test_rmse}.'

                # Q-value
                q_drift_train=sm.stats.acorr_ljungbox(drift_res, lags=5, boxpierce=True, return_df=True)
                q_drifttest=sm.stats.acorr_ljungbox(drift_fore,lags=5,boxpierce=True,return_df=True)
                q_Value_drift = f'Q-value (residual): {q_drift_train} and Q-value (Forecast):\n {q_drifttest}'

                # Error mean and variance
                emv_drift = (f'Drift Method: Mean of residual error is {np.mean(drift_res)} and Forecast error is : {np.mean(drift_fore)}.\n',
                             f'Drift Method: Variance of residual error is : {np.var(drift_res)} and Forecast error is {np.var(drift_fore)} ')

                # Create HTML components for model performance metrics
                mse_html_drift = html.Pre(mse_drift)
                rmse_html_drift = html.Pre(rmse_drift)
                q_html_drift = html.Pre(q_Value_drift)
                emv_html_drift = html.Pre(emv_drift)

                # Combine theccomponents into a single HTML div
                performance_div_drift = html.Div([
                    html.H4('Model Performance Metrics:'),
                    html.Div(mse_html_drift),
                    html.Div(rmse_html_drift),
                    html.Div(q_html_drift),
                    html.Div(emv_html_drift),
                ])
                return fig_drift, fig_tp1,  fig_1_drift, fig_1_drift_fore, performance_div_drift
            elif input9 == 'naive':
                train_naive = []
                for i in range(len(y_train[1:])):
                    train_naive.append(y_train.values[i-1])

                test_naive=[y_train.values[-1] for i in y_test]
                naive_fore_pd = pd.DataFrame(test_naive).set_index(y_test.index)

                # drift-model
                fig_naive = go.Figure()
                fig_naive.add_trace(go.Scatter(x=y_train.index, y=y_train, mode='lines', name='Train'))
                fig_naive.add_trace(go.Scatter(x=y_test.index, y=y_test, mode='lines', name='Test'))
                fig_naive.add_trace(go.Scatter(x=y_test.index, y=test_naive, mode='lines', name='Predicted'))
                fig_naive.update_layout(
                    xaxis_title='Time',
                    yaxis_title='Extent 10^6 sq km',
                    title='Naive method predictions',
                    xaxis=dict(tickangle=-90),
                    legend=dict(x=0, y=1, traceorder="normal",
                                font=dict(family="sans-serif", size=12, color="black"))
                )

                # testvpred-drift
                fig_tp2 = go.Figure()
                fig_tp2.add_trace(go.Scatter(x=y_test.index, y=y_test, mode='lines', name='Test'))
                fig_tp2.add_trace(go.Scatter(x=y_test.index, y=test_naive, mode='lines', name='Forecasted'))
                fig_tp2.update_layout(
                    xaxis_title='Time',
                    yaxis_title='Extent 10^6 sq km',
                    title='Naive method Forecast',
                    xaxis=dict(tickangle=-90),
                    legend=dict(x=0, y=1, traceorder="normal",
                                font=dict(family="sans-serif", size=12, color="black"))
                )

                #residual and forecast error
                naive_res= y_train[1:]-train_naive
                naive_fore1 = y_test-test_naive

                # acfres
                data1_naive_res = naive_res.values[1:]
                max_lag = 60
                acf_values1_naive = sm.tsa.acf(data1_naive_res, nlags=max_lag)
                lags = np.arange(0, max_lag + 1)
                fig_1_naive = go.Figure()
                fig_1_naive.add_trace(
                    go.Scatter(x=lags, y=acf_values1_naive, mode='markers', marker=dict(color='blue'), name='ACF'))
                fig_1_naive.add_trace(
                    go.Scatter(x=-1 * lags, y=acf_values1_naive, mode='markers', marker=dict(color='blue'),
                               name='ACF'))
                # Add stems
                for lag, acf_value in zip(lags, acf_values1_naive):
                    fig_1_naive.add_trace(
                        go.Scatter(x=[lag, lag], y=[0, acf_value], mode='lines', line=dict(color='blue', width=1)))
                for lag, acf_value in zip(-1 * lags, acf_values1_naive):
                    fig_1_naive.add_trace(
                        go.Scatter(x=[lag, lag], y=[0, acf_value], mode='lines', line=dict(color='blue', width=1)))
                # Update layout
                fig_1_naive.update_layout(
                    title="Autocorrelation Function (ACF) Stem Plot of Residual error",
                    xaxis_title="Lags",
                    yaxis_title="Autocorrelation",
                    plot_bgcolor='rgba(0,0,0,0)'
                )

                # acffore
                data1_naive_fore = naive_fore1.values
                max_lag = 60
                acf_values1_naive_fore = sm.tsa.acf(data1_naive_fore, nlags=max_lag)
                lags = np.arange(0, max_lag + 1)
                fig_1_naive_fore = go.Figure()
                fig_1_naive_fore.add_trace(
                    go.Scatter(x=lags, y=acf_values1_naive_fore, mode='markers', marker=dict(color='blue'),
                               name='ACF'))
                fig_1_naive_fore.add_trace(
                    go.Scatter(x=-1 * lags, y=acf_values1_naive_fore, mode='markers', marker=dict(color='blue'),
                               name='ACF'))
                # Add stems
                for lag, acf_value in zip(lags, acf_values1_naive_fore):
                    fig_1_naive_fore.add_trace(
                        go.Scatter(x=[lag, lag], y=[0, acf_value], mode='lines', line=dict(color='blue', width=1)))
                for lag, acf_value in zip(-1 * lags, acf_values1_naive_fore):
                    fig_1_naive_fore.add_trace(
                        go.Scatter(x=[lag, lag], y=[0, acf_value], mode='lines', line=dict(color='blue', width=1)))
                    # Update layout
                    fig_1_naive_fore.update_layout(
                        title="Autocorrelation Function (ACF) Stem Plot of forecasted error",
                        xaxis_title="Lags",
                        yaxis_title="Autocorrelation",
                        plot_bgcolor='rgba(0,0,0,0)'
                    )

                # Model performance on train and test data

                #MSE
                naive_train_mse=mean_squared_error(y_train[1:],train_naive)
                naive_test_mse=mean_squared_error(y_test,test_naive)
                mse_naive = f"The Mean squared error of Naive Method on train data: : {naive_train_mse} and the Mean Squared Error of MSE of Naive Method on test data is : {naive_test_mse}."

                # RMSE
                naive_train_rmse = mean_squared_error(y_train[1:], train_naive, squared=False)
                naive_test_rmse = mean_squared_error(y_test, test_naive, squared=False)
                rmse_naive = f'RMSE of Naive Method on train data: {naive_train_rmse} and RMSE of Naive Method on test data: {naive_test_rmse}.'

                # Q-value
                q_naive_train=acorr_ljungbox(naive_res, lags=5, boxpierce=True, return_df=True)
                q_naivetest=acorr_ljungbox(naive_fore1,lags=5,boxpierce=True,return_df=True)
                q_Value_naive = f'Q-value (residual): {q_naive_train} and Q-value (Forecast):\n {q_naivetest}'

                # Error mean and variance
                emv_naive = (f'Naive Method: Mean of residual error is {np.mean(naive_res)} and Forecast error is : {np.mean(naive_fore1)}.\n',
                             f'Naive Method: Variance of residual error is : {np.var(naive_res)} and Forecast error is {np.var(naive_fore1)} ')

                # Create HTML components for model performance metrics
                mse_html_naive = html.Pre(mse_naive)
                rmse_html_naive = html.Pre(rmse_naive)
                q_html_naive = html.Pre(q_Value_naive)
                emv_html_naive = html.Pre(emv_naive)

                # Combine theccomponents into a single HTML div
                performance_div_naive = html.Div([
                    html.H4('Model Performance Metrics:'),
                    html.Div(mse_html_naive),
                    html.Div(rmse_html_naive),
                    html.Div(q_html_naive),
                    html.Div(emv_html_naive),
                ])
                return fig_naive, fig_tp2, fig_1_naive, fig_1_naive_fore, performance_div_naive
            elif input9 == 'ses':
                ses = ets.ExponentialSmoothing(y_train, trend=None, damped_trend=False, seasonal=None).fit(
                    smoothing_level=0.5)
                train_ses= ses.forecast(steps=len(y_train))
                train_ses=pd.DataFrame(train_ses).set_index(y_train.index)

                test_ses= ses.forecast(steps=len(y_test))
                test_ses=pd.DataFrame(test_ses).set_index(y_test.index)

                # ses-model
                fig_ses = go.Figure()
                fig_ses.add_trace(go.Scatter(x=y_train.index, y=y_train, mode='lines', name='Train'))
                fig_ses.add_trace(go.Scatter(x=y_test.index, y=y_test, mode='lines', name='Test'))
                fig_ses.add_trace(go.Scatter(x=y_test.index, y=test_ses[0], mode='lines', name='Predicted'))
                fig_ses.update_layout(
                    xaxis_title='Time',
                    yaxis_title='Extent 10^6 sq km',
                    title='Simple Exponential Smoothing Method Predictions',
                    xaxis=dict(tickangle=-90),
                    legend=dict(x=0, y=1, traceorder="normal",
                                font=dict(family="sans-serif", size=12, color="black"))
                )

                # testvpred-drift
                fig_tp3 = go.Figure()
                fig_tp3.add_trace(go.Scatter(x=y_test.index, y=y_test, mode='lines', name='Test'))
                fig_tp3.add_trace(go.Scatter(x=y_test.index, y=test_ses[0], mode='lines', name='Forecasted'))
                fig_tp3.update_layout(
                    xaxis_title='Time',
                    yaxis_title='Extent 10^6 sq km',
                    title='Simple Exponential Smoothing Method Forecast',
                    xaxis=dict(tickangle=-90),
                    legend=dict(x=0, y=1, traceorder="normal",
                                font=dict(family="sans-serif", size=12, color="black"))
                )

                #residual and forecast error
                ses_res= y_train[2:]-train_ses[0]
                ses_fore= y_test-test_ses[0]

                # acfres
                data1_ses_res = ses_res.values[2:]
                max_lag = 60
                acf_values1_ses = sm.tsa.acf(data1_ses_res, nlags=max_lag)
                lags = np.arange(0, max_lag + 1)
                fig_1_ses = go.Figure()
                fig_1_ses.add_trace(
                    go.Scatter(x=lags, y=acf_values1_ses, mode='markers', marker=dict(color='blue'), name='ACF'))
                fig_1_ses.add_trace(
                    go.Scatter(x=-1 * lags, y=acf_values1_ses, mode='markers', marker=dict(color='blue'),
                               name='ACF'))
                # Add stems
                for lag, acf_value in zip(lags, acf_values1_ses):
                    fig_1_ses.add_trace(
                        go.Scatter(x=[lag, lag], y=[0, acf_value], mode='lines', line=dict(color='blue', width=1)))
                for lag, acf_value in zip(-1 * lags, acf_values1_ses):
                    fig_1_ses.add_trace(
                        go.Scatter(x=[lag, lag], y=[0, acf_value], mode='lines', line=dict(color='blue', width=1)))
                # Update layout
                fig_1_ses.update_layout(
                    title="Autocorrelation Function (ACF) Stem Plot of Residual error",
                    xaxis_title="Lags",
                    yaxis_title="Autocorrelation",
                    plot_bgcolor='rgba(0,0,0,0)'
                )

                # acffore
                data1_ses_fore = ses_fore.values
                max_lag = 60
                acf_values1_ses_fore = sm.tsa.acf(data1_ses_fore, nlags=max_lag)
                lags = np.arange(0, max_lag + 1)
                fig_1_ses_fore = go.Figure()
                fig_1_ses_fore.add_trace(
                    go.Scatter(x=lags, y=acf_values1_ses_fore, mode='markers', marker=dict(color='blue'),
                               name='ACF'))
                fig_1_ses_fore.add_trace(
                    go.Scatter(x=-1 * lags, y=acf_values1_ses_fore, mode='markers', marker=dict(color='blue'),
                               name='ACF'))
                # Add stems
                for lag, acf_value in zip(lags, acf_values1_ses_fore):
                    fig_1_ses_fore.add_trace(
                        go.Scatter(x=[lag, lag], y=[0, acf_value], mode='lines', line=dict(color='blue', width=1)))
                for lag, acf_value in zip(-1 * lags, acf_values1_ses_fore):
                    fig_1_ses_fore.add_trace(
                        go.Scatter(x=[lag, lag], y=[0, acf_value], mode='lines', line=dict(color='blue', width=1)))
                    # Update layout
                    fig_1_ses_fore.update_layout(
                        title="Autocorrelation Function (ACF) Stem Plot of forecasted error",
                        xaxis_title="Lags",
                        yaxis_title="Autocorrelation",
                        plot_bgcolor='rgba(0,0,0,0)'
                    )

                # Model performance on train and test data
                #MSE
                ses_train_mse=mean_squared_error(y_train,train_ses)
                ses_test_mse=mean_squared_error(y_test,test_ses)
                mse_ses = f"The Mean squared error of Simple Exponential Smoothing Method on train data: : {ses_train_mse} and the Mean Squared Error of MSE of Simple Exponential Smoothing Method on test data is : {ses_test_mse}."

                # RMSE
                ses_train_rmse = mean_squared_error(y_train, train_ses, squared=False)
                ses_test_rmse = mean_squared_error(y_test, test_ses, squared=False)
                rmse_ses = f'RMSE of Simple Exponential Smoothing Method on train data: {ses_train_rmse} and RMSE of Simple Exponential Smoothing Method on test data: {ses_test_rmse}.'

                # Q-value
                q_ses_train = acorr_ljungbox(ses_res[2:], lags=5, boxpierce=True, return_df=True)
                q_sestest = acorr_ljungbox(ses_fore, lags=5, boxpierce=True, return_df=True)
                q_Value_ses = f'Q-value (residual): {q_ses_train} and Q-value (Forecast):\n {q_sestest}'

                # Error mean and variance
                emv_ses = (f'Simple Exponential Smoothing Method: Mean of residual error is {np.mean(ses_res)} and Forecast error is : {np.mean(ses_fore)}.\n',
                           f'Simple Exponential Smoothing Method: Variance of residual error is : {np.var(ses_res)} and Forecast error is {np.var(ses_fore)} ')

                # Create HTML components for model performance metrics
                mse_html_ses = html.Pre(mse_ses)
                rmse_html_ses = html.Pre(rmse_ses)
                q_html_ses = html.Pre(q_Value_ses)
                emv_html_ses = html.Pre(emv_ses)

                # Combine theccomponents into a single HTML div
                performance_div_ses = html.Div([
                    html.H4('Model Performance Metrics:'),
                    html.Div(mse_html_ses),
                    html.Div(rmse_html_ses),
                    html.Div(q_html_ses),
                    html.Div(emv_html_ses),
                ])
                return fig_ses, fig_tp3, fig_1_ses, fig_1_ses_fore, performance_div_ses
        elif input8 == 'antartic':
            if input9 == 'average':
                train_pred_avg1=avg_one(y1_train)
                test_pred_avg1=avg_hstep(y1_train,y1_test)

                #avg-model
                fig_avg2 = go.Figure()
                fig_avg2.add_trace(go.Scatter(x=y1_train.index, y=y1_train, mode='lines', name='Train'))
                fig_avg2.add_trace(go.Scatter(x=y1_test.index, y=y1_test, mode='lines', name='Test'))
                fig_avg2.add_trace(go.Scatter(x=y1_test.index, y=test_pred_avg1, mode='lines', name='Predicted'))
                fig_avg2.update_layout(
                    xaxis_title='Time',
                    yaxis_title='Extent 10^6 sq km',
                    title='Average method predictions Antartic',
                    xaxis=dict(tickangle=-90),
                    legend=dict(x=0, y=1, traceorder="normal", font=dict(family="sans-serif", size=12, color="black"))
                )

                #testvpred
                fig_tp_ant= go.Figure()
                fig_tp_ant.add_trace(go.Scatter(x=y1_test.index, y=y1_test, mode='lines', name='Test'))
                fig_tp_ant .add_trace(go.Scatter(x=y1_test.index, y=test_pred_avg1, mode='lines', name='Forecasted'))
                fig_tp_ant .update_layout(
                    xaxis_title='Time',
                    yaxis_title='Extent 10^6 sq km',
                    title='Average method Forecast of Antartic',
                    xaxis=dict(tickangle=-90),
                    legend=dict(x=0, y=1, traceorder="normal", font=dict(family="sans-serif", size=12, color="black"))
                )

                #residual and forecast error
                avg_res_ant=y_train-train_pred_avg1
                avg_fore_ant=y_test-test_pred_avg1

                #acfres
                data1_avgres_ant = avg_res_ant.values[1:]
                max_lag = 60
                acf_values2_avg = sm.tsa.acf(data1_avgres_ant, nlags=max_lag)
                lags = np.arange(0, max_lag + 1)
                fig_2_avg = go.Figure()
                fig_2_avg.add_trace(
                    go.Scatter(x=lags, y=acf_values2_avg, mode='markers', marker=dict(color='blue'), name='ACF'))
                fig_2_avg.add_trace(
                    go.Scatter(x=-1 * lags, y=acf_values2_avg, mode='markers', marker=dict(color='blue'), name='ACF'))
                # Add stems
                for lag, acf_value in zip(lags, acf_values2_avg):
                    fig_2_avg.add_trace(
                        go.Scatter(x=[lag, lag], y=[0, acf_value], mode='lines', line=dict(color='blue', width=1)))
                for lag, acf_value in zip(-1 * lags, acf_values2_avg):
                    fig_2_avg.add_trace(
                        go.Scatter(x=[lag, lag], y=[0, acf_value], mode='lines', line=dict(color='blue', width=1)))
                # Update layout
                fig_2_avg.update_layout(
                    title="Autocorrelation Function (ACF) Stem Plot of Residual error",
                    xaxis_title="Lags",
                    yaxis_title="Autocorrelation",
                    plot_bgcolor='rgba(0,0,0,0)'
                )

                #acffore
                data_avg_fore_ant = avg_fore_ant.values
                max_lag = 60
                acf_values_avg_fore_ant = sm.tsa.acf(data_avg_fore_ant, nlags=max_lag)
                lags = np.arange(0, max_lag + 1)
                fig_2_avg_fore_ant = go.Figure()
                fig_2_avg_fore_ant.add_trace(
                    go.Scatter(x=lags, y=acf_values_avg_fore_ant, mode='markers', marker=dict(color='blue'), name='ACF'))
                fig_2_avg_fore_ant.add_trace(
                    go.Scatter(x=-1 * lags, y=acf_values_avg_fore_ant, mode='markers', marker=dict(color='blue'),
                               name='ACF'))
                # Add stems
                for lag, acf_value in zip(lags, acf_values_avg_fore_ant):
                    fig_2_avg_fore_ant.add_trace(
                        go.Scatter(x=[lag, lag], y=[0, acf_value], mode='lines', line=dict(color='blue', width=1)))
                for lag, acf_value in zip(-1 * lags, acf_values_avg_fore_ant):
                    fig_2_avg_fore_ant.add_trace(
                        go.Scatter(x=[lag, lag], y=[0, acf_value], mode='lines', line=dict(color='blue', width=1)))
                # Update layout
                fig_2_avg_fore_ant.update_layout(
                    title="Autocorrelation Function (ACF) Stem Plot of forecasted error",
                    xaxis_title="Lags",
                    yaxis_title="Autocorrelation",
                    plot_bgcolor='rgba(0,0,0,0)'
                )

                # MSE
                avg_train_ant_mse=mean_squared_error(y_train[1:],train_pred_avg1[1:])
                avg_test_ant_mse=mean_squared_error(y_test,test_pred_avg1)
                mse_avg_ant = f"The Mean sqaured error of Average method on train data: : {avg_train_ant_mse} and the Mean Squared Error of MSE of Average method on test data is : {avg_test_ant_mse}."

                #RMSE
                avg_train_ant_rmse= mean_squared_error(y_train[1:],train_pred_avg1[1:],squared=False)
                avg_test_ant_rmse= mean_squared_error(y_test,test_pred_avg1,squared=False)
                rmse_avg_ant = f'RMSE of Average method on train data: {avg_train_ant_rmse} and RMSE of Average method on test data: {avg_test_ant_rmse}.'

                #Q-value
                q_avg_train_ant=sm.stats.acorr_ljungbox(avg_res_ant.values[1:], lags=5, boxpierce=True,return_df=True)
                q_avgtest_ant=sm.stats.acorr_ljungbox(avg_fore_ant.values[1:],lags=5,boxpierce=True,return_df=True)
                q_Value_avg_ant = f'Q-value (residual): {q_avg_train_ant} and Q-value (Forecast):\n {q_avgtest_ant}'

                # Error mean and variance
                emv_avg_ant = (f'Average Method: Mean of residual error is {np.mean(avg_res_ant)} and Forecast error is : {np.mean(avg_fore_ant)}.\n',
                               f' Average Method: Variance of residual error is : {np.var(avg_res_ant)} and Forecast error is {np.var(avg_fore_ant)} ')

                # Create HTML components for model performance metrics
                mse_html_avg_ant = html.Pre(mse_avg_ant)
                rmse_html_avg_ant= html.Pre(rmse_avg_ant)
                q_html_avg_ant = html.Pre(q_Value_avg_ant)
                emv_html_avg_ant = html.Pre(emv_avg_ant)

                # Combine theccomponents into a single HTML div
                performance_div_avg_ant = html.Div([
                    html.H4('Model Performance Metrics:'),
                    html.Div(mse_html_avg_ant),
                    html.Div(rmse_html_avg_ant),
                    html.Div(q_html_avg_ant),
                    html.Div(emv_html_avg_ant),
                ])
                return fig_avg2, fig_tp_ant, fig_2_avg, fig_2_avg_fore_ant, performance_div_avg_ant

            elif input9 == 'drift':
                train_drift1 = []
                value = 0
                for i in range(len(y1_train)):
                    if i > 1:
                        slope_val = (y1_train[i - 1]-y1_train[0]) / (i-1)
                        y_predict = (slope_val * i) + y1_train[0]
                        train_drift1.append(y_predict)
                    else:
                        continue

                test_drift1= []
                for h in range(len(y1_test)):
                    slope_val = (y1_train.values[-1] - y1_train.values[0] ) /( len(y1_train) - 1 )
                    y1_predict= y1_train.values[-1] + ((h +1) * slope_val)
                    test_drift1.append(y1_predict)

                # drift-model
                fig_drift1 = go.Figure()
                fig_drift1.add_trace(go.Scatter(x=y1_train.index, y=y1_train, mode='lines', name='Train'))
                fig_drift1.add_trace(go.Scatter(x=y1_test.index, y=y1_test, mode='lines', name='Test'))
                fig_drift1.add_trace(go.Scatter(x=y1_test.index, y=test_drift1, mode='lines', name='Predicted'))
                fig_drift1.update_layout(
                    xaxis_title='Time',
                    yaxis_title='Extent 10^6 sq km',
                    title='Drift method predictions',
                    xaxis=dict(tickangle=-90),
                    legend=dict(x=0, y=1, traceorder="normal",
                                font=dict(family="sans-serif", size=12, color="black"))
                )

                # testvpred-drift
                fig_tp12 = go.Figure()
                fig_tp12.add_trace(go.Scatter(x=y1_test.index, y=y1_test, mode='lines', name='Test'))
                fig_tp12.add_trace(go.Scatter(x=y1_test.index, y=test_drift1, mode='lines', name='Forecasted'))
                fig_tp12.update_layout(
                        xaxis_title='Time',
                        yaxis_title='Extent 10^6 sq km',
                        title='Drift method Forecast',
                        xaxis=dict(tickangle=-90),
                        legend=dict(x=0, y=1, traceorder="normal",
                                    font=dict(family="sans-serif", size=12, color="black"))
                )

                # #residual and forecast error
                drift_res1 = y_train[2:]-train_drift1
                drift_fore1 = y_test-test_drift1

                # acfres
                data2_drift_res = drift_res1.values[1:]
                max_lag = 60
                acf_values2_drift = sm.tsa.acf(data2_drift_res, nlags=max_lag)
                lags = np.arange(0, max_lag + 1)
                fig_2_drift = go.Figure()
                fig_2_drift.add_trace(
                    go.Scatter(x=lags, y=acf_values2_drift, mode='markers', marker=dict(color='blue'), name='ACF'))
                fig_2_drift.add_trace(
                    go.Scatter(x=-1 * lags, y=acf_values2_drift, mode='markers', marker=dict(color='blue'),
                                   name='ACF'))
                # Add stems
                for lag, acf_value in zip(lags, acf_values2_drift):
                    fig_2_drift.add_trace(
                        go.Scatter(x=[lag, lag], y=[0, acf_value], mode='lines', line=dict(color='blue', width=1)))
                for lag, acf_value in zip(-1 * lags, acf_values2_drift):
                    fig_2_drift.add_trace(
                         go.Scatter(x=[lag, lag], y=[0, acf_value], mode='lines', line=dict(color='blue', width=1)))
                # Update layout
                fig_2_drift.update_layout(
                        title="Autocorrelation Function (ACF) Stem Plot of Residual error",
                        xaxis_title="Lags",
                        yaxis_title="Autocorrelation",
                        plot_bgcolor='rgba(0,0,0,0)'
                )

                # acffore
                data2_drift_fore = drift_res1.values
                max_lag = 60
                acf_values2_drift_fore = sm.tsa.acf(data2_drift_fore , nlags=max_lag)
                lags = np.arange(0, max_lag + 1)
                fig_2_drift_fore = go.Figure()
                fig_2_drift_fore.add_trace(
                    go.Scatter(x=lags, y=acf_values2_drift_fore, mode='markers', marker=dict(color='blue'),
                                   name='ACF'))
                fig_2_drift_fore.add_trace(
                    go.Scatter(x=-1 * lags, y=acf_values2_drift_fore, mode='markers', marker=dict(color='blue'),
                                   name='ACF'))
                # Add stems
                for lag, acf_value in zip(lags, acf_values2_drift_fore):
                    fig_2_drift_fore.add_trace(
                            go.Scatter(x=[lag, lag], y=[0, acf_value], mode='lines', line=dict(color='blue', width=1)))
                for lag, acf_value in zip(-1 * lags, acf_values2_drift_fore):
                    fig_2_drift_fore.add_trace(
                            go.Scatter(x=[lag, lag], y=[0, acf_value], mode='lines', line=dict(color='blue', width=1)))
                    # Update layout
                    fig_2_drift_fore.update_layout(
                        title="Autocorrelation Function (ACF) Stem Plot of forecasted error",
                        xaxis_title="Lags",
                        yaxis_title="Autocorrelation",
                        plot_bgcolor='rgba(0,0,0,0)'
                    )

                #Model performance on train and test data

                #MSE
                drift_train_mse2=mean_squared_error(y1_train[2:],train_drift1)
                drift_test_mse2=mean_squared_error(y1_test,test_drift1)
                mse_drift2 = f"The Mean squared error of Drift Method on train data: : {drift_train_mse2} and the Mean Squared Error of MSE of Drift Method on test data is : {drift_test_mse2}."

                # RMSE
                drift_train_rmse2 = mean_squared_error(y1_train[2:], train_drift1, squared=False)
                drift_test_rmse2 = mean_squared_error(y1_test, test_drift1, squared=False)
                rmse_drift2 = f'RMSE of Drift Method on train data: {drift_train_rmse2} and RMSE of Drift Method on test data: {drift_test_rmse2}.'

                # Q-value
                q_drift_train2=sm.stats.acorr_ljungbox(drift_res1, lags=5, boxpierce=True, return_df=True)
                q_drifttest2=sm.stats.acorr_ljungbox(drift_fore1,lags=5,boxpierce=True,return_df=True)
                q_Value_drift2 = f'Q-value (residual): {q_drift_train2} and Q-value (Forecast):\n {q_drifttest2}'

                # Error mean and variance
                emv_drift2 = (f'Drift Method: Mean of residual error is {np.mean(drift_res1)} and Forecast error is : {np.mean(drift_fore1)}.\n',
                              f' Drift Method: Variance of residual error is : {np.var(drift_res1)} and Forecast error is {np.var(drift_fore1)} ')

                # Create HTML components for model performance metrics
                mse_html_drift2 = html.Pre(mse_drift2)
                rmse_html_drift2 = html.Pre(rmse_drift2)
                q_html_drift2 = html.Pre(q_Value_drift2)
                emv_html_drift2 = html.Pre(emv_drift2)

                # Combine theccomponents into a single HTML div
                performance_div_drift2 = html.Div([
                    html.H4('Model Performance Metrics:'),
                    html.Div(mse_html_drift2),
                    html.Div(rmse_html_drift2),
                    html.Div(q_html_drift2),
                    html.Div(emv_html_drift2),
                ])
                return fig_drift1, fig_tp12, fig_2_drift, fig_2_drift_fore, performance_div_drift2

            elif input9 == 'naive':
                train_naive1 = []
                for i in range(len(y1_train[1:])):
                    train_naive1.append(y1_train.values[i-1])

                test_naive1=[y1_train.values[-1] for i in y1_test]
                naive_fore_pd = pd.DataFrame(test_naive1).set_index(y1_test.index)

                # drift-model
                fig_naive1 = go.Figure()
                fig_naive1.add_trace(go.Scatter(x=y1_train.index, y=y1_train, mode='lines', name='Train'))
                fig_naive1.add_trace(go.Scatter(x=y1_test.index, y=y1_test, mode='lines', name='Test'))
                fig_naive1.add_trace(go.Scatter(x=y1_test.index, y=test_naive1, mode='lines', name='Predicted'))
                fig_naive1.update_layout(
                    xaxis_title='Time',
                    yaxis_title='Extent 10^6 sq km',
                    title='Naive method predictions',
                    xaxis=dict(tickangle=-90),
                    legend=dict(x=0, y=1, traceorder="normal",
                                font=dict(family="sans-serif", size=12, color="black"))
                )

                # testvpred-drift
                fig_tp21 = go.Figure()
                fig_tp21.add_trace(go.Scatter(x=y1_test.index, y=y1_test, mode='lines', name='Test'))
                fig_tp21.add_trace(go.Scatter(x=y1_test.index, y=test_naive1, mode='lines', name='Forecasted'))
                fig_tp21.update_layout(
                    xaxis_title='Time',
                    yaxis_title='Extent 10^6 sq km',
                    title='Naive method Forecast',
                    xaxis=dict(tickangle=-90),
                    legend=dict(x=0, y=1, traceorder="normal",
                                font=dict(family="sans-serif", size=12, color="black"))
                )

                #residual and forecast error
                naive_res1= y1_train[1:]-train_naive1
                naive_fore2 = y1_test-test_naive1

                # acfres
                data2_naive_res = naive_res1.values[1:]
                max_lag = 60
                acf_values2_naive = sm.tsa.acf(data2_naive_res, nlags=max_lag)
                lags = np.arange(0, max_lag + 1)
                fig_2_naive = go.Figure()
                fig_2_naive.add_trace(
                    go.Scatter(x=lags, y=acf_values2_naive, mode='markers', marker=dict(color='blue'), name='ACF'))
                fig_2_naive.add_trace(
                    go.Scatter(x=-1 * lags, y=acf_values2_naive, mode='markers', marker=dict(color='blue'),
                               name='ACF'))
                # Add stems
                for lag, acf_value in zip(lags, acf_values2_naive):
                    fig_2_naive.add_trace(
                        go.Scatter(x=[lag, lag], y=[0, acf_value], mode='lines', line=dict(color='blue', width=1)))
                for lag, acf_value in zip(-1 * lags, acf_values2_naive):
                    fig_2_naive.add_trace(
                        go.Scatter(x=[lag, lag], y=[0, acf_value], mode='lines', line=dict(color='blue', width=1)))
                # Update layout
                fig_2_naive.update_layout(
                    title="Autocorrelation Function (ACF) Stem Plot of Residual error",
                    xaxis_title="Lags",
                    yaxis_title="Autocorrelation",
                    plot_bgcolor='rgba(0,0,0,0)'
                )

                # acffore
                data2_naive_fore = naive_fore2.values
                max_lag = 60
                acf_values2_naive_fore = sm.tsa.acf(data2_naive_fore, nlags=max_lag)
                lags = np.arange(0, max_lag + 1)
                fig_2_naive_fore = go.Figure()
                fig_2_naive_fore.add_trace(
                    go.Scatter(x=lags, y=acf_values2_naive_fore, mode='markers', marker=dict(color='blue'),
                               name='ACF'))
                fig_2_naive_fore.add_trace(
                    go.Scatter(x=-1 * lags, y=acf_values2_naive_fore, mode='markers', marker=dict(color='blue'),
                               name='ACF'))
                # Add stems
                for lag, acf_value in zip(lags, acf_values2_naive_fore):
                    fig_2_naive_fore.add_trace(
                        go.Scatter(x=[lag, lag], y=[0, acf_value], mode='lines', line=dict(color='blue', width=1)))
                for lag, acf_value in zip(-1 * lags, acf_values2_naive_fore):
                    fig_2_naive_fore.add_trace(
                        go.Scatter(x=[lag, lag], y=[0, acf_value], mode='lines', line=dict(color='blue', width=1)))
                    # Update layout
                    fig_2_naive_fore.update_layout(
                        title="Autocorrelation Function (ACF) Stem Plot of forecasted error",
                        xaxis_title="Lags",
                        yaxis_title="Autocorrelation",
                        plot_bgcolor='rgba(0,0,0,0)'
                    )

                # Model performance on train and test data

                #MSE
                naive_train_mse1=mean_squared_error(y1_train[1:],train_naive1)
                naive_test_mse1=mean_squared_error(y1_test,test_naive1)
                mse_naive1 = f"The Mean squared error of Naive Method on train data: : {naive_train_mse1} and the Mean Squared Error of MSE of Naive Method on test data is : {naive_test_mse1}."

                # RMSE
                naive_train_rmse1 = mean_squared_error(y1_train[1:], train_naive1, squared=False)
                naive_test_rmse1 = mean_squared_error(y1_test, test_naive1, squared=False)
                rmse_naive1 = f'RMSE of Naive Method on train data: {naive_train_rmse1} and RMSE of Naive Method on test data: {naive_test_rmse1}.'

                # Q-value
                q_naive_train1=acorr_ljungbox(naive_res1, lags=5, boxpierce=True, return_df=True)
                q_naivetest1=acorr_ljungbox(naive_fore2,lags=5,boxpierce=True,return_df=True)
                q_Value_naive1 = f'Q-value (residual): {q_naive_train1} and Q-value (Forecast):\n {q_naivetest1}'

                # Error mean and variance
                emv_naive1 = (f'Naive Method: Mean of residual error is {np.mean(naive_res1)} and Forecast error is : {np.mean(naive_fore2)}.\n'
                              f'Naive Method: Variance of residual error is : {np.var(naive_res1)} and Forecast error is {np.var(naive_fore2)} ')

                # Create HTML components for model performance metrics
                mse_html_naive1 = html.Pre(mse_naive1)
                rmse_html_naive1 = html.Pre(rmse_naive1)
                q_html_naive1 = html.Pre(q_Value_naive1)
                emv_html_naive1 = html.Pre(emv_naive1)

                # Combine theccomponents into a single HTML div
                performance_div_naive1 = html.Div([
                    html.H4('Model Performance Metrics:'),
                    html.Div(mse_html_naive1),
                    html.Div(rmse_html_naive1),
                    html.Div(q_html_naive1),
                    html.Div(emv_html_naive1),
                ])

                return fig_naive1, fig_tp21, fig_2_naive, fig_2_naive_fore, performance_div_naive1
            elif input9 == 'ses':
                ses1 = ets.ExponentialSmoothing(y1_train, trend=None, damped_trend=False, seasonal=None).fit(
                    smoothing_level=0.5)
                train_ses1= ses1.forecast(steps=len(y1_train))
                train_ses1=pd.DataFrame(train_ses1).set_index(y_train.index)

                test_ses1= ses1.forecast(steps=len(y1_test))
                test_ses1=pd.DataFrame(test_ses1).set_index(y1_test.index)

                # ses-model
                fig_ses1 = go.Figure()
                fig_ses1.add_trace(go.Scatter(x=y1_train.index, y=y1_train, mode='lines', name='Train'))
                fig_ses1.add_trace(go.Scatter(x=y1_test.index, y=y1_test, mode='lines', name='Test'))
                fig_ses1.add_trace(go.Scatter(x=y1_test.index, y=test_ses1[0], mode='lines', name='Predicted'))
                fig_ses1.update_layout(
                    xaxis_title='Time',
                    yaxis_title='Extent 10^6 sq km',
                    title='Simple Exponential Smoothing Method Predictions',
                    xaxis=dict(tickangle=-90),
                    legend=dict(x=0, y=1, traceorder="normal",
                                font=dict(family="sans-serif", size=12, color="black"))
                )

                # testvpred-drift
                fig_tp31 = go.Figure()
                fig_tp31.add_trace(go.Scatter(x=y1_test.index, y=y1_test, mode='lines', name='Test'))
                fig_tp31.add_trace(go.Scatter(x=y1_test.index, y=test_ses1[0], mode='lines', name='Forecasted'))
                fig_tp31.update_layout(
                    xaxis_title='Time',
                    yaxis_title='Extent 10^6 sq km',
                    title='Simple Exponential Smoothing Method Forecast',
                    xaxis=dict(tickangle=-90),
                    legend=dict(x=0, y=1, traceorder="normal",
                                font=dict(family="sans-serif", size=12, color="black"))
                )

                #residual and forecast error
                ses_res1= y1_train[2:]-train_ses1[0]
                ses_fore1= y1_test-test_ses1[0]

                # acfres
                data2_ses_res = ses_res1.values[2:]
                max_lag = 60
                acf_values2_ses = sm.tsa.acf(data2_ses_res, nlags=max_lag)
                lags = np.arange(0, max_lag + 1)
                fig_2_ses = go.Figure()
                fig_2_ses.add_trace(
                    go.Scatter(x=lags, y=acf_values2_ses, mode='markers', marker=dict(color='blue'), name='ACF'))
                fig_2_ses.add_trace(
                    go.Scatter(x=-1 * lags, y=acf_values2_ses, mode='markers', marker=dict(color='blue'),
                               name='ACF'))
                # Add stems
                for lag, acf_value in zip(lags, acf_values2_ses):
                    fig_2_ses.add_trace(
                        go.Scatter(x=[lag, lag], y=[0, acf_value], mode='lines', line=dict(color='blue', width=1)))
                for lag, acf_value in zip(-1 * lags, acf_values2_ses):
                    fig_2_ses.add_trace(
                        go.Scatter(x=[lag, lag], y=[0, acf_value], mode='lines', line=dict(color='blue', width=1)))
                # Update layout
                fig_2_ses.update_layout(
                    title="Autocorrelation Function (ACF) Stem Plot of Residual error",
                    xaxis_title="Lags",
                    yaxis_title="Autocorrelation",
                    plot_bgcolor='rgba(0,0,0,0)'
                )

                # acffore
                data2_ses_fore = ses_fore1.values
                max_lag = 60
                acf_values2_ses_fore = sm.tsa.acf(data2_ses_fore, nlags=max_lag)
                lags = np.arange(0, max_lag + 1)
                fig_2_ses_fore = go.Figure()
                fig_2_ses_fore.add_trace(
                    go.Scatter(x=lags, y=acf_values2_ses_fore, mode='markers', marker=dict(color='blue'),
                               name='ACF'))
                fig_2_ses_fore.add_trace(
                    go.Scatter(x=-1 * lags, y=acf_values2_ses_fore, mode='markers', marker=dict(color='blue'),
                               name='ACF'))
                # Add stems
                for lag, acf_value in zip(lags, acf_values2_ses_fore):
                    fig_2_ses_fore.add_trace(
                        go.Scatter(x=[lag, lag], y=[0, acf_value], mode='lines', line=dict(color='blue', width=1)))
                for lag, acf_value in zip(-1 * lags, acf_values2_ses_fore):
                    fig_2_ses_fore.add_trace(
                        go.Scatter(x=[lag, lag], y=[0, acf_value], mode='lines', line=dict(color='blue', width=1)))
                    # Update layout
                    fig_2_ses_fore.update_layout(
                        title="Autocorrelation Function (ACF) Stem Plot of forecasted error",
                        xaxis_title="Lags",
                        yaxis_title="Autocorrelation",
                        plot_bgcolor='rgba(0,0,0,0)'
                    )

                # Model performance on train and test data
                #MSE
                ses_train_mse1=mean_squared_error(y1_train,train_ses1)
                ses_test_mse1=mean_squared_error(y1_test,test_ses1)
                mse_ses1 = f"The Mean squared error of Simple Exponential Smoothing Method on train data: : {ses_train_mse1} and the Mean Squared Error of MSE of Simple Exponential Smoothing Method on test data is : {ses_test_mse1}."

                # RMSE
                ses_train_rmse1 = mean_squared_error(y1_train, train_ses1, squared=False)
                ses_test_rmse1 = mean_squared_error(y1_test, test_ses1, squared=False)
                rmse_ses1 = f'RMSE of Simple Exponential Smoothing Method on train data: {ses_train_rmse1} and RMSE of Simple Exponential Smoothing Method on test data: {ses_test_rmse1}.'

                # Q-value
                q_ses_train1 = acorr_ljungbox(ses_res1[2:], lags=5, boxpierce=True, return_df=True)
                q_sestest1 = acorr_ljungbox(ses_fore1, lags=5, boxpierce=True, return_df=True)
                q_Value_ses1 = f'Q-value (residual): {q_ses_train1} and Q-value (Forecast):\n {q_sestest1}'

                # Error mean and variance
                emv_ses1 = (
                    f'Simple Exponential Smoothing Method: Mean of residual error is {np.mean(ses_res1)} and Forecast error is : {np.mean(ses_fore1)}.\n'
                    f'Simple Exponential Smoothing Method: Variance of residual error is : {np.var(ses_res1)} and Forecast error is {np.var(ses_fore1)}'
                )
                # Create HTML components for model performance metrics
                mse_html_ses1 = html.Pre(mse_ses1)
                rmse_html_ses1 = html.Pre(rmse_ses1)
                q_html_ses1 = html.Pre(q_Value_ses1)
                emv_html_ses1 = html.Pre(emv_ses1)

                # Combine theccomponents into a single HTML div
                performance_div_ses1 = html.Div([
                    html.H4('Model Performance Metrics:'),
                    html.Div(mse_html_ses1),
                    html.Div(rmse_html_ses1),
                    html.Div(q_html_ses1),
                    html.Div(emv_html_ses1),
                ])


                return fig_ses1, fig_tp31, fig_2_ses, fig_2_ses_fore, performance_div_ses1

@my_app.callback(Output('f-value', 'children'),
               Output('p1-value', 'children'),
               Output('mlr-model', 'children'),
               Output('mlr-model1', 'figure'),
               Output('res-mlr', 'figure'),
               Output('fore-mlr', 'figure'),
               Output('perf-mlr', 'children'),
               [Input('dropdown_mlr', 'value')
                ]
)

def base_models(input10):
    if input10 == 'artic':
        x_train.drop([ 'Missing 10^6 sq km'], axis=1)
        X_train_sm = sm.add_constant(x_train)
        mlr_model = sm.OLS(y_train, X_train_sm).fit()

        # Make predictions on the test set
        X_test_sm = sm.add_constant(x_test)
        y_pred = mlr_model.predict(X_test_sm)

        pred_train = mlr_model.predict(X_train_sm)
        #fvalue and p-value
        f = f'The F-value is: {mlr_model.fvalue}'
        p = f'The P-value is : {mlr_model.f_pvalue}'

        # Fitting original model
        model_summary = html.Div([
            html.H4('Multi Linear Regression Model:'),
            html.Pre(mlr_model.summary().as_text()),
            html.Br(),
            ])

        residuals = y_test - y_pred
        fore = y_train - pred_train

        # mlr-model
        fig_mlr1 = go.Figure()
        fig_mlr1.add_trace(go.Scatter(x=y_train.index, y=y_train, mode='lines', name='Train'))
        fig_mlr1.add_trace(go.Scatter(x=y_test.index, y=y_test, mode='lines', name='Test'))
        fig_mlr1.add_trace(go.Scatter(x=y_test.index, y=y_pred, mode='lines', name='Predicted'))
        fig_mlr1.update_layout(
            xaxis_title='Time',
            yaxis_title='Extent 10^6 sq km',
            title='Multiple Linear Regression Model Predictions',
            xaxis=dict(tickangle=-90),
            legend=dict(x=0, y=1, traceorder="normal",
                        font=dict(family="sans-serif", size=12, color="black"))
        )

        # acfres
        data1_mlr = residuals.values[1:]
        max_lag = 60
        acf_values_mlr = sm.tsa.acf(data1_mlr, nlags=max_lag)
        lags = np.arange(0, max_lag + 1)
        fig_1_mlr = go.Figure()
        fig_1_mlr.add_trace(
            go.Scatter(x=lags, y=acf_values_mlr, mode='markers', marker=dict(color='blue'), name='ACF'))
        fig_1_mlr.add_trace(
            go.Scatter(x=-1 * lags, y=acf_values_mlr, mode='markers', marker=dict(color='blue'),
                       name='ACF'))
        # Add stems
        for lag, acf_value in zip(lags, acf_values_mlr):
            fig_1_mlr.add_trace(
                go.Scatter(x=[lag, lag], y=[0, acf_value], mode='lines', line=dict(color='blue', width=1)))
        for lag, acf_value in zip(-1 * lags, acf_values_mlr):
            fig_1_mlr.add_trace(
                go.Scatter(x=[lag, lag], y=[0, acf_value], mode='lines', line=dict(color='blue', width=1)))
        # Update layout
        fig_1_mlr.update_layout(
            title="Autocorrelation Function (ACF) Stem Plot of Residual error",
            xaxis_title="Lags",
            yaxis_title="Autocorrelation",
            plot_bgcolor='rgba(0,0,0,0)'
        )

        # acffore
        data1_mlr_fore = fore.values
        max_lag = 60
        acf_values1_fore = sm.tsa.acf(data1_mlr_fore, nlags=max_lag)
        lags = np.arange(0, max_lag + 1)
        fig_1_fore = go.Figure()
        fig_1_fore.add_trace(
            go.Scatter(x=lags, y=acf_values1_fore, mode='markers', marker=dict(color='blue'),
                       name='ACF'))
        fig_1_fore.add_trace(
            go.Scatter(x=-1 * lags, y=acf_values1_fore, mode='markers', marker=dict(color='blue'),
                       name='ACF'))
        # Add stems
        for lag, acf_value in zip(lags, acf_values1_fore):
            fig_1_fore.add_trace(
                go.Scatter(x=[lag, lag], y=[0, acf_value], mode='lines', line=dict(color='blue', width=1)))
        for lag, acf_value in zip(-1 * lags, acf_values1_fore):
            fig_1_fore.add_trace(
                go.Scatter(x=[lag, lag], y=[0, acf_value], mode='lines', line=dict(color='blue', width=1)))
            # Update layout
            fig_1_fore.update_layout(
                title="Autocorrelation Function (ACF) Stem Plot of forecasted error",
                xaxis_title="Lags",
                yaxis_title="Autocorrelation",
                plot_bgcolor='rgba(0,0,0,0)'
            )

        # Model performance on train and test data

        # MSE
        ML_train_mse = mean_squared_error(y_train, pred_train)
        ML_test_mse = mean_squared_error(y_test, y_pred)
        mse_ml = f"The Mean squared error of Multiple Linear Regression Method on train data: : {ML_train_mse} and the Mean Squared Error of MSE of Multiple Linear Regression Method on test data is : {ML_test_mse}."

        # RMSE
        ml_train_rmse = mean_squared_error(y_train, pred_train, squared=False)
        ml_test_rmse = mean_squared_error(y_test, y_pred, squared=False)
        rmse_ml = f'RMSE of Multiple Linear Regression Method on train data: {ml_train_rmse} and RMSE of Multiple Linear Regression Method on test data: {ml_test_rmse}.'

        # Q-value
        q_ml_train = sm.stats.acorr_ljungbox(residuals, lags=5, return_df=True)
        q_mltest = sm.stats.acorr_ljungbox(residuals, lags=5, return_df=True)
        q_Value_ml = f'Q-value (residual): {q_ml_train} and Q-value (Forecast):\n {q_mltest}'

        # Error mean and variance
        emv_ml = (f'Multiple Linear Regression Method: Mean of residual error is {np.mean(residuals)} and Forecast error is : {np.mean(fore)}.\n'
                  f'Multiple Linear Regression Method: Variance of residual error is : {np.var(residuals)} and Forecast error is {np.var(fore)} ')

        # Create HTML components for model performance metrics
        mse_html_art = html.Pre(mse_ml)
        rmse_html_art = html.Pre(rmse_ml)
        q_html_art = html.Pre(q_Value_ml)
        emv_html_art = html.Pre(emv_ml)

        # Combine theccomponents into a single HTML div
        performance_div_art = html.Div([
            html.H4('Model Performance Metrics:'),
            html.Div(mse_html_art),
            html.Div(rmse_html_art),
            html.Div(q_html_art),
            html.Div(emv_html_art),
        ])
        return f, p, model_summary, fig_mlr1, fig_1_mlr, fig_1_fore, performance_div_art

    elif input10 =='antartic':
        X1_train_sm = sm.add_constant(x1_train)
        model_ant = sm.OLS(y1_train, X1_train_sm).fit()

        # Make predictions on the test set
        X1_test_sm = sm.add_constant(x1_test)
        y_pred1 = model_ant.predict(X1_test_sm)

        pred_train_1 = model_ant.predict(X1_train_sm)

        # fvalue and p-value
        f1 = f'The F-value is: {model_ant.fvalue}'
        p1 = f'The P-value is : {model_ant.f_pvalue}'

        # Fitting original model
        model_summary_ant = html.Div([
            html.H4('Multi Linear Regression Model:'),
            html.Pre(model_ant.summary().as_text()),
            html.Br(),
        ])

        residuals1 = y1_test - y_pred1
        fore1 = y1_train - pred_train_1

        fig_mlr2 = go.Figure()
        fig_mlr2.add_trace(go.Scatter(x=y1_train.index, y=y1_train, mode='lines', name='Train'))
        fig_mlr2.add_trace(go.Scatter(x=y1_test.index, y=y1_test, mode='lines', name='Test'))
        fig_mlr2.add_trace(go.Scatter(x=y1_test.index, y=y_pred1, mode='lines', name='Predicted'))
        fig_mlr2.update_layout(
            xaxis_title='Time',
            yaxis_title='Extent 10^6 sq km',
            title='Multiple Linear Regression Model Predictions',
            xaxis=dict(tickangle=-90),
            legend=dict(x=0, y=1, traceorder="normal",
                        font=dict(family="sans-serif", size=12, color="black"))
        )

        # acfres
        data2_mlr = residuals1.values[1:]
        max_lag = 60
        acf_values_mlr2 = sm.tsa.acf(data2_mlr, nlags=max_lag)
        lags = np.arange(0, max_lag + 1)
        fig_2_mlr = go.Figure()
        fig_2_mlr.add_trace(
            go.Scatter(x=lags, y=acf_values_mlr2, mode='markers', marker=dict(color='blue'), name='ACF'))
        fig_2_mlr.add_trace(
            go.Scatter(x=-1 * lags, y=acf_values_mlr2, mode='markers', marker=dict(color='blue'),
                       name='ACF'))
        # Add stems
        for lag, acf_value in zip(lags, acf_values_mlr2):
            fig_2_mlr.add_trace(
                go.Scatter(x=[lag, lag], y=[0, acf_value], mode='lines', line=dict(color='blue', width=1)))
        for lag, acf_value in zip(-1 * lags, acf_values_mlr2):
            fig_2_mlr.add_trace(
                go.Scatter(x=[lag, lag], y=[0, acf_value], mode='lines', line=dict(color='blue', width=1)))
        # Update layout
        fig_2_mlr.update_layout(
            title="Autocorrelation Function (ACF) Stem Plot of Residual error",
            xaxis_title="Lags",
            yaxis_title="Autocorrelation",
            plot_bgcolor='rgba(0,0,0,0)'
        )

        # acffore
        data2_mlr_fore = fore1.values
        max_lag = 60
        acf_values2_fore = sm.tsa.acf(data2_mlr_fore, nlags=max_lag)
        lags = np.arange(0, max_lag + 1)
        fig_2_fore = go.Figure()
        fig_2_fore.add_trace(
            go.Scatter(x=lags, y=acf_values2_fore, mode='markers', marker=dict(color='blue'),
                       name='ACF'))
        fig_2_fore.add_trace(
            go.Scatter(x=-1 * lags, y=acf_values2_fore, mode='markers', marker=dict(color='blue'),
                       name='ACF'))
        # Add stems
        for lag, acf_value in zip(lags, acf_values2_fore):
            fig_2_fore.add_trace(
                go.Scatter(x=[lag, lag], y=[0, acf_value], mode='lines', line=dict(color='blue', width=1)))
        for lag, acf_value in zip(-1 * lags, acf_values2_fore):
            fig_2_fore.add_trace(
                go.Scatter(x=[lag, lag], y=[0, acf_value], mode='lines', line=dict(color='blue', width=1)))
            # Update layout
            fig_2_fore.update_layout(
                title="Autocorrelation Function (ACF) Stem Plot of forecasted error",
                xaxis_title="Lags",
                yaxis_title="Autocorrelation",
                plot_bgcolor='rgba(0,0,0,0)'
            )

        # Model performance on train and test data

        # MSE
        ML_train_mse2 = mean_squared_error(y1_train, pred_train_1)
        ML_test_mse2 = mean_squared_error(y1_test, y_pred1)
        mse_ml1 = f"The Mean squared error of Multiple Linear Regression Method on train data: : {ML_train_mse2} and the Mean Squared Error of MSE of Multiple Linear Regression Method on test data is : {ML_test_mse2}."

        # RMSE
        ml_train_rmse2 = mean_squared_error(y1_train, pred_train_1, squared=False)
        ml_test_rmse2 = mean_squared_error(y1_test, y_pred1, squared=False)
        rmse_ml2 = f'RMSE of Multiple Linear Regression Method on train data: {ml_train_rmse2} and RMSE of Multiple Linear Regression Method on test data: {ml_test_rmse2}.'

        # Q-value
        q_ml_train2 = sm.stats.acorr_ljungbox(residuals1, lags=5, return_df=True)
        q_mltest2 = sm.stats.acorr_ljungbox(residuals1, lags=5, return_df=True)
        q_Value_ml2 = f'Q-value (residual): {q_ml_train2} and Q-value (Forecast):\n {q_mltest2}'

        # Error mean and variance
        emv_ml2 = (f'Multiple Linear Regression Method: Mean of residual error is {np.mean(residuals1)} and Forecast error is : {np.mean(fore1)}.\n'
                   f'Multiple Linear Regression Method: Variance of residual error is : {np.var(residuals1)} and Forecast error is {np.var(fore1)} ')

        # Create HTML components for model performance metrics
        mse_html_ant = html.Pre(mse_ml1)
        rmse_html_ant = html.Pre(rmse_ml2)
        q_html_ant = html.Pre(q_Value_ml2)
        emv_html_ant= html.Pre(emv_ml2)

        # Combine theccomponents into a single HTML div
        performance_div_ant = html.Div([
            html.H4('Model Performance Metrics:'),
            html.Div(mse_html_ant),
            html.Div(rmse_html_ant),
            html.Div(q_html_ant),
            html.Div(emv_html_ant),
        ])
        return f1, p1, model_summary_ant, fig_mlr2, fig_2_mlr, fig_2_fore, performance_div_ant

# #gpac can't be shown in dash because of the dash properties
# # diffferncing-one
# diff1 = difference(df['Extent 10^6 sq km'], interval=365)
# diff_df = pd.DataFrame(diff1, index=df.index[365:])
# diff2 = difference(diff1, 1)
# diff_df1 = pd.DataFrame(diff1, df.index[365:])  # Adjust the index range
# diff_df2 = pd.DataFrame(diff2, df.index[366:])
# diff_df21 = pd.DataFrame(diff2, index=df.index[366:], columns=['Extent 10^6 sq km'])
# diff_train, diff_test = train_test_split(diff_df21, test_size=0.2, shuffle=False)
#
# # acfres
# df_acf_1 = diff_df21.values[1:]
# max_lag = 60
# acf_values_df = sm.tsa.acf(df_acf_1, nlags=max_lag)
# lags = np.arange(0, max_lag + 1)
# fig_df_arsa = go.Figure()
# fig_df_arsa.add_trace(
# go.Scatter(x=lags, y=acf_values_df, mode='markers', marker=dict(color='blue'), name='ACF'))
# fig_df_arsa.add_trace(go.Scatter(x=-1 * lags, y=acf_values_df, mode='markers', marker=dict(color='blue'),
#                    name='ACF'))
# # Add stems
# for lag, acf_value in zip(lags, acf_values_df):
#         fig_df_arsa.add_trace(go.Scatter(x=[lag, lag], y=[0, acf_value], mode='lines', line=dict(color='blue', width=1)))
# for lag, acf_value in zip(-1 * lags, acf_values_df):
#         fig_df_arsa.add_trace(go.Scatter(x=[lag, lag], y=[0, acf_value], mode='lines', line=dict(color='blue', width=1)))
#         # Update layout
# fig_df_arsa.update_layout(
#         title="Autocorrelation Function (ACF) Stem Plot of Residual error",
#         xaxis_title="Lags",
#         yaxis_title="Autocorrelation",
#         plot_bgcolor='rgba(0,0,0,0)'
#         )
#
# GPAC(fig_df_arsa,10,10)

# Define the confidence interval function
def conf_int(cov, params, na, nb):
    intervals = []
    for i in range(na):
        pos = params[i] + 2 * np.sqrt(cov[i][i])
        neg = params[i] - 2 * np.sqrt(cov[i][i])
        intervals.append((neg, params[i], pos))
    for i in range(nb):
        pos = params[na + i] + 2 * np.sqrt(cov[na + i][na + i])
        neg = params[na + i] - 2 * np.sqrt(cov[na + i][na + i])
        intervals.append((neg, params[na + i], pos))
    return intervals

# Zero-poles cancellation
def zero_poles_plotly(params, na):
    y_den = [1] + list(params[:na])
    e_num = [1] + list(params[na:])
    zeros = np.roots(e_num)
    poles = np.roots(y_den)
    result = f"The roots of numerator are {zeros} and The roots of denominator are {poles}"
    return result

# Chi-square test
def chi_test_plotly(na, nb, lags, Q, e):
    chi_statistic = (Q ** 2).sum().sum()
    dof = (Q.shape[na] - nb) * (Q.shape[na] - nb)
    alpha = 0.01
    chi_critical = chi2.ppf(1 - alpha, dof)
    if chi_statistic > chi_critical:
        result = 'The residuals are white'
    else:
        result = 'The residual is not white'
    return result



@my_app.callback(Output('acf-order', 'figure'),
               Output('Image', 'children'),
               Output('estimated1', 'children'),
               Output('model_sum', 'children'),
               Output('para-div', 'children'),
               Output('acf-arma-res', 'figure'),
               Output('acf-arma-fore', 'figure'),
               Output('acf-arma11', 'figure'),
               Output('acf-testvfore', 'figure'),
               Output('acf-trainvspred', 'figure'),
               Output('arma-performance', 'children'),
               Output('arma-chi', 'children'),
               Output('arma-confi', 'children'),
               Output('arma-zeropoles', 'children'),
                 [Input('dropdown_arsa', 'value')]
)

def arma_arima_Sarima(input11):
    if input11 == 'artic':
        # diffferncing-one
        diff1 = difference(df['Extent 10^6 sq km'], interval=365)
        diff_df = pd.DataFrame(diff1, index=df.index[365:])
        diff2 = difference(diff1, 1)
        diff_df1 = pd.DataFrame(diff1, df.index[365:])  # Adjust the index range
        diff_df2 = pd.DataFrame(diff2, df.index[366:])
        diff_df21 = pd.DataFrame(diff2, index=df.index[366:], columns=['Extent 10^6 sq km'])
        diff_train, diff_test = train_test_split(diff_df21, test_size=0.2, shuffle=False)

        # acfres
        df_acf_1 = diff_df21.values[1:]
        max_lag = 60
        acf_values_df = sm.tsa.acf(df_acf_1, nlags=max_lag)
        lags = np.arange(0, max_lag + 1)
        fig_df_arsa = go.Figure()
        fig_df_arsa.add_trace(
            go.Scatter(x=lags, y=acf_values_df, mode='markers', marker=dict(color='blue'), name='ACF'))
        fig_df_arsa.add_trace(
            go.Scatter(x=-1 * lags, y=acf_values_df, mode='markers', marker=dict(color='blue'),
                   name='ACF'))
        # Add stems
        for lag, acf_value in zip(lags, acf_values_df):
            fig_df_arsa.add_trace(
                go.Scatter(x=[lag, lag], y=[0, acf_value], mode='lines', line=dict(color='blue', width=1)))
        for lag, acf_value in zip(-1 * lags, acf_values_df):
            fig_df_arsa.add_trace(
            go.Scatter(x=[lag, lag], y=[0, acf_value], mode='lines', line=dict(color='blue', width=1)))
        # Update layout
        fig_df_arsa.update_layout(
            title="Autocorrelation Function (ACF)",
            xaxis_title="Lags",
            yaxis_title="Autocorrelation",
            plot_bgcolor='rgba(0,0,0,0)'
        )

        img1 = html.Div([
            html.Img(src= image9_path),
            html.P("We Could see that the order is (1,1)"),
            html.P("Accordingly, na = 1, nb =1  in which na(y-axis) has more constants and nb(x-axis) has more zeros."),
        ])

        # order ARMA(1,1)

        # 14: LMA algorithm
        na = 1
        nb = 1
        order = (na, 0, nb)  # ARIMA order: (p, d, q)
        arma_1_1 = sm.tsa.ARIMA(y_train, order=order, trend=None).fit()

        # Estimated parameters
        ar_params = []
        for i in range(na):
            ar_params.append(f"The AR coefficient a{i} is: {arma_1_1.params[i]}")

        ma_params = []
        for i in range(nb):
            ma_params.append(f"The MA coefficient a{i} is: {arma_1_1.params[i + na]}")

        # Create HTML components for AR and MA coefficients
        ar_html = html.Pre(ar_params)
        ma_html = html.Pre(ma_params)

        # Combine the components into a single HTML div
        params_div = html.Div([
            html.H4('Estimated Co-efficients:'),
            html.Div(ar_html),
            html.Div(ma_html),
        ])
        # Fitting original model
        arma_mode = html.Div([
            html.H4('First Arma Model:'),
            html.Pre(arma_1_1.summary().as_text()),
            html.Br(),
        ])

        # initialise values used in function
        mu = 0.01
        delta = 10 ** -6
        epsilon = 0.001
        mu_max = 10 ** 10
        max_iter = 100
        SSE, cov, params, var = step3(max_iter, mu, delta, epsilon, mu_max, na, nb, y)
        para_df = []
        para_df.append(f"The Estimated parameter of AR is: {params[0]}. ")
        para_df.append(f"The Estimated parameter of MA is: {params[1]}")
        paradf_html = html.Pre(para_df)
        paradf_div = html.Div([
            html.H4('Estimated Parameters:'),
            html.Div(paradf_html),
            html.P(
                "So, The parameter estimation employed by LMA(Levenberg Marquardt Algorithm) is shown. Since this is evident that the order is (1,1), the parameter estimation of AR model and also the MA model is shown.")
        ])

        # Prediction on train set
        total_length = 14691
        train_length = int(0.8 * total_length)
        test_length = total_length - train_length
        # For training set
        arma_1_1_train = arma_1_1.predict(start=0, end=train_length - 1)
        arma_1_1_res = y_train - arma_1_1_train
        # For test set
        arma_1_1_test = arma_1_1.predict(start=train_length, end=total_length - 1)
        arma_1_1_fore = y_test - arma_1_1_test

        #acfres
        df_acf_res_arma = arma_1_1_res
        max_lag = 60
        acf_values_df_arma = sm.tsa.acf(df_acf_res_arma, nlags=max_lag)
        lags = np.arange(0, max_lag + 1)
        fig_df_res_arma = go.Figure()
        fig_df_res_arma.add_trace(
            go.Scatter(x=lags, y=acf_values_df_arma, mode='markers', marker=dict(color='blue'), name='ACF'))
        fig_df_res_arma.add_trace(
            go.Scatter(x=-1 * lags, y=acf_values_df_arma , mode='markers', marker=dict(color='blue'),
                       name='ACF'))
        # Add stems
        for lag, acf_value in zip(lags, acf_values_df_arma ):
            fig_df_res_arma.add_trace(
                go.Scatter(x=[lag, lag], y=[0, acf_value], mode='lines', line=dict(color='blue', width=1)))
        for lag, acf_value in zip(-1 * lags, acf_values_df_arma ):
            fig_df_res_arma.add_trace(
                go.Scatter(x=[lag, lag], y=[0, acf_value], mode='lines', line=dict(color='blue', width=1)))
        # Update layout
        fig_df_res_arma.update_layout(
            title="Autocorrelation Function (ACF) of Residuals",
            xaxis_title="Lags",
            yaxis_title="Autocorrelation",
            plot_bgcolor='rgba(0,0,0,0)'
        )

        # acffore
        df_acf_fore_arma = arma_1_1_fore.values
        max_lag = 60
        acf_fore_values_df_arma = sm.tsa.acf(df_acf_fore_arma, nlags=max_lag)
        lags = np.arange(0, max_lag + 1)
        fig_df_fore_arma = go.Figure()
        fig_df_fore_arma.add_trace(
            go.Scatter(x=lags, y=acf_fore_values_df_arma , mode='markers', marker=dict(color='blue'), name='ACF'))
        fig_df_fore_arma.add_trace(
            go.Scatter(x=-1 * lags, y=acf_fore_values_df_arma , mode='markers', marker=dict(color='blue'),
                       name='ACF'))
        # Add stems
        for lag, acf_value in zip(lags, acf_fore_values_df_arma ):
            fig_df_fore_arma.add_trace(
                go.Scatter(x=[lag, lag], y=[0, acf_value], mode='lines', line=dict(color='blue', width=1)))
        for lag, acf_value in zip(-1 * lags, acf_fore_values_df_arma ):
            fig_df_fore_arma.add_trace(
                go.Scatter(x=[lag, lag], y=[0, acf_value], mode='lines', line=dict(color='blue', width=1)))
        # Update layout
        fig_df_fore_arma.update_layout(
            title="Autocorrelation Function (ACF) of Forecasted",
            xaxis_title="Lags",
            yaxis_title="Autocorrelation",
            plot_bgcolor='rgba(0,0,0,0)'
        )

        #full arma model
        fig_arma_model_df = go.Figure()
        fig_arma_model_df.add_trace(go.Scatter(x=y_train.index, y=y_train, mode='lines', name='Train'))
        fig_arma_model_df.add_trace(go.Scatter(x=y_test.index, y=y_test, mode='lines', name='Test'))
        fig_arma_model_df.add_trace(go.Scatter(x=y_test.index, y=arma_1_1_test, mode='lines', name='Predicted'))
        fig_arma_model_df.update_layout(
            xaxis_title='Time',
            yaxis_title='Extent 10^6 sq km',
            title='ARMA Model - TRAIN, TEST AND PREDICTED ',
            xaxis=dict(tickangle=-90),
            legend=dict(x=0, y=1, traceorder="normal",
                        font=dict(family="sans-serif", size=12, color="black"))
        )

        # full testvspred model
        fig_arma_model_df_test = go.Figure()
        fig_arma_model_df_test.add_trace(go.Scatter(x=y_test.index, y=y_test, mode='lines', name='Test'))
        fig_arma_model_df_test.add_trace(go.Scatter(x=y_test.index, y=arma_1_1_test, mode='lines', name='Predicted'))
        fig_arma_model_df_test.update_layout(
            xaxis_title='Time',
            yaxis_title='Extent 10^6 sq km',
            title='ARMA Model - TEST vs PREDICTED ',
            xaxis=dict(tickangle=-90),
            legend=dict(x=0, y=1, traceorder="normal",
                        font=dict(family="sans-serif", size=12, color="black"))
        )

        # full trainvspred model
        fig_arma_model_df_train = go.Figure()
        fig_arma_model_df_train.add_trace(go.Scatter(x=y_train.index, y=y_train, mode='lines', name='Test'))
        fig_arma_model_df_train.add_trace(go.Scatter(x=y_train.index, y=arma_1_1_test, mode='lines', name='Predicted'))
        fig_arma_model_df_train.update_layout(
            xaxis_title='Time',
            yaxis_title='Extent 10^6 sq km',
            title='ARMA Model - TRAIN vs PREDICTED ',
            xaxis=dict(tickangle=-90),
            legend=dict(x=0, y=1, traceorder="normal",
                        font=dict(family="sans-serif", size=12, color="black"))
        )

        # Model performance on train and test data

        # MSE
        arma11_train_mse = mean_squared_error(y_train, arma_1_1_train)
        arma11_test_mse = mean_squared_error(y_test, arma_1_1_test)
        mse_arma11 = f"The Mean squared error of ARMA(1,1) on train data: : {arma11_train_mse} and the Mean Squared Error of MSE of ARMA(1,1) on test data is : {arma11_test_mse}."

        # RMSE
        ar11_train_rmse = mean_squared_error(y_train, arma_1_1_train, squared=False)
        ar11_test_rmse = mean_squared_error(y_test, arma_1_1_test, squared=False)
        rmse_arma11 = f'RMSE of ARMA(1,1) on train data: {ar11_train_rmse} and RMSE of ARMA(1,1) on test data: {ar11_test_rmse}.'

        # Q-value
        q_arma11_train = acorr_ljungbox(arma_1_1_res, lags=5, boxpierce=True, return_df=True)
        q_ar11_test = acorr_ljungbox(arma_1_1_fore, lags=5, boxpierce=True, return_df=True)
        q_Value_arma11 = f'Q-value (residual): {q_arma11_train} and Q-value (Forecast):\n {q_ar11_test}'

        # Error mean and variance
        emv_arma11 = (f'ARMA(1,1): Mean of residual error is {np.mean(arma_1_1_res)} and Forecast error is : {np.mean(arma_1_1_fore)}.\n'
                      f'ARMA(1,1): Variance of residual error is : {np.var(arma_1_1_res)} and Forecast error is {np.var(arma_1_1_fore)} ')

        # Create HTML components for model performance metrics
        mse_html = html.Pre(mse_arma11)
        rmse_html = html.Pre(rmse_arma11)
        q_html = html.Pre(q_Value_arma11)
        emv_html = html.Pre(emv_arma11)

        # Combine theccomponents into a single HTML div
        performance_div = html.Div([
            html.H4('Model Performance Metrics:'),
            html.Div(mse_html),
            html.Div(rmse_html),
            html.Div(q_html),
            html.Div(emv_html),
        ])

        lags = 40
        Q1 = q_arma11_train
        error1 = arma_1_1_test
        chi = chi_test_plotly(na, nb, lags, Q1, error1)

        zer = zero_poles_plotly(params, na)

        con_f = conf_int(cov, params, na, nb)

        # Create HTML divs for the numerical data
        chi_html = html.Div([
            html.H4("Chi-Square Test Output:"),
            html.Pre(chi)
        ])

        zer_html = html.Div([
            html.H4("Zero-Poles Cancellation Output:"),
            html.Pre(zer)
        ])

        con_f_html = html.Div([
            html.H4("Confidence Interval Output:"),
            html.Pre(str(con_f))  # Convert DataFrame to string
        ])

        return fig_df_arsa, img1, params_div, arma_mode, paradf_div, fig_df_res_arma, fig_df_fore_arma, fig_arma_model_df, fig_arma_model_df_test, fig_arma_model_df_train, performance_div, chi_html, zer_html, con_f_html



    elif input11 == 'antartic':
        # diffferncing-one
        diff_ant1 = difference(df1['Extent 10^6 sq km'], interval=365)
        diff_df_ant = pd.DataFrame(diff_ant1, index=df1.index[365:])
        diff_ant2 = difference(diff_ant1, 1)
        diff_df1_ant = pd.DataFrame(diff_ant1, df1.index[365:])  # Adjust the index range
        diff_df2_ant = pd.DataFrame(diff_ant2, df1.index[366:])

        diff_df21_ant = pd.DataFrame(diff_ant2, index=df1.index[366:],
                                     columns=['Extent 10^6 sq km'])  # Adjust column name if needed

        # acf
        df_acf_12 = diff_df21_ant.values[1:]
        max_lag = 60
        acf_values_df1 = sm.tsa.acf(df_acf_12, nlags=max_lag)
        lags = np.arange(0, max_lag + 1)
        fig_df1_ant = go.Figure()
        fig_df1_ant.add_trace(
            go.Scatter(x=lags, y=acf_values_df1, mode='markers', marker=dict(color='blue'), name='ACF'))
        fig_df1_ant.add_trace(
            go.Scatter(x=-1 * lags, y=acf_values_df1, mode='markers', marker=dict(color='blue'),
                       name='ACF'))
        # Add stems
        for lag, acf_value in zip(lags, acf_values_df1):
            fig_df1_ant.add_trace(
                go.Scatter(x=[lag, lag], y=[0, acf_value], mode='lines', line=dict(color='blue', width=1)))
        for lag, acf_value in zip(-1 * lags, acf_values_df1):
            fig_df1_ant.add_trace(
                go.Scatter(x=[lag, lag], y=[0, acf_value], mode='lines', line=dict(color='blue', width=1)))
        # Update layout
        fig_df1_ant.update_layout(
            title="Autocorrelation Function (ACF)",
            xaxis_title="Lags",
            yaxis_title="Autocorrelation",
            plot_bgcolor='rgba(0,0,0,0)'
        )

        img2 = html.Div([
            html.Img(src=image10_path),
            html.P("We Could see that the order is (1,0)"),
            html.P("Accordingly, na = 1, nb =0 in which na(y-axis) has more constants and nb(x-axis) has more zeros."),
        ])

        # order ARMA(1,0)
        # 14: LMA algorithm
        na = 1 #y
        nb = 0 #x
        order = (na, 0, nb)  # ARIMA order: (p, d, q)
        arma_1_0 = sm.tsa.ARIMA(y1_train, order=order, trend=None).fit()
        # Estimated parameters
        ar_params1 = []
        for i in range(na):
            ar_params1.append(f"The AR coefficient a{i} is: {arma_1_0.params[i]}")

        ma_params1 = []
        for i in range(nb):
            ma_params1.append(f"The MA coefficient a{i} is: {arma_1_0.params[i + na]}")

        # Create HTML components for AR and MA coefficients
        ar_html1 = html.Pre(ar_params1)
        ma_html1 = html.Pre(ma_params1)

        # Combine the components into a single HTML div
        params_div1 = html.Div([
            html.H4('Estimated Co-effieceints:'),
            html.Div(ar_html1),
            html.Div(ma_html1),
        ])

        # Fitting original model
        arma_modes = html.Div([
                html.H4('First Arma Model:'),
                html.Pre(arma_1_0.summary().as_text()),
                html.Br(),
            ])

        # initialise values used in function
        mu = 0.01
        delta = 10 ** -6
        epsilon = 0.001
        mu_max = 10 ** 10
        max_iter = 100
        SSE, cov, params, var = step3(max_iter, mu, delta, epsilon, mu_max, na, nb, y)
        para = []
        para.append(f"The Estimated parameter of AR is: {params}")
        para_html = html.Pre(para)
        para_div = html.Div([
            html.H4('Estimated Parameters:'),
            html.Div(para_html),
            html.P("So, The parameter estimation employed by LMA(Levenberg Marquardt Algorithm) is shown. Since this is evident that the order is (1,0), the parameter estimation of AR model is shown.")
        ])



        # Prediction on train set
        total_length = 14691
        train_length = int(0.8 * total_length)
        test_length = total_length - train_length
        # For training set
        arma_1_0_train = arma_1_0.predict(start=0, end=train_length - 1)
        arma_1_0_res = y_train - arma_1_0_train
        # For test set
        arma_1_0_test = arma_1_0.predict(start=train_length, end=total_length - 1)
        arma_1_0_fore = y_test - arma_1_0_test

        # acfres
        df1_acf_res_arma = arma_1_0_res
        max_lag = 60
        acf_values_df1_arma = sm.tsa.acf(df1_acf_res_arma, nlags=max_lag)
        lags = np.arange(0, max_lag + 1)
        fig_df1_res_arma = go.Figure()
        fig_df1_res_arma.add_trace(
            go.Scatter(x=lags, y=acf_values_df1_arma, mode='markers', marker=dict(color='blue'), name='ACF'))
        fig_df1_res_arma.add_trace(
            go.Scatter(x=-1 * lags, y=acf_values_df1_arma, mode='markers', marker=dict(color='blue'),
                       name='ACF'))
        # Add stems
        for lag, acf_value in zip(lags, acf_values_df1_arma):
            fig_df1_res_arma.add_trace(
                go.Scatter(x=[lag, lag], y=[0, acf_value], mode='lines', line=dict(color='blue', width=1)))
        for lag, acf_value in zip(-1 * lags, acf_values_df1_arma):
            fig_df1_res_arma.add_trace(
                go.Scatter(x=[lag, lag], y=[0, acf_value], mode='lines', line=dict(color='blue', width=1)))
        # Update layout
        fig_df1_res_arma.update_layout(
            title="Autocorrelation Function (ACF) of Residuals",
            xaxis_title="Lags",
            yaxis_title="Autocorrelation",
            plot_bgcolor='rgba(0,0,0,0)'
        )

        # acffore
        df1_acf_fore_arma = arma_1_0_fore.values
        max_lag = 60
        acf_fore_values_df1_arma = sm.tsa.acf(df1_acf_fore_arma, nlags=max_lag)
        lags = np.arange(0, max_lag + 1)
        fig_df1_fore_arma = go.Figure()
        fig_df1_fore_arma.add_trace(
            go.Scatter(x=lags, y=acf_fore_values_df1_arma, mode='markers', marker=dict(color='blue'), name='ACF'))
        fig_df1_fore_arma.add_trace(
            go.Scatter(x=-1 * lags, y=acf_fore_values_df1_arma, mode='markers', marker=dict(color='blue'),
                       name='ACF'))
        # Add stems
        for lag, acf_value in zip(lags, acf_fore_values_df1_arma):
            fig_df1_fore_arma.add_trace(
                go.Scatter(x=[lag, lag], y=[0, acf_value], mode='lines', line=dict(color='blue', width=1)))
        for lag, acf_value in zip(-1 * lags, acf_fore_values_df1_arma):
            fig_df1_fore_arma.add_trace(
                go.Scatter(x=[lag, lag], y=[0, acf_value], mode='lines', line=dict(color='blue', width=1)))
        # Update layout
        fig_df1_fore_arma.update_layout(
            title="Autocorrelation Function (ACF) of forecasted",
            xaxis_title="Lags",
            yaxis_title="Autocorrelation",
            plot_bgcolor='rgba(0,0,0,0)'
        )

        # full arma model
        fig_arma_model_df1 = go.Figure()
        fig_arma_model_df1.add_trace(go.Scatter(x=y1_train.index, y=y1_train, mode='lines', name='Train'))
        fig_arma_model_df1.add_trace(go.Scatter(x=y1_test.index, y=y1_test, mode='lines', name='Test'))
        fig_arma_model_df1.add_trace(go.Scatter(x=y1_test.index, y=arma_1_0_test, mode='lines', name='Predicted'))
        fig_arma_model_df1.update_layout(
            xaxis_title='Time',
            yaxis_title='Extent 10^6 sq km',
            title='ARMA Model - TRAIN, TEST AND PREDICTED ',
            xaxis=dict(tickangle=-90),
            legend=dict(x=0, y=1, traceorder="normal",
                        font=dict(family="sans-serif", size=12, color="black"))
        )

        # full testvspred model
        fig_arma_model_df1_test = go.Figure()
        fig_arma_model_df1_test.add_trace(go.Scatter(x=y1_test.index, y=y1_test, mode='lines', name='Test'))
        fig_arma_model_df1_test.add_trace(go.Scatter(x=y1_test.index, y=arma_1_0_test, mode='lines', name='Predicted'))
        fig_arma_model_df1_test.update_layout(
            xaxis_title='Time',
            yaxis_title='Extent 10^6 sq km',
            title='ARMA Model - TEST vs PREDICTED ',
            xaxis=dict(tickangle=-90),
            legend=dict(x=0, y=1, traceorder="normal",
                        font=dict(family="sans-serif", size=12, color="black"))
        )

        # full trainvspred model
        fig_arma_model_df1_train = go.Figure()
        fig_arma_model_df1_train.add_trace(go.Scatter(x=y1_train.index, y=y1_train, mode='lines', name='Test'))
        fig_arma_model_df1_train.add_trace(go.Scatter(x=y1_train.index, y=arma_1_0_test, mode='lines', name='Predicted'))
        fig_arma_model_df1_train.update_layout(
            xaxis_title='Time',
            yaxis_title='Extent 10^6 sq km',
            title='ARMA Model - TRAIN vs PREDICTED ',
            xaxis=dict(tickangle=-90),
            legend=dict(x=0, y=1, traceorder="normal",
                        font=dict(family="sans-serif", size=12, color="black"))
        )

        # MSE
        arma10_train_mse = mean_squared_error(y_train, arma_1_0_train)
        arma10_test_mse = mean_squared_error(y_test, arma_1_0_test)
        mse_arma10 = f"The Mean squared error of ARMA(1,0) on train data: : {arma10_train_mse} and the Mean Squared Error of MSE of ARMA(1,0) on test data is : {arma10_test_mse}."

        # RMSE
        ar10_train_rmse = mean_squared_error(y_train, arma_1_0_train, squared=False)
        ar10_test_rmse = mean_squared_error(y_test, arma_1_0_test, squared=False)
        rmse_arma10 = f'RMSE of ARMA(1,0) on train data: {ar10_train_rmse} and RMSE of ARMA(1,0) on test data: {ar10_test_rmse}.'

        # Q-value
        q_arma10_train = acorr_ljungbox(arma_1_0_res, lags=5, boxpierce=True, return_df=True)
        q_ar10test = acorr_ljungbox(arma_1_0_fore, lags=5, boxpierce=True, return_df=True)
        q_Value_arma10 = f'Q-value (residual): {q_arma10_train} and Q-value (Forecast):\n {q_ar10test}'

        # Error mean and variance
        emv_arma10 = (f'ARMA(1,0): Mean of residual error is {np.mean(arma_1_0_res)} and Forecast error is : {np.mean(arma_1_0_fore)}.\n'
                      f'ARMA(1,0): Variance of residual error is : {np.var(arma_1_0_res)} and Forecast error is {np.var(arma_1_0_fore)} ')


        # Create HTML components for model performance metrics
        mse_html_df1 = html.Pre(mse_arma10)
        rmse_html_df1 = html.Pre(rmse_arma10)
        q_html_df1 = html.Pre(q_Value_arma10)
        emv_html_df1 = html.Pre(emv_arma10)

        # Combine theccomponents into a single HTML div
        performance_div_df1 = html.Div([
            html.H4('Model Performance Metrics:'),
            html.Div(mse_html_df1),
            html.Div(rmse_html_df1),
            html.Div(q_html_df1),
            html.Div(emv_html_df1),
        ])

        lags = 40
        Q1_10 = q_arma10_train
        error10 = arma_1_0_test
        chi_10 = chi_test_plotly(na, nb, lags, Q1_10, error10)

        zer_10 = zero_poles_plotly(params, na)

        con_f_10 = conf_int(cov, params, na, nb)

        # Create HTML divs for the numerical data
        chi_html_10 = html.Div([
            html.H4("Chi-Square Test Output:"),
            html.Pre(chi_10)
        ])

        zer_html_10 = html.Div([
            html.H4("Zero-Poles Cancellation Output:"),
            html.Pre(zer_10)
        ])

        con_f_html_10 = html.Div([
            html.H4("Confidence Interval Output:"),
            html.Pre(str(con_f_10))  # Convert DataFrame to string
        ])

        return fig_df1_ant, img2, params_div1, arma_modes, para_div, fig_df1_res_arma, fig_df1_fore_arma, fig_arma_model_df1, fig_arma_model_df1_test, fig_arma_model_df1_train, performance_div_df1, chi_html_10, zer_html_10, con_f_html_10


@my_app.callback(Output('Model-fit', 'children'),
               Output('Acf-res-sarima', 'figure'),
               Output('Acf-fore-sarima', 'figure'),
               Output('Perf-sarima', 'children'),
               Output('model-sarima', 'figure'),
               Output('testvpred-sarima', 'figure'),
               Output('trainvpred-sarima', 'figure'),
                 [Input('dropdown_sarima', 'value')]
)

def sarima_fore(input12):
    if input12 == 'artic':
            # SARIMA(1,0,1)(1,0,365)
            # order = (1, 0, 1)
            # SARIMA(1,0,1)(1,0,1,365)
            sarima = sm.tsa.statespace.SARIMAX(y_train, order=(1, 0, 1), seasonal_order=(1, 0, 1, 12),
                                       enforce_stationarity=False, enforce_invertibility=False)

            results = sarima.fit(disp=0)

            #out1
            # Fitting original model
            sarima_modes = html.Div([
                html.H4('Sarima Model:'),
                html.Pre(results.summary().as_text()),
                html.Br(),
            ])

            # predictions on train data
            sarima_train = results.get_prediction(start=0, end=len(y_train), dynamic=False)
            Sarima_pred = sarima_train.predicted_mean
            sarima_res = y_train - Sarima_pred.values[1:]

            # forecast
            sarima_test = results.predict(start=0, end=(len(y_test)))
            sarima_fore = y_test - sarima_test.values[1:]

            # acfres
            df_acf_res_sarima = sarima_res.values
            max_lag = 60
            acf_values_df_sarima = sm.tsa.acf(df_acf_res_sarima, nlags=max_lag)
            lags = np.arange(0, max_lag + 1)
            fig_df_res_sarima = go.Figure()
            fig_df_res_sarima.add_trace(
                go.Scatter(x=lags, y=acf_values_df_sarima, mode='markers', marker=dict(color='blue'), name='ACF'))
            fig_df_res_sarima.add_trace(
                go.Scatter(x=-1 * lags, y=acf_values_df_sarima, mode='markers', marker=dict(color='blue'),
                           name='ACF'))
            # Add stems
            for lag, acf_value in zip(lags, acf_values_df_sarima):
                fig_df_res_sarima.add_trace(
                    go.Scatter(x=[lag, lag], y=[0, acf_value], mode='lines', line=dict(color='blue', width=1)))
            for lag, acf_value in zip(-1 * lags, acf_values_df_sarima):
                fig_df_res_sarima.add_trace(
                    go.Scatter(x=[lag, lag], y=[0, acf_value], mode='lines', line=dict(color='blue', width=1)))
            # Update layout
            fig_df_res_sarima.update_layout(
                title="Autocorrelation Function (ACF) of Residuals",
                xaxis_title="Lags",
                yaxis_title="Autocorrelation",
                plot_bgcolor='rgba(0,0,0,0)'
            )

            # acffore
            df_acf_fore_sarima = sarima_fore.values
            max_lag = 60
            acf_values_df_sarima_fore = sm.tsa.acf(df_acf_fore_sarima, nlags=max_lag)
            lags = np.arange(0, max_lag + 1)
            fig_df_fore_sarima = go.Figure()
            fig_df_fore_sarima.add_trace(
                go.Scatter(x=lags, y=acf_values_df_sarima_fore, mode='markers', marker=dict(color='blue'), name='ACF'))
            fig_df_fore_sarima.add_trace(
                go.Scatter(x=-1 * lags, y=acf_values_df_sarima_fore, mode='markers', marker=dict(color='blue'),
                           name='ACF'))
            # Add stems
            for lag, acf_value in zip(lags,acf_values_df_sarima_fore):
                fig_df_fore_sarima.add_trace(
                    go.Scatter(x=[lag, lag], y=[0, acf_value], mode='lines', line=dict(color='blue', width=1)))
            for lag, acf_value in zip(-1 * lags, acf_values_df_sarima_fore):
                fig_df_fore_sarima.add_trace(
                    go.Scatter(x=[lag, lag], y=[0, acf_value], mode='lines', line=dict(color='blue', width=1)))
            # Update layout
            fig_df_fore_sarima.update_layout(
                title="Autocorrelation Function (ACF) of Forecasted",
                xaxis_title="Lags",
                yaxis_title="Autocorrelation",
                plot_bgcolor='rgba(0,0,0,0)'
            )

            # Model performance on train and test data
            # MSE
            sarima_train_mse = mean_squared_error(y_train, Sarima_pred[1:])
            sarima_test_mse = mean_squared_error(y_test, sarima_test[1:])
            mse_sarima_11 = f"The Mean squared error of SARIMA(1,0,1)(1,0,1,365) on train data: : {sarima_train_mse} and the Mean Squared Error of MSE of SARIMA(1,0,1)(1,0,1,365) on test data is : {sarima_test_mse}."

            # RMSE
            sarima_train_rmse = mean_squared_error(y_train, Sarima_pred[1:], squared=False)
            sarima_test_rmse = mean_squared_error(y_test, sarima_test[1:], squared=False)
            rmse_sarima_11 = f'RMSE of SARIMA(1,0,1)(1,0,1,365) on train data: {sarima_train_rmse} and RMSE of SARIMA(1,0,1)(1,0,1,365) on test data: {sarima_test_rmse}.'

            # Q-value
            q_sarima_train = acorr_ljungbox(sarima_res, lags=5, boxpierce=True, return_df=True)
            q_sarimatest = acorr_ljungbox(sarima_fore, lags=5, boxpierce=True, return_df=True)
            q_Value_sarima11 = f'Q-value (residual): {q_sarima_train} and Q-value (Forecast):\n {q_sarimatest}'

            # Error mean and variance
            emv_sarima11 = (
                f'SARIMA(1,0,1)(1,0,1,365): Mean of residual error is {np.mean(sarima_res)} and Forecast error is : {np.mean(sarima_fore)}.\n'
                f'SARIMA(1,0,1)(1,0,1,365): Variance of residual error is : {np.var(sarima_res)} and Forecast error is {np.var(sarima_res)} ')

            # Covariance matrix
            covariance = f'Covariance matrix\n, {results.cov_params()}'

            # standard error
            standard_error = f'Standard error:\n, {results.bse}'


            # Create HTML components for model performance metrics
            mse_html_sarima = html.Pre(mse_sarima_11)
            rmse_html_sarima = html.Pre(rmse_sarima_11)
            q_html_sarima = html.Pre(q_Value_sarima11)
            emv_html_sarima = html.Pre(emv_sarima11)
            covariance_html_sarima = html.Pre(covariance)
            standard_error_sarima = html.Pre(standard_error)


            # Combine theccomponents into a single HTML div
            performance_div_saarima = html.Div([
                html.H4('Model Performance Metrics:'),
                html.Div(mse_html_sarima),
                html.Div(rmse_html_sarima),
                html.Div(q_html_sarima),
                html.Div(emv_html_sarima),
                html.Div(covariance_html_sarima),
                html.Div(standard_error_sarima)
            ])

            # full arma model
            fig_sarima_model_df = go.Figure()
            fig_sarima_model_df.add_trace(go.Scatter(x=y_train.index, y=y_train, mode='lines', name='Train'))
            fig_sarima_model_df.add_trace(go.Scatter(x=y_test.index, y=y_test, mode='lines', name='Test'))
            fig_sarima_model_df.add_trace(go.Scatter(x=y_test.index, y=sarima_test[1:], mode='lines', name='Predicted'))
            fig_sarima_model_df.update_layout(
                xaxis_title='Time',
                yaxis_title='Extent 10^6 sq km',
                title='SARIMA Model - TRAIN, TEST AND PREDICTED ',
                xaxis=dict(tickangle=-90),
                legend=dict(x=0, y=1, traceorder="normal",
                            font=dict(family="sans-serif", size=12, color="black"))
            )

            # full testvspred model
            fig_sarima_model_df_test = go.Figure()
            fig_sarima_model_df_test.add_trace(go.Scatter(x=y_test.index, y=y_test, mode='lines', name='Test'))
            fig_sarima_model_df_test.add_trace(
                go.Scatter(x=y_test.index, y=sarima_test[1:], mode='lines', name='Predicted'))
            fig_sarima_model_df_test.update_layout(
                xaxis_title='Time',
                yaxis_title='Extent 10^6 sq km',
                title='SARIMA Model - TEST vs PREDICTED ',
                xaxis=dict(tickangle=-90),
                legend=dict(x=0, y=1, traceorder="normal",
                            font=dict(family="sans-serif", size=12, color="black"))
            )

            # full trainvspred model
            fig_sarima_model_df_train = go.Figure()
            fig_sarima_model_df_train.add_trace(go.Scatter(x=y_train.index, y=y_train, mode='lines', name='Test'))
            fig_sarima_model_df_train.add_trace(
                go.Scatter(x=y_train.index, y=Sarima_pred[1:], mode='lines', name='Predicted'))
            fig_sarima_model_df_train.update_layout(
                xaxis_title='Time',
                yaxis_title='Extent 10^6 sq km',
                title='SARIMA Model - TRAIN vs PREDICTED ',
                xaxis=dict(tickangle=-90),
                legend=dict(x=0, y=1, traceorder="normal",
                            font=dict(family="sans-serif", size=12, color="black"))
            )

            return sarima_modes, fig_df_res_sarima, fig_df_fore_sarima, performance_div_saarima, fig_sarima_model_df, fig_sarima_model_df_test, fig_sarima_model_df_train

    elif input12 == 'antartic':
            # SARIMA(1,0,1)(0,0,365)
            # order = (1, 0, 1)
            # SARIMA(1,0,1)(0,0,1,365)
            sarima = sm.tsa.statespace.SARIMAX(y1_train, order=(1, 0, 1), seasonal_order=(0, 0, 1, 12),
                                               enforce_stationarity=False, enforce_invertibility=False)
            results1 = sarima.fit(disp=0)

            # out1
            # Fitting original model
            sarima_modes1 = html.Div([
                html.H4('Sarima Model:'),
                html.Pre(results1.summary().as_text()),
                html.Br(),
            ])

            # predictions on train data
            sarima_train1 = results1.get_prediction(start=0, end=len(y1_train), dynamic=False)
            sarima_pred1 = sarima_train1.predicted_mean
            sarima_res1 = y1_train - sarima_pred1.values[1:]

            # forecast
            sarima_test1 = results1.predict(start=0, end=(len(y1_test)))
            sarima_fore1 = y1_test - sarima_test1.values[1:]

            # acfres
            df1_acf_res_sarima = sarima_res1.values
            max_lag = 60
            acf_values_df1_sarima = sm.tsa.acf(df1_acf_res_sarima, nlags=max_lag)
            lags = np.arange(0, max_lag + 1)
            fig_df1_res_sarima = go.Figure()
            fig_df1_res_sarima.add_trace(
                go.Scatter(x=lags, y=acf_values_df1_sarima, mode='markers', marker=dict(color='blue'), name='ACF'))
            fig_df1_res_sarima.add_trace(
                go.Scatter(x=-1 * lags, y=acf_values_df1_sarima, mode='markers', marker=dict(color='blue'),
                           name='ACF'))
            # Add stems
            for lag, acf_value in zip(lags, acf_values_df1_sarima):
                fig_df1_res_sarima.add_trace(
                    go.Scatter(x=[lag, lag], y=[0, acf_value], mode='lines', line=dict(color='blue', width=1)))
            for lag, acf_value in zip(-1 * lags, acf_values_df1_sarima):
                fig_df1_res_sarima.add_trace(
                    go.Scatter(x=[lag, lag], y=[0, acf_value], mode='lines', line=dict(color='blue', width=1)))
            # Update layout
            fig_df1_res_sarima.update_layout(
                title="Autocorrelation Function (ACF) of Residuals",
                xaxis_title="Lags",
                yaxis_title="Autocorrelation",
                plot_bgcolor='rgba(0,0,0,0)'
            )

            # acffore
            df1_acf_fore_sarima = sarima_fore1.values
            max_lag = 60
            acf_values_df1_sarima_fore = sm.tsa.acf(df1_acf_fore_sarima, nlags=max_lag)
            lags = np.arange(0, max_lag + 1)
            fig_df1_fore_sarima = go.Figure()
            fig_df1_fore_sarima.add_trace(
                go.Scatter(x=lags, y=acf_values_df1_sarima_fore, mode='markers', marker=dict(color='blue'), name='ACF'))
            fig_df1_fore_sarima.add_trace(
                go.Scatter(x=-1 * lags, y=acf_values_df1_sarima_fore, mode='markers', marker=dict(color='blue'),
                           name='ACF'))
            # Add stems
            for lag, acf_value in zip(lags, acf_values_df1_sarima_fore):
                fig_df1_fore_sarima.add_trace(
                    go.Scatter(x=[lag, lag], y=[0, acf_value], mode='lines', line=dict(color='blue', width=1)))
            for lag, acf_value in zip(-1 * lags, acf_values_df1_sarima_fore):
                fig_df1_fore_sarima.add_trace(
                    go.Scatter(x=[lag, lag], y=[0, acf_value], mode='lines', line=dict(color='blue', width=1)))
            # Update layout
            fig_df1_fore_sarima.update_layout(
                title="Autocorrelation Function (ACF) of Forecasted",
                xaxis_title="Lags",
                yaxis_title="Autocorrelation",
                plot_bgcolor='rgba(0,0,0,0)'
            )

            # Model performance on train and test data

            # MSE
            sarima_train_mse1 = mean_squared_error(y1_train, sarima_pred1[1:])
            sarima_test_mse1 = mean_squared_error(y1_test, sarima_test1[1:])
            mse_sarima_ant = f"The Mean squared error of SARIMA(1,0,1)(0,0,1,365) on train data: : {sarima_train_mse1}.\n the Mean Squared Error of MSE of SARIMA(1,0,1)(0,0,1,365) on test data is : {sarima_test_mse1}."

            # RMSE
            sarima_train_rmse1 = mean_squared_error(y1_train, sarima_pred1[1:], squared=False)
            sarima_test_rmse1 = mean_squared_error(y1_test, sarima_test1[1:], squared=False)
            rmse_sarima_ant = f'RMSE of SARIMA(1,0,1)(0,0,1,365) on train data: {sarima_train_rmse1}.\n RMSE of SARIMA(1,0,1)(0,0,1,365) on test data: {sarima_test_rmse1}.'

            # Q-value
            q_sarima_train1 = acorr_ljungbox(sarima_res1, lags=5, boxpierce=True, return_df=True)
            q_sarimatest1 = acorr_ljungbox(sarima_fore1, lags=5, boxpierce=True, return_df=True)
            q_Value_sarima_ant = f'Q-value (residual): {q_sarima_train1} and Q-value (Forecast):\n {q_sarimatest1}'

            # Error mean and variance
            emv_sarima_ant = (
                f'SARIMA(1,0,1)(0,0,1,365): Mean of residual error is {np.mean(sarima_res1)} and Forecast error is : {np.mean(sarima_fore1)}.\n ',
                f'SARIMA(1,0,1)(0,0,1,365): Variance of residual error is : {np.var(sarima_res1)} and Forecast error is {np.var(sarima_res1)} ')

            # Covariance matrix
            covariance_ant = f'Covariance matrix\n {results1.cov_params()}'

            # standard error
            standard_error_ant = f'Standard error:\n {results1.bse}'

            # Create HTML components for model performance metrics
            mse_html_sarima_ant = html.Pre(mse_sarima_ant)
            rmse_html_sarima_ant = html.Pre(rmse_sarima_ant)
            q_html_sarima_ant = html.Pre(q_Value_sarima_ant)
            emv_html_sarima_ant = html.Pre(emv_sarima_ant)
            covariance_html_sarima_ant = html.Pre(covariance_ant)
            standard_error_sarima_ant = html.Pre(standard_error_ant)

            # Combine theccomponents into a single HTML div
            performance_div_saarima = html.Div([
                html.H4('Model Performance Metrics:'),
                html.Div(mse_html_sarima_ant),
                html.Div(rmse_html_sarima_ant),
                html.Div(q_html_sarima_ant),
                html.Div(emv_html_sarima_ant),
                html.Div(covariance_html_sarima_ant),
                html.Div(standard_error_sarima_ant)
            ])

            # full arma model
            fig_sarima_model_df1 = go.Figure()
            fig_sarima_model_df1.add_trace(go.Scatter(x=y1_train.index, y=y1_train, mode='lines', name='Train'))
            fig_sarima_model_df1.add_trace(go.Scatter(x=y1_test.index, y=y1_test, mode='lines', name='Test'))
            fig_sarima_model_df1.add_trace(go.Scatter(x=y1_test.index, y=sarima_test1[1:], mode='lines', name='Predicted'))
            fig_sarima_model_df1.update_layout(
                xaxis_title='Time',
                yaxis_title='Extent 10^6 sq km',
                title='SARIMA Model - TRAIN, TEST AND PREDICTED ',
                xaxis=dict(tickangle=-90),
                legend=dict(x=0, y=1, traceorder="normal",
                            font=dict(family="sans-serif", size=12, color="black"))
            )

            # full testvspred model
            fig_sarima_model_df1_test = go.Figure()
            fig_sarima_model_df1_test.add_trace(go.Scatter(x=y1_test.index, y=y1_test, mode='lines', name='Test'))
            fig_sarima_model_df1_test.add_trace(
                go.Scatter(x=y1_test.index, y=sarima_test1[1:], mode='lines', name='Predicted'))
            fig_sarima_model_df1_test.update_layout(
                xaxis_title='Time',
                yaxis_title='Extent 10^6 sq km',
                title='SARIMA Model - TEST vs PREDICTED ',
                xaxis=dict(tickangle=-90),
                legend=dict(x=0, y=1, traceorder="normal",
                            font=dict(family="sans-serif", size=12, color="black"))
            )

            # full trainvspred model
            fig_sarima_model_df1_train = go.Figure()
            fig_sarima_model_df1_train.add_trace(go.Scatter(x=y1_train.index, y=y1_train, mode='lines', name='Test'))
            fig_sarima_model_df1_train.add_trace(
                go.Scatter(x=y1_train.index, y=sarima_pred1[1:], mode='lines', name='Predicted'))
            fig_sarima_model_df1_train.update_layout(
                xaxis_title='Time',
                yaxis_title='Extent 10^6 sq km',
                title='SARIMA Model - TRAIN vs PREDICTED ',
                xaxis=dict(tickangle=-90),
                legend=dict(x=0, y=1, traceorder="normal",
                            font=dict(family="sans-serif", size=12, color="black"))
            )

            return sarima_modes1, fig_df1_res_sarima, fig_df1_fore_sarima, performance_div_saarima, fig_sarima_model_df1, fig_sarima_model_df1_test, fig_sarima_model_df1_train

@my_app.callback(
               Output('testvsprediction', 'figure'),
                 [Input('dropdown_sarimap', 'value')]
)

def forecast_final(input13):
    if input13 == 'artic':
        sarima = sm.tsa.statespace.SARIMAX(y_test, order=(1, 0, 1), seasonal_order=(1, 0, 1, 12),
                                           enforce_stationarity=False, enforce_invertibility=False)
        predicted1 = sarima.fit().predict()

        # Create traces
        trace_actual = go.Scatter(x=y_test.index, y=y_test.values, mode='lines', name='Actual')
        trace_predicted = go.Scatter(x=y_test.index, y=predicted1, mode='lines', name='Predicted')

        # Create layout
        layout = go.Layout(title='Actual vs Predicted',
                           xaxis=dict(title='Index'),
                           yaxis=dict(title='Extent 10^6 sq km'))

        # Create figure
        fig = go.Figure(data=[trace_actual, trace_predicted], layout=layout)
        return fig

    elif input13 == 'antartic':
        sarima1 = sm.tsa.statespace.SARIMAX(y1_test, order=(1, 0, 1), seasonal_order=(1, 0, 1, 12),
                                           enforce_stationarity=False, enforce_invertibility=False)
        predicted2 = sarima1.fit().predict()

        # Create traces
        trace_actual1 = go.Scatter(x=y1_test.index, y=y1_test.values, mode='lines', name='Actual')
        trace_predicted1 = go.Scatter(x=y1_test.index, y=predicted2, mode='lines', name='Predicted')

        # Create layout
        layout1 = go.Layout(title='Actual vs Predicted in Southern Hemisphere(Antartic)',
                           xaxis=dict(title='Index'),
                           yaxis=dict(title='Extent 10^6 sq km'))

        # Create figure
        fig1 = go.Figure(data=[trace_actual1, trace_predicted1], layout=layout1)

        return fig1


my_app.run_server(port=9900, host='0.0.0.0')
