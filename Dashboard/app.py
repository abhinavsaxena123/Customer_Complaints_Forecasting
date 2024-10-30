import pickle
import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go

# Load the saved Holt-Winters model
with open('../Saved_Models/holt_winters.pkl', 'rb') as file:
    hw_model = pickle.load(file)

# Load the saved Auto ARIMA model
with open('../Saved_Models/auto_arima.pkl', 'rb') as f:
    auto_arima_model = pickle.load(f)

# Create the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div(children=[
    html.H1(children='Customer Complaints Forecasting', style={'textAlign': 'center', 'color': '#333', 'fontSize': '36px'}),
    
    # Date input for starting date
    html.Div([
        html.Label('Select the start date for forecasting:', style={'fontSize': '18px'}),
        dcc.Input(
            id='start-date',
            type='text',
            value=pd.Timestamp.now().strftime('%Y-%m-%d'),  # Default to today's date
            style={'margin': '10px', 'fontSize': '16px', 'width': '200px'}
        ),
    ], style={'marginBottom': '20px', 'textAlign': 'center'}),
    
    # Date input for end date
    html.Div([
        html.Label('Select the end date for forecasting:', style={'fontSize': '18px'}),
        dcc.Input(
            id='end-date',
            type='text',
            value=(pd.Timestamp.now() + pd.Timedelta(days=79)).strftime('%Y-%m-%d'),  # Default to 79 days from today
            style={'margin': '10px', 'fontSize': '16px', 'width': '200px'}
        ),
    ], style={'marginBottom': '20px', 'textAlign': 'center'}),

    # Dropdown for model selection
    html.Div([
        html.Label('Select Forecasting Model:', style={'fontSize': '18px'}),
        dcc.Dropdown(
            id='model-dropdown',
            options=[
                {'label': 'Holt-Winters', 'value': 'holt_winters'},
                {'label': 'Auto ARIMA', 'value': 'auto_arima'}
            ],
            value='holt_winters',  # Default model
            style={'width': '300px', 'marginLeft': '220px', 'marginTop':'15px'}
        ),
    ], style={'marginBottom': '20px', 'textAlign': 'center'}),

    # Button to submit
    html.Div([
        html.Button('Forecast', id='forecast-button', n_clicks=0, style={'margin': '10px', 'fontSize': '16px'}),
    ], style={'textAlign': 'center'}),

    # Graph for displaying forecast
    dcc.Graph(id='forecast-graph', style={'height': '60vh'})
], style={'padding': '20px', 'backgroundColor': '#f9f9f9', 'borderRadius': '10px', 'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.2)'})

# Callback to update the forecast graph
@app.callback(
    Output('forecast-graph', 'figure'),
    Input('forecast-button', 'n_clicks'),
    Input('start-date', 'value'),
    Input('end-date', 'value'),
    Input('model-dropdown', 'value')
)
def update_graph(n_clicks, start_date, end_date, model):
    if n_clicks > 0:  # Only update when the button is clicked
        # Convert input dates to datetime
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        # Calculate the number of steps for forecasting
        steps = (end_date - start_date).days + 1  # Include end date

        # Generate forecast based on selected model
        if model == 'holt_winters':
            holt_winters_forecast = hw_model.forecast(steps=steps)
            forecast_index = pd.date_range(start=start_date, periods=steps, freq='D')

            # Create the figure for Holt-Winters
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=forecast_index, y=holt_winters_forecast,
                                     mode='lines+markers', name='Holt-Winters Forecast'))

        elif model == 'auto_arima':
            # Generate the forecast using Auto ARIMA
            auto_arima_forecast = auto_arima_model.predict(n_periods=steps)
            forecast_index = pd.date_range(start=start_date, periods=steps, freq='D')

            # Create the figure for Auto ARIMA
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=forecast_index, y=auto_arima_forecast,
                                     mode='lines+markers', name='Auto ARIMA Forecast'))

        # Update layout
        fig.update_layout(title=f'{model.replace("_", " ").title()} Forecast',
                          xaxis_title='Date',
                          yaxis_title='Forecasted Value',
                          template='plotly_white',
                          height=600)

        return fig

    # Return empty figure initially
    return go.Figure()

if __name__ == '__main__':
    app.run_server(debug=True)

