from flask import Flask, render_template, request
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as pyo
import pickle
import os
from prophet import Prophet

app = Flask(__name__)

# Load full dataset once
df_full = pd.read_csv("data/crime_rates.csv")
df_full.dropna(subset=['Date', 'Primary Type'], inplace=True)
df_full['Date'] = pd.to_datetime(df_full['Date'], format="%d/%m/%Y %I:%M:%S %p", errors='coerce')
df_full = df_full[df_full['Date'].notna()]
df_full = df_full[df_full['Date'] >= '2010-01-01']

# Get unique crime types
crime_types = sorted(df_full['Primary Type'].unique())

@app.route('/', methods=['GET', 'POST'])
def index():
    result_plot = None
    error_message = ""

    if request.method == 'POST':
        model_name = request.form['model']
        selected_crime_type = request.form['crime_type']
        start_date = request.form['start_date']
        end_date = request.form['end_date']

        # Filter dataset based on selected crime type
        df_crime = df_full[df_full['Primary Type'] == selected_crime_type].copy()

        df_crime['Day'] = df_crime['Date'].dt.date
        daily_crime = df_crime.groupby('Day').size().reset_index(name='CrimeCount')
        daily_crime = daily_crime.rename(columns={'Day': 'ds', 'CrimeCount': 'y'})
        daily_crime['ds'] = pd.to_datetime(daily_crime['ds'])  # datetime format

        if daily_crime.empty:
            error_message = f"No data available for {selected_crime_type}."
            return render_template('index.html', crime_types=crime_types, result=None, error=error_message)

        # Preprocessing based on model selection
        if model_name == 'NoOutliers_model.pkl':
            # Remove outliers (you can improve this logic if needed)
            q1 = daily_crime['y'].quantile(0.25)
            q3 = daily_crime['y'].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            daily_crime = daily_crime[(daily_crime['y'] >= lower_bound) & (daily_crime['y'] <= upper_bound)]

        # Build Prophet model
        model = Prophet()

        if model_name == 'holidays_model.pkl':
            # Add holidays (example holidays can be customized)
            holidays = pd.DataFrame({
                'holiday': 'public_holiday',
                'ds': pd.to_datetime([
                    '2016-01-01', '2016-12-25', '2017-01-01', '2017-12-25', 
                    '2018-01-01', '2018-12-25'
                ]),
                'lower_window': 0,
                'upper_window': 1,
            })
            model = Prophet(holidays=holidays)

        # Fit model
        model.fit(daily_crime)

        # Future dataframe generation
        future_dates = pd.date_range(start=start_date, end=end_date, freq='D')  # 'MS' = month start
        future_df = pd.DataFrame({'ds': future_dates})

        # Predict
        forecast = model.predict(future_df)
        forecast_range = forecast[['ds', 'yhat']]

        avg_prediction = round(forecast_range['yhat'].mean(), 2)

        # Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=forecast_range['ds'], y=forecast_range['yhat'],
                                 mode='lines+markers', name='Prediction'))
        fig.update_layout(
            title=f'{selected_crime_type} Prediction ({model_name.split("_")[0].capitalize()} Model)',
            xaxis_title='Date',
            yaxis_title='Predicted Crimes'
        )
        result_plot = pyo.plot(fig, include_plotlyjs=False, output_type='div')

        return render_template('result.html', plot=result_plot, avg=avg_prediction, selected_crime=selected_crime_type)

    return render_template('index.html', crime_types=crime_types, result=None, error=error_message)

if __name__ == '__main__':
    app.run(debug=True)