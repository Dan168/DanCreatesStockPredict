from flask import Flask, render_template
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import datetime as dt
import yfinance as yf

import matplotlib.pyplot as plt

app = Flask(__name__)


def Poly_reg(ticker, weeks):
    # Get today's date and 12 weeks prior
    today = dt.date.today()
    lag = today - dt.timedelta(weeks=weeks)

    # Download historical data from yahoo finance to train model based on dates
    df = yf.download(ticker, lag, today)
    df.reset_index(inplace=True)

    # Convert the date column to ordinal value (int to work with)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Date'] = df['Date'].map(dt.datetime.toordinal)

    X = df['Date']
    y = df['Close']

    X = X.values.reshape(-1, 1)

    poly_reg = PolynomialFeatures(degree=2)
    X = poly_reg.fit_transform(X)

    lr = LinearRegression()
    lr.fit(X, y)

    # Create a collection of 10 dates + 1 day each iteration (1 day, 2 day etc)
    prediction_dates = [dt.date.today() + dt.timedelta(days=i) for i in range(1, 11)]
    prediction_dates_ord = [date.toordinal() for date in prediction_dates]
    prediction_dates_poly = poly_reg.transform([[ordinal] for ordinal in prediction_dates_ord])

    # For each date in the above collection, convert to ordinal and predict a price
    predictions = [str(lr.predict(poly_features.reshape(1, -1))[0]) for poly_features in prediction_dates_poly]

    return predictions


def Get_close_price(ticker):
    today = dt.date.today()
    df = yf.download(ticker, today - dt.timedelta(days=3), today)
    df.reset_index(inplace=True)
    return df['Close'][0]


@app.route('/', methods=['GET', 'POST'])
def Home():
    return render_template('index.html')


@app.route('/TSLA', methods=['GET', 'POST'])
def TSLA():
    return render_template('TSLA.html', predictions=Poly_reg('TSLA', 3), last_close_price=Get_close_price('TSLA'))


@app.route('/AAPL', methods=['GET', 'POST'])
def AAPL():
    # Get today's date and 12 weeks prior
    today = dt.date.today()
    lag = today - dt.timedelta(weeks=12)

    # Download historical data from yahoo finance to train model based on dates
    df = yf.download('AAPL', lag, today)
    df.reset_index(inplace=True)

    # Convert the date column to ordinal value (int to work with)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Date'] = df['Date'].map(dt.datetime.toordinal)

    X = df['Date']
    y = df['Close']

    X = X.values.reshape(-1, 1)

    lr = LinearRegression()
    lr.fit(X, y)

    # Create a collection of 10 dates + 1 day each iteration (1 day, 2 day etc)
    prediction_dates = [dt.date.today() + dt.timedelta(days=i) for i in range(1, 11)]

    # For each date in the above collection, convert to ordinal and predict a price
    predictions = [str(lr.predict([[date.toordinal()]])[0]) for date in prediction_dates]

    last_close_price = df['Close'][0]

    return render_template('AAPL.html', predictions=predictions, last_close_price=Get_close_price('AAPL'))


@app.route('/BTC', methods=['GET', 'POST'])
def BTC():
    return render_template('BTC.html', predictions=Poly_reg('BTC-USD', 3), last_close_price=Get_close_price('BTC-USD'))


if __name__ == "__main__":
    app.run(debug=False, host=0.0.0.0)
