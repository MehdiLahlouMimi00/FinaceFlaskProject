import re
from flask import Flask, render_template, request
import yfinance as yf
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from plotly.offline import plot
import pandas as pd
import numpy as np

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def dashboard():
    default_ticker = 'BTC-USD'
    period = '1y'
    line_color = '#0000ff'

    if request.method == 'POST':
        tickers = request.form.get('tickers', default_ticker)
        period = request.form.get('period', period)
        line_color = request.form.get('lineColor', line_color)
    else:
        tickers = request.args.get('tickers', default_ticker)
        period = request.args.get('period', period)
        line_color = request.args.get('lineColor', line_color)

    # Validate and filter tickers for cryptocurrency format
    tickers_list = tickers.split(',')
    valid_tickers = [ticker for ticker in tickers_list if re.match(r'^[A-Z]{1,5}-USD$', ticker)]
    if not valid_tickers:
        return render_template('error.html', error="No valid cryptocurrency tickers provided.")

    try:
        data = yf.download(valid_tickers, period=period, interval="1d")
        if data.empty:
            return render_template('error.html', error="No data received for the provided tickers.")
    except Exception as e:
        return render_template('error.html', error=str(e))

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.7, 0.3],
        subplot_titles=('Historical Price', 'Buy Indicator'),
        specs=[[{"type": "xy"}, {"type": "domain"}]]
    )

    is_multi_index = isinstance(data.columns, pd.MultiIndex)
    reference_value = 100000

    plot_config = {
        'modeBarButtonsToAdd': [
            'drawline',
            'drawopenpath',
            'drawclosedpath',
            'drawcircle',
            'drawrect',
            'eraseshape'
        ],
    }

    for ticker in valid_tickers:
        try:
            if is_multi_index:
                close_prices = data[(ticker, 'Close')].dropna()
            else:
                close_prices = data['Close'].dropna()

            # Ichimoku Cloud Calculation
            high_prices = data[(ticker, 'High')].dropna() if is_multi_index else data['High'].dropna()
            low_prices = data[(ticker, 'Low')].dropna() if is_multi_index else data['Low'].dropna()

            nine_period_high = high_prices.rolling(window=9).max()
            nine_period_low = low_prices.rolling(window=9).min()
            tenkan_sen = (nine_period_high + nine_period_low) / 2

            twenty_six_period_high = high_prices.rolling(window=26).max()
            twenty_six_period_low = low_prices.rolling(window=26).min()
            kijun_sen = (twenty_six_period_high + twenty_six_period_low) / 2

            senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
            senkou_span_b = ((high_prices.rolling(window=52).max() + low_prices.rolling(window=52).min()) / 2).shift(26)

            chikou_span = close_prices.shift(-26)

            # Add Ichimoku Cloud traces
            fig.add_trace(
                go.Scatter(x=close_prices.index, y=tenkan_sen, line=dict(color='red', width=1.5), name='Tenkan Sen'),
                row=1, col=1)
            fig.add_trace(
                go.Scatter(x=close_prices.index, y=kijun_sen, line=dict(color='blue', width=1.5), name='Kijun Sen'),
                row=1, col=1)
            fig.add_trace(
                go.Scatter(x=close_prices.index[25:], y=senkou_span_a[25:], line=dict(color='green', width=1.5),
                           name='Senkou Span A'), row=1, col=1)
            fig.add_trace(go.Scatter(x=close_prices.index[25:], y=senkou_span_b[25:], fill='tonexty',
                                     line=dict(color='orange', width=1.5), name='Senkou Span B'), row=1, col=1)
            fig.add_trace(go.Scatter(x=close_prices.index, y=chikou_span, line=dict(color='purple', width=1.5),
                                     name='Chikou Span'), row=1, col=1)

            # MACD Calculation
            exp1 = close_prices.ewm(span=12, adjust=False).mean()
            exp2 = close_prices.ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()

            # Adding MACD to the chart
            fig.add_trace(
                go.Scatter(x=close_prices.index, y=macd, line=dict(color='blue', width=2), name='MACD'),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=close_prices.index, y=signal, line=dict(color='red', width=2), name='Signal'),
                row=1, col=1
            )

            current_metric_value = close_prices.iloc[-1]
            value_indicator = (current_metric_value / reference_value) * 100

            fig.add_trace(go.Scatter(
                x=close_prices.index,
                y=close_prices,
                mode='lines',
                name=ticker
            ), row=1, col=1)

            # Calculate and add the trend line
            x_days = np.arange(len(close_prices.index))
            y_values = close_prices.values
            slope, intercept = np.polyfit(x_days, y_values, 1)
            trend_line = slope * x_days + intercept

            # Add the trend line
            fig.add_trace(
                go.Scatter(
                    x=close_prices.index,
                    y=trend_line,
                    mode='lines',
                    name=f'Trend for {ticker}',
                    line=dict(color='purple', width=4, dash='dot')
                ),
                row=1, col=1
            )

            def add_drawing_tools(fig, line_color):
                fig.update_layout(
                    dragmode='drawrect',
                    newshape=dict(line=dict(color=line_color, width=2)),  # Aplica a cor escolhida
                    title_text='Interactive Cryptocurrency Financial Dashboard'
                )
                return fig

            def interpolate_color(value, min_value, max_value, start_hue, end_hue):
                # Limit the value to the defined range
                value = max(min(value, max_value), min_value)
                # Calculate the proportion of the value in the range
                ratio = (value - min_value) / (max_value - min_value)
                # Interpolate the hue between the colors green (120) and red (0)
                hue = (1 - ratio) * start_hue + ratio * end_hue
                # Return color in HSL format
                return f'hsl({hue}, 100%, 50%)'

            # Make sure the value is above 0 to start the gradient and use the function to set the color
            if value_indicator > 0:
                color = interpolate_color(value_indicator, 0.1, 100, 120, 0)
            else:
                color = 'grey'  # Valor é 0, use a cor cinza

            # Add the gauge chart with dynamic calculation
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=value_indicator,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Buy Signal Strength"},
                    gauge={'axis': {'range': [None, 100]}, 'bar': {'color': color}}
                ),
                row=1, col=2
            )

        except KeyError:
            return render_template('error.html', error=f"Data for {ticker} is not available or the ticker is invalid.")

        # Use a dynamic title based on user input
        title = 'Financial Dashboard'
        if len(tickers_list) == 1:
            title += ' for ' + tickers_list[0]
        else:
            title += ' for Multiple Assets'

        fig.update_layout(title=title)

        def highlight_significant_data_points(fig, prices, window=10):
            # Finding local maxima and minima
            local_max = prices[(prices.shift(1) < prices) & (prices.shift(-1) < prices)]
            local_min = prices[(prices.shift(1) > prices) & (prices.shift(-1) > prices)]

            # Add markers for local maxima
            fig.add_trace(
                go.Scatter(
                    x=local_max.index,
                    y=local_max,
                    mode='markers',
                    marker=dict(color='red', size=10),
                    name='Local Max'
                ),
                row=1, col=1
            )

            # Add markers for local minimums
            fig.add_trace(
                go.Scatter(
                    x=local_min.index,
                    y=local_min,
                    mode='markers',
                    marker=dict(color='green', size=10),
                    name='Local Min'
                ),
                row=1, col=1
            )

        # Apply the min and max function
        highlight_significant_data_points(fig, close_prices)

        fig = add_drawing_tools(fig, line_color)

        graph_html = plot(fig, output_type='div', config=plot_config)
        return render_template('dashboard.html', graph_html=graph_html, tickers=tickers)

if __name__ == '__main__':
    app.run(debug=False)
