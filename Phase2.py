import plotly.graph_objects as go
from plotly.subplots import make_subplots
import webbrowser
import os
from Phase1 import get_stock_data  # Capital P import

def plot_stock_dashboard(df, ticker="AAPL"):
    # Subplots structure (Row 1: Price + Moving Averages, Row 2: RSI)
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.08,
        subplot_titles=(f'{ticker} Stock Price & Moving Averages', 'Relative Strength Index (RSI)'),
        row_heights=[0.7, 0.3]
    )

    # 1. Close Price Chart
    fig.add_trace(
        go.Scatter(x=df.index, y=df['Close'], name='Close Price', line=dict(color='#00F0FF', width=2)),
        row=1, col=1
    )

    # SMA 5 Line
    fig.add_trace(
        go.Scatter(x=df.index, y=df['SMA_5'], name='SMA 5', line=dict(color='#FF9900', width=1.5)),
        row=1, col=1
    )

    # SMA 20 Line
    fig.add_trace(
        go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20', line=dict(color='#E100FF', width=1.5)),
        row=1, col=1
    )

    # 2. RSI Indicator Chart
    fig.add_trace(
        go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='#00FF66', width=1.5)),
        row=2, col=1
    )

    # Overbought (70) and Oversold (30) Reference Lines
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    # Styling (Dark Mode)
    fig.update_layout(
        template='plotly_dark',
        height=700,
        title_text=f"{ticker} Technical Analysis Visualizer",
        showlegend=True,
        xaxis2_title="Date",
        yaxis_title="Price (USD)",
        yaxis2_title="RSI Value"
    )

    # Chart ko HTML file mein save karke browser mein auto-open karna
    file_path = "stock_chart.html"
    fig.write_html(file_path)
    print(f"\nChart saved to '{file_path}'. Opening in browser...")
    webbrowser.open('file://' + os.path.realpath(file_path))

if __name__ == "__main__":
    df = get_stock_data("AAPL")
    plot_stock_dashboard(df, ticker="AAPL")