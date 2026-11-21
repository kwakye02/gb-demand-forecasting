"""Dash app for the GB day-ahead demand forecast.

It reads one JSON file produced by gb_demand_forecasting.reports.export_dashboard. Set the
DASHBOARD_DATA environment variable to a file path or an https URL; with nothing set it reads
the dashboard.json next to this file.
"""

import json
import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
from dash import Dash, Input, Output, dcc, html
from plotly.subplots import make_subplots

BUNDLED_SOURCE = Path(__file__).resolve().parent / "dashboard.json"
EXPORTER_SOURCE = Path(__file__).resolve().parent.parent / "outputs" / "dashboard" / "dashboard.json"
DEFAULT_SOURCE = BUNDLED_SOURCE if BUNDLED_SOURCE.exists() else EXPORTER_SOURCE
DATA_SOURCE = os.environ.get("DASHBOARD_DATA", str(DEFAULT_SOURCE))
REQUEST_TIMEOUT_SECONDS = 20
SCHEMA_VERSION = 1
MW_PER_GW = 1000

MODEL_COLOUR = "#1565c0"
NESO_COLOUR = "#e08a1e"
ACTUAL_COLOUR = "#1f2937"
MUTED_COLOUR = "#6b7280"
BORDER_COLOUR = "#e5e7eb"
GRID_COLOUR = "#f1f2f4"

PAGE_STYLE = {
    "maxWidth": "1100px",
    "margin": "0 auto",
    "padding": "24px 16px 48px",
    "fontFamily": "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
    "color": "#111827",
}
CARD_STYLE = {
    "flex": "1 1 140px",
    "boxSizing": "border-box",
    "border": f"1px solid {BORDER_COLOUR}",
    "borderRadius": "8px",
    "padding": "12px 16px",
}
ROW_STYLE = {"display": "flex", "gap": "12px", "flexWrap": "wrap", "margin": "12px 0 8px"}
GRAPH_CONFIG = {"displayModeBar": False}


def load_dashboard_data(source=DATA_SOURCE):
    """Read the dashboard JSON from a URL or a local file."""
    if str(source).startswith(("http://", "https://")):
        response = requests.get(source, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.json()
    with open(source) as file:
        return json.load(file)


def period_label(period):
    """Clock time at the start of a settlement period, e.g. period 1 is 00:00."""
    minutes = (int(period) - 1) * 30
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def format_utc(timestamp):
    return pd.Timestamp(timestamp).strftime("%d %b %Y %H:%M UTC")


def format_date(date):
    return pd.Timestamp(date).strftime("%a %d %b %Y")


def stat_card(label, value, note=None):
    children = [
        html.Div(label, style={"fontSize": "12px", "color": MUTED_COLOUR}),
        html.Div(value, style={"fontSize": "22px", "fontWeight": "600", "margin": "2px 0"}),
    ]
    if note:
        children.append(html.Div(note, style={"fontSize": "12px", "color": MUTED_COLOUR}))
    return html.Div(children, style=CARD_STYLE)


def style_figure(figure, y_title=None, height=340):
    figure.update_layout(
        margin={"l": 60, "r": 16, "t": 16, "b": 40},
        height=height,
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.1},
    )
    if y_title:
        figure.update_layout(yaxis_title=y_title)
    figure.update_xaxes(showgrid=False, linecolor=BORDER_COLOUR)
    figure.update_yaxes(gridcolor=GRID_COLOUR, gridwidth=0.5, zeroline=False)
    return figure


def label_every_four_hours(figure, labels):
    figure.update_xaxes(tickmode="array", tickvals=labels[::8], tickangle=0)
    return figure


def forecast_figure(latest_forecast):
    periods = latest_forecast["periods"]
    labels = [period_label(p["settlement_period"]) for p in periods]
    values = [p["forecast_mw"] / MW_PER_GW for p in periods]
    figure = go.Figure()
    figure.add_scatter(
        x=labels, y=values, mode="lines", name="Our model",
        line={"color": MODEL_COLOUR, "width": 3}, hovertemplate="%{y:.1f} GW",
    )
    peak_label = period_label(latest_forecast["peak_period"])
    figure.add_scatter(
        x=[peak_label], y=[latest_forecast["peak_mw"] / MW_PER_GW], mode="markers", name="Peak",
        marker={"color": MODEL_COLOUR, "size": 10}, hoverinfo="skip", showlegend=False,
    )
    figure.update_layout(showlegend=False)
    return label_every_four_hours(style_figure(figure, "Demand (GW)"), labels)


def accuracy_figure(daily_scores):
    scores = pd.DataFrame(daily_scores)
    days = pd.to_datetime(scores["date"]).dt.strftime("%d %b")
    figure = go.Figure()
    figure.add_bar(x=days, y=scores["mae"], name="Our model", marker_color=MODEL_COLOUR)
    figure.add_bar(x=days, y=scores["neso_mae"], name="NESO", marker_color=NESO_COLOUR)
    figure.update_layout(barmode="group")
    figure.update_xaxes(type="category", tickangle=0)
    return style_figure(figure, "Mean absolute error (MW)")


def day_review_figure(day_rows):
    day = pd.DataFrame(day_rows).sort_values("settlement_period")
    labels = [period_label(p) for p in day["settlement_period"]]
    figure = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32],
                           vertical_spacing=0.06)
    figure.add_scatter(x=labels, y=day["actual_mw"] / MW_PER_GW, name="Actual", mode="lines",
                       line={"color": ACTUAL_COLOUR, "width": 4}, row=1, col=1)
    figure.add_scatter(x=labels, y=day["forecast_mw"] / MW_PER_GW, name="Our model", mode="lines",
                       line={"color": MODEL_COLOUR, "width": 2.5}, row=1, col=1)
    figure.add_scatter(x=labels, y=day["neso_mw"] / MW_PER_GW, name="NESO", mode="lines",
                       line={"color": NESO_COLOUR, "width": 2.5, "dash": "dash"}, row=1, col=1)
    figure.add_scatter(x=labels, y=day["forecast_mw"] - day["actual_mw"], name="Model error",
                       mode="lines", line={"color": MODEL_COLOUR, "width": 2},
                       showlegend=False, row=2, col=1)
    figure.add_scatter(x=labels, y=day["neso_mw"] - day["actual_mw"], name="NESO error",
                       mode="lines", line={"color": NESO_COLOUR, "width": 2, "dash": "dash"},
                       showlegend=False, row=2, col=1)
    figure.add_hline(y=0, line_color=ACTUAL_COLOUR, line_width=1, row=2, col=1)
    style_figure(figure, height=520)
    figure.update_yaxes(title_text="Demand (GW)", row=1, col=1)
    figure.update_yaxes(title_text="Error (MW)", row=2, col=1)
    return label_every_four_hours(figure, labels)


def section(title, *children):
    return html.Div([html.H2(title, style={"fontSize": "20px", "margin": "32px 0 4px"}), *children])


def forecast_section(latest):
    return section(
        f"Forecast for {format_date(latest['target_date'])}",
        html.Div(
            [
                stat_card("Forecast peak demand", f"{latest['peak_mw'] / MW_PER_GW:.1f} GW",
                          f"{period_label(latest['peak_period'])} UK"),
                stat_card("Forecast daily energy", f"{latest['energy_gwh']:,.0f} GWh"),
            ],
            style=ROW_STYLE,
        ),
        dcc.Graph(figure=forecast_figure(latest), config=GRAPH_CONFIG),
    )


def accuracy_section(daily_scores):
    if not daily_scores:
        return section("Forecast performance", html.P("No days have been evaluated yet."))
    scores = pd.DataFrame(daily_scores)
    return section(
        "Forecast performance",
        html.Div(
            [
                stat_card("Days evaluated", f"{len(scores)}"),
                stat_card("Model MAE", f"{scores['mae'].mean():,.0f} MW"),
                stat_card("NESO MAE", f"{scores['neso_mae'].mean():,.0f} MW"),
                stat_card("Model mean bias", f"{scores['bias'].mean():+,.0f} MW", "negative = under"),
                stat_card("NESO mean bias", f"{scores['neso_bias'].mean():+,.0f} MW", "negative = under"),
            ],
            style=ROW_STYLE,
        ),
        dcc.Graph(figure=accuracy_figure(daily_scores), config=GRAPH_CONFIG),
    )


def day_review_section(day_review):
    if not day_review:
        return section("Historical forecast comparison", html.P("No complete days to compare yet."))
    dates = sorted({row["date"] for row in day_review}, reverse=True)
    return section(
        "Historical forecast comparison",
        dcc.Dropdown(
            id="review-date",
            options=[{"label": format_date(d), "value": d} for d in dates],
            value=dates[0],
            clearable=False,
            style={"maxWidth": "260px"},
        ),
        dcc.Graph(id="review-graph", config=GRAPH_CONFIG),
        dcc.Store(id="review-data", data=day_review),
    )


def status_section(status):
    return section(
        "Operational status",
        html.Div(
            [
                stat_card("Forecast generated", format_utc(status["last_forecast_issued_utc"])),
                stat_card("Latest forecast date", format_date(status["last_forecast_target_date"])),
                stat_card(
                    "Latest evaluation date",
                    format_date(status["last_scored_date"]) if status["last_scored_date"] else "None yet",
                ),
                stat_card("Model version", f"{status['model_version']}",
                          f"trained to {status['model_trained_to']}"),
            ],
            style=ROW_STYLE,
        ),
    )


def footer():
    return html.P(
        "NESO forecast: newest version published by 09:00 UK time on the day before.",
        style={"fontSize": "13px", "color": MUTED_COLOUR, "marginTop": "40px"},
    )


def build_layout():
    """Build the page. Dash calls this on every page load, so the data is always current."""
    try:
        data = load_dashboard_data()
    except (OSError, ValueError, requests.RequestException) as error:
        return html.Div(
            [html.H1("GB electricity demand forecast"), html.P(f"The data could not be loaded: {error}")],
            style=PAGE_STYLE,
        )
    if data.get("schema_version") != SCHEMA_VERSION:
        return html.Div(
            [html.H1("GB electricity demand forecast"),
             html.P("The data file is in a format this dashboard does not understand.")],
            style=PAGE_STYLE,
        )

    return html.Div(
        [
            html.H1("GB electricity demand forecast", style={"fontSize": "28px", "margin": "0"}),
            forecast_section(data["latest_forecast"]),
            accuracy_section(data["daily_scores"]),
            day_review_section(data["day_review"]),
            status_section(data["pipeline_status"]),
            footer(),
        ],
        style=PAGE_STYLE,
    )


app = Dash(__name__, title="GB demand forecast")
server = app.server
app.layout = build_layout


@app.callback(Output("review-graph", "figure"), Input("review-date", "value"), Input("review-data", "data"))
def update_day_review(selected_date, day_review):
    rows = [row for row in day_review if row["date"] == selected_date]
    return day_review_figure(rows)


if __name__ == "__main__":
    app.run(debug=False, port=8050)
