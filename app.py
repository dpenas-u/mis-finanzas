from __future__ import annotations

import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from finanzas_dashboard import (
    DEFAULT_SHEET_URL,
    LABEL_MODES,
    PROJECTION_SCENARIOS,
    account_monthly_summary,
    balance_over_time,
    build_projection_history,
    combine_accounts,
    filter_period,
    label_monthly_comparison,
    label_monthly_evolution,
    label_summary,
    label_trend_summary,
    load_accounts_from_sheet,
    month_label_breakdown,
    monthly_balance_over_time,
    monthly_summary,
    projection_from_component_history,
)

PALETTE = [
    "#0f766e",
    "#d97706",
    "#0f172a",
    "#84cc16",
    "#dc2626",
    "#2563eb",
    "#a16207",
    "#0891b2",
]


@st.cache_data(ttl=300, show_spinner=False)
def load_dashboard_data(sheet_ref: str):
    return load_accounts_from_sheet(sheet_ref)


def format_eur(value: float) -> str:
    if pd.isna(value):
        return "n/d"
    formatted = f"{value:,.2f} €"
    return formatted.replace(",", "X").replace(".", ",").replace("X", ".")


def format_pct(value: float) -> str:
    if pd.isna(value):
        return "n/d"
    return f"{value * 100:.1f}%"


def period_preset_dates(
    preset: str,
    min_date,
    max_date,
) -> tuple:
    max_ts = pd.Timestamp(max_date)
    if preset == "Últimos 3 meses":
        start_ts = max_ts - pd.DateOffset(months=3) + pd.DateOffset(days=1)
    elif preset == "Últimos 6 meses":
        start_ts = max_ts - pd.DateOffset(months=6) + pd.DateOffset(days=1)
    elif preset == "Este año":
        start_ts = pd.Timestamp(year=max_ts.year, month=1, day=1)
    else:
        start_ts = pd.Timestamp(min_date)

    start_date = max(pd.Timestamp(min_date), start_ts).date()
    return start_date, pd.Timestamp(max_date).date()


def default_sheet_ref() -> str:
    env_value = os.getenv("MIS_FINANZAS_SHEET_URL")
    if env_value:
        return env_value

    try:
        secret_value = st.secrets.get("MIS_FINANZAS_SHEET_URL")
    except Exception:
        secret_value = None
    return secret_value or DEFAULT_SHEET_URL


def previous_period_metrics(
    accounts_df: pd.DataFrame,
    start_date,
    end_date,
) -> dict:
    duration_days = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days
    prev_end = pd.Timestamp(start_date) - pd.Timedelta(days=1)
    prev_start = prev_end - pd.Timedelta(days=duration_days)
    prev_df = filter_period(accounts_df, prev_start, prev_end)
    if prev_df.empty:
        return {}
    prev_income = float(prev_df["income_amount"].sum())
    prev_expense = float(prev_df["expense_abs"].sum())
    prev_net = float(prev_df["quantity"].sum())
    return {
        "income": prev_income,
        "expense": prev_expense,
        "net": prev_net,
        "savings_rate": prev_net / prev_income if prev_income > 0 else None,
    }


def delta_eur(current: float, previous: float | None) -> str | None:
    if previous is None:
        return None
    diff = current - previous
    sign = "+" if diff >= 0 else ""
    return f"{sign}{format_eur(diff)}"


def delta_pct_points(current: float | None, previous: float | None) -> str | None:
    if current is None or previous is None or pd.isna(current) or pd.isna(previous):
        return None
    diff = current - previous
    sign = "+" if diff >= 0 else ""
    return f"{sign}{diff * 100:.1f} pp"


def compact_text_columns(columns: list[str]) -> dict[str, st.column_config.TextColumn]:
    return {
        column: st.column_config.TextColumn(
            column,
            width="medium",
            max_chars=72,
        )
        for column in columns
    }


def apply_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(15,118,110,0.10), transparent 32%),
                radial-gradient(circle at top right, rgba(217,119,6,0.10), transparent 28%),
                linear-gradient(180deg, #f5f2e8 0%, #f8fafc 48%, #eef4f3 100%);
            color: #15202b;
            font-family: "Avenir Next", "Segoe UI", sans-serif;
        }
        [data-testid="stSidebar"] {
            background: rgba(255, 255, 255, 0.84);
            border-right: 1px solid rgba(15, 23, 42, 0.08);
        }
        [data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.82);
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 14px;
            padding: 10px 12px;
            box-shadow: 0 14px 40px rgba(15, 23, 42, 0.06);
        }
        [data-testid="stMetricLabel"] p {
            font-size: 0.78rem;
            line-height: 1.15;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.15rem;
            line-height: 1.15;
        }
        .hero {
            padding: 1.1rem 1.35rem;
            border-radius: 20px;
            background:
                linear-gradient(135deg, rgba(15,118,110,0.95), rgba(15,23,42,0.95)),
                #0f172a;
            color: #f8fafc;
            box-shadow: 0 20px 45px rgba(15, 23, 42, 0.18);
            margin-bottom: 1rem;
        }
        .hero-kicker {
            letter-spacing: 0.12em;
            text-transform: uppercase;
            font-size: 0.66rem;
            opacity: 0.76;
            margin-bottom: 0.35rem;
        }
        .hero h1 {
            font-family: "Iowan Old Style", "Palatino Linotype", serif;
            font-size: clamp(1.55rem, 4vw, 2.05rem);
            line-height: 1.05;
            margin: 0 0 0.35rem 0;
        }
        .hero p {
            margin: 0;
            max-width: 70ch;
            color: rgba(248, 250, 252, 0.86);
            font-size: 0.92rem;
            line-height: 1.35;
        }
        .section-caption {
            color: #475569;
            margin-top: -0.1rem;
            margin-bottom: 0.9rem;
            font-size: 0.9rem;
        }
        .pill-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin-top: 0.95rem;
        }
        .pill {
            padding: 0.38rem 0.7rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.14);
            border: 1px solid rgba(255,255,255,0.18);
            font-size: 0.82rem;
        }
        [data-testid="stDataFrame"] {
            font-size: 0.86rem;
        }
        [data-testid="stDataFrame"] div {
            line-height: 1.25;
        }
        @media (max-width: 700px) {
            .hero {
                padding: 0.95rem 1rem;
            }
            .pill {
                font-size: 0.76rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def plot_monthly_net(monthly_df: pd.DataFrame) -> go.Figure:
    chart_df = monthly_df.copy()
    chart_df["sentido"] = np.where(chart_df["net"] >= 0, "Neto positivo", "Neto negativo")
    fig = px.bar(
        chart_df,
        x="month_start",
        y="net",
        color="sentido",
        color_discrete_map={
            "Neto positivo": "#0f766e",
            "Neto negativo": "#dc2626",
        },
    )
    fig.update_traces(hovertemplate="%{x|%b %Y}<br><b>%{y:,.2f} €</b><extra></extra>")
    fig.update_layout(
        title="Neto mensual",
        xaxis_title="",
        yaxis_title="Euros",
        legend_title="",
        separators=",.",
        margin=dict(l=10, r=10, t=52, b=10),
    )
    return fig


def plot_income_vs_expense(monthly_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(
        x=monthly_df["month_start"],
        y=monthly_df["income"],
        name="Ingresos",
        marker_color="#0f766e",
        hovertemplate="%{x|%b %Y}<br><b>Ingresos: %{y:,.2f} €</b><extra></extra>",
    )
    fig.add_bar(
        x=monthly_df["month_start"],
        y=monthly_df["expense"],
        name="Gastos",
        marker_color="#d97706",
        hovertemplate="%{x|%b %Y}<br><b>Gastos: %{y:,.2f} €</b><extra></extra>",
    )
    fig.update_layout(
        title="Ingresos vs gastos",
        xaxis_title="",
        yaxis_title="Euros",
        barmode="group",
        separators=",.",
        margin=dict(l=10, r=10, t=52, b=10),
    )
    return fig


def plot_daily_balance(balance_df: pd.DataFrame) -> go.Figure:
    fig = px.line(
        balance_df,
        x="date",
        y="balance",
        markers=True,
        color_discrete_sequence=["#0f172a"],
    )
    fig.update_traces(hovertemplate="%{x|%d %b %Y}<br><b>Saldo: %{y:,.2f} €</b><extra></extra>")
    fig.update_layout(
        title="Evolución del saldo acumulado",
        xaxis_title="",
        yaxis_title="Saldo (€)",
        showlegend=False,
        separators=",.",
        margin=dict(l=10, r=10, t=52, b=10),
    )
    return fig


def plot_monthly_balance(balance_df: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        balance_df,
        x="month_start",
        y="balance",
        color_discrete_sequence=["#2563eb"],
    )
    fig.update_traces(hovertemplate="%{x|%b %Y}<br><b>Patrimonio: %{y:,.2f} €</b><extra></extra>")
    fig.update_layout(
        title="Patrimonio acumulado por mes",
        xaxis_title="",
        yaxis_title="Saldo (€)",
        showlegend=False,
        separators=",.",
        margin=dict(l=10, r=10, t=52, b=10),
    )
    return fig


def plot_savings_rate(monthly_df: pd.DataFrame) -> go.Figure:
    chart_df = monthly_df.copy()
    chart_df["savings_rate_pct"] = chart_df["savings_rate"] * 100
    fig = px.line(
        chart_df,
        x="month_start",
        y="savings_rate_pct",
        markers=True,
        color_discrete_sequence=["#0f766e"],
    )
    fig.update_traces(hovertemplate="%{x|%b %Y}<br><b>Ahorro: %{y:.1f}%</b><extra></extra>")
    fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="#64748b")
    fig.update_layout(
        title="Tasa de ahorro mensual",
        xaxis_title="",
        yaxis_title="% sobre ingresos",
        showlegend=False,
        separators=",.",
        margin=dict(l=10, r=10, t=52, b=10),
    )
    return fig


def plot_account_monthly_net(account_monthly_df: pd.DataFrame) -> go.Figure:
    fig = px.line(
        account_monthly_df,
        x="month_start",
        y="net",
        color="account",
        markers=True,
        color_discrete_sequence=PALETTE,
    )
    fig.update_traces(hovertemplate="%{x|%b %Y}<br><b>Neto: %{y:,.2f} €</b><extra></extra>")
    fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="#64748b")
    fig.update_layout(
        title="Neto mensual por cuenta",
        xaxis_title="",
        yaxis_title="Euros",
        legend_title="Cuenta",
        separators=",.",
        margin=dict(l=10, r=10, t=52, b=10),
    )
    return fig


def plot_expense_pie(pie_df: pd.DataFrame, label_column: str) -> go.Figure:
    fig = px.pie(
        pie_df,
        values="expense",
        names=label_column,
        hole=0.45,
        color_discrete_sequence=PALETTE,
    )
    fig.update_layout(
        title="Distribución de gastos por categoría",
        margin=dict(l=10, r=10, t=52, b=10),
    )
    return fig


def plot_label_totals(summary_df: pd.DataFrame, label_column: str) -> go.Figure:
    top_df = summary_df.head(12).sort_values("gasto_total", ascending=True)
    fig = px.bar(
        top_df,
        x="gasto_total",
        y=label_column,
        orientation="h",
        color_discrete_sequence=["#d97706"],
    )
    fig.update_traces(hovertemplate="<b>%{y}</b><br>%{x:,.2f} €<extra></extra>")
    fig.update_layout(
        title="Dónde se va el dinero",
        xaxis_title="Gasto acumulado (€)",
        yaxis_title="",
        showlegend=False,
        separators=",.",
        margin=dict(l=10, r=10, t=52, b=10),
    )
    return fig


def plot_label_evolution(evolution_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(
        x=evolution_df["month_start"],
        y=evolution_df["expense"],
        name="Gasto",
        marker_color="#d97706",
        hovertemplate="%{x|%b %Y}<br><b>Gasto: %{y:,.2f} €</b><extra></extra>",
    )
    fig.add_scatter(
        x=evolution_df["month_start"],
        y=evolution_df["net"],
        mode="lines+markers",
        name="Neto",
        line=dict(color="#0f766e", width=3),
        hovertemplate="%{x|%b %Y}<br><b>Neto: %{y:,.2f} €</b><extra></extra>",
    )
    fig.update_layout(
        title="Evolución mensual de la categoría",
        xaxis_title="",
        yaxis_title="Euros",
        separators=",.",
        margin=dict(l=10, r=10, t=52, b=10),
    )
    return fig


def plot_label_comparison(
    comparison_df: pd.DataFrame,
    label_column: str,
    metric_column: str,
    metric_label: str,
) -> go.Figure:
    fig = px.line(
        comparison_df,
        x="month_start",
        y=metric_column,
        color=label_column,
        markers=True,
        color_discrete_sequence=PALETTE,
    )
    fig.update_traces(hovertemplate="%{x|%b %Y}<br><b>%{y:,.2f}</b><extra></extra>")
    if metric_column in {"net", "cambio_vs_anterior"}:
        fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="#64748b")
    fig.update_layout(
        title=f"Evolución mensual por categoría: {metric_label.lower()}",
        xaxis_title="",
        yaxis_title=metric_label,
        legend_title="Categoría",
        separators=",.",
        margin=dict(l=10, r=10, t=52, b=10),
    )
    return fig


def plot_projection_balance(
    historical_balance: pd.DataFrame,
    projection_df: pd.DataFrame,
    scenario_band: pd.DataFrame | None = None,
    scenario_name: str = "Base",
) -> go.Figure:
    fig = go.Figure()
    if not historical_balance.empty:
        fig.add_scatter(
            x=historical_balance["month_start"],
            y=historical_balance["balance"],
            mode="lines+markers",
            name="Balance real",
            line=dict(color="#0f172a", width=3),
        )
    if scenario_band is not None and not scenario_band.empty:
        fig.add_scatter(
            x=scenario_band["month_start"],
            y=scenario_band["balance_high"],
            mode="lines",
            line=dict(width=0),
            hoverinfo="skip",
            showlegend=False,
        )
        fig.add_scatter(
            x=scenario_band["month_start"],
            y=scenario_band["balance_low"],
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(15, 118, 110, 0.14)",
            line=dict(width=0),
            hoverinfo="skip",
            name="Rango entre escenarios",
        )
    if not projection_df.empty:
        fig.add_scatter(
            x=projection_df["month_start"],
            y=projection_df["balance_projection"],
            mode="lines+markers",
            name=f"Escenario: {scenario_name}",
            line=dict(color="#0f766e", width=3, dash="dash"),
        )
    fig.update_traces(hovertemplate="%{x|%b %Y}<br><b>%{y:,.2f} €</b><extra></extra>")
    fig.update_layout(
        title="Balance histórico y proyección",
        xaxis_title="",
        yaxis_title="Saldo (€)",
        separators=",.",
        margin=dict(l=10, r=10, t=52, b=10),
    )
    return fig


def plot_projection_components(projection_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(
        x=projection_df["month_start"],
        y=projection_df["fixed_expense"],
        name="Gasto fijo",
        marker_color="#0f172a",
    )
    fig.add_bar(
        x=projection_df["month_start"],
        y=projection_df["variable_expense"],
        name="Gasto variable",
        marker_color="#d97706",
    )
    fig.add_bar(
        x=projection_df["month_start"],
        y=projection_df["occasional_expense"],
        name="Gasto puntual",
        marker_color="#f59e0b",
    )
    fig.add_scatter(
        x=projection_df["month_start"],
        y=projection_df["income"],
        name="Ingresos proyectados",
        mode="lines+markers",
        line=dict(color="#0f766e", width=3),
    )
    fig.add_scatter(
        x=projection_df["month_start"],
        y=projection_df["net"],
        name="Neto proyectado",
        mode="lines+markers",
        line=dict(color="#2563eb", width=3, dash="dot"),
    )
    fig.update_traces(hovertemplate="%{x|%b %Y}<br><b>%{y:,.2f} €</b><extra></extra>")
    fig.update_layout(
        title="Composición del flujo proyectado",
        xaxis_title="",
        yaxis_title="Euros",
        barmode="stack",
        separators=",.",
        margin=dict(l=10, r=10, t=52, b=10),
    )
    return fig


def plot_scenario_outcomes(scenario_results: dict[str, pd.DataFrame]) -> go.Figure:
    rows = []
    for name, result in scenario_results.items():
        if result.empty:
            continue
        rows.append(
            {
                "scenario": name,
                "final_balance": float(result["balance_projection"].iloc[-1]),
                "avg_net": float(result["net"].mean()),
            }
        )
    outcomes = pd.DataFrame(rows).sort_values("final_balance", ascending=True)
    fig = px.bar(
        outcomes,
        x="final_balance",
        y="scenario",
        orientation="h",
        color="avg_net",
        color_continuous_scale=["#dc2626", "#d97706", "#0f766e"],
    )
    fig.update_traces(hovertemplate="<b>%{y}</b><br>%{x:,.2f} €<extra></extra>")
    fig.update_layout(
        title="Balance final por escenario",
        xaxis_title="Balance estimado al final del horizonte (€)",
        yaxis_title="",
        separators=",.",
        margin=dict(l=10, r=10, t=52, b=10),
    )
    return fig


def main() -> None:
    st.set_page_config(
        page_title="Mis Finanzas",
        page_icon="📊",
        layout="wide",
    )
    apply_styles()

    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">Dashboard local conectado a Google Sheets</div>
            <h1>Mis Finanzas</h1>
            <p>
                Vista única para seguir saldo, gasto, labels y tendencia de ahorro por cuenta.
                La app replica el notebook y añade filtros interactivos, tabla de movimientos y
                una proyección mensual interpretable.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Origen de datos")
        sheet_ref = st.text_input(
            "Google Sheet público",
            value=default_sheet_ref(),
            help="Puedes pegar la URL completa o solo el ID del documento.",
        )
        if st.button("Actualizar datos", use_container_width=True):
            load_dashboard_data.clear()

    try:
        with st.spinner("Descargando y preparando datos..."):
            accounts, metadata = load_dashboard_data(sheet_ref)
    except Exception as exc:
        st.error(f"No se pudo cargar el Google Sheet: {exc}")
        st.stop()

    all_accounts = list(accounts.keys())
    full_df = combine_accounts(accounts, all_accounts)
    min_date = full_df["date"].min().date()
    max_date = full_df["date"].max().date()

    with st.sidebar:
        st.header("Filtros")
        selected_accounts = st.multiselect(
            "Cuentas",
            options=all_accounts,
            default=all_accounts,
        )
        period_preset = st.selectbox(
            "Periodo rápido",
            options=["Todo", "Últimos 3 meses", "Últimos 6 meses", "Este año", "Personalizado"],
            index=0,
            help="Usa Personalizado si quieres ajustar el rango manualmente.",
        )
        preset_start, preset_end = period_preset_dates(period_preset, min_date, max_date)
        if period_preset == "Personalizado":
            start_date, end_date = st.slider(
                "Periodo personalizado",
                min_value=min_date,
                max_value=max_date,
                value=(min_date, max_date),
                format="DD/MM/YYYY",
            )
        else:
            start_date, end_date = preset_start, preset_end
            st.caption(
                f"Periodo aplicado: {start_date.strftime('%d/%m/%Y')} - {end_date.strftime('%d/%m/%Y')}"
            )
        label_mode = st.radio(
            "Agrupación de categorías",
            options=list(LABEL_MODES.keys()),
            help="Normalizadas agrupa labels equivalentes ignorando mayúsculas.",
        )

    if not selected_accounts:
        st.warning("Selecciona al menos una cuenta.")
        st.stop()

    accounts_df = combine_accounts(accounts, selected_accounts)
    period_df = filter_period(accounts_df, start_date, end_date)

    if period_df.empty:
        st.warning("No hay movimientos en el periodo seleccionado.")
        st.stop()

    label_column = LABEL_MODES[label_mode]
    available_labels = sorted(period_df[label_column].dropna().unique().tolist())

    with st.sidebar:
        selected_labels = st.multiselect(
            "Filtrar categorías",
            options=available_labels,
            help="Afecta a categorías, al gráfico circular y a la tabla de movimientos.",
        )

    labels_df = period_df.copy()
    if selected_labels:
        labels_df = labels_df.loc[labels_df[label_column].isin(selected_labels)].copy()

    monthly_df = monthly_summary(period_df)
    account_monthly_df = account_monthly_summary(period_df)
    balance_df = balance_over_time(accounts_df, start_date, end_date)
    monthly_balance_df = monthly_balance_over_time(accounts_df, start_date, end_date)
    overview_label_summary = label_summary(period_df, label_column)

    current_balance = float(
        accounts_df.loc[accounts_df["date"] <= pd.Timestamp(end_date), "quantity"].sum()
    )
    period_net = float(period_df["quantity"].sum())
    period_income = float(period_df["income_amount"].sum())
    period_expense = float(period_df["expense_abs"].sum())
    savings_rate = period_net / period_income if period_income > 0 else np.nan
    avg_monthly_expense = float(monthly_df["expense"].mean()) if not monthly_df.empty else 0.0
    avg_monthly_net = float(monthly_df["net"].mean()) if not monthly_df.empty else 0.0

    per_account_balance = (
        accounts_df.loc[accounts_df["date"] <= pd.Timestamp(end_date)]
        .groupby("account", as_index=False)["quantity"]
        .sum()
        .rename(columns={"quantity": "balance"})
        .sort_values("balance", ascending=False)
    )
    best_month = (
        monthly_df.loc[monthly_df["net"].idxmax()]
        if not monthly_df.empty
        else None
    )
    highest_expense_month = (
        monthly_df.loc[monthly_df["expense"].idxmax()]
        if not monthly_df.empty
        else None
    )
    top_label_row = (
        overview_label_summary.iloc[0]
        if not overview_label_summary.empty
        else None
    )
    top_3_share = (
        float(overview_label_summary.head(3)["gasto_total"].sum()) / period_expense
        if period_expense > 0 and not overview_label_summary.empty
        else np.nan
    )

    prev = previous_period_metrics(accounts_df, start_date, end_date)

    source_text = (
        "Google Sheets en directo"
        if metadata["source"] == "google_sheets"
        else "Copia local de respaldo"
    )
    loaded_at = metadata.get("loaded_at")
    loaded_at_str = loaded_at.strftime("%H:%M") if loaded_at else "?"
    st.markdown(
        f"""
        <div class="pill-row">
            <div class="pill">Fuente: {source_text}</div>
            <div class="pill">Actualizado: {loaded_at_str}</div>
            <div class="pill">Cuentas: {len(all_accounts)}</div>
            <div class="pill">Periodo disponible: {min_date.strftime("%d/%m/%Y")} – {max_date.strftime("%d/%m/%Y")}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if metadata.get("warning"):
        st.warning(metadata["warning"])
    if metadata.get("ignored_sheets"):
        st.info(
            "Se han ignorado hojas no transaccionales: "
            + ", ".join(metadata["ignored_sheets"])
        )

    metric_cols = st.columns(5)
    metric_cols[0].metric("Balance a fecha fin", format_eur(current_balance))
    metric_cols[1].metric(
        "Neto del periodo",
        format_eur(period_net),
        delta=delta_eur(period_net, prev.get("net")),
        delta_color="normal",
        help="Comparado con el periodo anterior de igual duración.",
    )
    metric_cols[2].metric(
        "Ingresos del periodo",
        format_eur(period_income),
        delta=delta_eur(period_income, prev.get("income")),
        delta_color="normal",
        help="Comparado con el periodo anterior de igual duración.",
    )
    metric_cols[3].metric(
        "Gastos del periodo",
        format_eur(period_expense),
        delta=delta_eur(period_expense, prev.get("expense")),
        delta_color="inverse",
        help="Verde si gastas menos que el periodo anterior.",
    )
    metric_cols[4].metric(
        "Tasa de ahorro",
        format_pct(savings_rate),
        delta=delta_pct_points(savings_rate, prev.get("savings_rate")),
        delta_color="normal",
        help="Comparado con el periodo anterior de igual duración.",
    )

    secondary_cols = st.columns(3)
    secondary_cols[0].metric("Gasto medio mensual", format_eur(avg_monthly_expense))
    secondary_cols[1].metric("Neto medio mensual", format_eur(avg_monthly_net))
    secondary_cols[2].dataframe(
        per_account_balance.assign(balance=per_account_balance["balance"].map(format_eur)),
        hide_index=True,
        use_container_width=True,
        height=180,
        column_config=compact_text_columns(["account"]),
    )

    tab_summary, tab_labels, tab_projection, tab_movements = st.tabs(
        ["Resumen", "Categorías", "Proyección", "Movimientos"]
    )

    with tab_summary:
        st.markdown(
            '<p class="section-caption">Esta vista replica el notebook con una capa interactiva: neto mensual, distribución por label y evolución del saldo.</p>',
            unsafe_allow_html=True,
        )
        pulse_cols = st.columns(4)
        pulse_cols[0].metric(
            "Mejor mes neto",
            (
                f"{best_month['month_start'].strftime('%Y-%m')} · {format_eur(float(best_month['net']))}"
                if best_month is not None
                else "n/d"
            ),
        )
        pulse_cols[1].metric(
            "Mes con más gasto",
            (
                f"{highest_expense_month['month_start'].strftime('%Y-%m')} · {format_eur(float(highest_expense_month['expense']))}"
                if highest_expense_month is not None
                else "n/d"
            ),
        )
        pulse_cols[2].metric(
            "Label que más pesa",
            (
                f"{top_label_row[label_column]} · {format_eur(float(top_label_row['gasto_total']))}"
                if top_label_row is not None
                else "n/d"
            ),
        )
        pulse_cols[3].metric(
            "Concentración top 3",
            format_pct(top_3_share),
        )

        row_1 = st.columns(2)
        row_1[0].plotly_chart(plot_monthly_net(monthly_df), use_container_width=True)
        row_1[1].plotly_chart(plot_income_vs_expense(monthly_df), use_container_width=True)

        row_2 = st.columns(2)
        row_2[0].plotly_chart(plot_daily_balance(balance_df), use_container_width=True)
        row_2[1].plotly_chart(plot_monthly_balance(monthly_balance_df), use_container_width=True)

        row_3 = st.columns(2)
        row_3[0].plotly_chart(plot_savings_rate(monthly_df), use_container_width=True)
        row_3[1].plotly_chart(
            plot_account_monthly_net(account_monthly_df),
            use_container_width=True,
        )

        month_options = sorted(labels_df["month_key"].dropna().unique().tolist(), reverse=True)
        if month_options:
            selected_month = st.selectbox(
                "Mes para ver la distribución de gastos por label",
                options=month_options,
                index=0,
            )
            pie_df = month_label_breakdown(labels_df, label_column, selected_month)
            if pie_df.empty:
                st.info("No hay gastos con labels en ese mes para el filtro actual.")
            else:
                st.plotly_chart(
                    plot_expense_pie(pie_df, label_column),
                    use_container_width=True,
                )
        else:
            st.info("No hay labels disponibles para mostrar la distribución mensual.")

    with tab_labels:
        st.markdown(
            '<p class="section-caption">Analítica por categorías: peso económico, comparación temporal y señales de cambio reciente.</p>',
            unsafe_allow_html=True,
        )
        label_summary_df = label_summary(labels_df, label_column)
        if label_summary_df.empty:
            st.info("No hay datos para las categorías seleccionadas.")
        else:
            trend_df = label_trend_summary(labels_df, label_column)
            label_row = st.columns([1.1, 1.9])
            label_row[0].plotly_chart(
                plot_label_totals(label_summary_df, label_column),
                use_container_width=True,
            )
            selected_label = label_row[1].selectbox(
                "Categoría a analizar",
                options=label_summary_df[label_column].tolist(),
                index=0,
            )
            evolution_df = label_monthly_evolution(labels_df, label_column, selected_label)
            label_row[1].plotly_chart(
                plot_label_evolution(evolution_df),
                use_container_width=True,
            )

            st.markdown("**Comparar evolución en el tiempo**")
            comparison_controls = st.columns([2.2, 1])
            default_compare_labels = label_summary_df[label_column].head(5).tolist()
            compare_labels = comparison_controls[0].multiselect(
                "Categorías a comparar",
                options=label_summary_df[label_column].tolist(),
                default=default_compare_labels,
                help="Elige varias categorías para ver cómo cambian mes a mes.",
            )
            metric_options = {
                "Gasto": ("expense", "Euros"),
                "Ingresos": ("income", "Euros"),
                "Neto": ("net", "Euros"),
                "Movimientos": ("movements", "Movimientos"),
            }
            metric_name = comparison_controls[1].selectbox(
                "Métrica",
                options=list(metric_options.keys()),
                index=0,
            )
            metric_column, metric_axis_label = metric_options[metric_name]
            comparison_df = label_monthly_comparison(
                labels_df,
                label_column,
                compare_labels,
            )
            if comparison_df.empty:
                st.info("Selecciona al menos una categoría con datos para comparar su evolución.")
            else:
                st.plotly_chart(
                    plot_label_comparison(
                        comparison_df,
                        label_column,
                        metric_column,
                        metric_axis_label,
                    ),
                    use_container_width=True,
                )

            st.markdown("**Tendencias recientes por categoría**")
            if trend_df.empty:
                st.info("No hay gastos suficientes para calcular tendencias por categoría.")
            else:
                trend_table = trend_df.assign(
                    gasto_total=trend_df["gasto_total"].map(format_eur),
                    ultimo_mes=trend_df["ultimo_mes"].map(format_eur),
                    media_reciente=trend_df["media_reciente"].map(format_eur),
                    media_anterior=trend_df["media_anterior"].map(format_eur),
                    cambio_vs_anterior=trend_df["cambio_vs_anterior"].map(format_pct),
                )
                st.dataframe(
                    trend_table,
                    hide_index=True,
                    use_container_width=True,
                    height=300,
                    column_config={
                        **compact_text_columns([label_column]),
                        "gasto_total": st.column_config.TextColumn("Gasto total"),
                        "ultimo_mes": st.column_config.TextColumn("Último mes"),
                        "media_reciente": st.column_config.TextColumn("Media reciente"),
                        "media_anterior": st.column_config.TextColumn("Media anterior"),
                        "cambio_vs_anterior": st.column_config.TextColumn("Cambio"),
                        "movimientos": st.column_config.NumberColumn("Movimientos"),
                    },
                )

            st.markdown("**Resumen completo de categorías**")
            st.dataframe(
                label_summary_df.assign(
                    gasto_total=label_summary_df["gasto_total"].map(format_eur),
                    ingreso_total=label_summary_df["ingreso_total"].map(format_eur),
                    neto=label_summary_df["neto"].map(format_eur),
                    ticket_medio=label_summary_df["ticket_medio"].map(format_eur),
                ),
                hide_index=True,
                use_container_width=True,
                height=360,
                column_config=compact_text_columns([label_column]),
            )

    with tab_projection:
        st.markdown(
            '<p class="section-caption">La proyección separa gasto fijo recurrente, gasto variable y movimientos puntuales. Compara el escenario base con otros más conservadores o permisivos.</p>',
            unsafe_allow_html=True,
        )
        with st.expander("¿Cómo funciona la proyección?"):
            st.markdown(
                """
                Cada categoría de gasto e ingreso se clasifica según su regularidad en el histórico seleccionado:

                | Tipo | Criterio | Proyección |
                |---|---|---|
                | **Gasto fijo recurrente** | ≥ 75 % de los meses, poca variación | Media; los escenarios no lo tocan |
                | **Gasto variable recurrente** | Frecuente pero importe variable | Sujeto al multiplicador del escenario |
                | **Gasto puntual** | < 50 % de los meses | Los escenarios lo modulan más |
                | **Ingreso recurrente** | ≥ 60 % de los meses | Tendencia lineal o media |
                | **Ingreso variable** | Entre 25 % y 60 % de los meses | Media simple |

                Los **escenarios** aplican multiplicadores distintos a los componentes variables para modelar distintos comportamientos.
                El gasto fijo se replica siempre igual independientemente del escenario.
                """
            )
        if len(monthly_df) < 2:
            st.info("Hace falta al menos dos meses con datos para proyectar.")
        else:
            projection_controls = st.columns(3)
            max_lookback = len(monthly_df)
            lookback_months = projection_controls[0].slider(
                "Meses históricos para la proyección",
                min_value=2,
                max_value=max_lookback,
                value=min(6, max_lookback),
            )
            horizon_months = projection_controls[1].slider(
                "Meses a proyectar",
                min_value=1,
                max_value=18,
                value=6,
            )
            method_label = projection_controls[2].radio(
                "Método",
                options=["Media reciente", "Tendencia lineal"],
                horizontal=False,
            )
            method_key = "linear" if method_label == "Tendencia lineal" else "average"
            projection_inputs = build_projection_history(
                period_df,
                label_column=label_column,
                lookback_months=lookback_months,
            )
            history_components = projection_inputs["history"]
            expense_profile = projection_inputs["expense_profile"]
            income_profile = projection_inputs["income_profile"]
            scenario_name = st.selectbox(
                "Escenario de comportamiento",
                options=list(PROJECTION_SCENARIOS.keys()),
                index=1,
                help="Afecta solo a los componentes variables de la proyección.",
            )
            st.caption(PROJECTION_SCENARIOS[scenario_name]["description"])

            projection_df = projection_from_component_history(
                history_df=history_components,
                current_balance=current_balance,
                horizon_months=horizon_months,
                method=method_key,
                scenario=PROJECTION_SCENARIOS[scenario_name],
            )
            scenario_results = {
                name: projection_from_component_history(
                    history_df=history_components,
                    current_balance=current_balance,
                    horizon_months=horizon_months,
                    method=method_key,
                    scenario=scenario,
                )
                for name, scenario in PROJECTION_SCENARIOS.items()
            }
            scenario_band = pd.DataFrame({"month_start": projection_df["month_start"]})
            balance_matrix = pd.DataFrame(
                {
                    name: result["balance_projection"].to_numpy()
                    for name, result in scenario_results.items()
                }
            )
            scenario_band["balance_low"] = balance_matrix.min(axis=1)
            scenario_band["balance_high"] = balance_matrix.max(axis=1)

            historical_start = history_components["month_start"].min()
            historical_balance = monthly_balance_over_time(
                accounts_df,
                historical_start,
                end_date,
            )

            projection_row = st.columns(2)
            projection_row[0].plotly_chart(
                plot_projection_balance(
                    historical_balance,
                    projection_df,
                    scenario_band=scenario_band,
                    scenario_name=scenario_name,
                ),
                use_container_width=True,
            )
            projection_row[1].plotly_chart(
                plot_projection_components(projection_df),
                use_container_width=True,
            )
            st.plotly_chart(
                plot_scenario_outcomes(scenario_results),
                use_container_width=True,
            )

            projected_final_balance = float(projection_df["balance_projection"].iloc[-1])
            projected_avg_net = float(projection_df["net"].mean())
            projected_avg_expense = float(projection_df["expense"].mean())
            projected_fixed_expense = float(projection_df["fixed_expense"].mean())

            projection_metrics = st.columns(4)
            projection_metrics[0].metric(
                "Balance proyectado al final",
                format_eur(projected_final_balance),
            )
            projection_metrics[1].metric(
                "Neto medio proyectado",
                format_eur(projected_avg_net),
            )
            projection_metrics[2].metric(
                "Gasto medio proyectado",
                format_eur(projected_avg_expense),
            )
            projection_metrics[3].metric(
                "Base fija mensual",
                format_eur(projected_fixed_expense),
            )

            profile_cols = st.columns(2)
            profile_cols[0].markdown("**Gastos detectados en el histórico usado**")
            profile_cols[0].dataframe(
                expense_profile.assign(
                    coverage_ratio=expense_profile["coverage_ratio"].map(lambda value: format_pct(value)),
                    avg_monthly=expense_profile["avg_monthly"].map(format_eur),
                    avg_active_month=expense_profile["avg_active_month"].map(format_eur),
                    std_active_month=expense_profile["std_active_month"].map(format_eur),
                    last_month_amount=expense_profile["last_month_amount"].map(format_eur),
                ),
                hide_index=True,
                use_container_width=True,
                height=340,
                column_config=compact_text_columns([label_column, "classification"]),
            )
            profile_cols[1].markdown("**Ingresos detectados en el histórico usado**")
            profile_cols[1].dataframe(
                income_profile.assign(
                    coverage_ratio=income_profile["coverage_ratio"].map(lambda value: format_pct(value)),
                    avg_monthly=income_profile["avg_monthly"].map(format_eur),
                    avg_active_month=income_profile["avg_active_month"].map(format_eur),
                    std_active_month=income_profile["std_active_month"].map(format_eur),
                    last_month_amount=income_profile["last_month_amount"].map(format_eur),
                ),
                hide_index=True,
                use_container_width=True,
                height=340,
                column_config=compact_text_columns([label_column, "classification"]),
            )

            projection_table = projection_df.copy()
            projection_table["month_start"] = projection_table["month_start"].dt.strftime("%Y-%m")
            for column in [
                "fixed_expense",
                "variable_expense",
                "occasional_expense",
                "recurring_income",
                "variable_income",
                "income",
                "expense",
                "net",
                "balance_projection",
            ]:
                projection_table[column] = projection_table[column].map(format_eur)
            st.dataframe(projection_table, hide_index=True, use_container_width=True)

    with tab_movements:
        st.markdown(
            '<p class="section-caption">Tabla de movimientos del periodo. Usa el buscador para filtrar por categoría, cuenta o comentario. Si tienes filtro de categorías activo en la barra lateral, también se aplica aquí.</p>',
            unsafe_allow_html=True,
        )
        movements_df = labels_df[
            ["date", "account", "quantity", label_column, "comment", "direction"]
        ].rename(columns={label_column: "label"})
        movements_df = movements_df.sort_values("date", ascending=False).reset_index(drop=True)

        search_col, count_col, download_col = st.columns([3, 1, 1])
        search_query = search_col.text_input(
            "Buscar",
            placeholder="Filtrar por categoría, cuenta o comentario…",
            label_visibility="collapsed",
        )
        if search_query:
            mask = (
                movements_df["label"].str.contains(search_query, case=False, na=False)
                | movements_df["comment"].str.contains(search_query, case=False, na=False)
                | movements_df["account"].str.contains(search_query, case=False, na=False)
            )
            movements_df = movements_df[mask].reset_index(drop=True)
        count_col.metric("Movimientos", len(movements_df))

        export_df = movements_df.copy()
        export_df["date"] = export_df["date"].dt.strftime("%Y-%m-%d")
        download_col.download_button(
            "Descargar CSV",
            data=export_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="movimientos_filtrados.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.dataframe(
            movements_df.assign(
                date=movements_df["date"].dt.strftime("%Y-%m-%d"),
                quantity=movements_df["quantity"].map(format_eur),
            ),
            hide_index=True,
            use_container_width=True,
            height=520,
            column_config={
                **compact_text_columns(["account", "label", "comment", "direction"]),
                "comment": st.column_config.TextColumn(
                    "comment",
                    width="large",
                    max_chars=96,
                ),
            },
        )


if __name__ == "__main__":
    main()
