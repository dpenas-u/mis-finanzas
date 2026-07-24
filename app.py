from __future__ import annotations

import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from finanzas_dashboard import (
    DEFAULT_SHEET_URL,
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

# --- Sistema de tema -------------------------------------------------------
# Acento índigo (marca/patrimonio) + semántica universal de finanzas:
# verde = ingresos, rojo = gasto. Paleta categórica validada (CVD-safe) para
# series múltiples. Neutros con leve sesgo frío/índigo.
LIGHT_THEME = {
    "brand": "#4f46e5",
    "brand_strong": "#4338ca",
    "income": "#16a34a",
    "expense": "#dc2626",
    "neg": "#dc2626",
    "warn": "#d97706",
    "ink": "#1a1d29",
    "ink2": "#565d70",
    "muted": "#8b91a3",
    "grid": "#eef1f7",
    "axis": "#d4d8e4",
    "surface": "#ffffff",
    "band": "rgba(79, 70, 229, 0.14)",
    "cat": ["#4f46e5", "#0891b2", "#e87ba4", "#eda100", "#16a34a", "#eb6834", "#7c3aed", "#dc2626"],
    "seq": ["#dc2626", "#d97706", "#16a34a"],
}
DARK_THEME = {
    "brand": "#818cf8",
    "brand_strong": "#a5b4fc",
    "income": "#34d399",
    "expense": "#f87171",
    "neg": "#f87171",
    "warn": "#f0a94a",
    "ink": "#eceef5",
    "ink2": "#a6acbd",
    "muted": "#6f7688",
    "grid": "#242835",
    "axis": "#333949",
    "surface": "#1b1e26",
    "band": "rgba(129, 140, 248, 0.18)",
    "cat": ["#818cf8", "#22b8cf", "#e87ba4", "#eda100", "#34d399", "#f0784a", "#a78bfa", "#f87171"],
    "seq": ["#f87171", "#f0a94a", "#34d399"],
}
ACTIVE = LIGHT_THEME
FONT_STACK = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'


def resolve_theme(override: str | None = None) -> str:
    """Devuelve 'light' o 'dark'.

    `override` viene del botón de la app: 'Claro'/'Oscuro' fuerzan el tema;
    'Auto' (o None) sigue al tema del sistema/navegador vía Streamlit, que es
    el mismo que usan los widgets nativos → sin desincronización.
    """
    if override == "Claro":
        return "light"
    if override == "Oscuro":
        return "dark"
    try:
        theme_type = st.context.theme.type
    except Exception:
        theme_type = None
    if theme_type in ("light", "dark"):
        return theme_type
    try:
        base = st.get_option("theme.base")
    except Exception:
        base = None
    return "dark" if base == "dark" else "light"


def set_theme(theme: str) -> None:
    global ACTIVE
    ACTIVE = DARK_THEME if theme == "dark" else LIGHT_THEME


def base_layout(
    fig: go.Figure,
    title: str | None = None,
    *,
    y_title: str = "",
    x_title: str = "",
    legend: bool = True,
) -> go.Figure:
    """Plantilla común: fondo transparente, rejilla recesiva y tipografía system-ui."""
    t = ACTIVE
    fig.update_layout(
        font=dict(family=FONT_STACK, color=t["ink2"], size=13),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=t["cat"],
        separators=",.",
        margin=dict(l=8, r=14, t=54 if title else 14, b=46 if legend else 10),
        showlegend=legend,
        hoverlabel=dict(
            bgcolor=t["surface"],
            bordercolor=t["grid"],
            font=dict(color=t["ink"], family=FONT_STACK, size=12),
        ),
    )
    if title:
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(color=t["ink"], size=16),
                x=0.01,
                xanchor="left",
                y=0.98,
                yanchor="top",
            )
        )
    if legend:
        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.16,
                xanchor="left",
                x=0,
                title_text="",
                font=dict(size=12, color=t["ink2"]),
            )
        )
    fig.update_xaxes(
        title_text=x_title,
        showgrid=False,
        zeroline=False,
        linecolor=t["axis"],
        tickcolor=t["axis"],
        tickfont=dict(color=t["muted"], size=11),
        title_font=dict(color=t["ink2"], size=12),
    )
    fig.update_yaxes(
        title_text=y_title,
        gridcolor=t["grid"],
        zeroline=False,
        showline=False,
        tickfont=dict(color=t["muted"], size=11),
        title_font=dict(color=t["ink2"], size=12),
    )
    return fig


def kpi_card(label: str, value: str, *, delta_html: str = "") -> str:
    """Tarjeta KPI. `label` puede incluir HTML (p. ej. el icono)."""
    return (
        f'<div class="kpi"><div class="lab">{label}</div>'
        f'<div class="val tnum">{value}</div>{delta_html}</div>'
    )


def kpi_delta_eur(current: float, previous: float | None, less_is_better: bool = False) -> str:
    if previous is None:
        return ""
    diff = current - previous
    good = (diff <= 0) if less_is_better else (diff >= 0)
    arrow = "▲" if diff >= 0 else "▼"
    css = "up" if good else "down"
    return f'<div class="delta {css}">{arrow} {delta_eur(current, previous)} vs anterior</div>'


def kpi_delta_pp(current: float | None, previous: float | None) -> str:
    text = delta_pct_points(current, previous)
    if text is None:
        return ""
    diff = (current or 0.0) - (previous or 0.0)
    arrow = "▲" if diff >= 0 else "▼"
    css = "up" if diff >= 0 else "down"
    return f'<div class="delta {css}">{arrow} {text}</div>'


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


def apply_styles(theme: str) -> None:
    if theme == "dark":
        tok = {
            "page": "#12141a", "surface": "#1b1e26", "surface2": "#161922",
            "ink": "#eceef5", "ink2": "#a6acbd", "muted": "#6f7688", "hairline": "#2a2e3a",
            "brand": "#818cf8", "brand_strong": "#a5b4fc", "brand_wash": "rgba(129,140,248,.16)",
            "income": "#34d399", "expense": "#f87171",
            "income_wash": "rgba(52,211,153,.15)", "expense_wash": "rgba(248,113,113,.15)",
            "shadow": "0 1px 2px rgba(0,0,0,.4), 0 14px 32px rgba(0,0,0,.5)",
        }
    else:
        tok = {
            "page": "#f4f6fb", "surface": "#ffffff", "surface2": "#f7f9fd",
            "ink": "#1a1d29", "ink2": "#565d70", "muted": "#8b91a3", "hairline": "#e6e9f1",
            "brand": "#4f46e5", "brand_strong": "#4338ca", "brand_wash": "rgba(79,70,229,.10)",
            "income": "#16a34a", "expense": "#dc2626",
            "income_wash": "rgba(22,163,74,.10)", "expense_wash": "rgba(220,38,38,.09)",
            "shadow": "0 1px 2px rgba(20,24,40,.05), 0 10px 26px rgba(20,24,40,.06)",
        }
    st.markdown(
        f"""
        <style>
        :root {{
            --page:{tok["page"]}; --surface:{tok["surface"]}; --surface2:{tok["surface2"]};
            --ink:{tok["ink"]}; --ink2:{tok["ink2"]}; --muted:{tok["muted"]}; --hairline:{tok["hairline"]};
            --brand:{tok["brand"]}; --brand-strong:{tok["brand_strong"]}; --brand-wash:{tok["brand_wash"]};
            --income:{tok["income"]}; --expense:{tok["expense"]};
            --income-wash:{tok["income_wash"]}; --expense-wash:{tok["expense_wash"]};
            --shadow:{tok["shadow"]};
            --sans:{FONT_STACK};
        }}
        html, body, .stApp, [class*="css"] {{ font-family: var(--sans); }}
        .stApp {{ background: var(--page); color: var(--ink); }}
        .block-container {{ padding-top: 1.6rem; max-width: 1180px; }}
        [data-testid="stSidebar"] {{
            background: var(--surface);
            border-right: 1px solid var(--hairline);
        }}
        [data-testid="stSidebar"] * {{ color: var(--ink); }}
        .stButton button {{
            background: var(--brand); color: #fff; border: 0; border-radius: 10px;
            font-weight: 600;
        }}
        .stButton button:hover {{ background: var(--brand-strong); color: #fff; }}
        .stDownloadButton button {{
            border: 1px solid var(--hairline); border-radius: 10px; color: var(--ink);
        }}

        /* Tarjetas contenedoras (st.container(border=True)) */
        [data-testid="stVerticalBlockBorderWrapper"] {{
            background: var(--surface);
            border: 1px solid var(--hairline) !important;
            border-radius: 14px;
            box-shadow: var(--shadow);
        }}

        /* Cabecera de la app */
        .apphead {{
            display: flex; align-items: center; justify-content: space-between;
            gap: 16px; flex-wrap: wrap; margin-bottom: 18px;
        }}
        .brand {{ display: flex; align-items: center; gap: 11px; }}
        .brand .mk {{
            width: 36px; height: 36px; border-radius: 11px; color: #fff;
            background: linear-gradient(135deg, var(--brand), var(--brand-strong));
            display: grid; place-items: center; font-weight: 800; font-size: 1rem;
            box-shadow: 0 6px 16px var(--brand-wash);
        }}
        .brand .bt {{ font-weight: 730; font-size: 1.1rem; color: var(--ink); letter-spacing: -.01em; }}
        .brand .bs {{ font-size: .75rem; color: var(--muted); }}
        .fresh {{ font-size: .77rem; color: var(--muted); display: inline-flex; align-items: center; gap: 7px; }}
        .fresh .liv {{ width: 7px; height: 7px; border-radius: 50%; background: var(--income); }}

        /* Hero: patrimonio + cuentas */
        .netcard .k, .accts .k {{ font-size: .8rem; color: var(--ink2); font-weight: 600; margin-bottom: 6px; }}
        .netcard .big {{ font-size: 2.15rem; font-weight: 760; letter-spacing: -.02em; line-height: 1.05; color: var(--ink); }}
        .netcard .tr {{ margin-top: 8px; font-size: .82rem; font-weight: 650; }}
        .acct {{ display: flex; align-items: center; justify-content: space-between; gap: 12px;
            padding: 9px 0; border-bottom: 1px solid var(--hairline); }}
        .acct:last-child {{ border-bottom: 0; }}
        .acct .nm {{ display: inline-flex; align-items: center; gap: 9px; font-size: .92rem; color: var(--ink); }}
        .acct .sw {{ width: 9px; height: 9px; border-radius: 3px; flex: none; }}
        .acct .vl {{ font-weight: 680; font-size: .95rem; color: var(--ink); }}

        /* KPIs */
        .kpi-row {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }}
        @media (max-width: 760px) {{ .kpi-row {{ grid-template-columns: repeat(2, 1fr); }} }}
        .kpi {{
            background: var(--surface); border: 1px solid var(--hairline);
            border-radius: 14px; padding: 15px 16px; box-shadow: var(--shadow);
        }}
        .kpi .lab {{ font-size: .74rem; color: var(--ink2); font-weight: 600; margin-bottom: 7px;
            display: inline-flex; align-items: center; gap: 7px; }}
        .kpi .ic {{ width: 22px; height: 22px; border-radius: 7px; display: grid; place-items: center;
            font-size: .8rem; font-weight: 800; }}
        .kpi .ic.inc {{ background: var(--income-wash); color: var(--income); }}
        .kpi .ic.exp {{ background: var(--expense-wash); color: var(--expense); }}
        .kpi .ic.acc {{ background: var(--brand-wash); color: var(--brand); }}
        .kpi .val {{ font-size: 1.32rem; font-weight: 730; letter-spacing: -.01em; line-height: 1; color: var(--ink); }}
        .kpi .delta {{ font-size: .73rem; font-weight: 650; margin-top: 6px; }}
        .kpi .up {{ color: var(--income); }}
        .kpi .down {{ color: var(--expense); }}
        .tnum {{ font-variant-numeric: tabular-nums; }}

        /* Selector de periodo / tema (st.segmented_control) */
        [data-testid="stSegmentedControl"] button[aria-checked="true"] {{
            background: var(--brand); color: #fff; border-color: var(--brand);
        }}

        /* Tabs */
        [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 1px solid var(--hairline); }}
        [data-baseweb="tab"] {{ font-weight: 600; color: var(--ink2); }}
        [aria-selected="true"][data-baseweb="tab"] {{ color: var(--brand); }}

        .section-caption {{ color: var(--ink2); margin: -0.1rem 0 0.9rem; font-size: .9rem; }}
        [data-testid="stDataFrame"] {{ font-size: .86rem; }}
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
            "Neto positivo": ACTIVE["income"],
            "Neto negativo": ACTIVE["expense"],
        },
    )
    fig.update_traces(
        marker_line_width=0,
        hovertemplate="%{x|%b %Y}<br><b>%{y:,.2f} €</b><extra></extra>",
    )
    return base_layout(fig, "Neto mensual", y_title="Euros")


def plot_income_vs_expense(monthly_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(
        x=monthly_df["month_start"],
        y=monthly_df["income"],
        name="Ingresos",
        marker_color=ACTIVE["income"],
        hovertemplate="%{x|%b %Y}<br><b>Ingresos: %{y:,.2f} €</b><extra></extra>",
    )
    fig.add_bar(
        x=monthly_df["month_start"],
        y=monthly_df["expense"],
        name="Gastos",
        marker_color=ACTIVE["expense"],
        hovertemplate="%{x|%b %Y}<br><b>Gastos: %{y:,.2f} €</b><extra></extra>",
    )
    fig.update_layout(barmode="group")
    return base_layout(fig, "Ingresos vs gastos", y_title="Euros")


def plot_daily_balance(balance_df: pd.DataFrame) -> go.Figure:
    fig = px.line(
        balance_df,
        x="date",
        y="balance",
        color_discrete_sequence=[ACTIVE["brand"]],
    )
    fig.update_traces(
        line=dict(width=2.5),
        fill="tozeroy",
        fillcolor=ACTIVE["band"],
        hovertemplate="%{x|%d %b %Y}<br><b>Saldo: %{y:,.2f} €</b><extra></extra>",
    )
    return base_layout(fig, "Evolución del saldo acumulado", y_title="Saldo (€)", legend=False)


def plot_monthly_balance(balance_df: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        balance_df,
        x="month_start",
        y="balance",
        color_discrete_sequence=[ACTIVE["brand"]],
    )
    fig.update_traces(
        marker_line_width=0,
        hovertemplate="%{x|%b %Y}<br><b>Patrimonio: %{y:,.2f} €</b><extra></extra>",
    )
    return base_layout(fig, "Patrimonio acumulado por mes", y_title="Saldo (€)", legend=False)


def plot_savings_rate(monthly_df: pd.DataFrame) -> go.Figure:
    chart_df = monthly_df.copy()
    chart_df["savings_rate_pct"] = chart_df["savings_rate"] * 100
    fig = px.line(
        chart_df,
        x="month_start",
        y="savings_rate_pct",
        markers=True,
        color_discrete_sequence=[ACTIVE["brand"]],
    )
    fig.update_traces(
        line=dict(width=2.5),
        hovertemplate="%{x|%b %Y}<br><b>Ahorro: %{y:.1f}%</b><extra></extra>",
    )
    fig.add_hline(y=0, line_width=1, line_dash="dot", line_color=ACTIVE["muted"])
    return base_layout(fig, "Tasa de ahorro mensual", y_title="% sobre ingresos", legend=False)


def plot_account_monthly_net(account_monthly_df: pd.DataFrame) -> go.Figure:
    fig = px.line(
        account_monthly_df,
        x="month_start",
        y="net",
        color="account",
        markers=True,
        color_discrete_sequence=ACTIVE["cat"],
    )
    fig.update_traces(hovertemplate="%{x|%b %Y}<br><b>Neto: %{y:,.2f} €</b><extra></extra>")
    fig.add_hline(y=0, line_width=1, line_dash="dot", line_color=ACTIVE["muted"])
    return base_layout(fig, "Neto mensual por cuenta", y_title="Euros")


def plot_expense_pie(pie_df: pd.DataFrame, label_column: str) -> go.Figure:
    fig = px.pie(
        pie_df,
        values="expense",
        names=label_column,
        hole=0.55,
        color_discrete_sequence=ACTIVE["cat"],
    )
    fig.update_traces(
        textposition="inside",
        textinfo="percent",
        insidetextfont=dict(color="#ffffff", size=12),
        marker=dict(line=dict(color=ACTIVE["surface"], width=2)),
        hovertemplate="<b>%{label}</b><br>%{value:,.2f} € · %{percent}<extra></extra>",
    )
    fig.update_layout(
        legend=dict(font=dict(color=ACTIVE["ink2"], size=12)),
    )
    return base_layout(fig, "Distribución de gastos por categoría", legend=True)


def plot_label_totals(summary_df: pd.DataFrame, label_column: str) -> go.Figure:
    top_df = summary_df.head(12).sort_values("gasto_total", ascending=True)
    fig = px.bar(
        top_df,
        x="gasto_total",
        y=label_column,
        orientation="h",
        color_discrete_sequence=[ACTIVE["brand"]],
    )
    fig.update_traces(
        marker_line_width=0,
        hovertemplate="<b>%{y}</b><br>%{x:,.2f} €<extra></extra>",
    )
    return base_layout(fig, "Dónde se va el dinero", x_title="Gasto acumulado (€)", legend=False)


def plot_label_evolution(evolution_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(
        x=evolution_df["month_start"],
        y=evolution_df["expense"],
        name="Gasto",
        marker_color=ACTIVE["expense"],
        marker_line_width=0,
        hovertemplate="%{x|%b %Y}<br><b>Gasto: %{y:,.2f} €</b><extra></extra>",
    )
    fig.add_scatter(
        x=evolution_df["month_start"],
        y=evolution_df["net"],
        mode="lines+markers",
        name="Neto",
        line=dict(color=ACTIVE["brand"], width=3),
        hovertemplate="%{x|%b %Y}<br><b>Neto: %{y:,.2f} €</b><extra></extra>",
    )
    return base_layout(fig, "Evolución mensual de la categoría", y_title="Euros")


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
        color_discrete_sequence=ACTIVE["cat"],
    )
    fig.update_traces(hovertemplate="%{x|%b %Y}<br><b>%{y:,.2f}</b><extra></extra>")
    if metric_column in {"net", "cambio_vs_anterior"}:
        fig.add_hline(y=0, line_width=1, line_dash="dot", line_color=ACTIVE["muted"])
    return base_layout(
        fig,
        f"Evolución mensual por categoría: {metric_label.lower()}",
        y_title=metric_label,
    )


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
            line=dict(color=ACTIVE["ink"], width=3),
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
            fillcolor=ACTIVE["band"],
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
            line=dict(color=ACTIVE["brand"], width=3, dash="dash"),
        )
    fig.update_traces(hovertemplate="%{x|%b %Y}<br><b>%{y:,.2f} €</b><extra></extra>")
    return base_layout(fig, "Balance histórico y proyección", y_title="Saldo (€)")


def plot_projection_components(projection_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(
        x=projection_df["month_start"],
        y=projection_df["fixed_expense"],
        name="Gasto fijo",
        marker_color=ACTIVE["ink2"],
        marker_line=dict(color=ACTIVE["surface"], width=1),
    )
    fig.add_bar(
        x=projection_df["month_start"],
        y=projection_df["variable_expense"],
        name="Gasto variable",
        marker_color=ACTIVE["warn"],
        marker_line=dict(color=ACTIVE["surface"], width=1),
    )
    fig.add_bar(
        x=projection_df["month_start"],
        y=projection_df["occasional_expense"],
        name="Gasto puntual",
        marker_color=ACTIVE["expense"],
        marker_line=dict(color=ACTIVE["surface"], width=1),
    )
    fig.add_scatter(
        x=projection_df["month_start"],
        y=projection_df["income"],
        name="Ingresos proyectados",
        mode="lines+markers",
        line=dict(color=ACTIVE["income"], width=3),
    )
    fig.add_scatter(
        x=projection_df["month_start"],
        y=projection_df["net"],
        name="Neto proyectado",
        mode="lines+markers",
        line=dict(color=ACTIVE["brand"], width=3, dash="dot"),
    )
    fig.update_traces(hovertemplate="%{x|%b %Y}<br><b>%{y:,.2f} €</b><extra></extra>")
    fig.update_layout(barmode="stack")
    return base_layout(fig, "Composición del flujo proyectado", y_title="Euros")


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
        color_continuous_scale=ACTIVE["seq"],
    )
    fig.update_traces(
        marker_line_width=0,
        hovertemplate="<b>%{y}</b><br>%{x:,.2f} €<extra></extra>",
    )
    fig.update_layout(
        coloraxis_colorbar=dict(
            title=dict(text="Neto medio", font=dict(color=ACTIVE["ink2"], size=12)),
            tickfont=dict(color=ACTIVE["muted"], size=11),
        )
    )
    return base_layout(
        fig,
        "Balance final por escenario",
        x_title="Balance estimado al final del horizonte (€)",
        legend=False,
    )


def chart_card(fig, slot=None) -> None:
    """Renderiza una gráfica dentro de una tarjeta con borde.

    Asigna una key única por posición (reiniciada en cada run) para evitar
    colisiones de ID cuando dos gráficas comparten datos/parámetros.
    """
    chart_card._n += 1
    container = (slot or st).container(border=True)
    with container:
        st.plotly_chart(fig, width="stretch", key=f"chart_{chart_card._n}")


chart_card._n = 0


def main() -> None:
    st.set_page_config(
        page_title="Mis Finanzas",
        page_icon="📊",
        layout="wide",
    )
    chart_card._n = 0
    with st.sidebar:
        theme_choice = st.segmented_control(
            "Tema",
            options=["Auto", "Claro", "Oscuro"],
            default="Auto",
            key="theme_choice",
            help="Auto sigue el tema de tu sistema. Cámbialo a claro u oscuro cuando quieras.",
        )
    theme = resolve_theme(theme_choice)
    set_theme(theme)
    apply_styles(theme)

    with st.sidebar:
        st.header("Origen de datos")
        sheet_ref = st.text_input(
            "Google Sheet público",
            value=default_sheet_ref(),
            help="Puedes pegar la URL completa o solo el ID del documento.",
        )
        if st.button("Actualizar datos", width="stretch"):
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
        agrupar_categorias = st.checkbox(
            "Agrupar categorías equivalentes",
            value=False,
            help="Une categorías que solo se diferencian en mayúsculas o espacios.",
        )
    label_column = "label_normalized" if agrupar_categorias else "label_display"

    if not selected_accounts:
        st.warning("Selecciona al menos una cuenta.")
        st.stop()

    # ---- Cabecera + selector de periodo (parte superior) ----
    source_text = (
        "Google Sheets en directo"
        if metadata["source"] == "google_sheets"
        else "Copia local de respaldo"
    )
    loaded_at = metadata.get("loaded_at")
    loaded_at_str = loaded_at.strftime("%H:%M") if loaded_at else "?"
    live_dot = '<span class="liv"></span>' if metadata["source"] == "google_sheets" else ""
    st.markdown(
        f"""
        <div class="apphead">
            <div class="brand">
                <div class="mk">M</div>
                <div>
                    <div class="bt">Mis Finanzas</div>
                    <div class="bs">{len(all_accounts)} cuentas · {len(full_df):,} movimientos</div>
                </div>
            </div>
            <div class="fresh">{live_dot} {source_text} · actualizado {loaded_at_str}</div>
        </div>
        """.replace(",", "."),
        unsafe_allow_html=True,
    )
    if metadata.get("warning"):
        st.warning(metadata["warning"])

    period_label = st.segmented_control(
        "Periodo",
        options=["Todo", "Este año", "Últimos 6 meses", "Últimos 3 meses", "Personalizado"],
        default="Todo",
        key="period_choice",
        label_visibility="collapsed",
    ) or "Todo"
    if period_label == "Personalizado":
        start_date, end_date = st.slider(
            "Rango personalizado",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="DD/MM/YYYY",
        )
    else:
        start_date, end_date = period_preset_dates(period_label, min_date, max_date)

    accounts_df = combine_accounts(accounts, selected_accounts)
    period_df = filter_period(accounts_df, start_date, end_date)

    if period_df.empty:
        st.warning("No hay movimientos en el periodo seleccionado.")
        st.stop()

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
    avg_monthly_net = float(monthly_df["net"].mean()) if not monthly_df.empty else 0.0

    per_account_balance = (
        accounts_df.loc[accounts_df["date"] <= pd.Timestamp(end_date)]
        .groupby("account", as_index=False)["quantity"]
        .sum()
        .rename(columns={"quantity": "balance"})
        .sort_values("balance", ascending=False)
    )

    prev = previous_period_metrics(accounts_df, start_date, end_date)

    # ---- Hero: patrimonio total + saldo por cuenta ----
    trend_up = period_net >= 0
    trend = (
        f'<div class="tr" style="color:var(--{"income" if trend_up else "expense"})">'
        f'{"▲" if trend_up else "▼"} {format_eur(period_net)} en el periodo</div>'
    )
    swatches = ACTIVE["cat"]
    acct_rows = "".join(
        f'<div class="acct"><div class="nm">'
        f'<span class="sw" style="background:{swatches[i % len(swatches)]}"></span>{row.account}</div>'
        f'<div class="vl tnum">{format_eur(row.balance)}</div></div>'
        for i, row in enumerate(per_account_balance.itertuples())
    )
    st.markdown(
        f"""
        <style>
        .hero{{display:grid;grid-template-columns:1.05fr 1.45fr;gap:16px;margin-bottom:14px}}
        @media(max-width:760px){{.hero{{grid-template-columns:1fr}}}}
        .hero .card{{background:var(--surface);border:1px solid var(--hairline);
            border-radius:16px;box-shadow:var(--shadow)}}
        </style>
        <div class="hero">
            <div class="card netcard" style="padding:20px;display:flex;flex-direction:column;justify-content:center">
                <div class="k">Patrimonio total</div>
                <div class="big tnum">{format_eur(current_balance)}</div>
                {trend}
            </div>
            <div class="card accts" style="padding:16px 18px">
                <div class="k">Saldo por cuenta</div>
                {acct_rows}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- 4 KPIs claros (verde = ingresos, rojo = gasto) ----
    kpi_html = '<div class="kpi-row">' + "".join(
        [
            kpi_card(
                '<span class="ic inc">+</span> Ingresos',
                format_eur(period_income),
                delta_html=kpi_delta_eur(period_income, prev.get("income")),
            ),
            kpi_card(
                '<span class="ic exp">−</span> Gastos',
                format_eur(period_expense),
                delta_html=kpi_delta_eur(period_expense, prev.get("expense"), less_is_better=True),
            ),
            kpi_card(
                '<span class="ic acc">≈</span> Neto mensual medio',
                format_eur(avg_monthly_net),
                delta_html=f'<div class="delta" style="color:var(--muted)">Ahorro total {format_eur(period_net)}</div>',
            ),
            kpi_card(
                '<span class="ic acc">%</span> Tasa de ahorro',
                format_pct(savings_rate),
                delta_html=kpi_delta_pp(savings_rate, prev.get("savings_rate")),
            ),
        ]
    ) + "</div>"
    st.markdown(kpi_html, unsafe_allow_html=True)

    tab_summary, tab_labels, tab_projection, tab_movements = st.tabs(
        ["Resumen", "Categorías", "Proyección", "Movimientos"]
    )

    with tab_summary:
        row_1 = st.columns(2)
        chart_card(plot_income_vs_expense(monthly_df), row_1[0])
        chart_card(plot_daily_balance(balance_df), row_1[1])

        chart_card(plot_label_totals(overview_label_summary, label_column))

        with st.expander("Ver más detalle"):
            row_2 = st.columns(2)
            chart_card(plot_monthly_net(monthly_df), row_2[0])
            chart_card(plot_savings_rate(monthly_df), row_2[1])

            row_3 = st.columns(2)
            chart_card(plot_monthly_balance(monthly_balance_df), row_3[0])
            chart_card(plot_account_monthly_net(account_monthly_df), row_3[1])

            month_options = sorted(labels_df["month_key"].dropna().unique().tolist(), reverse=True)
            if month_options:
                selected_month = st.selectbox(
                    "Distribución de gastos por categoría en el mes",
                    options=month_options,
                    index=0,
                )
                pie_df = month_label_breakdown(labels_df, label_column, selected_month)
                if pie_df.empty:
                    st.info("No hay gastos con categorías en ese mes para el filtro actual.")
                else:
                    chart_card(plot_expense_pie(pie_df, label_column))

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
            chart_card(plot_label_totals(label_summary_df, label_column), label_row[0])
            selected_label = label_row[1].selectbox(
                "Categoría a analizar",
                options=label_summary_df[label_column].tolist(),
                index=0,
            )
            evolution_df = label_monthly_evolution(labels_df, label_column, selected_label)
            chart_card(plot_label_evolution(evolution_df), label_row[1])

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
                chart_card(
                    plot_label_comparison(
                        comparison_df,
                        label_column,
                        metric_column,
                        metric_axis_label,
                    )
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
                    width="stretch",
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
                width="stretch",
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
            chart_card(
                plot_projection_balance(
                    historical_balance,
                    projection_df,
                    scenario_band=scenario_band,
                    scenario_name=scenario_name,
                ),
                projection_row[0],
            )
            chart_card(plot_projection_components(projection_df), projection_row[1])
            chart_card(plot_scenario_outcomes(scenario_results))

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
                width="stretch",
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
                width="stretch",
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
            st.dataframe(projection_table, hide_index=True, width="stretch")

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
            width="stretch",
        )
        st.dataframe(
            movements_df.assign(
                date=movements_df["date"].dt.strftime("%Y-%m-%d"),
                quantity=movements_df["quantity"].map(format_eur),
            ),
            hide_index=True,
            width="stretch",
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
