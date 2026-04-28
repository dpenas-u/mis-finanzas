from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
import requests

DEFAULT_SHEET_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1Cn8mXJnuFatoVghd_lA_zaTYnGqVcBFGRbGGLsMUqQY/edit?gid=2102906247#gid=2102906247"
)

KNOWN_COLUMNS = ["year", "month", "day", "quantity", "label", "comment"]
REQUIRED_COLUMNS = {"year", "month", "day", "quantity", "label"}
LABEL_MODES = {
    "Sin espacios extra": "label_display",
    "Normalizadas": "label_normalized",
}
PROJECTION_SCENARIOS = {
    "Ahorro activo": {
        "recurring_income": 1.00,
        "variable_income": 0.98,
        "fixed_expense": 1.00,
        "variable_expense": 0.88,
        "occasional_expense": 0.72,
        "description": "Supone que contienes el gasto flexible y reduces compras puntuales.",
    },
    "Base": {
        "recurring_income": 1.00,
        "variable_income": 1.00,
        "fixed_expense": 1.00,
        "variable_expense": 1.00,
        "occasional_expense": 1.00,
        "description": "Replica tu comportamiento medio reciente sin forzar ajustes.",
    },
    "Prudente": {
        "recurring_income": 1.00,
        "variable_income": 0.94,
        "fixed_expense": 1.00,
        "variable_expense": 1.08,
        "occasional_expense": 1.18,
        "description": "Asume ingresos variables algo menores y más colchón en el gasto flexible.",
    },
    "Gasto relajado": {
        "recurring_income": 1.00,
        "variable_income": 1.00,
        "fixed_expense": 1.00,
        "variable_expense": 1.15,
        "occasional_expense": 1.32,
        "description": "Modela meses con menos contención en gasto discrecional y compras puntuales.",
    },
}


def extract_sheet_id(sheet_ref: str) -> str:
    sheet_ref = sheet_ref.strip()
    match = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", sheet_ref)
    if match:
        return match.group(1)
    if re.fullmatch(r"[a-zA-Z0-9-_]{20,}", sheet_ref):
        return sheet_ref
    raise ValueError(
        "No se pudo extraer el identificador del Google Sheet. "
        "Usa la URL pública completa o el ID del documento."
    )


def build_export_url(sheet_ref: str) -> str:
    return (
        f"https://docs.google.com/spreadsheets/d/{extract_sheet_id(sheet_ref)}/export?format=xlsx"
    )


def download_workbook_bytes(
    sheet_ref: str,
    cache_path: str | Path = ".cache/mis_finanzas.xlsx",
    timeout: int = 30,
) -> tuple[bytes, dict[str, Any]]:
    cache_file = Path(cache_path)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    export_url = build_export_url(sheet_ref)
    metadata: dict[str, Any] = {
        "source": "google_sheets",
        "warning": None,
        "cache_path": str(cache_file),
        "export_url": export_url,
        "loaded_at": datetime.now(timezone.utc),
    }

    try:
        response = requests.get(
            export_url,
            timeout=timeout,
            headers={"User-Agent": "mis-finanzas-dashboard/1.0"},
        )
        response.raise_for_status()
        workbook_bytes = response.content
        cache_file.write_bytes(workbook_bytes)
        metadata["download_size_bytes"] = len(workbook_bytes)
        return workbook_bytes, metadata
    except requests.RequestException as exc:
        if cache_file.exists():
            metadata["source"] = "cache_local"
            metadata["warning"] = (
                "No se pudo descargar la versión más reciente de Google Sheets; "
                "se usa la última copia local."
            )
            metadata["fallback_reason"] = str(exc)
            metadata["download_size_bytes"] = cache_file.stat().st_size
            return cache_file.read_bytes(), metadata
        raise RuntimeError(
            "No se pudo descargar el Google Sheet y no existe una copia local de respaldo."
        ) from exc


def normalize_account_sheet(
    df_raw: pd.DataFrame,
    account_name: str,
) -> pd.DataFrame | None:
    renamed_columns = {
        column: str(column).strip().lower()
        for column in df_raw.columns
        if not pd.isna(column)
    }
    df = df_raw.rename(columns=renamed_columns).copy()
    if not REQUIRED_COLUMNS.issubset(df.columns):
        return None

    normalized = pd.DataFrame(index=df.index)
    for column in KNOWN_COLUMNS:
        normalized[column] = df[column] if column in df.columns else pd.NA

    normalized["year"] = (
        normalized["year"].astype("string").str.extract(r"(\d{4})", expand=False)
    )
    normalized["month"] = (
        normalized["month"]
        .astype("string")
        .str.extract(r"(\d{1,2})", expand=False)
        .str.zfill(2)
    )
    normalized["day"] = (
        normalized["day"]
        .astype("string")
        .str.extract(r"(\d{1,2})", expand=False)
        .str.zfill(2)
    )
    normalized["quantity"] = pd.to_numeric(normalized["quantity"], errors="coerce")
    normalized["comment"] = normalized["comment"].astype("string").fillna("")

    label_display = (
        normalized["label"]
        .astype("string")
        .fillna("Sin label")
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    label_display = label_display.mask(label_display.eq(""), "Sin label")

    normalized["date"] = pd.to_datetime(
        normalized["year"] + "-" + normalized["month"] + "-" + normalized["day"],
        errors="coerce",
    )
    normalized["label_display"] = label_display
    normalized["label_normalized"] = label_display.str.casefold()
    normalized["account"] = account_name
    normalized = normalized.dropna(subset=["date", "quantity"]).copy()
    normalized = normalized.sort_values("date").reset_index(drop=True)
    normalized["month_start"] = normalized["date"].dt.to_period("M").dt.to_timestamp()
    normalized["month_key"] = normalized["month_start"].dt.strftime("%Y-%m")
    normalized["direction"] = np.where(normalized["quantity"] >= 0, "Ingreso", "Gasto")
    normalized["income_amount"] = np.where(
        normalized["quantity"] > 0,
        normalized["quantity"],
        0.0,
    )
    normalized["expense_abs"] = np.where(
        normalized["quantity"] < 0,
        normalized["quantity"].abs(),
        0.0,
    )
    return normalized


def load_accounts_from_sheet(
    sheet_ref: str,
    cache_path: str | Path = ".cache/mis_finanzas.xlsx",
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    workbook_bytes, metadata = download_workbook_bytes(sheet_ref, cache_path=cache_path)
    workbook = pd.ExcelFile(BytesIO(workbook_bytes), engine="openpyxl")

    accounts: dict[str, pd.DataFrame] = {}
    ignored_sheets: list[str] = []
    for sheet_name in workbook.sheet_names:
        df_raw = pd.read_excel(workbook, sheet_name=sheet_name)
        normalized = normalize_account_sheet(df_raw, sheet_name)
        if normalized is None or normalized.empty:
            ignored_sheets.append(sheet_name)
            continue
        accounts[sheet_name] = normalized

    if not accounts:
        raise RuntimeError(
            "No se encontraron hojas con columnas compatibles "
            "(year, month, day, quantity, label)."
        )

    combined = pd.concat(accounts.values(), ignore_index=True).sort_values("date")
    metadata["accounts"] = list(accounts.keys())
    metadata["ignored_sheets"] = ignored_sheets
    metadata["min_date"] = combined["date"].min()
    metadata["max_date"] = combined["date"].max()
    return accounts, metadata


def combine_accounts(
    accounts: dict[str, pd.DataFrame],
    selected_accounts: list[str],
) -> pd.DataFrame:
    frames = [accounts[account] for account in selected_accounts if account in accounts]
    if not frames:
        return pd.DataFrame(columns=list(next(iter(accounts.values())).columns))
    return pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)


def filter_period(
    df: pd.DataFrame,
    start_date: datetime | pd.Timestamp,
    end_date: datetime | pd.Timestamp,
) -> pd.DataFrame:
    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize()
    mask = (df["date"] >= start_ts) & (df["date"] <= end_ts)
    return df.loc[mask].copy()


def monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=["month_start", "net", "income", "expense", "transactions", "savings_rate"]
        )

    monthly = (
        df.groupby("month_start", as_index=False)
        .agg(
            net=("quantity", "sum"),
            income=("income_amount", "sum"),
            expense=("expense_abs", "sum"),
            transactions=("quantity", "size"),
        )
        .sort_values("month_start")
        .reset_index(drop=True)
    )
    monthly["savings_rate"] = np.where(
        monthly["income"] > 0,
        monthly["net"] / monthly["income"],
        np.nan,
    )
    return monthly


def account_monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["account", "month_start", "net", "income", "expense"])

    return (
        df.groupby(["account", "month_start"], as_index=False)
        .agg(
            net=("quantity", "sum"),
            income=("income_amount", "sum"),
            expense=("expense_abs", "sum"),
        )
        .sort_values(["month_start", "account"])
        .reset_index(drop=True)
    )


def complete_months(monthly_df: pd.DataFrame) -> pd.DataFrame:
    if monthly_df.empty:
        return monthly_df.copy()

    full_index = pd.date_range(
        monthly_df["month_start"].min(),
        monthly_df["month_start"].max(),
        freq="MS",
    )
    completed = (
        monthly_df.set_index("month_start")
        .reindex(full_index, fill_value=0)
        .rename_axis("month_start")
        .reset_index()
    )
    if "savings_rate" in completed.columns:
        completed["savings_rate"] = np.where(
            completed["income"] > 0,
            completed["net"] / completed["income"],
            np.nan,
        )
    return completed


def balance_over_time(
    df: pd.DataFrame,
    start_date: datetime | pd.Timestamp,
    end_date: datetime | pd.Timestamp,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["date", "delta", "balance"])

    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize()
    opening_balance = df.loc[df["date"] < start_ts, "quantity"].sum()

    period = (
        df.loc[(df["date"] >= start_ts) & (df["date"] <= end_ts)]
        .groupby("date", as_index=False)["quantity"]
        .sum()
        .sort_values("date")
    )
    if period.empty:
        return pd.DataFrame(
            {
                "date": [start_ts],
                "delta": [0.0],
                "balance": [opening_balance],
            }
        )

    period = period.rename(columns={"quantity": "delta"})
    period["balance"] = opening_balance + period["delta"].cumsum()
    if period.iloc[0]["date"] > start_ts:
        period = pd.concat(
            [
                pd.DataFrame(
                    {"date": [start_ts], "delta": [0.0], "balance": [opening_balance]}
                ),
                period,
            ],
            ignore_index=True,
        )
    return period


def monthly_balance_over_time(
    df: pd.DataFrame,
    start_date: datetime | pd.Timestamp,
    end_date: datetime | pd.Timestamp,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["month_start", "balance"])

    start_month = pd.Timestamp(start_date).to_period("M").to_timestamp()
    end_month = pd.Timestamp(end_date).to_period("M").to_timestamp()
    monthly = (
        df.groupby("month_start", as_index=False)["quantity"]
        .sum()
        .sort_values("month_start")
        .rename(columns={"quantity": "delta"})
    )
    monthly["balance"] = monthly["delta"].cumsum()
    return monthly.loc[
        (monthly["month_start"] >= start_month) & (monthly["month_start"] <= end_month)
    ].reset_index(drop=True)


def label_summary(df: pd.DataFrame, label_column: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                label_column,
                "movimientos",
                "gasto_total",
                "ingreso_total",
                "neto",
                "ticket_medio",
            ]
        )

    summary = (
        df.groupby(label_column, as_index=False)
        .agg(
            movimientos=("quantity", "size"),
            gasto_total=("expense_abs", "sum"),
            ingreso_total=("income_amount", "sum"),
            neto=("quantity", "sum"),
            ticket_medio=("quantity", "mean"),
        )
        .sort_values(["gasto_total", "movimientos"], ascending=[False, False])
        .reset_index(drop=True)
    )
    return summary


def month_label_breakdown(
    df: pd.DataFrame,
    label_column: str,
    month_key: str,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[label_column, "expense"])

    filtered = df.loc[(df["month_key"] == month_key) & (df["expense_abs"] > 0)].copy()
    if filtered.empty:
        return pd.DataFrame(columns=[label_column, "expense"])
    return (
        filtered.groupby(label_column, as_index=False)["expense_abs"]
        .sum()
        .rename(columns={"expense_abs": "expense"})
        .sort_values("expense", ascending=False)
        .reset_index(drop=True)
    )


def label_monthly_evolution(
    df: pd.DataFrame,
    label_column: str,
    selected_label: str,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["month_start", "expense", "net"])

    filtered = df.loc[df[label_column] == selected_label].copy()
    if filtered.empty:
        return pd.DataFrame(columns=["month_start", "expense", "net"])

    evolution = (
        filtered.groupby("month_start", as_index=False)
        .agg(
            expense=("expense_abs", "sum"),
            net=("quantity", "sum"),
        )
        .sort_values("month_start")
        .reset_index(drop=True)
    )
    return evolution


def recurring_label_profile(
    df: pd.DataFrame,
    label_column: str,
    months_index: pd.Series | pd.Index | list[pd.Timestamp],
    flow: str = "expense",
) -> pd.DataFrame:
    months = pd.Index(pd.to_datetime(months_index)).sort_values().unique()
    if df.empty or len(months) == 0:
        return pd.DataFrame(
            columns=[
                label_column,
                "classification",
                "active_months",
                "coverage_ratio",
                "avg_monthly",
                "avg_active_month",
                "std_active_month",
                "cv_active_month",
                "last_month_amount",
            ]
        )

    if flow == "expense":
        value_column = "expense_abs"
    elif flow == "income":
        value_column = "income_amount"
    else:
        raise ValueError("flow debe ser 'expense' o 'income'.")

    filtered = df.loc[
        df["month_start"].isin(months) & (df[value_column] > 0),
        [label_column, "month_start", value_column],
    ].copy()
    if filtered.empty:
        return pd.DataFrame(
            columns=[
                label_column,
                "classification",
                "active_months",
                "coverage_ratio",
                "avg_monthly",
                "avg_active_month",
                "std_active_month",
                "cv_active_month",
                "last_month_amount",
            ]
        )

    pivot = filtered.pivot_table(
        index=label_column,
        columns="month_start",
        values=value_column,
        aggfunc="sum",
        fill_value=0.0,
    ).reindex(columns=months, fill_value=0.0)

    active_months = (pivot > 0).sum(axis=1)
    coverage_ratio = active_months / len(months)
    active_values = pivot.mask(pivot <= 0)
    avg_monthly = pivot.mean(axis=1)
    avg_active_month = active_values.mean(axis=1).fillna(0.0)
    std_active_month = active_values.std(axis=1, ddof=0).fillna(0.0)
    cv_active = np.where(
        avg_active_month > 0,
        std_active_month / avg_active_month,
        np.nan,
    )
    last_month_amount = pivot.iloc[:, -1]

    if flow == "expense":
        classification = np.select(
            [
                (coverage_ratio >= 0.75) & (cv_active <= 0.35),
                coverage_ratio >= 0.50,
            ],
            ["Fijo recurrente", "Recurrente variable"],
            default="Puntual",
        )
        order = {
            "Fijo recurrente": 0,
            "Recurrente variable": 1,
            "Puntual": 2,
        }
    else:
        classification = np.select(
            [
                coverage_ratio >= 0.60,
                coverage_ratio >= 0.25,
            ],
            ["Ingreso recurrente", "Ingreso variable"],
            default="Puntual",
        )
        order = {
            "Ingreso recurrente": 0,
            "Ingreso variable": 1,
            "Puntual": 2,
        }

    profile = pd.DataFrame(
        {
            label_column: pivot.index,
            "classification": classification,
            "active_months": active_months.to_numpy(),
            "coverage_ratio": coverage_ratio.to_numpy(),
            "avg_monthly": avg_monthly.to_numpy(),
            "avg_active_month": avg_active_month.to_numpy(),
            "std_active_month": std_active_month.to_numpy(),
            "cv_active_month": cv_active,
            "last_month_amount": last_month_amount.to_numpy(),
        }
    )
    profile["classification_order"] = profile["classification"].map(order).fillna(9)
    return (
        profile.sort_values(
            ["classification_order", "avg_monthly", "coverage_ratio"],
            ascending=[True, False, False],
        )
        .drop(columns="classification_order")
        .reset_index(drop=True)
    )


def _component_monthly_series(
    df: pd.DataFrame,
    label_column: str,
    months: pd.Index,
    labels: pd.Series | list[str],
    value_column: str,
) -> pd.Series:
    base = pd.Series(0.0, index=months, dtype=float)
    label_list = pd.Series(labels).dropna().tolist()
    if not label_list:
        return base

    series = (
        df.loc[
            df["month_start"].isin(months) & df[label_column].isin(label_list),
            ["month_start", value_column],
        ]
        .groupby("month_start")[value_column]
        .sum()
    )
    return base.add(series, fill_value=0.0).reindex(months, fill_value=0.0)


def build_projection_history(
    df: pd.DataFrame,
    label_column: str,
    lookback_months: int,
) -> dict[str, pd.DataFrame]:
    monthly = complete_months(monthly_summary(df))
    history = monthly.tail(lookback_months).reset_index(drop=True)
    if history.empty:
        empty = pd.DataFrame()
        return {
            "history": empty,
            "expense_profile": empty,
            "income_profile": empty,
        }

    months = pd.Index(history["month_start"])
    expense_profile = recurring_label_profile(df, label_column, months, flow="expense")
    income_profile = recurring_label_profile(df, label_column, months, flow="income")

    components = pd.DataFrame({"month_start": months})
    components["fixed_expense"] = _component_monthly_series(
        df,
        label_column,
        months,
        expense_profile.loc[
            expense_profile["classification"] == "Fijo recurrente",
            label_column,
        ],
        "expense_abs",
    ).to_numpy()
    components["variable_expense"] = _component_monthly_series(
        df,
        label_column,
        months,
        expense_profile.loc[
            expense_profile["classification"] == "Recurrente variable",
            label_column,
        ],
        "expense_abs",
    ).to_numpy()
    components["occasional_expense"] = _component_monthly_series(
        df,
        label_column,
        months,
        expense_profile.loc[
            expense_profile["classification"] == "Puntual",
            label_column,
        ],
        "expense_abs",
    ).to_numpy()
    components["recurring_income"] = _component_monthly_series(
        df,
        label_column,
        months,
        income_profile.loc[
            income_profile["classification"] == "Ingreso recurrente",
            label_column,
        ],
        "income_amount",
    ).to_numpy()
    components["variable_income"] = _component_monthly_series(
        df,
        label_column,
        months,
        income_profile.loc[
            income_profile["classification"] != "Ingreso recurrente",
            label_column,
        ],
        "income_amount",
    ).to_numpy()
    components["expense"] = (
        components["fixed_expense"]
        + components["variable_expense"]
        + components["occasional_expense"]
    )
    components["income"] = components["recurring_income"] + components["variable_income"]
    components["net"] = components["income"] - components["expense"]

    return {
        "history": components,
        "expense_profile": expense_profile,
        "income_profile": income_profile,
    }


def _project_series(
    series: pd.Series,
    horizon_months: int,
    method: str,
    allow_trend: bool = True,
) -> np.ndarray:
    values = pd.Series(series, dtype=float).fillna(0.0).to_numpy()
    if horizon_months <= 0:
        return np.array([], dtype=float)
    if len(values) == 0:
        return np.zeros(horizon_months, dtype=float)
    if method == "linear" and allow_trend and len(values) >= 2:
        x_hist = np.arange(len(values), dtype=float)
        x_future = np.arange(len(values), len(values) + horizon_months, dtype=float)
        slope, intercept = np.polyfit(x_hist, values, 1)
        projected = slope * x_future + intercept
    else:
        projected = np.full(horizon_months, values.mean(), dtype=float)
    return np.clip(projected, 0.0, None)


def projection_from_component_history(
    history_df: pd.DataFrame,
    current_balance: float,
    horizon_months: int,
    method: str,
    scenario: dict[str, Any] | None = None,
) -> pd.DataFrame:
    if history_df.empty:
        return pd.DataFrame(
            columns=[
                "month_start",
                "fixed_expense",
                "variable_expense",
                "occasional_expense",
                "recurring_income",
                "variable_income",
                "expense",
                "income",
                "net",
                "balance_projection",
            ]
        )

    scenario = scenario or PROJECTION_SCENARIOS["Base"]
    last_month = history_df["month_start"].max()
    future_months = pd.date_range(
        last_month + pd.offsets.MonthBegin(1),
        periods=horizon_months,
        freq="MS",
    )
    projection = pd.DataFrame({"month_start": future_months})
    projection["fixed_expense"] = (
        _project_series(
            history_df["fixed_expense"],
            horizon_months,
            method="average",
            allow_trend=False,
        )
        * scenario["fixed_expense"]
    )
    projection["variable_expense"] = (
        _project_series(
            history_df["variable_expense"],
            horizon_months,
            method=method,
            allow_trend=True,
        )
        * scenario["variable_expense"]
    )
    projection["occasional_expense"] = (
        _project_series(
            history_df["occasional_expense"],
            horizon_months,
            method="average",
            allow_trend=False,
        )
        * scenario["occasional_expense"]
    )
    projection["recurring_income"] = (
        _project_series(
            history_df["recurring_income"],
            horizon_months,
            method=method,
            allow_trend=True,
        )
        * scenario["recurring_income"]
    )
    projection["variable_income"] = (
        _project_series(
            history_df["variable_income"],
            horizon_months,
            method="average",
            allow_trend=False,
        )
        * scenario["variable_income"]
    )
    projection["expense"] = (
        projection["fixed_expense"]
        + projection["variable_expense"]
        + projection["occasional_expense"]
    )
    projection["income"] = projection["recurring_income"] + projection["variable_income"]
    projection["net"] = projection["income"] - projection["expense"]
    projection["balance_projection"] = current_balance + projection["net"].cumsum()
    return projection


def projection_from_history(
    monthly_df: pd.DataFrame,
    current_balance: float,
    horizon_months: int,
    lookback_months: int,
    method: str,
) -> pd.DataFrame:
    history = complete_months(monthly_df).tail(lookback_months).reset_index(drop=True)
    if history.empty:
        return pd.DataFrame(
            columns=[
                "month_start",
                "income",
                "expense",
                "net",
                "balance_projection",
                "balance_optimistic",
                "balance_conservative",
            ]
        )

    last_month = history["month_start"].max()
    future_months = pd.date_range(
        last_month + pd.offsets.MonthBegin(1),
        periods=horizon_months,
        freq="MS",
    )

    if method == "linear" and len(history) >= 2:
        x_hist = np.arange(len(history), dtype=float)
        x_future = np.arange(len(history), len(history) + horizon_months, dtype=float)
        income_coef = np.polyfit(x_hist, history["income"].to_numpy(dtype=float), 1)
        expense_coef = np.polyfit(x_hist, history["expense"].to_numpy(dtype=float), 1)
        projected_income = income_coef[0] * x_future + income_coef[1]
        projected_expense = expense_coef[0] * x_future + expense_coef[1]
    else:
        projected_income = np.full(horizon_months, history["income"].mean())
        projected_expense = np.full(horizon_months, history["expense"].mean())

    projected_income = np.clip(projected_income, 0.0, None)
    projected_expense = np.clip(projected_expense, 0.0, None)
    projected_net = projected_income - projected_expense
    net_volatility = float(history["net"].std(ddof=0)) if len(history) > 1 else 0.0

    projection = pd.DataFrame(
        {
            "month_start": future_months,
            "income": projected_income,
            "expense": projected_expense,
            "net": projected_net,
        }
    )
    projection["balance_projection"] = current_balance + projection["net"].cumsum()
    projection["balance_optimistic"] = current_balance + (
        projection["net"] + net_volatility * 0.35
    ).cumsum()
    projection["balance_conservative"] = current_balance + (
        projection["net"] - net_volatility * 0.35
    ).cumsum()
    return projection
