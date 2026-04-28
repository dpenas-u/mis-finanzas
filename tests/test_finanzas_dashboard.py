from __future__ import annotations

import unittest

import pandas as pd

from finanzas_dashboard import (
    balance_over_time,
    build_projection_history,
    monthly_summary,
    normalize_account_sheet,
    recurring_label_profile,
)


class FinanzasDashboardTest(unittest.TestCase):
    def test_normalize_account_sheet_keeps_excel_contract(self) -> None:
        raw = pd.DataFrame(
            {
                "year": [2026, 2026, None],
                "month": [1, 1, 1],
                "day": [2, 3, 4],
                "quantity": [1000, -25.5, -10],
                "label": [" Nomina  ", " Cafe   Bar ", "ignored"],
                "comment": ["enero", None, "bad date"],
            }
        )

        normalized = normalize_account_sheet(raw, "Cuenta")

        self.assertIsNotNone(normalized)
        assert normalized is not None
        self.assertEqual(len(normalized), 2)
        self.assertEqual(normalized.loc[0, "label_display"], "Nomina")
        self.assertEqual(normalized.loc[1, "label_display"], "Cafe Bar")
        self.assertEqual(normalized.loc[0, "account"], "Cuenta")
        self.assertEqual(normalized.loc[1, "direction"], "Gasto")

    def test_monthly_summary_and_balance(self) -> None:
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-01-01", "2026-01-10", "2026-02-01"]),
                "month_start": pd.to_datetime(["2026-01-01", "2026-01-01", "2026-02-01"]),
                "quantity": [1000.0, -200.0, -50.0],
                "income_amount": [1000.0, 0.0, 0.0],
                "expense_abs": [0.0, 200.0, 50.0],
            }
        )

        monthly = monthly_summary(df)
        balance = balance_over_time(df, "2026-01-05", "2026-02-28")

        self.assertEqual(monthly.loc[0, "net"], 800.0)
        self.assertEqual(monthly.loc[0, "savings_rate"], 0.8)
        self.assertEqual(balance.iloc[-1]["balance"], 750.0)

    def test_recurring_profile_classifies_fixed_expense(self) -> None:
        months = pd.to_datetime(["2026-01-01", "2026-02-01", "2026-03-01", "2026-04-01"])
        df = pd.DataFrame(
            {
                "month_start": list(months) + list(months[:2]),
                "label_display": ["Alquiler"] * 4 + ["Restaurante"] * 2,
                "expense_abs": [800.0, 800.0, 810.0, 800.0, 40.0, 55.0],
                "income_amount": [0.0] * 6,
            }
        )

        profile = recurring_label_profile(df, "label_display", months, flow="expense")

        alquiler = profile.loc[profile["label_display"] == "Alquiler"].iloc[0]
        restaurante = profile.loc[profile["label_display"] == "Restaurante"].iloc[0]
        self.assertEqual(alquiler["classification"], "Fijo recurrente")
        self.assertEqual(restaurante["classification"], "Recurrente variable")

    def test_projection_history_splits_components(self) -> None:
        df = pd.DataFrame(
            {
                "month_start": pd.to_datetime(
                    [
                        "2026-01-01",
                        "2026-02-01",
                        "2026-03-01",
                        "2026-01-01",
                        "2026-03-01",
                    ]
                ),
                "quantity": [1000.0, 1000.0, 1000.0, -300.0, -450.0],
                "income_amount": [1000.0, 1000.0, 1000.0, 0.0, 0.0],
                "expense_abs": [0.0, 0.0, 0.0, 300.0, 450.0],
                "label_display": ["Nomina", "Nomina", "Nomina", "Viaje", "Viaje"],
            }
        )

        result = build_projection_history(df, "label_display", lookback_months=3)

        self.assertIn("history", result)
        self.assertEqual(result["history"]["recurring_income"].sum(), 3000.0)
        self.assertEqual(result["history"]["variable_expense"].sum(), 750.0)


if __name__ == "__main__":
    unittest.main()
