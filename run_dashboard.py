from __future__ import annotations

from pathlib import Path
import sys

from streamlit.web import cli as stcli


if __name__ == "__main__":
    app_path = Path(__file__).with_name("app.py")
    sys.argv = ["streamlit", "run", str(app_path)]
    raise SystemExit(stcli.main())
