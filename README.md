# Mis Finanzas

Dashboard local en `Streamlit` para seguir la evolucion de tus cuentas a partir de un Google Sheet publico.

## Crear el entorno

```bash
conda env create -f environment.yml
conda activate mis-finanzas-dashboard
```

## Levantar la app

```bash
python run_dashboard.py
```

Alternativa equivalente:

```bash
streamlit run app.py
```

La app quedara disponible en `http://localhost:8501`.

## Origen de datos

El input sigue siendo un Google Sheet publico exportado como Excel. La app espera hojas
transaccionales con estas columnas:

- `year`
- `month`
- `day`
- `quantity`
- `label`
- `comment` opcional

Puedes usar el valor por defecto incluido en la app, pegar otra URL publica en la barra
lateral o definir `MIS_FINANZAS_SHEET_URL` como variable de entorno. Si usas secretos de
Streamlit, crea `.streamlit/secrets.toml` localmente; ese archivo queda excluido de Git.

## Como funciona

- Descarga el Google Sheet publico como `.xlsx`.
- Detecta automaticamente las hojas transaccionales con columnas `year`, `month`, `day`, `quantity` y `label`.
- Ignora hojas auxiliares, como la de renta.
- Excluye la ultima columna-resumen del Excel y solo usa las columnas utiles.
- Si Google Sheets falla temporalmente, reutiliza la ultima copia local almacenada en `.cache/mis_finanzas.xlsx`.

## Lo que incluye

- Filtros por cuenta, periodo y modo de agrupacion de labels.
- Resumen de ingresos, gastos, tasa de ahorro y balances.
- Graficos equivalentes al notebook original:
  - neto mensual
  - distribucion de gastos por label y mes
  - evolucion del saldo acumulado
  - patrimonio acumulado por mes
  - evolucion mensual de un label concreto
- Proyeccion de ingresos, gastos, neto y balance futuro.
- Escenarios de proyeccion basados en gasto fijo recurrente, gasto variable y movimientos puntuales.

## Control de versiones

El proyecto esta preparado para Git. Se excluyen archivos locales o generados como:

- `.cache/`
- `__pycache__/`
- `.pytest_cache/`
- entornos virtuales
- `.streamlit/secrets.toml`

Comandos utiles:

```bash
git status
git add .
git commit -m "Initial dashboard version"
```

## Tests

La logica financiera principal tiene tests unitarios basicos. Para ejecutarlos dentro del
entorno del proyecto:

```bash
python -m unittest discover -s tests
```
