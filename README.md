# Pronóstico de Series Financieras: S&P 500 vs. Dow Jones

Análisis y pronóstico de series de tiempo sobre dos de los principales índices bursátiles de EE.UU. — **S&P 500 (`^GSPC`)** y **Dow Jones (`^DJI`)** — comparando tres enfoques de forecasting: **Holt-Winters**, **ARIMA rolling (walk-forward)** y una red neuronal **LSTM**.

> Proyecto del curso *IA Generativa para la Gestión*. Equipo: Tomás Allende · Juan Costa · Felipe Castillo.

---

## 🎯 Objetivo

Determinar si la relación entre el S&P 500 y el Dow Jones es **explotable** (cointegración) y qué modelo entrega el mejor pronóstico del precio de cierre diario en un contexto realista.

## 🧰 Stack

`Python` · `yfinance` · `pandas` · `NumPy` · `statsmodels` · `pmdarima` · `scikit-learn` · `TensorFlow/Keras` · `Matplotlib` · `seaborn`

## 📊 Datos

| | |
|---|---|
| **Tickers** | `^GSPC` (S&P 500), `^DJI` (Dow Jones) |
| **Fuente** | Yahoo Finance vía `yfinance` |
| **Período** | 2015-01-01 → 2025-09-16 (~10,7 años, frecuencia diaria) |
| **Variable** | Precio de cierre (`Close`) |
| **Partición** | 80% entrenamiento / 20% prueba |

## 🔬 Metodología

1. **Análisis exploratorio** — visualización, estadística descriptiva y descomposición de la serie.
2. **Estacionariedad** — test aumentado de Dickey-Fuller (ADF) sobre cada serie.
3. **Relación entre índices** — correlación y test de **cointegración de Engle-Granger**.
4. **Modelamiento y comparación** de tres enfoques:
   - **Holt-Winters** (suavizamiento exponencial) — línea base.
   - **ARIMA rolling** — pronóstico walk-forward, reentrenando con cada dato observado.
   - **LSTM** — red neuronal recurrente sobre datos escalados (`MinMaxScaler`).

## 📈 Resultados

| Modelo | MAPE | R² | Comentario |
|---|---|---|---|
| Holt-Winters | 12,75 % | negativo | Baseline; proyecta una recta, no capta volatilidad. |
| ARIMA (rolling) | ~0,67 % | 0,99 | Sigue la serie con desfase mínimo de un día. |
| **LSTM** | ~0,67 % | 0,99 | Ligeramente mejor (MAE más bajo). |

**Conclusiones clave:**
- Ambas series son **no estacionarias** con tendencia alcista de largo plazo y correlación muy alta.
- En un mercado eficiente, el mejor pronóstico del precio de mañana es cercano al precio de hoy: los modelos *rolling* (ARIMA y LSTM) capturan exactamente eso y superan ampliamente al baseline.
- Pronosticar a **largo horizonte** es arriesgado: el error se acumula y los cambios de régimen del mercado no son predecibles desde los precios pasados.

## ▶️ Cómo ejecutar

```bash
pip install -r requirements.txt
jupyter notebook sp500-dji-forecasting.ipynb
```

El notebook descarga los datos automáticamente desde Yahoo Finance.

---

> **Transparencia:** el notebook incluye una declaración de uso de IA (apoyo en estructura, redacción y depuración de código); la metodología, los modelos y la interpretación son del equipo.
