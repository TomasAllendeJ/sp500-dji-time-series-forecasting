# Pronóstico de Series Financieras: S&P 500 vs. Dow Jones

Análisis y pronóstico de series de tiempo sobre dos de los principales índices bursátiles de EE.UU. — **S&P 500 (`^GSPC`)** y **Dow Jones (`^DJI`)** — comparando tres enfoques de forecasting: **Holt-Winters**, **ARIMA rolling (walk-forward)** y una red neuronal **LSTM**.

> Proyecto del curso *IA Generativa para la Gestión*. Equipo: [Tomás Allende](https://github.com/TomasAllendeJ) · Juan Costa · [Felipe Castillo](https://github.com/CastilloFelipe).

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
| **Período** | 2015-09-16 → 2025-09-16 (10 años, frecuencia diaria) |
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

**Relación entre índices:** la correlación entre el S&P 500 y el Dow Jones es de 0,99, pero
el test de Engle-Granger da un p-value de **0,39**: las series **no están cointegradas**. Por
eso no sirve una estrategia de *pairs trading* basada en que el diferencial vuelva a cerrarse,
y usar un índice como cobertura del otro es más riesgoso de lo que sugiere la correlación.

**Conclusiones clave:**
- Ambas series son **no estacionarias** con tendencia alcista de largo plazo y correlación muy alta.
- En un mercado eficiente, el mejor pronóstico del precio de mañana es cercano al precio de hoy: los modelos *rolling* (ARIMA y LSTM) capturan exactamente eso y superan ampliamente al baseline.
- Pronosticar a **largo horizonte** es arriesgado: el error se acumula y los cambios de régimen del mercado no son predecibles desde los precios pasados.

## ⚠️ Limitaciones y lectura de los resultados

- **Pronóstico a un día.** ARIMA y LSTM se evalúan con un esquema *rolling*: predicen solo el
  día siguiente y se actualizan con el precio real. Un MAPE de 0,67% es esperable en ese
  esquema, porque el precio de mañana se parece mucho al de hoy. La comparación justa es con
  un pronóstico ingenuo (precio de mañana = precio de hoy); superarlo es lo que indicaría una
  capacidad predictiva real.
- **R² sobre precios.** El R² de 0,99 se calcula sobre el nivel del índice, que tiene una
  tendencia fuerte. Medido sobre los retornos diarios, el resultado sería mucho más modesto.
- **Horizontes largos.** Holt-Winters muestra cómo crece el error al proyectar muchos días sin
  datos nuevos; los cambios de régimen del mercado no se pueden anticipar con precios pasados.
- **Sin costos de transacción.** Las conclusiones sobre estrategias no consideran comisiones,
  *spread* ni deslizamiento de precios.
- **Datos descargados al ejecutar.** El notebook baja los datos de Yahoo Finance con fechas
  fijas, pero el proveedor puede corregir cifras históricas, así que los resultados pueden
  variar levemente entre ejecuciones.

## ▶️ Cómo ejecutar

```bash
pip install -r requirements.txt
jupyter notebook sp500-dji-forecasting.ipynb
```

El notebook descarga los datos automáticamente desde Yahoo Finance.

---

> **Transparencia:** el notebook incluye una declaración de uso de IA (apoyo en estructura, redacción y depuración de código); la metodología, los modelos y la interpretación son del equipo.
