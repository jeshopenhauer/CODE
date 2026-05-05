"""
config.py - Archivo de configuración centralizado

Modifica los parámetros aquí para personalizar el análisis.
"""

# =====================================================================
# CONFIGURACIÓN DE EMPRESAS
# =====================================================================
# Formato: "key": {"nombre": str, "ticker": str, "sector": str, "activo": bool}
# Establece "activo": False para deshabilitar una empresa
EMPRESAS = {
    "XOM": {
        "nombre": "Exxon Mobil",
        "ticker": "XOM",
        "sector": "Energy",
        "activo": True,
    },
    "CVX": {
        "nombre": "Chevron Corporation",
        "ticker": "CVX",
        "sector": "Energy",
        "activo": True,
    },
    "SHEL": {
        "nombre": "Shell PLC",
        "ticker": "SHEL",
        "sector": "Energy",
        "activo": True,
    },
    "BP": {
        "nombre": "BP plc",
        "ticker": "BP",
        "sector": "Energy",
        "activo": True,
    },
    "KO": {
        "nombre": "The Coca-Cola Company",
        "ticker": "KO",
        "sector": "Consumer Staples",
        "activo": True,
    },
    "NVDA": {
        "nombre": "NVIDIA Corporation",
        "ticker": "NVDA",
        "sector": "Technology",
        "activo": True,
    },
    "ADBE": {
        "nombre": "Adobe Inc.",
        "ticker": "ADBE",
        "sector": "Technology",
        "activo": True,
    },
    "TSLA": {
        "nombre": "Tesla Inc.",
        "ticker": "TSLA",
        "sector": "Technology",
        "activo": True,
    },
    "INTC": {
        "nombre": "Intel Corporation",
        "ticker": "INTC",
        "sector": "Technology",
        "activo": True,
    },
    "PLTR": {
        "nombre": "Palantir Technologies",
        "ticker": "PLTR",
        "sector": "Technology",
        "activo": True,
    },
    "WDAY": {
        "nombre": "Workday Inc.",
        "ticker": "WDAY",
        "sector": "Technology",
        "activo": True,
    },
    "GENI": {
        "nombre": "Genius Sports Limited",
        "ticker": "GENI",
        "sector": "Technology",
        "activo": True,
    },
    "TEAM": {
        "nombre": "Atlassian Corporation",
        "ticker": "TEAM",
        "sector": "Technology",
        "activo": True,
    },
}

# =====================================================================
# PARÁMETROS DE EXTRACCIÓN DE DATOS
# =====================================================================
DIAS_MINIMOS = 15           # Exluye opciones que caducan en < 15 días (ruido gamma)
DIAS_MAXIMOS = 365          # Incluye opciones hasta 1 año vista

RUTA_CSV = "datos_{company}.csv"
RUTA_GRAFICO = "{company}_analisis.png"

# =====================================================================
# PARÁMETROS DE ANÁLISIS CUANTITATIVO
# =====================================================================
# Tasas de interés
TASA_RIESGO_LIBRE = 0.04    # 4% anual (r en Black-Scholes)

# Árbol binomial
PASOS_ARBOL = 100           # Número de pasos en el árbol (trade-off: precisión vs velocidad)

# Suavizado de distribuciones
SIGMA_GAUSSIANA = 2.0       # Desviación estándar del filtro Gaussiano

# Grid de precios
RANGO_PRECIO_MIN = 0.6      # Mín: 60% del spot actual
RANGO_PRECIO_MAX = 1.4      # Máx: 140% del spot actual
PUNTOS_GRID = 250           # Número de puntos en el grid de precios

# =====================================================================
# PARÁMETROS DE VISUALIZACIÓN
# =====================================================================
# Tamaño de figuras
FIGURA_ANCHO = 16
FIGURA_ALTO = 10

# Ratios de subplots (ridge plot vs barplot)
RATIO_ANCHO = [3, 1]        # 75% ridge plot, 25% barplot
ESPACIADO = 0.3

# Colores
COLORMAP_RIDGE = "viridis"
COLORMAP_BARRAS = "plasma"

# Resolución de salida
DPI_SALIDA = 300

# =====================================================================
# PARÁMETROS DE DETECCIÓN DE MÁXIMOS
# =====================================================================
ORDEN_MAXIMOS = 5           # Número de puntos a cada lado para definir máximoNUMERO_MAXIMOS = 2        # Detectar los 2 máximos más altos por distribución

# =====================================================================
# LOGGING Y VERBOSIDAD
# =====================================================================
VERBOSE = True              # Mostrar detalles durante ejecución
MOSTRAR_GRAFICOS = True     # plt.show() al final de cada análisis
GUARDAR_PNG = True          # Guardar PNG de cada gráfico

# =====================================================================
# TOLERANCIAS NUMÉRICAS
# =====================================================================
MIN_STRIKES = 3             # Mínimo de strikes para considerar un vencimiento
MIN_AREA_PDF = 1e-9         # Área mínima para considerar una PDF válida
RANGO_IV_MIN = 1e-4         # IV mínima permitida
RANGO_IV_MAX = 2.5          # IV máxima permitida (250% volatilidad)

# Rango de probabilidades (filtro contra arbitraje)
PROB_MINIMA = 0.01
PROB_MAXIMA = 0.99

if __name__ == "__main__":
    print("Configuración del Sistema:")
    print(f"  - Empresas: {len(EMPRESAS)}")
    print(f"  - Rango de vencimientos: {DIAS_MINIMOS}-{DIAS_MAXIMOS} días")
    print(f"  - Pasos del árbol: {PASOS_ARBOL}")
    print(f"  - Tasa libre de riesgo: {TASA_RIESGO_LIBRE*100:.1f}%")
