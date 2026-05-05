import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.interpolate import CubicSpline, make_interp_spline
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema
import traceback
import yfinance as yf
import warnings

warnings.filterwarnings('ignore')

# Fix para versiones de NumPy (2.0 vs 1.2x)
try:
    integrate_func = np.trapezoid
except AttributeError:
    integrate_func = np.trapz


# =====================================================================
# FUNCIONES MATEMÁTICAS (nivel de módulo — reutilizables y testeables)
# =====================================================================

def bs_price(S, K, T, r, sigma, tipe='c'):
    """Precio Black-Scholes para call ('c') o put ('p')."""
    if T <= 0:
        return max(S - K, 0) if tipe == 'c' else max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if tipe == 'c':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def solve_iv(S, K, T, r, mkt_price, tipe):
    """
    Calcula la volatilidad implícita mediante Brent.
    Devuelve np.nan si el precio es incompatible con el modelo o si
    el spread bid/ask es demasiado amplio (señal de baja liquidez).
    """
    # Precio de mercado inválido
    if mkt_price <= 0:
        return np.nan
    # Precio inferior al valor intrínseco → arbitraje, se descarta
    intrinsic = max(S - K, 0) if tipe == 'c' else max(K - S, 0)
    if mkt_price < intrinsic * 0.999:
        return np.nan
    try:
        f_low  = bs_price(S, K, T, r, 0.001, tipe) - mkt_price
        f_high = bs_price(S, K, T, r, 5.0,   tipe) - mkt_price
        # Sin cambio de signo → precio fuera del rango BS
        if f_low * f_high > 0:
            return np.nan
        return brentq(
            lambda sig: bs_price(S, K, T, r, sig, tipe) - mkt_price,
            0.001, 5.0, xtol=1e-6
        )
    except Exception:
        return np.nan


# =====================================================================
# FUNCIONES DE VALIDACIÓN DEL ÁRBOL IMPLÍCITO
# =====================================================================

def valorar_opcion_con_arbol_implicito(S_tree, P_path, K, r, T, tipo='call'):
    """
    Valora una opción usando el árbol binomial implícito construido.
    
    A diferencia de Black-Scholes (que asume lognormalidad), este método
    usa directamente las probabilidades de trayectoria extraídas del mercado,
    permitiendo distribuciones no-lognormales (múltiples picos, asimetrías).
    
    Args:
        S_tree: Matriz (N+1, N+1) con precios en cada nodo
        P_path: Matriz (N+1, N+1) con probabilidades de trayectoria risk-neutral
        K: Strike de la opción
        r: Tasa libre de riesgo
        T: Tiempo hasta vencimiento (años)
        tipo: 'call' o 'put'
    
    Returns:
        Precio de la opción en t=0
    """
    N = S_tree.shape[0] - 1
    payoffs = np.zeros(N + 1)
    
    # Calcular payoffs en el vencimiento
    for j in range(N + 1):
        S_final = S_tree[N, j]
        if S_final > 0:  # Nodo válido
            if tipo == 'call':
                payoffs[j] = max(S_final - K, 0)
            else:  # put
                payoffs[j] = max(K - S_final, 0)
    
    # Valor esperado bajo medida Q implícita, descontado
    precio = np.sum(P_path[N, :] * payoffs) * np.exp(-r * T)
    
    return precio


def validar_arbol_implicito(df_vencimiento, S_tree, P_path, S0, r, T):
    """
    Valida que el árbol implícito reproduzca los precios de mercado.
    
    Esta validación NO usa Black-Scholes (que impondría lognormalidad),
    sino que verifica directamente que las probabilidades extraídas
    del smile reproduzcan los precios observados.
    
    Returns:
        DataFrame con comparación strike por strike
    """
    N = S_tree.shape[0] - 1
    resultados = []
    
    for _, row in df_vencimiento.iterrows():
        K = row['Strike']
        
        # Precio de mercado (mid de call)
        call_mid = (row['Call_Bid'] + row['Call_Ask']) / 2
        
        # Valorar con el árbol implícito
        try:
            call_arbol = valorar_opcion_con_arbol_implicito(
                S_tree, P_path, K, r, T, tipo='call'
            )
            
            error_abs = abs(call_arbol - call_mid)
            error_rel = (error_abs / call_mid * 100) if call_mid > 0 else np.nan
            
            resultados.append({
                'Strike': K,
                'Mercado': call_mid,
                'Árbol_Implícito': call_arbol,
                'Error_Abs': error_abs,
                'Error_Rel_%': error_rel
            })
        except Exception as e:
            print(f"    [!] Error valorando K={K}: {e}")
    
    return pd.DataFrame(resultados)


def plotear_arbol(S_tree, P_path, S0, T, nombre_empresa, dias, max_niveles=None):
    """
    Visualiza la evolución de la distribución como superficie de contorno 3D.
    Usa escala logarítmica en probabilidad para visualizar mejor nodos extremos.
    
    - Eje X: Pasos temporales (N)
    - Eje Y: Precio del activo ($)
    - Eje Z: Log(Densidad de probabilidad) para mejor visualización
    - Color: Precio (viridis)
    
    Args:
        S_tree: Matriz de precios (N+1, N+1)
        P_path: Matriz de probabilidades de trayectoria (N+1, N+1)
        S0: Precio spot inicial
        T: Tiempo total (años)
        nombre_empresa: Nombre para el título
        dias: Días hasta vencimiento
        max_niveles: Número máximo de niveles a mostrar (None = todos)
    """
    from mpl_toolkits.mplot3d import Axes3D
    from scipy.ndimage import gaussian_filter1d
    from scipy.interpolate import griddata
    
    N = S_tree.shape[0] - 1
    N_plot = N if max_niveles is None else min(N, max_niveles)
    
    print(f"        [*] Generando superficie 3D con escala logarítmica ({N_plot+1} pasos)...")
    
    # Grid de precios común
    S_min = S_tree[S_tree > 0].min()
    S_max = S_tree[S_tree > 0].max()
    S_grid = np.linspace(S_min, S_max, 200)
    
    # Seleccionar pasos temporales (submuestreo)
    step = max(1, N_plot // 80)  # Máximo 80 pasos temporales
    time_steps = list(range(0, N_plot + 1, step))
    
    # Construir matriz 2D de densidades
    Z_matrix = np.zeros((len(S_grid), len(time_steps)))
    
    for idx_t, n in enumerate(time_steps):
        # Extraer nodos válidos
        mask = S_tree[n, :] > 0
        s_vals = S_tree[n, mask]
        p_vals = P_path[n, mask]
        
        if len(s_vals) < 2:
            continue
        
        # Ordenar
        orden = np.argsort(s_vals)
        s_vals = s_vals[orden]
        p_vals = p_vals[orden]
        
        # Interpolar al grid
        pdf = np.interp(S_grid, s_vals, p_vals, left=0, right=0)
        
        # Suavizar
        sigma_smooth = max(0.5, len(s_vals) * 0.008)
        pdf = gaussian_filter1d(pdf, sigma=sigma_smooth)
        
        # Normalizar
        area = integrate_func(pdf, S_grid)
        if area > 1e-9:
            pdf = pdf / area
        
        # Guardar en matriz
        Z_matrix[:, idx_t] = pdf
    
    # Aplicar transformación logarítmica para mejor visualización
    # log(pdf + epsilon) para evitar log(0)
    epsilon = 1e-8
    Z_log = np.log10(Z_matrix + epsilon)
    
    # Crear mallas para superficie
    X_mesh, Y_mesh = np.meshgrid(time_steps, S_grid)
    
    # Crear figura 3D
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Superficie con color por PRECIO (eje Y)
    # Normalizar colores por precio
    norm = plt.Normalize(vmin=S_grid.min(), vmax=S_grid.max())
    colors = plt.cm.viridis(norm(Y_mesh))
    
    # Plot de superficie
    surf = ax.plot_surface(X_mesh, Y_mesh, Z_log,
                          facecolors=colors,
                          shade=True,
                          alpha=0.85,
                          antialiased=True,
                          linewidth=0,
                          rasterized=True)
    
    # Línea de precio actual
    z_min = Z_log[~np.isinf(Z_log)].min() if np.any(~np.isinf(Z_log)) else -8
    ax.plot([0, N_plot], [S0, S0], [z_min, z_min],
           color='red', linewidth=4, linestyle='--',
           label=f'Precio actual: ${S0:.2f}', zorder=1000)
    
    # Etiquetas
    ax.set_xlabel('Pasos temporales (N)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('Precio de la acción ($)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_zlabel('log₁₀(Densidad de probabilidad)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title(f'Superficie de Probabilidad 3D: {nombre_empresa} ({dias} días)\n'
                f'Escala logarítmica para visualizar nodos extremos',
                fontsize=14, fontweight='bold', pad=20)
    
    # Colorbar manual para precios
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.1, shrink=0.6, aspect=15)
    cbar.set_label('Precio del activo ($)', rotation=270, labelpad=20,
                   fontsize=11, fontweight='bold')
    
    # Marca de S0 en colorbar
    cbar.ax.axhline(S0, color='red', linewidth=2, linestyle='--', alpha=0.8)
    cbar.ax.text(1.5, S0, 'Actual', va='center', ha='left',
                fontsize=9, color='red', fontweight='bold')
    
    # Leyenda
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    
    # Ángulo de vista óptimo
    ax.view_init(elev=25, azim=135)
    
    # Grid
    ax.grid(True, alpha=0.25, linestyle=':', linewidth=0.5)
    
    # Paneles semi-transparentes
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('gray')
    ax.yaxis.pane.set_edgecolor('gray')
    ax.zaxis.pane.set_edgecolor('gray')
    
    plt.tight_layout()
    
    # Guardar
    archivo = f'{nombre_empresa}_superficie_3d_{dias}d.png'
    plt.savefig(archivo, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"        [✓] Superficie 3D guardada: {archivo}")
    
    plt.close()


def construir_smile(df_t, S0):
    """
    Interpolación del smile con extrapolación FLAT en los extremos.

    - >= 4 puntos : CubicSpline (reemplaza interp1d deprecada de SciPy 1.14+)
    - < 4 puntos  : interpolación lineal mediante make_interp_spline(k=1)

    La extrapolación flat evita IV negativas o explosivas fuera del rango
    de strikes observados, lo que produciría nodos colapsados en el árbol.
    """
    strikes = df_t['Strike'].values.astype(float)
    ivs     = df_t['IV'].values.astype(float)
    iv_izq, iv_der = ivs[0], ivs[-1]

    if len(df_t) >= 4:
        spline = CubicSpline(strikes, ivs, extrapolate=False)
    else:
        spline = make_interp_spline(strikes, ivs, k=1)

    def smile_func(K_query):
        K_arr    = np.atleast_1d(np.asarray(K_query, dtype=float))
        result   = spline(K_arr)
        # Extrapolación flat manual (CubicSpline devuelve NaN fuera del rango)
        result   = np.where(np.isnan(result) | (K_arr < strikes[0]),
                            np.where(K_arr < strikes[0], iv_izq, iv_der),
                            result)
        result   = np.where(K_arr > strikes[-1], iv_der, result)
        # Garantía: IV siempre positiva
        result   = np.clip(result, 0.01, 3.0)
        return float(result[0]) if np.ndim(K_query) == 0 else result

    return smile_func


def obtener_tasa_libre_riesgo():
    """
    Intenta obtener la tasa libre de riesgo actual (T-Bill 13 semanas, ^IRX)
    desde Yahoo Finance. Si falla, usa un fallback de 0.04 con aviso.
    """
    try:
        tbill = yf.Ticker("^IRX")
        hist  = tbill.history(period="5d")
        if not hist.empty:
            r = hist['Close'].iloc[-1] / 100.0
            print(f"[*] Tasa libre de riesgo (^IRX): {r:.4f} ({r*100:.2f}%)")
            return r
    except Exception:
        pass
    r_fallback = 0.04
    print(f"[!] No se pudo obtener ^IRX. Usando r={r_fallback:.2f} (fallback).")
    return r_fallback


def validar_csv(df, columnas_requeridas):
    """
    Valida que el DataFrame tenga las columnas necesarias y que
    DaysToExpiration contenga al menos un valor positivo.
    Lanza ValueError con mensaje descriptivo si algo falla.
    """
    faltantes = [c for c in columnas_requeridas if c not in df.columns]
    if faltantes:
        raise ValueError(
            f"El CSV no tiene las columnas requeridas: {faltantes}\n"
            f"Columnas encontradas: {list(df.columns)}"
        )
    if df['DaysToExpiration'].le(0).all():
        raise ValueError("DaysToExpiration no contiene valores positivos.")
    return True


# =====================================================================
# FUNCIÓN PRINCIPAL: ANÁLISIS DE UNA EMPRESA
# =====================================================================

def analizar_empresa(nombre_empresa, ticker_symbol, archivo_csv=None, N=300, 
                     validar_precios=True, plotear_arbol_bool=False, max_niveles_arbol=None):
    """
    Función genérica que analiza una empresa y genera su gráfico.

    Args:
        nombre_empresa : Nombre corto (ej: "XOM")
        ticker_symbol  : Ticker de Yahoo Finance (ej: "XOM")
        archivo_csv    : Ruta del CSV. Si None, usa "datos_{nombre_empresa}.csv"
        N              : Pasos del árbol (100 es buen equilibrio velocidad/precisión)
        validar_precios: Si True, valida que el árbol reproduzca precios de mercado
        plotear_arbol_bool: Si True, genera visualización del árbol binomial
        max_niveles_arbol: Niveles máximos a mostrar en plot (None = todos)
    """

    COLUMNAS_REQUERIDAS = [
        'DaysToExpiration', 'Strike',
        'Call_Bid', 'Call_Ask',
        'Put_Bid',  'Put_Ask'
    ]

    if archivo_csv is None:
        archivo_csv = f"datos_{nombre_empresa}.csv"

    print(f"\n{'='*60}")
    print(f"[*] Analizando : {nombre_empresa}")
    print(f"[*] Archivo    : {archivo_csv}")
    print(f"[*] Pasos árbol: {N}")
    print(f"[*] Validación : {'SÍ' if validar_precios else 'NO'}")
    print(f"[*] Plot árbol : {'SÍ' if plotear_arbol_bool else 'NO'}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # 1. CARGAR DATOS
    # ------------------------------------------------------------------
    # CORRECCIÓN: r se obtiene dinámicamente del mercado, no hardcodeado
    r = obtener_tasa_libre_riesgo()

    try:
        ticker_yf = yf.Ticker(ticker_symbol)
        hist      = ticker_yf.history(period="5d")   # Más robusto que "1d"
        if hist.empty:
            raise ValueError(f"No se obtuvieron datos de precio para '{ticker_symbol}'.")
        S0 = float(hist['Close'].iloc[-1])
        print(f"[*] Spot Actual (S0): ${S0:.2f}")

        df_master = pd.read_csv(archivo_csv)
        # Limpiar espacios en nombres de columnas
        df_master.columns = [c.strip() for c in df_master.columns]

        # CORRECCIÓN: validación temprana del CSV con mensaje descriptivo
        validar_csv(df_master, COLUMNAS_REQUERIDAS)

        # Filtrar vencimientos inválidos (días <= 0 no tienen sentido)
        df_master = df_master[df_master['DaysToExpiration'] > 0].copy()
        print(f"[✓] Datos cargados: {len(df_master)} filas")

    except ValueError as e:
        print(f"[!] Error de validación: {e}")
        return False
    except Exception as e:
        print(f"[!] Error cargando datos: {e}")
        return False

    S_grid       = np.linspace(S0 * 0.6, S0 * 1.4, 300)
    vencimientos = sorted(df_master['DaysToExpiration'].unique())

    # ------------------------------------------------------------------
    # 2. CALIBRACIÓN DEL SMILE POR VENCIMIENTO
    # ------------------------------------------------------------------
    all_pdfs  = []
    v_validos = []
    
    # Almacenar tablas de validación
    tablas_validacion = {}

    for dias in vencimientos:
        T    = dias / 365.0
        df_t = df_master[df_master['DaysToExpiration'] == dias].copy()

        # Mid: call para K >= S0 (mayor liquidez ATM/OTM), put para K < S0
        df_t['Mid'] = np.where(
            df_t['Strike'] >= S0,
            (df_t['Call_Bid'] + df_t['Call_Ask']) / 2,
            (df_t['Put_Bid']  + df_t['Put_Ask'])  / 2
        )

        # CORRECCIÓN: descartar filas con bid/ask inválidos o spread nulo
        df_t = df_t[
            (df_t['Call_Bid'] >= 0) & (df_t['Call_Ask'] > 0) &
            (df_t['Put_Bid']  >= 0) & (df_t['Put_Ask']  > 0) &
            (df_t['Mid'] > 0)
        ].copy()

        # IV con el tipo de opción coherente con el Mid calculado arriba
        df_t['IV'] = df_t.apply(
            lambda x: solve_iv(
                S0, x['Strike'], T, r, x['Mid'],
                'c' if x['Strike'] >= S0 else 'p'
            ),
            axis=1
        )
        df_t = df_t.dropna(subset=['IV'])

        # CORRECCIÓN: filtrar IVs fuera de rango razonable (< 1% o > 300%)
        df_t = df_t[(df_t['IV'] >= 0.01) & (df_t['IV'] <= 3.0)]

        # Ordenar por strike (necesario para splines y np.interp)
        df_t = df_t.sort_values('Strike').reset_index(drop=True)

        # CORRECCIÓN: eliminar duplicados de strike (rompen CubicSpline)
        df_t = df_t.drop_duplicates(subset='Strike').reset_index(drop=True)

        if len(df_t) < 3:
            print(f"    [!] T={dias}d: menos de 3 strikes válidos, se omite.")
            continue

        print(f"    [+] Calibrando T={dias}d con {len(df_t)} strikes...")

        try:
            # CORRECCIÓN: construir_smile ya no usa interp1d deprecada
            smile   = construir_smile(df_t, S0)
            sig_atm = float(smile(S0))
            sig_atm = np.clip(sig_atm, 0.01, 3.0)

            dt      = T / N
            df_step = np.exp(r * dt)

            # ----------------------------------------------------------
            # 3. CONSTRUCCIÓN DEL ÁRBOL — PASADA 1: malla de precios
            #    Se completa ENTERA antes de calcular probabilidades.
            #    Esto elimina el error de leer nodos aún no escritos.
            # ----------------------------------------------------------
            S_tree = np.zeros((N + 1, N + 1))
            S_tree[0, 0] = S0

            for n in range(N):
                k      = n + 1
                F_next = S0 * np.exp(r * k * dt)

                if k % 2 == 0:
                    mid = k // 2
                    S_tree[k, mid] = F_next
                    for j in range(mid, k):
                        vol = float(smile(S_tree[k - 1, j]))
                        S_tree[k, j + 1] = S_tree[k, j] * np.exp(vol * np.sqrt(dt))
                    for j in range(mid, 0, -1):
                        vol = float(smile(S_tree[k - 1, j - 1]))
                        S_tree[k, j - 1] = S_tree[k, j] * np.exp(-vol * np.sqrt(dt))
                else:
                    up_i, dn_i = (k + 1) // 2, (k - 1) // 2
                    S_tree[k, up_i] = F_next * np.exp(sig_atm * np.sqrt(dt))
                    S_tree[k, dn_i] = F_next * np.exp(-sig_atm * np.sqrt(dt))
                    for j in range(up_i, k):
                        vol = float(smile(S_tree[k - 1, j]))
                        S_tree[k, j + 1] = S_tree[k, j] * np.exp(vol * np.sqrt(dt))
                    for j in range(dn_i, 0, -1):
                        vol = float(smile(S_tree[k - 1, j - 1]))
                        S_tree[k, j - 1] = S_tree[k, j] * np.exp(-vol * np.sqrt(dt))

            # ----------------------------------------------------------
            # PASADA 2: probabilidades de trayectoria
            #    Ahora S_tree está completamente relleno → up y dn
            #    siempre son valores válidos, nunca cero.
            # ----------------------------------------------------------
            P_path = np.zeros((N + 1, N + 1))
            P_path[0, 0] = 1.0

            for n in range(N):
                for j in range(n + 1):
                    if P_path[n, j] == 0.0:
                        continue  # Nodo no alcanzable → no contribuye
                    up     = S_tree[n + 1, j + 1]
                    dn     = S_tree[n + 1, j]
                    spread = up - dn
                    if spread < 1e-10:
                        # Nodos demasiado juntos (colapso numérico): distribuir 50/50
                        q = 0.5
                    else:
                        q = (S_tree[n, j] * df_step - dn) / spread
                        q = np.clip(q, 0.01, 0.99)
                    P_path[n + 1, j]     += P_path[n, j] * (1.0 - q)
                    P_path[n + 1, j + 1] += P_path[n, j] * q

            # ----------------------------------------------------------
            # VALIDACIÓN: Verificar que el árbol reproduce precios
            # ----------------------------------------------------------
            if validar_precios:
                print(f"        [*] Validando precios observados vs. árbol implícito...")
                df_validacion = validar_arbol_implicito(df_t, S_tree, P_path, S0, r, T)
                
                if not df_validacion.empty:
                    mae = df_validacion['Error_Abs'].mean()
                    rmse = np.sqrt((df_validacion['Error_Abs'] ** 2).mean())
                    error_rel_mean = df_validacion['Error_Rel_%'].mean()
                    
                    # Determinar estado según error
                    if error_rel_mean < 5.0:
                        estado = "✓✓✓"
                    elif error_rel_mean < 10.0:
                        estado = "✓✓"
                    elif error_rel_mean < 20.0:
                        estado = "✓"
                    else:
                        estado = "⚠"
                    
                    print(f"        [✓] MAE = ${mae:.4f} | RMSE = ${rmse:.4f} | Error = {error_rel_mean:.2f}% {estado}")
                    
                    # Almacenar resultados
                    tablas_validacion[dias] = {
                        'df': df_validacion,
                        'mae': mae,
                        'rmse': rmse,
                        'error_rel_mean': error_rel_mean,
                        'n_strikes': len(df_validacion)
                    }
                    
                    # Guardar tabla detallada si hay error alto
                    if error_rel_mean > 10.0:
                        print(f"        [!] Error > 10%, mostrando detalles:")
                        print(df_validacion[['Strike', 'Mercado', 'Árbol_Implícito', 'Error_Rel_%']].head(10).to_string(index=False))

            # ----------------------------------------------------------
            # PLOT DEL ÁRBOL BINOMIAL
            # ----------------------------------------------------------
            if plotear_arbol_bool:
                print(f"        [*] Generando visualización del árbol...")
                plotear_arbol(S_tree, P_path, S0, T, nombre_empresa, dias, max_niveles_arbol)

            # ----------------------------------------------------------
            # MAPEO A LA PDF CONTINUA
            #    Filtrar nodos válidos del paso final antes de np.interp.
            #    S_tree[N,:] tiene ceros en posiciones no usadas;
            #    np.interp requiere x estrictamente creciente.
            # ----------------------------------------------------------
            mask    = S_tree[N, :] > 0
            s_final = S_tree[N, mask]
            p_final = P_path[N, mask]

            # Ordenar por precio (garantía adicional)
            orden   = np.argsort(s_final)
            s_final = s_final[orden]
            p_final = p_final[orden]

            # CORRECCIÓN: eliminar duplicados en s_final antes de np.interp
            # (nodos colapsados producirían un x no estrictamente creciente)
            _, idx_uniq = np.unique(s_final, return_index=True)
            s_final = s_final[idx_uniq]
            p_final = p_final[idx_uniq]

            pdf = np.interp(S_grid, s_final, p_final, left=0.0, right=0.0)

            # Suavizado adaptativo: sigma proporcional al ancho efectivo
            # de la distribución para no sobre/sub-suavizar entre vencimientos
            n_activos    = max(1, np.sum(pdf > np.max(pdf) * 0.01))
            sigma_smooth = max(1.0, n_activos * 0.008)
            pdf = gaussian_filter1d(pdf, sigma=sigma_smooth)

            area = integrate_func(pdf, S_grid)
            if area > 1e-9:
                all_pdfs.append(pdf / area)
                v_validos.append(dias)
            else:
                print(f"    [!] T={dias}d: área nula tras suavizado, se omite.")

            # Liberar memoria explícitamente (útil con N grande)
            del S_tree, P_path

        except (ValueError, RuntimeError) as e:
            print(f"    [!] Error numérico en T={dias}d: {e}")
            print(traceback.format_exc())
        except Exception as e:
            print(f"    [!] Error inesperado en T={dias}d: {e}")
            print(traceback.format_exc())

    # ------------------------------------------------------------------
    # 3.5 REPORTE DE VALIDACIÓN
    # ------------------------------------------------------------------
    if validar_precios and tablas_validacion:
        print(f"\n{'='*70}")
        print("REPORTE DE VALIDACIÓN: ÁRBOL IMPLÍCITO vs. PRECIOS DE MERCADO")
        print(f"{'='*70}\n")
        
        # Tabla resumen
        print(f"{'Vencimiento':<15} {'Strikes':<10} {'MAE ($)':<12} {'RMSE ($)':<12} {'Error%':<10} {'Estado':<8}")
        print("-" * 70)
        
        for dias in sorted(tablas_validacion.keys()):
            data = tablas_validacion[dias]
            mae = data['mae']
            rmse = data['rmse']
            error = data['error_rel_mean']
            n = data['n_strikes']
            
            if error < 5.0:
                estado = "✓✓✓"
            elif error < 10.0:
                estado = "✓✓"
            elif error < 20.0:
                estado = "✓"
            else:
                estado = "⚠"
            
            print(f"{dias:>3} días         {n:<10} {mae:<12.4f} {rmse:<12.4f} {error:<10.2f} {estado:<8}")
        
        # Estadísticas globales
        errores_globales = [v['error_rel_mean'] for v in tablas_validacion.values()]
        error_global_mean = np.mean(errores_globales)
        error_global_std = np.std(errores_globales)
        
        print("-" * 70)
        print(f"{'PROMEDIO GLOBAL':<15} {'':<10} {'':<12} {'':<12} {error_global_mean:<10.2f}")
        print(f"{'DESV. ESTÁNDAR':<15} {'':<10} {'':<12} {'':<12} {error_global_std:<10.2f}")
        print()
        
        # Interpretación
        print("INTERPRETACIÓN:")
        if error_global_mean < 5.0:
            print("  ✓✓✓ Excelente: El árbol implícito reproduce fielmente los precios de mercado")
        elif error_global_mean < 10.0:
            print("  ✓✓ Bueno: El árbol captura la estructura del smile con error aceptable")
        elif error_global_mean < 20.0:
            print("  ✓ Aceptable: El árbol captura la forma general, con desviaciones moderadas")
        else:
            print("  ⚠ Revisar: Errores elevados, posible problema en calibración o datos")
        
        print()
        print("NOTA: Este árbol permite distribuciones NO-LOGNORMALES (múltiples picos,")
        print("      asimetrías) extraídas directamente del smile de volatilidad observado.")
        print(f"{'='*70}\n")
    
    # ------------------------------------------------------------------
    # 4. PLOTEO CON AUTOFOCUS Y LEYENDA INTEGRADA
    # ------------------------------------------------------------------
    if not all_pdfs:
        print("[!] Error fatal: No se pudo generar ninguna distribución.")
        return False

    print(f"\n[✓] Se generaron {len(all_pdfs)} distribuciones\n")

    fig, ax = plt.subplots(figsize=(14, 9))
    colors  = plt.cm.turbo(np.linspace(0.05, 0.95, len(all_pdfs)))

    # Zoom dinámico
    suma_pdfs       = np.sum(all_pdfs, axis=0)
    umbral          = np.max(suma_pdfs) * 0.001
    indices_activos = np.where(suma_pdfs > umbral)[0]

    if len(indices_activos) > 0:
        x_min_real = S_grid[indices_activos[0]]
        x_max_real = S_grid[indices_activos[-1]]
    else:
        x_min_real, x_max_real = S_grid[0], S_grid[-1]

    rango_real = x_max_real - x_min_real
    x_min_plot = x_min_real - rango_real * 0.03
    x_max_plot = x_max_real + rango_real * 0.05

    # Offset basado en el máximo global para separación uniforme entre curvas
    max_global = max(np.max(p) for p in all_pdfs)

    for i, (pdf, d) in enumerate(zip(all_pdfs, v_validos)):
        off = i * (max_global * 0.3)

        ax.fill_between(S_grid, off, pdf + off,
                        color=colors[i], alpha=0.75,
                        zorder=len(all_pdfs) - i, label=f"{d} días")
        ax.plot(S_grid, pdf + off,
                color='white', lw=0.8, zorder=len(all_pdfs) - i)

        # Anotar los dos picos más altos
        maxima_indices = argrelextrema(pdf, np.greater, order=5)[0]
        if len(maxima_indices) > 0:
            top_two_idx = maxima_indices[np.argsort(-pdf[maxima_indices])[:2]]
            top_two_idx = np.sort(top_two_idx)

            for idx in top_two_idx:
                price    = S_grid[idx]
                peak_val = pdf[idx]
                y_pos    = peak_val + off

                ax.plot(price, y_pos, 'o', color='white', markersize=5,
                        zorder=len(all_pdfs) - i + 1,
                        markeredgecolor='black', markeredgewidth=0.8)
                ax.plot([x_min_plot, price], [y_pos, y_pos],
                        ls=':', color=colors[i], alpha=0.8, zorder=0)
                ax.text(
                    x_min_real - rango_real * 0.01, y_pos,
                    f'${price:.2f}',
                    va='center', ha='right', fontsize=9,
                    fontweight='bold', color=colors[i],
                    bbox=dict(facecolor='white', alpha=0.8,
                              edgecolor='none', pad=1)
                )

    ax.axvline(S0, color='red', ls='--', linewidth=2.5, alpha=0.8,
               label=f'Precio Actual: ${S0:.2f}', zorder=10)

    ax.set_title(
        f"Evolución de Expectativas del Mercado: {nombre_empresa} (${S0:.2f})",
        fontsize=14, fontweight='bold'
    )
    ax.set_xlabel("Precio de la Acción ($)", fontsize=11, fontweight='bold')
    ax.set_yticks([])
    ax.set_xlim(x_min_plot, x_max_plot)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.legend(title="Vencimientos", fontsize=10, title_fontsize=11,
              loc='lower right', framealpha=1, edgecolor='gray', borderpad=1)

    plt.tight_layout()
    archivo_salida = f'{nombre_empresa}_analisis.png'
    plt.savefig(archivo_salida, dpi=300, bbox_inches='tight')
    print(f"[✓] Gráfico guardado: {archivo_salida}")

    plt.show()
    return True


# =====================================================================
# SI SE EJECUTA DIRECTAMENTE
# =====================================================================
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Uso: python algoritmo.py <nombre_empresa> [ticker_symbol] [N_pasos] [--sin-validacion] [--plot-arbol]")
        print("Ej:  python algoritmo.py XOM XOM 300")
        print("     python algoritmo.py XOM XOM 300 --plot-arbol")
        print("     python algoritmo.py XOM XOM 300 --sin-validacion --plot-arbol")
        sys.exit(1)

    nombre_empresa = sys.argv[1]
    ticker_symbol  = sys.argv[2] if len(sys.argv) > 2 else nombre_empresa
    N_pasos        = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else 300
    validar        = '--sin-validacion' not in sys.argv
    plot_arbol     = '--plot-arbol' in sys.argv

    analizar_empresa(nombre_empresa, ticker_symbol, N=N_pasos, 
                    validar_precios=validar, plotear_arbol_bool=plot_arbol)