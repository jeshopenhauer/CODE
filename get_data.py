import yfinance as yf
import pandas as pd
import datetime
import warnings
warnings.filterwarnings('ignore')

# Importar configuración
from config import EMPRESAS, DIAS_MINIMOS, DIAS_MAXIMOS, RUTA_CSV

print(f"\n{'='*60}")
print(f"[*] MOTOR DE EXTRACCIÓN DE SUPERFICIE 3D - MULTI EMPRESA")
print(f"[*] Total de empresas configuradas: {len(EMPRESAS)}")
print(f"[*] Rango temporal: {DIAS_MINIMOS} a {DIAS_MAXIMOS} días")
print(f"{'='*60}\n")

# =====================================================================
# FUNCIÓN GENÉRICA DE DESCARGA
# =====================================================================
def descargar_datos_empresa(nombre_empresa, ticker_symbol):
    """
    Descarga datos de opciones para una empresa específica.
    Retorna True si éxito, False si falla.
    """
    print(f"\n[*] Procesando: {nombre_empresa} ({ticker_symbol})")
    print("-" * 50)
    
    try:
        # Conexión y obtención del Spot
        print(f"    [*] Conectando con Yahoo Finance...")
        activo = yf.Ticker(ticker_symbol)
        
        spot_price = activo.history(period="1d")['Close'].iloc[-1]
        print(f"    [✓] Precio Spot actual: ${spot_price:.2f}")
        
    except Exception as e:
        print(f"    [!] Error al obtener el Spot: {e}")
        return False
    
    try:
        # Obtener vencimientos disponibles
        fechas_disponibles = activo.options
        if not fechas_disponibles:
            print(f"    [!] No hay opciones disponibles para {ticker_symbol}.")
            return False
        
        fecha_hoy = datetime.date.today()
        vencimientos_filtrados = []
        
        # Filtrar por rango de días
        for fecha_str in fechas_disponibles:
            f_date = datetime.datetime.strptime(fecha_str, "%Y-%m-%d").date()
            dias_restantes = (f_date - fecha_hoy).days
            
            if DIAS_MINIMOS <= dias_restantes <= DIAS_MAXIMOS:
                vencimientos_filtrados.append((fecha_str, dias_restantes))
        
        if not vencimientos_filtrados:
            print(f"    [!] No hay vencimientos en el rango {DIAS_MINIMOS}-{DIAS_MAXIMOS} días.")
            return False
        
        print(f"    [✓] {len(vencimientos_filtrados)} vencimientos encontrados")
        
    except Exception as e:
        print(f"    [!] Error obteniendo vencimientos: {e}")
        return False
    
    try:
        # Procesar cada vencimiento
        lista_dataframes = []
        
        for fecha_str, dias_restantes in vencimientos_filtrados:
            try:
                cadena = activo.option_chain(fecha_str)
                calls = cadena.calls.copy()
                puts = cadena.puts.copy()
                
                # Renombrar columnas
                calls = calls.rename(columns={'bid': 'Call_Bid', 'ask': 'Call_Ask', 'lastPrice': 'Call_Last'})
                puts = puts.rename(columns={'bid': 'Put_Bid', 'ask': 'Put_Ask', 'lastPrice': 'Put_Last'})
                
                # Cruzar datos
                opciones_df = pd.merge(
                    calls[['strike', 'Call_Bid', 'Call_Ask', 'Call_Last']],
                    puts[['strike', 'Put_Bid', 'Put_Ask', 'Put_Last']],
                    on='strike',
                    how='inner'
                )
                opciones_df = opciones_df.rename(columns={'strike': 'Strike'})
                
                # ================================================================
                # CORRECCIÓN: Imputación de liquidez con SPREAD REALISTA
                # ================================================================
                for opt_type in ['Call', 'Put']:
                    # Rellenar NaN con 0
                    opciones_df[f'{opt_type}_Bid'] = opciones_df[f'{opt_type}_Bid'].fillna(0)
                    opciones_df[f'{opt_type}_Ask'] = opciones_df[f'{opt_type}_Ask'].fillna(0)
                    opciones_df[f'{opt_type}_Last'] = opciones_df[f'{opt_type}_Last'].fillna(0)
                    
                    # Detectar casos donde tenemos Last pero no Bid/Ask
                    mask_sin_cotizacion = (
                        (opciones_df[f'{opt_type}_Bid'] <= 0) | 
                        (opciones_df[f'{opt_type}_Ask'] <= 0)
                    ) & (opciones_df[f'{opt_type}_Last'] > 0)
                    
                    if mask_sin_cotizacion.any():
                        # SOLUCIÓN: Reconstruir spread realista
                        # Spread típico en opciones: 2-5% del precio mid
                        # Para opciones baratas (< $1): spread mínimo $0.05
                        # Para opciones caras: spread 3% del precio
                        
                        last_price = opciones_df.loc[mask_sin_cotizacion, f'{opt_type}_Last']
                        
                        # Spread adaptativo
                        spread_pct = 0.03  # 3% por defecto
                        spread_min = 0.05  # $0.05 mínimo
                        
                        spread_absoluto = last_price * spread_pct
                        spread_absoluto = spread_absoluto.clip(lower=spread_min)
                        
                        # Reconstruir Bid/Ask centrado en Last
                        opciones_df.loc[mask_sin_cotizacion, f'{opt_type}_Bid'] = last_price - spread_absoluto / 2
                        opciones_df.loc[mask_sin_cotizacion, f'{opt_type}_Ask'] = last_price + spread_absoluto / 2
                        
                        # Garantizar Bid >= 0.01 (precio mínimo)
                        opciones_df[f'{opt_type}_Bid'] = opciones_df[f'{opt_type}_Bid'].clip(lower=0.01)
                
                # Filtro: solo mantener donde hay bid > 0 (liquidez mínima)
                opciones_df = opciones_df[
                    (opciones_df['Call_Bid'] > 0) & 
                    (opciones_df['Put_Bid'] > 0)
                ]
                
                if len(opciones_df) == 0:
                    print(f"    [!] Vencimiento {fecha_str}: sin strikes líquidos")
                    continue
                
                # Añadir metadata temporal
                opciones_df['ExpirationDate'] = fecha_str
                opciones_df['DaysToExpiration'] = dias_restantes
                
                # Reordenar columnas
                opciones_df = opciones_df[[
                    'ExpirationDate', 'DaysToExpiration', 'Strike', 
                    'Call_Bid', 'Call_Ask', 'Put_Bid', 'Put_Ask'
                ]]
                
                lista_dataframes.append(opciones_df)
                print(f"        {fecha_str}: {len(opciones_df)} strikes")
                
            except Exception as e:
                print(f"    [!] Error en vencimiento {fecha_str}: {e}")
                continue
        
        if not lista_dataframes:
            print(f"    [!] No se pudo procesar ningún vencimiento para {ticker_symbol}.")
            return False
        
        # Guardar archivo
        df_superficie = pd.concat(lista_dataframes, ignore_index=True)
        archivo_salida = RUTA_CSV.format(company=nombre_empresa)
        df_superficie.to_csv(archivo_salida, index=False)
        
        print(f"    [✓] ÉXITO. {len(df_superficie)} filas guardadas en '{archivo_salida}'")
        
        # NUEVO: Verificar calidad de datos
        spread_call = ((df_superficie['Call_Ask'] - df_superficie['Call_Bid']) / 
                      ((df_superficie['Call_Ask'] + df_superficie['Call_Bid']) / 2)).mean()
        spread_put = ((df_superficie['Put_Ask'] - df_superficie['Put_Bid']) / 
                     ((df_superficie['Put_Ask'] + df_superficie['Put_Bid']) / 2)).mean()
        
        print(f"    [i] Spread promedio: Call={spread_call:.1%}, Put={spread_put:.1%}")
        
        return True
        
    except Exception as e:
        print(f"    [!] Error crítico procesando empresa: {e}")
        return False

# =====================================================================
# EJECUTAR DESCARGA PARA TODAS LAS EMPRESAS ACTIVAS
# =====================================================================
resultados = {}
empresas_inactivas = {}

for empresa_key, empresa_info in EMPRESAS.items():
    # Verificar si la empresa está activa
    if not empresa_info.get("activo", True):
        empresas_inactivas[empresa_key] = "⊘ Deshabilitada en config"
        continue
    
    resultado = descargar_datos_empresa(empresa_key, empresa_info["ticker"])
    resultados[empresa_key] = "✓" if resultado else "✗"

# =====================================================================
# RESUMEN FINAL
# =====================================================================
print(f"\n{'='*60}")
print(f"[*] RESUMEN DE DESCARGA")
print(f"{'='*60}")

print("\n[EXITOSAS]")
for empresa, estado in resultados.items():
    if estado == "✓":
        print(f"    {estado} {empresa}")

print("\n[FALLIDAS]")
for empresa, estado in resultados.items():
    if estado == "✗":
        print(f"    {estado} {empresa}")

if empresas_inactivas:
    print("\n[DESHABILITADAS]")
    for empresa, razon in empresas_inactivas.items():
        print(f"    {razon}: {empresa}")

total_activas = len(resultados)
total_exitosas = sum(1 for v in resultados.values() if v == "✓")
print(f"\n{'='*60}")
print(f"[✓] Resumen: {total_exitosas}/{total_activas} activas descargadas correctamente")
print(f"{'='*60}\n")