"""
AgroInsight India - Tablero del estado de los datos (Proyecto Integrador de Ciencia de Datos, Fase 4).

Universidad de La Salle - Maestría en Inteligencia Artificial.
Dataset: Crop Yield in Indian States (Kaggle, akshatgupta7), registros 2015-2020.

El tablero cuenta la historia de los datos en este orden:
    1. Resumen            -> de dónde se partió, qué se quitó y con qué se terminó
    2. Datos originales   -> diagnóstico de calidad de dataset_raw.csv (nulos, ceros, espacios, atípicos)
    3. Datos finales      -> el conjunto limpio con el que se trabajó (responde a los filtros)
    4. Explorar registros -> tabla filtrable y descarga en CSV
    5. Uso en modelos     -> KPI del modelo, simulador de predicción y segmentos (módulos de la guía)

Ejecutar desde la raíz del repositorio:
    streamlit run dashboard/app.py

Entradas:
    data/raw/dataset_raw.csv (Colab 1),
    dashboard/assets/dashboard_data.parquet, dashboard/assets/dashboard_meta.json (Colab 4),
    models/best_regression_model.pkl, models/best_classification_model.pkl,
    models/kmeans_model.pkl, models/yield_class_labeler.pkl (Colabs 2 y 3)
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ------------------------------------------------------------------
# Rutas (relativas al archivo: funciona en Colab, local y Streamlit Cloud)
# ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))
import crop_features as cf  # noqa: E402  - necesario para deserializar los .pkl

ASSETS = ROOT / 'dashboard' / 'assets'
MODELS = ROOT / 'models'
RAW_PATH = ROOT / 'data' / 'raw' / 'dataset_raw.csv'

st.set_page_config(page_title='AgroInsight India', page_icon='🌾', layout='wide',
                   initial_sidebar_state='expanded')

# ------------------------------------------------------------------
# Paleta (validada para daltonismo) y estilo
# ------------------------------------------------------------------
AZUL, AMARILLO, AQUA, ROJO = '#2a78d6', '#eda100', '#1baf7a', '#e34948'
GRIS, TINTA, TINTA_2 = '#898781', '#0b0b0b', '#52514e'
COLORES_SEG = [AZUL, AMARILLO, AQUA, ROJO]
SIMBOLOS_SEG = ['circle', 'diamond', 'square', 'cross']
CLASES = ['Bajo', 'Medio', 'Alto']
COLOR_CLASE = {'Bajo': '#86b6ef', 'Medio': '#2a78d6', 'Alto': '#104281'}   # rampa ordinal azul
NOMBRE_CONJ = {'train': 'Entrenamiento', 'val': 'Validación', 'test': 'Prueba'}
NUM_RAW = ['Area', 'Production', 'Annual_Rainfall', 'Fertilizer', 'Pesticide', 'Yield']
CAT_RAW = ['Crop', 'Season', 'State']

st.markdown(f"""
<style>
.block-container {{padding-top: 1.6rem; padding-bottom: 2rem;}}
.hero {{background: linear-gradient(120deg, #104281 0%, #2a78d6 60%, #1baf7a 100%);
       color: #ffffff; padding: 1.1rem 1.5rem; border-radius: 14px; margin-bottom: 1rem;}}
.hero h1 {{color: #ffffff; font-size: 1.75rem; margin: 0 0 .2rem 0; padding: 0;}}
.hero p {{color: #eef4fc; margin: 0; font-size: .95rem;}}
div[data-testid="stMetric"] {{background: #fcfcfb; border: 1px solid rgba(11,11,11,.10);
       border-radius: 12px; padding: .7rem .9rem;}}
div[data-testid="stMetricLabel"] p {{color: #52514e; font-size: .85rem;}}
.flujo {{display: flex; flex-wrap: wrap; gap: .4rem; align-items: stretch; margin: .3rem 0 1rem 0;}}
.paso {{flex: 1 1 150px; background: #fcfcfb; border: 1px solid rgba(11,11,11,.10); border-radius: 12px;
       padding: .75rem .8rem; text-align: center;}}
.paso.base {{border-top: 5px solid {AZUL};}}
.paso.quita {{border-top: 5px solid {ROJO};}}
.paso.cero {{border-top: 5px solid {GRIS};}}
.paso.fin {{border-top: 5px solid {AQUA};}}
.paso .num {{font-size: 1.55rem; font-weight: 700; color: {TINTA}; line-height: 1.2;}}
.paso .txt {{font-size: .85rem; color: {TINTA_2}; margin-top: .2rem;}}
.flecha {{align-self: center; color: {GRIS}; font-size: 1.3rem;}}
.hallazgo {{border-left: 5px solid var(--c); background: #fcfcfb; border-radius: 10px;
       padding: .6rem .9rem; margin-bottom: .5rem; border-top: 1px solid rgba(11,11,11,.08);
       border-right: 1px solid rgba(11,11,11,.08); border-bottom: 1px solid rgba(11,11,11,.08);}}
.hallazgo b {{color: {TINTA};}}
.hallazgo span {{color: {TINTA_2}; font-size: .9rem;}}
.nota {{color: #52514e; font-size: .85rem;}}
</style>
""", unsafe_allow_html=True)


def estilo(fig, titulo=None, h=340, leyenda=True):
    """Estilo común: rejilla discreta, leyenda arriba, sin fondo."""
    fig.update_layout(
        height=h, margin=dict(l=8, r=8, t=56 if titulo else 12, b=8),
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', size=12, color=TINTA),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', barcornerradius=4,
        legend=dict(orientation='h', yanchor='bottom', y=1.0, x=0, title=None) if leyenda else None,
        showlegend=leyenda, hoverlabel=dict(font_size=12))
    if titulo:
        fig.update_layout(title=dict(text=titulo, x=0, font=dict(size=15)))
    fig.update_xaxes(showgrid=False, linecolor='#c3c2b7', title_font_color=TINTA_2)
    fig.update_yaxes(gridcolor='#e1e0d9', zeroline=False, title_font_color=TINTA_2)
    return fig


def grafico(fig):
    st.plotly_chart(fig, width='stretch', theme=None, config={'displaylogo': False})


def hallazgo(titulo, texto, color=AZUL):
    st.markdown(f'<div class="hallazgo" style="--c:{color}"><b>{titulo}</b><br><span>{texto}</span></div>',
                unsafe_allow_html=True)


def miles(n):
    """Entero con punto de miles (formato colombiano)."""
    return f'{int(n):,}'.replace(',', '.')


# ------------------------------------------------------------------
# Carga de datos y modelos (una sola vez por sesión del servidor)
# ------------------------------------------------------------------
@st.cache_data(show_spinner='Cargando datos...')
def cargar_final():
    df = pd.read_parquet(ASSETS / 'dashboard_data.parquet')
    meta = json.loads((ASSETS / 'dashboard_meta.json').read_text(encoding='utf-8'))
    return df, meta


@st.cache_data(show_spinner='Cargando datos originales...')
def cargar_original():
    return pd.read_csv(RAW_PATH) if RAW_PATH.exists() else None


@st.cache_resource(show_spinner='Cargando modelos...')
def cargar_modelos():
    return {
        'reg': joblib.load(MODELS / 'best_regression_model.pkl'),
        'cls': joblib.load(MODELS / 'best_classification_model.pkl'),
        'km': joblib.load(MODELS / 'kmeans_model.pkl'),
        'labeler': joblib.load(MODELS / 'yield_class_labeler.pkl'),
    }


def pct_atipicos(s):
    """% de valores fuera de [Q1 - 1,5·IQR, Q3 + 1,5·IQR]."""
    s = s.dropna()
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return float(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).mean() * 100)


@st.cache_data
def diagnostico(tabla):
    """Una fila por columna: tipo, nulos, ceros, textos con espacios, atípicos y asimetría."""
    filas = []
    for c in tabla.columns:
        s = tabla[c]
        es_num = pd.api.types.is_numeric_dtype(s)
        cont = c in NUM_RAW
        filas.append({
            'Variable': c,
            'Tipo': 'Numérica' if es_num else 'Categórica',
            'Valores distintos': int(s.nunique()),
            'Nulos (%)': float(s.isna().mean() * 100),
            'Ceros': int((s == 0).sum()) if es_num else np.nan,
            'Textos con espacios': np.nan if es_num else int((s.astype(str) != s.astype(str).str.strip()).sum()),
            'Atípicos IQR (%)': pct_atipicos(s) if cont else np.nan,
            'Atípicos IQR en log (%)': pct_atipicos(np.log1p(s)) if cont else np.nan,
            'Asimetría': float(s.skew()) if cont else np.nan,
        })
    return pd.DataFrame(filas)


@st.cache_data
def limpieza(raw):
    """Reaplica las reglas fila a fila del Colab 2 para contar cuánto quitó cada paso."""
    d = raw.copy()
    espacios = int(sum((d[c].astype(str) != d[c].astype(str).str.strip()).sum() for c in CAT_RAW))
    for c in CAT_RAW:
        d[c] = d[c].astype(str).str.strip()
    dup = int(d.duplicated().sum())
    d = d.drop_duplicates()
    cero = (d['Production'] == 0) | (d['Yield'] == 0)
    n_cero = int(cero.sum())
    d = d[~cero]
    coco = d['Crop'] == 'Coconut'
    n_coco = int(coco.sum())
    d = d[~coco]
    return {'espacios': espacios, 'duplicados': dup, 'ceros': n_cero, 'coconut': n_coco, 'final': len(d)}


df, meta = cargar_final()
raw = cargar_original()
modelos = cargar_modelos()
SEGMENTOS = sorted(meta['segmentos'], key=lambda s: s['cluster'])
ORDEN_SEG = [s['nombre'] for s in SEGMENTOS]
COLOR_SEG = dict(zip(ORDEN_SEG, COLORES_SEG))
SIMBOLO_SEG = dict(zip(ORDEN_SEG, SIMBOLOS_SEG))
REG = pd.DataFrame(meta['regresion']['metricas'])
CLS = pd.DataFrame(meta['clasificacion']['metricas'])


def metrica(tabla, modelo, conjunto, col):
    fila = tabla[(tabla['modelo'] == modelo) & (tabla['conjunto'] == conjunto)]
    return float(fila[col].iloc[0]) if len(fila) else float('nan')


# ------------------------------------------------------------------
# Encabezado
# ------------------------------------------------------------------
r = meta['resumen']
st.markdown(f"""
<div class="hero">
  <h1>🌾 AgroInsight India · Estado de los datos</h1>
  <p>Cómo llegaron los datos, qué se corrigió y con qué datos se trabajó al final
  · Rendimiento de cultivos en India {r['anios'][0]}–{r['anios'][1]}
  · Fuente: Kaggle <i>Crop Yield in Indian States</i></p>
</div>""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Filtros dinámicos (barra lateral): aplican a los datos finales
# ------------------------------------------------------------------
sb = st.sidebar
sb.header('Filtros')
sb.caption('Aplican a «Datos finales», «Explorar registros» y a los segmentos. '
           '«Resumen» y «Datos originales» muestran siempre el conjunto completo.')
anio_min, anio_max = int(df['Crop_Year'].min()), int(df['Crop_Year'].max())
anios = sb.slider('Años de cosecha', anio_min, anio_max, (anio_min, anio_max))
temporadas = sb.multiselect('Temporada', sorted(df['Season'].unique()), placeholder='Todas')
grupos = sb.multiselect('Grupo de cultivo', sorted(df['Crop_group'].unique()), placeholder='Todos')
opciones_cultivo = sorted(df.loc[df['Crop_group'].isin(grupos) if grupos else slice(None), 'Crop'].unique())
cultivos = sb.multiselect('Cultivo', opciones_cultivo, placeholder='Todos')
estados = sb.multiselect('Estado', sorted(df['State'].unique()), placeholder='Todos')

mask = df['Crop_Year'].between(*anios)
for col, sel in [('Season', temporadas), ('Crop_group', grupos), ('Crop', cultivos), ('State', estados)]:
    if sel:
        mask &= df[col].isin(sel)
dff = df[mask]
sb.metric('Registros seleccionados', miles(len(dff)), f'{len(dff) / len(df):.0%} de los datos finales',
          delta_color='off')


def sin_datos():
    if dff.empty:
        st.warning('Ningún registro cumple los filtros. Amplíe la selección en la barra lateral.')
        return True
    return False


tab_res, tab_ori, tab_fin, tab_exp, tab_mod = st.tabs(
    ['📌 Resumen', '🔍 Datos originales', '✅ Datos finales', '📋 Explorar registros', '🧠 Uso en modelos'])

# ==================================================================
# Pestaña 1: Resumen (de los datos originales a los finales)
# ==================================================================
with tab_res:
    if raw is None:
        st.warning(f'No se encontró {RAW_PATH.relative_to(ROOT)}: la comparación con los datos originales '
                   'no está disponible. Copie el archivo generado por el Colab 1.')
    else:
        pasos = limpieza(raw)
        n_raw, n_fin = len(raw), len(df)
        quitados = n_raw - n_fin

        k = st.columns(4)
        k[0].metric('Registros originales', miles(n_raw), help='dataset_raw.csv: años 2015–2020')
        k[1].metric('Registros finales', miles(n_fin), f'−{miles(quitados)} eliminados', delta_color='off')
        k[2].metric('Datos conservados', f'{n_fin / n_raw:.1%}')
        k[3].metric('Valores de texto corregidos', miles(pasos['espacios']),
                    help='Espacios sobrantes al final de Crop, Season y State')

        st.markdown('##### Recorrido de los datos')
        tarjetas = [
            ('base', miles(n_raw), 'registros originales'),
            ('cero' if pasos['duplicados'] == 0 else 'quita', f'−{pasos["duplicados"]}', 'duplicados'),
            ('quita', f'−{pasos["ceros"]}', 'con producción o rendimiento = 0'),
            ('quita', f'−{pasos["coconut"]}', 'de coco (otra unidad de medida)'),
            ('fin', miles(n_fin), 'registros finales'),
        ]
        html = '<div class="flujo">'
        for i, (clase, num, txt) in enumerate(tarjetas):
            html += f'<div class="paso {clase}"><div class="num">{num}</div><div class="txt">{txt}</div></div>'
            if i < len(tarjetas) - 1:
                html += '<div class="flecha">→</div>'
        st.markdown(html + '</div>', unsafe_allow_html=True)
        if pasos['final'] != n_fin:
            st.caption(f'Aviso: al reaplicar las reglas quedan {miles(pasos["final"])} registros y los datos del '
                       f'tablero tienen {miles(n_fin)}. Revise que los archivos sean de la misma ejecución.')

        c1, c2 = st.columns([1.15, 1])
        with c1:
            st.markdown('##### Antes y después')
            d_raw, d_fin = diagnostico(raw), diagnostico(df[[c for c in raw.columns if c in df.columns]])
            y_raw, y_fin = d_raw.set_index('Variable').loc['Yield'], d_fin.set_index('Variable').loc['Yield']
            texto_raw = sum((raw[c].astype(str) != raw[c].astype(str).str.strip()).sum() for c in CAT_RAW)
            texto_fin = sum((df[c].astype(str) != df[c].astype(str).str.strip()).sum() for c in CAT_RAW)
            comp = pd.DataFrame([
                ('Registros', miles(n_raw), miles(n_fin)),
                ('Cultivos', raw['Crop'].nunique(), df['Crop'].nunique()),
                ('Estados', raw['State'].nunique(), df['State'].nunique()),
                ('Valores nulos', int(raw.isna().sum().sum()), int(df[raw.columns.intersection(df.columns)].isna().sum().sum())),
                ('Registros con producción o rendimiento = 0',
                 int(((raw['Production'] == 0) | (raw['Yield'] == 0)).sum()),
                 int(((df['Production'] == 0) | (df['Yield'] == 0)).sum())),
                ('Textos con espacios sobrantes', int(texto_raw), int(texto_fin)),
                ('Registros duplicados', int(raw.duplicated().sum()), int(df[raw.columns.intersection(df.columns)].duplicated().sum())),
                ('Rendimiento máximo (t/ha)', f'{raw["Yield"].max():,.0f}', f'{df["Yield"].max():,.0f}'),
                ('Asimetría del rendimiento', f'{y_raw["Asimetría"]:.1f}', f'{y_fin["Asimetría"]:.1f}'),
            ], columns=['Indicador', 'Original', 'Final'])
            comp[['Original', 'Final']] = comp[['Original', 'Final']].astype(str)
            st.dataframe(comp, width='stretch', hide_index=True)
        with c2:
            st.markdown('##### Qué se hizo y por qué')
            hallazgo('Texto normalizado',
                     f'{miles(pasos["espacios"])} valores tenían espacios al final (p. ej. «Kharif     »). '
                     'Se recortaron para que cada categoría se cuente una sola vez.', AZUL)
            hallazgo('Ceros en el rendimiento eliminados',
                     f'{pasos["ceros"]} registros con producción o rendimiento = 0 y área positiva. Son la variable '
                     'a predecir: imputarlos sería inventar el resultado. Se eliminan (menos del 1 %).', ROJO)
            hallazgo('Coco excluido',
                     f'{pasos["coconut"]} registros de Coconut se miden en número de nueces, no en toneladas. '
                     'Mezclarlos distorsionaba todo el análisis.', ROJO)
            hallazgo('Atípicos conservados',
                     'Fincas muy grandes o estados muy lluviosos son datos reales: no se borran, se usa escala '
                     'logarítmica. Solo en entrenamiento se recorta el rendimiento al p1–p99 de cada cultivo.', AQUA)

# ==================================================================
# Pestaña 2: Datos originales (diagnóstico de calidad)
# ==================================================================
with tab_ori:
    if raw is None:
        st.warning('Datos originales no disponibles (falta data/raw/dataset_raw.csv).')
    else:
        st.subheader('¿En qué estado llegaron los datos?')
        st.markdown(f'<p class="nota">dataset_raw.csv · {miles(len(raw))} registros × {raw.shape[1]} columnas · '
                    f'años {raw["Crop_Year"].min()}–{raw["Crop_Year"].max()} · sin ningún filtro ni corrección.</p>',
                    unsafe_allow_html=True)

        diag = diagnostico(raw)
        st.dataframe(diag, width='stretch', hide_index=True, column_config={
            'Nulos (%)': st.column_config.NumberColumn(format='%.1f %%'),
            'Ceros': st.column_config.NumberColumn(format='%d'),
            'Textos con espacios': st.column_config.NumberColumn(format='%d'),
            'Atípicos IQR (%)': st.column_config.ProgressColumn(format='%.1f %%', min_value=0, max_value=30),
            'Atípicos IQR en log (%)': st.column_config.ProgressColumn(format='%.1f %%', min_value=0, max_value=30),
            'Asimetría': st.column_config.NumberColumn(format='%.1f')})
        st.caption('Atípicos IQR: valores por fuera de Q1 − 1,5·IQR y Q3 + 1,5·IQR. Asimetría > 1: '
                   'distribución con cola larga a la derecha.')

        # Hallazgos calculados en vivo (texto recortado solo para agrupar; la tabla de arriba muestra el original)
        rt = raw.assign(**{c: raw[c].astype(str).str.strip() for c in CAT_RAW})
        con_area = raw[raw['Area'] > 0]
        anual = con_area.assign(f_ha=con_area['Fertilizer'] / con_area['Area']).groupby('Crop_Year')['f_ha']
        cv_max = float((anual.std() / anual.mean()).max())
        sin_cero = rt[(rt['Production'] > 0) & (rt['Yield'] > 0) & (rt['Crop'] != 'Coconut')]
        cociente = sin_cero['Yield'] / (sin_cero['Production'] / sin_cero['Area'])
        c1, c2 = st.columns(2)
        with c1:
            hallazgo('Sin nulos, pero con ceros «disfrazados»',
                     f'0 valores vacíos. Sin embargo, {int(((raw["Production"] == 0) | (raw["Yield"] == 0)).sum())} '
                     'registros reportan producción o rendimiento = 0 con área sembrada positiva.', AMARILLO)
            hallazgo('Una unidad de medida distinta',
                     f'Coconut tiene un rendimiento mediano de {rt.loc[rt["Crop"] == "Coconut", "Yield"].median():,.0f} '
                     'por ha porque se cuenta en nueces. El resto de cultivos está en toneladas.', ROJO)
        with c2:
            hallazgo('Insumos calculados por la fuente',
                     f'El fertilizante por hectárea es igual para todos los registros de un mismo año '
                     f'(variación dentro del año ≤ {cv_max:.1%}). Mide la tendencia anual, no la práctica de cada finca.',
                     AMARILLO)
            hallazgo('Rendimiento coherente con producción / área',
                     f'La mediana de Yield ÷ (Production ÷ Area) es {cociente.median():.3f}. Las diferencias son '
                     'pequeñas y repartidas entre cultivos: se conserva el rendimiento reportado por la fuente.', AQUA)

        c1, c2 = st.columns(2)
        with c1:
            d = diag.dropna(subset=['Atípicos IQR (%)']).melt(
                id_vars='Variable', value_vars=['Atípicos IQR (%)', 'Atípicos IQR en log (%)'],
                var_name='Escala', value_name='pct')
            d['Escala'] = d['Escala'].map({'Atípicos IQR (%)': 'Escala original', 'Atípicos IQR en log (%)': 'Escala logarítmica'})
            fig = px.bar(d, x='pct', y='Variable', color='Escala', orientation='h', barmode='group',
                         color_discrete_map={'Escala original': ROJO, 'Escala logarítmica': AZUL},
                         labels={'pct': '% de registros atípicos', 'Variable': ''})
            fig.update_traces(hovertemplate='%{y}: %{x:.1f} %<extra></extra>')
            grafico(estilo(fig, 'Atípicos por variable: la escala log los reduce'))
        with c2:
            top = rt[rt['Yield'] > 0].groupby('Crop')['Yield'].median().nlargest(8).sort_values()
            fig = px.bar(x=top.values, y=top.index, orientation='h', log_x=True,
                         labels={'x': 'Rendimiento mediano (escala log)', 'y': ''})
            fig.update_traces(marker_color=[ROJO if c == 'Coconut' else AZUL for c in top.index],
                              hovertemplate='%{y}: %{x:,.1f} por ha<extra></extra>')
            grafico(estilo(fig, 'Cultivos con mayor rendimiento: Coconut se sale de escala', leyenda=False))

        c1, c2 = st.columns(2)
        with c1:
            ceros = rt[(rt['Production'] == 0) | (rt['Yield'] == 0)]
            por_cultivo = ceros['Crop'].value_counts().head(10).sort_values()
            fig = px.bar(x=por_cultivo.values, y=por_cultivo.index, orientation='h',
                         labels={'x': 'Registros con producción o rendimiento = 0', 'y': ''})
            fig.update_traces(marker_color=AMARILLO, hovertemplate='%{y}: %{x} registros<extra></extra>')
            fig.update_xaxes(dtick=1)
            grafico(estilo(fig, 'Dónde aparecen los ceros (cultivos con más casos)', leyenda=False))
            st.caption('Los ceros se concentran en algunos cultivos y estados: la falta depende de lo observado '
                       '(mecanismo MAR), no es completamente al azar.')
        with c2:
            fig = px.histogram(x=cociente.clip(0.5, 1.5).values, nbins=40,
                               labels={'x': 'Yield ÷ (Production ÷ Area)'})
            fig.update_traces(marker_color='#9ec5f4', marker_line_color='#fcfcfb', marker_line_width=1,
                              hovertemplate='%{x}: %{y} registros<extra></extra>')
            fig.add_vline(x=1, line_color=TINTA, line_dash='dot', annotation_text='Coincidencia exacta',
                          annotation_font_color=TINTA)
            fig.update_yaxes(title='Registros')
            grafico(estilo(fig, 'Coherencia del rendimiento reportado', leyenda=False))
            st.caption('Valores recortados al rango 0,5–1,5 para la lectura.')

# ==================================================================
# Pestaña 3: Datos finales (responde a los filtros)
# ==================================================================
with tab_fin:
    if not sin_datos():
        st.subheader('Datos finales con los que se trabajó')
        k = st.columns(5)
        k[0].metric('Registros', miles(len(dff)), help='Combinaciones cultivo × año × temporada × estado')
        k[1].metric('Cultivos', dff['Crop'].nunique())
        k[2].metric('Estados', dff['State'].nunique())
        k[3].metric('Área sembrada', f'{dff["Area"].sum() / 1e6:,.1f} M ha')
        k[4].metric('Lluvia mediana', f'{dff["Annual_Rainfall"].median():,.0f} mm')

        c1, c2 = st.columns(2)
        with c1:
            por_anio = dff['Crop_Year'].value_counts().sort_index()
            fig = px.bar(x=por_anio.index.astype(str), y=por_anio.values,
                         labels={'x': 'Año de cosecha', 'y': 'Registros'})
            fig.update_traces(marker_color=AZUL, hovertemplate='%{x}: %{y:,} registros<extra></extra>')
            grafico(estilo(fig, 'Registros por año', leyenda=False))
            st.caption('2020 casi no tiene registros en la fuente; sus cifras son poco representativas.')
        with c2:
            por_grupo = dff['Crop_group'].value_counts().sort_values()
            fig = px.bar(x=por_grupo.values, y=por_grupo.index, orientation='h',
                         labels={'x': 'Registros', 'y': ''})
            fig.update_traces(marker_color=AZUL, hovertemplate='%{y}: %{x:,} registros<extra></extra>')
            grafico(estilo(fig, 'Registros por grupo de cultivo', leyenda=False))

        c1, c2 = st.columns(2)
        with c1:
            top = dff['State'].value_counts().head(12).sort_values()
            fig = px.bar(x=top.values, y=top.index, orientation='h', labels={'x': 'Registros', 'y': ''})
            fig.update_traces(marker_color=AZUL, hovertemplate='%{y}: %{x:,} registros<extra></extra>')
            grafico(estilo(fig, 'Estados con más registros', leyenda=False))
        with c2:
            orden = dff.groupby('Crop_group')['Yield'].median().sort_values().index.tolist()
            fig = px.box(dff, x='Crop_group', y='Yield', log_y=True, points=False,
                         category_orders={'Crop_group': orden},
                         labels={'Crop_group': '', 'Yield': 'Rendimiento (t/ha, escala log)'})
            fig.update_traces(marker_color=AZUL, line_color=AZUL, fillcolor='rgba(42,120,214,.25)')
            grafico(estilo(fig, 'Rendimiento por grupo de cultivo', leyenda=False))

        c1, c2 = st.columns(2)
        with c1:
            part = dff['Conjunto'].map(NOMBRE_CONJ).value_counts().reindex(list(NOMBRE_CONJ.values())).dropna()
            color_conj = dict(zip(NOMBRE_CONJ.values(), [AZUL, AMARILLO, AQUA]))
            fig = px.bar(x=part.index, y=part.values, text=[f'{v / part.sum():.0%}' for v in part.values],
                         labels={'x': '', 'y': 'Registros'})
            fig.update_traces(marker_color=[color_conj[n] for n in part.index], textposition='outside',
                              hovertemplate='%{x}: %{y:,} registros<extra></extra>')
            fig.update_yaxes(range=[0, part.max() * 1.15])
            grafico(estilo(fig, 'Partición 70 / 15 / 15 (estratificada por cultivo)', leyenda=False))
        with c2:
            cl = pd.crosstab(dff['Conjunto'].map(NOMBRE_CONJ), dff['Yield_class'], normalize='index') * 100
            cl = cl.reindex(columns=CLASES, fill_value=0).reindex([v for v in NOMBRE_CONJ.values() if v in cl.index])
            cl = cl.reset_index(names='Conjunto').melt(id_vars='Conjunto', var_name='Clase', value_name='pct')
            fig = px.bar(cl, x='pct', y='Conjunto', color='Clase', orientation='h',
                         color_discrete_map=COLOR_CLASE, category_orders={'Clase': CLASES},
                         labels={'pct': '% de registros', 'Conjunto': ''})
            fig.update_traces(marker_line_color='#fcfcfb', marker_line_width=2,
                              hovertemplate='%{y}<br>%{fullData.name}: %{x:.1f} %<extra></extra>')
            fig.update_layout(barmode='stack')
            grafico(estilo(fig, 'Clase de rendimiento (Bajo / Medio / Alto) por conjunto'))
            st.caption('Las tres clases quedan balanceadas y con la misma proporción en cada conjunto.')

        st.markdown('##### Variables del conjunto final')
        dicc = pd.DataFrame([
            ('Crop', 'Original', 'Categórica', 'Predictor', 'Cultivo (sin espacios sobrantes)'),
            ('Season', 'Original', 'Categórica', 'Predictor', 'Temporada de siembra'),
            ('State', 'Original', 'Categórica', 'Predictor', 'Estado de India'),
            ('Area', 'Original', 'Numérica', 'Predictor', 'Hectáreas sembradas (escala log en el modelo)'),
            ('Annual_Rainfall', 'Original', 'Numérica', 'Predictor', 'Lluvia anual del estado en mm (escala log)'),
            ('Crop_Year', 'Original', 'Numérica', 'Solo filtro', 'Su efecto ya lo recogen los insumos por ha'),
            ('Yield', 'Original', 'Numérica', 'Objetivo (regresión)', 'Rendimiento en t/ha reportado por la fuente'),
            ('Production', 'Original', 'Numérica', 'Excluida', 'Fuga de datos: Yield ≈ Production / Area'),
            ('Fertilizer, Pesticide', 'Original', 'Numérica', 'Excluidas', 'Son Área × dosis anual (redundantes)'),
            ('Crop_group', 'Nueva', 'Categórica', 'Predictor', 'Cereales, legumbres, oleaginosas, etc.'),
            ('Fert_por_ha, Pest_por_ha', 'Nueva', 'Numérica', 'Predictor', 'Insumos por hectárea (varían solo por año)'),
            ('Rain_anomaly', 'Nueva', 'Numérica', 'Predictor', 'Lluvia del año frente a la media del estado'),
            ('Yield_class', 'Nueva', 'Categórica', 'Objetivo (clasificación)', 'Bajo / Medio / Alto según los terciles de su cultivo'),
        ], columns=['Variable', 'Origen', 'Tipo', 'Uso', 'Descripción'])
        st.dataframe(dicc, width='stretch', hide_index=True)

# ==================================================================
# Pestaña 4: Explorar registros
# ==================================================================
with tab_exp:
    st.subheader('Registros finales (filtrados)')
    cols = ['Crop', 'Crop_group', 'Crop_Year', 'Season', 'State', 'Area', 'Production', 'Annual_Rainfall',
            'Fert_por_ha', 'Pest_por_ha', 'Yield', 'Yield_class', 'Conjunto']
    st.dataframe(dff[cols], width='stretch', hide_index=True, height=440)
    st.download_button('⬇️ Descargar CSV', dff[cols].to_csv(index=False).encode('utf-8'),
                       file_name='agroinsight_datos_finales.csv', mime='text/csv')
    with st.expander('Notas metodológicas'):
        st.markdown(f"""
* **Fuente:** Kaggle *Crop Yield in Indian States* (akshatgupta7), CC BY 4.0. Alcance 2015–2020 por infraestructura de Colab.
* **Limpieza:** espacios sobrantes recortados; se eliminaron los registros con producción o rendimiento cero (mecanismo MAR) y los de Coconut (unidades no comparables).
* **Atípicos:** no se eliminan; Área y Lluvia se usan en escala logarítmica y el rendimiento se recorta al p1–p99 de cada cultivo solo en entrenamiento.
* **Partición:** 70 / 15 / 15 estratificada por cultivo (semilla 42). Todo lo que se aprende de los datos (escalado, One-Hot, terciles, medianas, recortes) se ajusta solo con entrenamiento.
* **Limitaciones:** fertilizante y pesticida tienen una dosis fija por año en la fuente y la lluvia es un valor anual por estado.
* Datos del tablero generados el {meta['generado']}.
""")

# ==================================================================
# Pestaña 5: Uso en modelos (KPI del modelo, simulador y segmentos)
# ==================================================================
with tab_mod:
    st.subheader('Para qué se usaron los datos finales')
    st.markdown('<p class="nota">Resumen corto de los modelos entrenados con estos datos. '
                'Métricas sobre el conjunto de prueba, que el modelo no vio al entrenar.</p>', unsafe_allow_html=True)
    mr, mc = meta['regresion']['mejor'], meta['clasificacion']['mejor']
    mae_x, mae_b = metrica(REG, mr, 'test', 'MAE'), metrica(REG, 'Baseline_OLS', 'test', 'MAE')
    f1_x = metrica(CLS, mc, 'test', 'F1_macro')
    k = st.columns(3)
    k[0].metric('Error medio al estimar el rendimiento', f'{mae_x:.2f} t/ha',
                f'{(mae_x / mae_b - 1) * 100:+.0f} % frente al modelo base', delta_color='inverse',
                help=f'MAE de {mr} en prueba')
    k[1].metric('Acierto al clasificar Bajo / Medio / Alto', f'{f1_x:.2f}', help=f'F1 macro de {mc} en prueba (máximo 1)')
    k[2].metric('Segmentos de productores', meta['clustering']['k'],
                f'silueta {meta["clustering"]["silueta_test"]:.2f}', delta_color='off')

    # -------------------- Simulador de predicción en vivo --------------------
    st.markdown('##### Simulador de predicción')
    combos = meta['combos']
    c = st.columns(5)
    lista_estados = sorted(combos)
    estado = c[0].selectbox('Estado', lista_estados,
                            index=lista_estados.index('Karnataka') if 'Karnataka' in lista_estados else 0)
    lista_cultivos = sorted(combos[estado])
    cultivo = c[1].selectbox('Cultivo', lista_cultivos,
                             index=lista_cultivos.index('Rice') if 'Rice' in lista_cultivos else 0)
    temporada = c[2].selectbox('Temporada', combos[estado][cultivo])
    lista_anios = sorted(int(a) for a in meta['insumos_anio'])
    anio = lista_anios[-2] if len(lista_anios) > 1 else lista_anios[0]   # 2020 tiene muy pocos registros
    historico = df[(df['State'] == estado) & (df['Crop'] == cultivo)]
    area_def = float(historico['Area'].median()) if len(historico) else 1000.0
    lluvia_def = meta['lluvia_estado_anio'].get(f'{estado}|{anio}', meta['lluvia_estado'].get(estado, 1200.0))
    ll_min, ll_max = float(np.floor(df['Annual_Rainfall'].min())), float(np.ceil(df['Annual_Rainfall'].max()))
    area = c[3].number_input('Área sembrada (ha)', min_value=1.0, max_value=20_000_000.0,
                             value=max(1.0, float(round(area_def))), step=100.0, format='%.0f')
    lluvia = c[4].number_input('Precipitación anual (mm)', min_value=float(int(ll_min * 0.8)),
                               max_value=float(int(ll_max * 1.1)), value=float(round(lluvia_def)), step=10.0,
                               format='%.0f')

    grupo = cf.crop_group(cultivo)
    insumo = meta['insumos_anio'][str(anio)]
    X = pd.DataFrame([{'Area': area, 'Annual_Rainfall': lluvia, 'Fert_por_ha': insumo['Fert_por_ha'],
                       'Pest_por_ha': insumo['Pest_por_ha'], 'Crop': cultivo, 'Season': temporada,
                       'State': estado, 'Crop_group': grupo}])[meta['features']]
    y_hat = float(max(modelos['reg'].predict(X)[0], 0.0))
    proba = modelos['cls'].predict_proba(X)[0]
    clase = CLASES[int(np.argmax(proba))]
    km = modelos['km']
    referencia = km['crop_median'].get(cultivo, km['group_median'].get(grupo, y_hat or 1.0))
    y_rel = float(np.clip(np.log(max(y_hat, 1e-6) / referencia), -km['yield_rel_clip'], km['yield_rel_clip']))
    z = km['scaler'].transform(pd.DataFrame([[np.log1p(area), np.log1p(lluvia), y_rel]], columns=km['features']))
    seg = next(s for s in SEGMENTOS if s['cluster'] == int(km['kmeans'].predict(z)[0]))

    k = st.columns(4)
    k[0].metric('Rendimiento estimado', f'{y_hat:,.2f} t/ha', f'{np.exp(y_rel):.2f}× el típico del cultivo',
                delta_color='off')
    k[1].metric('Producción estimada', f'{y_hat * area:,.0f} t', help='Rendimiento estimado × área')
    k[2].metric('Clase de rendimiento', clase, f'probabilidad {proba.max():.0%}', delta_color='off')
    k[3].metric('Segmento', seg['nombre'], delta_color='off')
    st.caption(f'Insumos por hectárea fijados en la mediana de {anio} (en la fuente solo cambian entre años).')

    # -------------------- Segmentación --------------------
    st.markdown('##### Segmentos de unidades productivas (K-Means, registros filtrados)')
    if not sin_datos():
        c1, c2 = st.columns([1.4, 1])
        with c1:
            muestra = dff.sample(min(len(dff), 2500), random_state=42)
            fig = px.scatter(muestra, x='PC1', y='PC2', color='Segmento', symbol='Segmento',
                             color_discrete_map=COLOR_SEG, symbol_map=SIMBOLO_SEG,
                             category_orders={'Segmento': ORDEN_SEG}, opacity=0.7,
                             hover_data={'Crop': True, 'State': True, 'Crop_Year': True, 'Yield': ':.2f',
                                         'PC1': False, 'PC2': False, 'Segmento': False})
            fig.update_traces(marker=dict(size=7, line=dict(width=1, color='#fcfcfb')))
            var = meta['clustering']['pca_varianza']
            fig.update_xaxes(title=f'PC1 ({var[0]:.0%} de la varianza)')
            fig.update_yaxes(title=f'PC2 ({var[1]:.0%} de la varianza)')
            grafico(estilo(fig, h=440))
        with c2:
            for s in SEGMENTOS:
                st.markdown(f'<div class="hallazgo" style="--c:{COLOR_SEG[s["nombre"]]}"><b>{s["nombre"]} · '
                            f'{s["pct"]:.0f} % de los registros</b><br><span>{s["accion"]}</span></div>',
                            unsafe_allow_html=True)
        tabla = dff.groupby('Segmento').agg(
            Registros=('Crop', 'size'), Area_mediana_ha=('Area', 'median'),
            Lluvia_media_mm=('Annual_Rainfall', 'mean'), Rend_rel_medio=('Yield_rel', 'mean'),
            Pct_Alto=('Yield_class', lambda s: (s == 'Alto').mean() * 100)).reindex(ORDEN_SEG).dropna(how='all')
        tabla['Veces_tipico'] = np.exp(tabla['Rend_rel_medio'])
        st.dataframe(tabla[['Registros', 'Area_mediana_ha', 'Lluvia_media_mm', 'Veces_tipico', 'Pct_Alto']],
                     width='stretch', column_config={
                         'Registros': st.column_config.NumberColumn(format='%d'),
                         'Area_mediana_ha': st.column_config.NumberColumn('Área mediana (ha)', format='%.0f'),
                         'Lluvia_media_mm': st.column_config.NumberColumn('Lluvia media (mm)', format='%.0f'),
                         'Veces_tipico': st.column_config.NumberColumn('Rendimiento vs. típico', format='%.2f×'),
                         'Pct_Alto': st.column_config.NumberColumn('% clase Alto', format='%.1f')})
