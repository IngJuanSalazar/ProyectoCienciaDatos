"""
crop_features.py - Transformaciones propias del proyecto Crop Yield in Indian States.

Generado por el Colab 2. Para cargar los .pkl en otros cuadernos o en el dashboard:
    import sys; sys.path.insert(0, '<BASE_DIR>/src'); import crop_features
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

LABELS = ['Bajo', 'Medio', 'Alto']


def _norm(texto):
    """Normaliza nombres: minúsculas y un solo espacio entre palabras."""
    return ' '.join(str(texto).lower().split())


# ------------------------------------------------------------------
# Agrupación agronómica de cultivos (conocimiento del dominio)
# ------------------------------------------------------------------
_GRUPOS = {
    'Cereales': ['Rice', 'Wheat', 'Maize', 'Jowar', 'Bajra', 'Ragi', 'Small millets',
                 'Barley', 'Other Cereals'],
    'Legumbres': ['Arhar/Tur', 'Gram', 'Moong(Green Gram)', 'Urad', 'Masoor', 'Horse-gram',
                  'Khesari', 'Moth', 'Cowpea(Lobia)', 'Peas & beans (Pulses)','Other Summer Pulses',
                  'Other Rabi pulses', 'Other Kharif pulses', 'Guar seed'],
    'Oleaginosas': ['Groundnut', 'Rapeseed &Mustard', 'Sesamum', 'Soyabean', 'Sunflower',
                    'Castor seed', 'Linseed', 'Niger seed', 'Safflower', 'other oilseeds',
                    'Oilseeds total'],
    'Fibras': ['Cotton(lint)', 'Jute', 'Mesta', 'Sannhamp'],
    'Hortalizas_tuberculos_frutas': ['Potato', 'Onion', 'Sweet potato', 'Tapioca', 'Garlic',
                                     'Banana'],
    'Especias': ['Dry chillies', 'Turmeric', 'Ginger', 'Coriander', 'Black pepper',
                 'Cardamom'],
    'Plantacion_comerciales': ['Sugarcane', 'Arecanut', 'Cashewnut', 'Tobacco', 'Coconut'],
}
CROP_GROUPS = {_norm(c): g for g, cultivos in _GRUPOS.items() for c in cultivos}


def crop_group(crop):
    """Devuelve el grupo agronómico del cultivo ('Otros' si no está mapeado)."""
    return CROP_GROUPS.get(_norm(crop), 'Otros')


def add_row_features(df):
    """Variables derivadas calculadas FILA A FILA (no usan otros registros => sin fuga)."""
    out = df.copy()
    out['Fert_por_ha'] = out['Fertilizer'] / out['Area']     # kg de fertilizante por hectárea
    out['Pest_por_ha'] = out['Pesticide'] / out['Area']      # kg de pesticida por hectárea
    out['Crop_group'] = out['Crop'].map(crop_group)          # grupo agronómico
    return out

# ------------------------------------------------------------------
# Anomalía de lluvia respecto al promedio del estado (aprendido en TRAIN)
# ------------------------------------------------------------------
class RainAnomaly(BaseEstimator, TransformerMixin):
    """Evaluar la lluvia promedio por estado.

    0 = año normal para ese estado; +0.20 = 20 % más lluvioso; -0.20 = 20 % más seco.
    La media de cada estado se calcula sobre valores únicos estado-lluvia para que los
    estados con muchos cultivos registrados no pesen más. Estados no vistos en train
    usan la media global de train.
    """

    def fit(self, X, y=None):
        X = pd.DataFrame(X, columns=['State', 'Annual_Rainfall'])
        unicos = X.drop_duplicates(['State', 'Annual_Rainfall'])
        self.state_mean_ = unicos.groupby('State')['Annual_Rainfall'].mean().to_dict()
        self.global_mean_ = float(unicos['Annual_Rainfall'].mean())
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=['State', 'Annual_Rainfall'])
        ref = X['State'].map(self.state_mean_).fillna(self.global_mean_).astype(float)
        anomalia = X['Annual_Rainfall'].astype(float) / ref - 1
        return pd.DataFrame({'Rain_anomaly': anomalia.values}, index=X.index)

    def get_feature_names_out(self, input_features=None):
        return np.array(['Rain_anomaly'], dtype=object)
# ------------------------------------------------------------------
# Umbrales de la clase de rendimiento relativo (aprendidos en TRAIN)
# ------------------------------------------------------------------
class YieldTercileLabeler(BaseEstimator):
    """Evaluacion de redimiento bajo, medio o alto para el cultivo.

    Los terciles se calculan solo con train. Cultivos con menos de `min_n` registros en
    train usan los terciles de su grupo agronómico; si tampoco existe, los globales.
    """

    def __init__(self, min_n=12):
        self.min_n = min_n

    @staticmethod
    def _terciles(serie):
        return float(serie.quantile(1 / 3)), float(serie.quantile(2 / 3))

    def fit(self, df):
        n = df['Crop'].value_counts()
        self.crop_thr_ = {c: self._terciles(g['Yield']) for c, g in df.groupby('Crop')
                          if n[c] >= self.min_n}
        self.group_thr_ = {c: self._terciles(g['Yield']) for c, g in df.groupby('Crop_group')}
        self.global_thr_ = self._terciles(df['Yield'])
        return self

    def thresholds(self, df):
        lo = df['Crop'].map({k: v[0] for k, v in self.crop_thr_.items()})
        hi = df['Crop'].map({k: v[1] for k, v in self.crop_thr_.items()})
        lo = lo.fillna(df['Crop_group'].map({k: v[0] for k, v in self.group_thr_.items()}))
        hi = hi.fillna(df['Crop_group'].map({k: v[1] for k, v in self.group_thr_.items()}))
        return lo.fillna(self.global_thr_[0]), hi.fillna(self.global_thr_[1])

    def transform(self, df):
        lo, hi = self.thresholds(df)
        y = df['Yield']
        etiqueta = np.where(y <= lo, 'Bajo', np.where(y <= hi, 'Medio', 'Alto'))
        return pd.Series(pd.Categorical(etiqueta, categories=LABELS, ordered=True),
                         index=df.index, name='Yield_class')


# ------------------------------------------------------------------
# Recorte por percentiles (winsorization) de Yield dentro de cada cultivo
# ------------------------------------------------------------------
class CropWinsorizer(BaseEstimator):
    """Es un rendimiento extremo poco creible para el cultivo.

    Se aplica SOLO al objetivo de entrenamiento: validación y prueba se evalúan con el
    valor real para no maquillar las métricas. Cultivos con menos de `min_n` registros
    en train no se recortan.
    """

    def __init__(self, lower=0.01, upper=0.99, min_n=12):
        self.lower = lower
        self.upper = upper
        self.min_n = min_n

    def fit(self, df):
        n = df['Crop'].value_counts()
        self.limits_ = {c: (float(g['Yield'].quantile(self.lower)),
                            float(g['Yield'].quantile(self.upper)))
                        for c, g in df.groupby('Crop') if n[c] >= self.min_n}
        return self

    def transform(self, df):
        # Cultivos sin límites aprendidos: -inf / +inf (no se recortan)
        lo = df['Crop'].map({k: v[0] for k, v in self.limits_.items()}).fillna(-np.inf)
        hi = df['Crop'].map({k: v[1] for k, v in self.limits_.items()}).fillna(np.inf)
        return df['Yield'].clip(lower=lo, upper=hi).rename('Yield_w')
