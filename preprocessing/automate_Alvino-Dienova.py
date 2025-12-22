import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

df = pd.read_csv('abalone.data', header=None, names=[
    'sex',
    'length',
    'diameter',
    'height',
    'whole_weight',
    'shucked_weight',
    'viscera_weight',
    'shell_weight',
    'rings',
])

def bin_rings(rings):
  if rings < 6:
    return 1
  elif rings < 14:
    return 2
  else:
    return 3

df['rings'] = df['rings'].apply(bin_rings)

ct = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), ['sex']),
        ('num', StandardScaler(), ['length', 'diameter', 'height', 'whole_weight', 'shucked_weight', 'viscera_weight', 'shell_weight'])
    ],
)

df_preprocessing = pd.DataFrame(ct.fit_transform(df), columns=ct.get_feature_names_out())
df_preprocessing['rings'] = df['rings']

df_preprocessing.to_csv('preprocessing/abalone_preprocessing.csv', index=False)
