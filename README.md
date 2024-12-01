# NER To Features
Implemets a function that given a DataFrame with a (long) text column, converts all entities inside said column to features, based on if they pass a certain user defined cutoff. The function returns a mapping between cutoffs and features, so one can experiment with different cutoffs to see how the features they generate effect the final model.

# How To Use?
Add something along these lines to your python script:
```py
import pandas as pd
from ntf import ner_to_feature

df = pd.load_data_func()
datasets = ner_to_feature(df, 'text_column', 'ner_model_name', [1, 2, 3, 4, 5, 10], True)

for c, d in datasets.items():
    print(f'For cutoff of {c} we have a dataframe with {len(d.columns)} features.')
    print(d.head())
```