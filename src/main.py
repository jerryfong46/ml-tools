from generate_features import *
import pandas as pd

df = pd.read_csv('data/sample_customer_data.csv')
exclude_cols = ['customer_id', 'TARGET']

df_numerical = df.select_dtypes(include=['int', 'float'])
df_categorical = df.select_dtypes(include=['object'])

df_num_interactions = create_interaction_features(df_numerical.drop(columns=exclude_cols))
df_cat_interactions, cat_mappings = process_categorical_columns(df_categorical)

# Save the category mappings to a JSON file
with open('category_mappings.json', 'w') as f:
    json.dump(cat_mappings, f)

df_interactions = pd.concat([df_num_interactions, df_cat_interactions], axis=1)
print(df_interactions.head())

