
import pandas as pd
from pp01_paths import *
from collections import Counter
import re

df_judges = pd.read_csv(processed_data_path + r"/df_judges.csv")
df_d2v_vecs = pd.read_csv(processed_data_path + r"/topic_modelling/kanoon_doc2vec.csv")

filtered_df = df_judges[['kanoon_id', 'judge_court']].copy()
filtered_df['judge_court'] = filtered_df['judge_court'].apply(lambda x: eval(x) if isinstance(x, str) else [])
filtered_df = filtered_df.explode('judge_court')
filtered_df['judge'] = filtered_df['judge_court'].apply(lambda x: x[0].strip() if isinstance(x, tuple) else None)
df_cleaned_judges = filtered_df[['kanoon_id', 'judge']].dropna()

df_cleaned_judges['kanoon_id'] = df_cleaned_judges['kanoon_id'].astype(str)
df_d2v_vecs['kanoon_id'] = df_d2v_vecs['kanoon_id'].astype(str)

df_merged = pd.merge(df_cleaned_judges, df_d2v_vecs, on="kanoon_id", how="inner")

vector_cols = [col for col in df_d2v_vecs.columns if col != 'kanoon_id']

judge_avg_vectors = df_merged.groupby('judge')[vector_cols].mean().reset_index()

df_case_with_judge_vectors = pd.merge(df_cleaned_judges, judge_avg_vectors, on='judge', how='left')

df_case_avg_vectors = df_case_with_judge_vectors.groupby('kanoon_id')[vector_cols].mean().reset_index()


df_case_avg_vectors.to_csv("case_avg_vectors.csv", index=False)

print("Final DataFrame Shape:", df_case_avg_vectors.shape)
print("Sample Output:\n", df_case_avg_vectors.head())