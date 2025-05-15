import pandas as pd
from pp01_paths import *
from collections import Counter
import re

# all data
df_judges = pd.read_csv(processed_data_path + r"/df_judges.csv")
df_d2v_vecs = pd.read_csv(processed_data_path + r"/topic_modelling/kanoon_doc2vec.csv")
df_air_pollution = pd.read_csv('/Users/anchalbhardwaj/Downloads/India_Pollution-main/01_Data/Analysis/03_processed_data/topic_modelling/gptoutput_flat_27jun_notext.csv', low_memory=False)

# process df_judges to extract kanoon_id and judge
filtered_df = df_judges[['kanoon_id', 'judge_court']].copy()
filtered_df['judge_court'] = filtered_df['judge_court'].apply(lambda x: eval(x) if isinstance(x, str) else [])
filtered_df = filtered_df.explode('judge_court')
filtered_df['judge'] = filtered_df['judge_court'].apply(lambda x: x[0].strip() if isinstance(x, tuple) else None)
filtered_df = filtered_df.dropna(subset=['judge'])
df_cleaned_judges = filtered_df[['kanoon_id', 'judge']]

# extract unique judges who ruled on air pollution cases
judge_columns = [col for col in df_air_pollution.columns if re.search(r'gpt4_judge_list_\d+', col)]
df_air_pollution_judges = df_air_pollution.melt(
    id_vars=['kanoon_id'], 
    value_vars=judge_columns,
    var_name='Judge_Column', 
    value_name='judge'
).dropna(subset=['judge'])

# ensure all ids are strings for consistent merging
df_air_pollution["kanoon_id"] = df_air_pollution["kanoon_id"].astype(str).str.strip()
df_cleaned_judges["kanoon_id"] = df_cleaned_judges["kanoon_id"].astype(str).str.strip()

df_air_pollution_judges["kanoon_id"] = df_air_pollution_judges["kanoon_id"].astype(str).str.strip()

# make sure kanoon_id is consistently formatted across all relevant DataFrames
df_air_pollution["kanoon_id"] = df_air_pollution["kanoon_id"].astype(str).str.strip()
df_cleaned_judges["kanoon_id"] = df_cleaned_judges["kanoon_id"].astype(str).str.strip()

# get all unique air pollution case IDs
air_pollution_case_ids = set(df_air_pollution["kanoon_id"].drop_duplicates())
print("Unique air pollution case IDs:", len(air_pollution_case_ids))

# get all judges who ruled on air pollution cases
air_pollution_judge_ids = set(df_air_pollution_judges["judge"].unique())

# get all cases ruled on by these judges
cases_by_air_judges = df_cleaned_judges[df_cleaned_judges["judge"].isin(air_pollution_judge_ids)]
print("Cases by air judges:", len(cases_by_air_judges))

# filter non-air pollution cases by removing air pollution case IDs
non_air_cases_by_air_judges = cases_by_air_judges[~cases_by_air_judges["kanoon_id"].isin(air_pollution_case_ids)]
print("Non-air cases by air judges:", len(non_air_cases_by_air_judges))

# check if any air pollution cases were not removed
expected_removed_cases = air_pollution_case_ids
actual_removed_cases = set(cases_by_air_judges["kanoon_id"]) - set(non_air_cases_by_air_judges["kanoon_id"])

missing_removals = expected_removed_cases - actual_removed_cases
# print("Number of air cases not removed:", len(missing_removals))
# print("Sample missing removals:", list(missing_removals)[:10])

# final double-checks
unique_cases_by_air_judges = cases_by_air_judges["kanoon_id"].nunique()
unique_non_air_cases_by_air_judges = non_air_cases_by_air_judges["kanoon_id"].nunique()
# print("Unique cases by air judges:", unique_cases_by_air_judges)
# print("Unique non-air cases by air judges:", unique_non_air_cases_by_air_judges)
# print("Expected difference (should be 12615):", unique_cases_by_air_judges - unique_non_air_cases_by_air_judges)

panel_cases = non_air_cases_by_air_judges.groupby('kanoon_id').filter(lambda x: len(x) > 1)['kanoon_id'].unique()
print("Panel cases length:", len(panel_cases))

single_judge_cases = non_air_cases_by_air_judges[
    ~non_air_cases_by_air_judges['kanoon_id'].isin(panel_cases)
]

original_case_count = len(single_judge_cases)
if original_case_count == 0:
    print("No single-judge, non-air pollution cases found.")
else: 
    # ensure both df have the same type for the merge column
    single_judge_cases["kanoon_id"] = single_judge_cases["kanoon_id"].astype(str).str.strip()
    df_d2v_vecs["kanoon_id"] = df_d2v_vecs["kanoon_id"].astype(str).str.strip()

    df_merged = pd.merge(single_judge_cases, df_d2v_vecs, on="kanoon_id", how="left") 
    #print("d2v len", (df_d2v_vecs.head(5)))
    duplicate_kanoon_ids = df_merged['kanoon_id'].duplicated().sum()
   
    if duplicate_kanoon_ids == 0:
        print("No duplicate kanoon IDs found in df_merged")
    else: 
        print(f"Warning: {duplicate_kanoon_ids} duplicate kanoon ids found in df_merged")
    df_merged_case_count = len(df_merged)
    
    total_dropped_percentage = (original_case_count - df_merged_case_count) / original_case_count * 100
    #print(f"Percentage of cases dropped due to missing d2v vectors: {total_dropped_percentage:.2f}%")

    d2v_cols = [col for col in df_d2v_vecs.columns if col != 'kanoon_id']
    
    df_merged_with_vectors = pd.merge(single_judge_cases, df_d2v_vecs, on="kanoon_id", how="left")
    df_final = df_merged_with_vectors.dropna(subset=d2v_cols, how='all')

    # print(f"Number of cases with at least one vector score: {len(df_final)}")
    # print(f"Percentage of cases retained with at least one vector score: {len(df_final) / original_case_count * 100:.2f}%")

    judge_vectors = df_final.groupby('judge')[d2v_cols].mean().reset_index()
    judge_vectors.columns = ['judge'] + [f'avg_{col}' for col in d2v_cols]

    # add case counts
    judge_case_counts = df_final.groupby('judge')['kanoon_id'].count().reset_index()
    judge_case_counts.columns = ['judge', 'non_air_pollution_single_judge_cases']
    judge_vectors = pd.merge(judge_vectors, judge_case_counts, on='judge', how='left')

    # sort by number of cases
    judge_vectors = judge_vectors.sort_values('non_air_pollution_single_judge_cases', ascending=False)
    
     # summary statistics
    # print("\nSample of vector averages for first judge:")
    # print(judge_vectors.iloc[0])
    # print("\nDistribution of non-air pollution single-judge cases per judge:")
    # print(judge_vectors['non_air_pollution_single_judge_cases'].describe())

    judge_vectors.to_csv('finalVectors.csv', index=False)


