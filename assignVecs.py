import pandas as pd
from pp01_paths import *
from collections import Counter
import re

# all file paths
df_judges = pd.read_csv(processed_data_path + r"/df_judges.csv")
#df_lsa_vecs = pd.read_csv(processed_data_path + r"/topic_modelling/kanoon_lsa.csv")
df_d2v_vecs = pd.read_csv(processed_data_path + r"/topic_modelling/kanoon_doc2vec.csv")

# reading and filtering df_judges (all cases)
filtered_df = df_judges[['kanoon_id', 'judge_court']].copy()
print("filtered_df: ", len(filtered_df))
filtered_df['judge_court'] = filtered_df['judge_court'].apply(lambda x: eval(x) if isinstance(x, str) else [])
filtered_df = filtered_df.explode('judge_court')
filtered_df['judge'] = filtered_df['judge_court'].apply(lambda x: x[0].strip() if isinstance(x, tuple) else None)
filtered_df = filtered_df.dropna(subset=['judge'])
df_cleaned_judges = filtered_df[['kanoon_id', 'judge']]
print(df_cleaned_judges.head(5))
print("df_cleaned_judges: ", len(df_cleaned_judges))

# reading air pollution cases
df_air_pollution = pd.read_csv('/Users/anchalbhardwaj/Downloads/India_Pollution-main/01_Data/Analysis/03_processed_data/topic_modelling/gptoutput_flat_27jun_notext.csv', low_memory= False)

print("df_air_pollution: ", len(df_air_pollution))

# select judge columns
judge_columns = [col for col in df_air_pollution.columns if re.search(r'gpt4_judge_list_\d+', col)]

# melt the dataframe with dynamically selected judge columns
df_air_pollution_judges = df_air_pollution.melt(
    id_vars=['kanoon_id'], 
    value_vars=judge_columns,
    var_name='Judge_Column', 
    value_name='judge'
)

# drop rows with NaN judges and reset index
df_air_pollution_judges = df_air_pollution_judges.dropna(subset=['judge']).reset_index(drop=True)

print("df_air_pollution_judges: ", len(df_air_pollution_judges))

# unique list of air pollution judges
unique_air_pollution_judges = pd.DataFrame(df_air_pollution_judges['judge'].unique())
unique_air_pollution_judges.rename(columns={0: 'judge'}, inplace=True) 
print("unique_air_pollution_judges: ", len(unique_air_pollution_judges))

# ensure all IDs are strings for consistent merging
df_cleaned_judges['kanoon_id'] = df_cleaned_judges['kanoon_id'].astype(str)
print("df_cleaned_judges: ", len(df_cleaned_judges))

df_air_pollution_judges['kanoon_id'] = df_air_pollution_judges['kanoon_id'].astype(str)
df_air_pollution['kanoon_id'] = df_air_pollution['kanoon_id'].astype(str)

#df_lsa_vecs['kanoon_id'] = df_lsa_vecs['kanoon_id'].astype(str)
df_d2v_vecs['kanoon_id'] = df_d2v_vecs['kanoon_id'].astype(str)

# filter for only cases by air pollution judges
# potential issue - typographical errors, not 100% accurate match (to look into)
unique_air_pollution_judges["judge_flag"] = 1
cases_by_air_judges = pd.merge(df_cleaned_judges, unique_air_pollution_judges, on = ["judge"], how = "left")
cases_by_air_judges = cases_by_air_judges[cases_by_air_judges["judge_flag"] == 1]
#cases_by_air_judges = df_cleaned_judges[df_cleaned_judges['judge'].isin(unique_air_pollution_judges)]
print("number of unique cases by air judges by kanoon id:", len(cases_by_air_judges["kanoon_id"].unique()))
print("number of unique judges in air cases", len(cases_by_air_judges["judge"].unique()))
print("cases_by_air_judges length:", len(cases_by_air_judges))


# remove air pollution cases to get only non-air pollution cases
# difference from cases_by_air_judges of 16k, should be 12k
df_air_pollution["air_flag"] = 1
non_air_cases_by_air_judges = pd.merge(cases_by_air_judges, df_air_pollution, how = "left")
print("non air cases before dropping:", len(non_air_cases_by_air_judges))
# drops the air pollutions cases but keeps non-air cases for air judges
non_air_cases_by_air_judges = non_air_cases_by_air_judges[non_air_cases_by_air_judges["air_flag"] != 1]

# non_air_cases_by_air_judges = cases_by_air_judges[
#     ~cases_by_air_judges['kanoon_id'].isin(df_air_pollution['kanoon_id'])
# 
print("non_air_cases_by_air_judges length:", len(non_air_cases_by_air_judges))

# remove panel decisions from non-air pollution cases
panel_cases = non_air_cases_by_air_judges.groupby('kanoon_id').filter(lambda x: len(x) > 1)['kanoon_id'].unique()
print("Panel cases length:", len(panel_cases))

single_judge_cases = non_air_cases_by_air_judges[
    ~non_air_cases_by_air_judges['kanoon_id'].isin(panel_cases)
]

original_case_count = len(single_judge_cases)
if original_case_count == 0:
    print("No single-judge, non-air pollution cases found.")
else: 
    # merge with vector data
    #df_merged0 = pd.merge(single_judge_cases, df_lsa_vecs, on="kanoon_id", how="left")
    df_merged = pd.merge(single_judge_cases, df_d2v_vecs, on="kanoon_id", how="left") 

    print("d2v len", (df_d2v_vecs.head(5)))
    #print("LSA len", (df_lsa_vecs.head(5)))

    # find duplicate kanoon IDs
    duplicate_kanoon_ids = df_merged['kanoon_id'].duplicated().sum()
    if duplicate_kanoon_ids == 0:
        print("No duplicate kanoon IDs found in df_merged")
    else: 
        print(f"Warning: {duplicate_kanoon_ids} duplicate kanoon ids found in df_merged")

    # find percentage of cases dropped
    #df_merged0_case_count = len(df_merged0)
    df_merged_case_count = len(df_merged)

    #lsa_dropped_percentage = (original_case_count - df_merged0_case_count) / original_case_count * 100
    total_dropped_percentage = (original_case_count - df_merged_case_count) / original_case_count * 100
    #print(f"Percentage of cases dropped due to missing lsa vectors: {lsa_dropped_percentage:.2f}%")
    print(f"Percentage of cases dropped due to missing d2v vectors: {total_dropped_percentage:.2f}%")

    # define columns for vector calculations
    #lsa_cols = [col for col in df_lsa_vecs.columns if col != 'kanoon_id']
    d2v_cols = [col for col in df_d2v_vecs.columns if col != 'kanoon_id']

    # find non-na versions of the data frames
    #df_merged_lsa = pd.merge(single_judge_cases, df_lsa_vecs, on="kanoon_id", how="left")
    df_merged_with_vectors = pd.merge(single_judge_cases, df_d2v_vecs, on="kanoon_id", how="left")
    df_final = df_merged_with_vectors.dropna(subset=d2v_cols, how='all')

    print(f"Number of cases with at least one vector score: {len(df_final)}")
    print(f"Percentage of cases retained with at least one vector score: {len(df_final) / original_case_count * 100:.2f}%")

    # calculate average vectors by judge (using only non-air pollution, single-judge cases)
    # lsa_avg_by_judge = df_final.groupby('judge')[lsa_cols].mean().reset_index()
    # lsa_avg_by_judge.columns = ['judge'] + [f'avg_{col}' for col in lsa_cols]

    d2v_avg_by_judge = df_final.groupby('judge')[d2v_cols].mean().reset_index()
    d2v_avg_by_judge.columns = ['judge'] + [f'avg_{col}' for col in d2v_cols]

    # merge the averages together - change
    judge_vectors = pd.merge(lsa_avg_by_judge, d2v_avg_by_judge, on='judge', how='inner')

    # add case counts
    judge_case_counts = df_final.groupby('judge')['kanoon_id'].count().reset_index()
    judge_case_counts.columns = ['judge', 'non_air_pollution_single_judge_cases']
    judge_vectors = pd.merge(judge_vectors, judge_case_counts, on='judge', how='left')

    # sort by number of cases
    judge_vectors = judge_vectors.sort_values('non_air_pollution_single_judge_cases', ascending=False)
    
    
    # summary statistics
    print(f"Number of air pollution judges: {len(unique_air_pollution_judges)}")
    print(f"Number of air pollution judges with non-air pollution single-judge cases: {len(judge_vectors)}")
    print("\nSample of vector averages for first judge:")
    print(judge_vectors.iloc[0])
    print("\nDistribution of non-air pollution single-judge cases per judge:")
    print(judge_vectors['non_air_pollution_single_judge_cases'].describe())


    
# # create the final merged dataset
# def create_final_merged_dataset(df_air_pollution, judge_vectors, df_lsa_vecs, df_d2v_vecs):

#     df_air_pollution['kanoon_id'] = df_air_pollution['kanoon_id'].astype(str)
    
#     air_pollution_judge_cols = [col for col in df_air_pollution.columns if 'gpt4_judge_list' in col]
    
#     # create a clean judge column for air pollution cases
#     df_air_pollution['judge'] = df_air_pollution[air_pollution_judge_cols].apply(
#         lambda x: next((j for j in x if pd.notna(j)), None),
#         axis=1
#     )
    
#     # merge LSA vectors
#     result_df = pd.merge(
#         df_air_pollution,
#         df_lsa_vecs,
#         on='kanoon_id',
#         how='left'
#     )
    
#     # merge d2v vectors
#     result_df = pd.merge(
#         result_df,
#         df_d2v_vecs,
#         on='kanoon_id',
#         how='left'
#     )
    
#     # merge with judge vectors (average vectors by judge)
#     result_df = pd.merge(
#         result_df,
#         judge_vectors,
#         on='judge',
#         how='left'
#     )
    
#     lsa_cols = [col for col in df_lsa_vecs.columns if col != 'kanoon_id']
#     d2v_cols = [col for col in df_d2v_vecs.columns if col != 'kanoon_id']
    
#     # not sure if you need this?
#     result_df['has_direct_vectors'] = result_df[lsa_cols + d2v_cols].notna().any(axis=1)
#     result_df['has_judge_avg_vectors'] = result_df[[f'avg_{col}' for col in lsa_cols + d2v_cols]].notna().any(axis=1)
    
#     #fill in!!- not sure if you need this?
#     for col in lsa_cols + d2v_cols:
#         result_df[col] = result_df[col].fillna(result_df[f'avg_{col}'])
    
#     return result_df

# # create the final merged dataset
# final_result_df = create_final_merged_dataset(df_air_pollution, judge_vectors, df_lsa_vecs, df_d2v_vecs)

# # print summary statistics
# print("\nFinal Dataset Summary:")
# print(f"Total air pollution cases: {len(final_result_df)}")
# print(f"Cases with direct vectors: {final_result_df['has_direct_vectors'].sum()}")
# print(f"Cases with judge-averaged vectors: {final_result_df['has_judge_avg_vectors'].sum()}")
# print(f"Cases with any vectors (direct or averaged): {final_result_df[['has_direct_vectors', 'has_judge_avg_vectors']].any(axis=1).sum()}")