import numpy as np
import pandas as pd

#read data
df = pd.read_csv('../data/mlbootcamp5_train.csv', sep=';', index_col='id');

# 1. Check gender.
#       |    id|  count
# ======================
# female|     1|  45530
#   male|     2|  24470
print("\n1. Check gender via mean height: ")
print(df.groupby(['gender'])['height'].describe()[['count', 'mean']])

# 2. Test alco.
# woman use alco less often mеn
print("\n2. Test alco: ")
print(df.groupby(['gender'])['alco'].mean())

# 3. Test gender smoke.
# Result = 12
# via crosstab
print("\n3. Test gender smoke: ")
print("  Crosstab: ")
print(pd.crosstab(df['gender'], df['smoke']))
# calculate result
print("\n  Result: ", end='')
smoke_male_ratio = df[df['gender'] == 2]['smoke'].value_counts(normalize=True)[1]
smoke_female_ratio = df[df['gender'] == 1]['smoke'].value_counts(normalize=True)[1]
print(round(smoke_male_ratio / smoke_female_ratio))

# 4. Test smoke age difference.
# Result = 20 month
print("\n4. Test smoke age difference: ")
not_smoke_age, smoke_age = df.groupby(['smoke'])['age'].median().values
print(round(abs(not_smoke_age - smoke_age) / 30), " month")

# 5. Test CCZ.
# create new feature
print("\n.5 Test HeartScore")
df['age_years'] = round(df['age'] / 365)
# select smoke man with age in range of 60 to 64
df_man_smoke_60_64 = df[
    (df['smoke'] == 1)
    & (df['age_years'] >= 60)
    & (df['age_years'] <= 64)].copy()
# create two groups
sub_df1 = df_man_smoke_60_64[
    (df_man_smoke_60_64['ap_hi'] < 120)
    & (df_man_smoke_60_64['cholesterol'] == 1)].copy()
sub_df2 = df_man_smoke_60_64[
    (df_man_smoke_60_64['ap_hi'] >= 160)
    & (df_man_smoke_60_64['ap_hi'] < 180)
    & (df_man_smoke_60_64['cholesterol'] == 3)].copy()
# calculate result
hs_res = round(
    sub_df2['cardio'].value_counts(normalize=True)[1]
    / sub_df1['cardio'].value_counts(normalize=True)[1])
print('  HeartScore ratio of two groups is ', hs_res)

# 6. Body Mass Index (BMI)
print("\n6. Body Mass Index")
df['BMI'] = df['weight'] / (df['height'] / 100) ** 2
# check asserts
print("  ", df['BMI'].median() > 25)
print("  ", df[df['gender'] == 1]['BMI'].mean()
    < df[df['gender'] == 2]['BMI'].mean())
print("  ", df[df['cardio'] == 0]['BMI'].mean()
    > df[df['cardio'] == 1]['BMI'].mean())
gr1 = df[
    (df['gender'] == 2)
    & (df['cardio'] == 0)
    & (df['alco'] == 0)].copy()
gr2 = df[
    (df['gender'] == 1)
    & (df['cardio'] == 0)
    & (df['alco'] == 0)].copy()
print("  ", gr1['BMI'].mean() < gr2['BMI'].mean())

# 7. Clear data frame
print("\n7. Clear data frame")
clear_df = df.drop(df[ df['ap_lo'] > df['ap_hi'] ].index)
h_lo_limit = clear_df['height'].quantile(0.025)
h_hi_limit = clear_df['height'].quantile(0.975)
w_lo_limit = clear_df['weight'].quantile(0.025)
w_hi_limit = clear_df['weight'].quantile(0.975)
clear_df = clear_df.drop( clear_df[clear_df['height'] < h_lo_limit ].index)
clear_df = clear_df.drop( clear_df[clear_df['height'] > h_hi_limit ].index)
clear_df = clear_df.drop( clear_df[clear_df['weight'] < w_lo_limit ].index)
clear_df = clear_df.drop( clear_df[clear_df['weight'] > w_hi_limit ].index)

clear_data_count = clear_df.shape[0]
dirty_data_count = df.shape[0]
print("  Waste data percentage: ", end='')
print(round((dirty_data_count - clear_data_count) / dirty_data_count * 100))