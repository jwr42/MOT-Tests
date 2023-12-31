# EXPLORATORY DATA ANALYSIS OF MOT RESULTS
# Goal: find features that could be used to predict MOT results

# Accessed the MOT results data from data.gov.uk
# URL: https://www.data.gov.uk/dataset/e3939ef8-30c7-4ca8-9c7c-ad9475cc9b2f/anonymised-mot-tests-and-results
# Date Accessed: 2023-09-27

# import necessary dependencies
import pandas as pd
import matplotlib.pyplot as plt

# IMPORT DATA

# file with full dataset is too large, sampling first 1M rows of the datasets instead
df = pd.read_csv("data/test_result.csv", sep="|", nrows=1000000)
# Data Source: https://data.dft.gov.uk/anonymised-mot-test/test_data/dft_test_result_2022.zip

# Multiple test types in the datasets, some of which are re-tests
test_type_details = pd.read_csv("data/mdr_test_type.csv", header=0, sep="|")
# Data Source: https://data.dft.gov.uk/anonymised-mot-test/lookup.zip

# multiple test outcomes in the dataset details in a lookup table
test_result_details = pd.read_csv("data/mdr_test_outcome.csv", header=0, sep="|")
# Data Source: https://data.dft.gov.uk/anonymised-mot-test/lookup.zip
test_result_codes = test_result_details.result_code.unique().tolist()

# CLEAN DATA

# transform the date columns into datetime datatypes
df.test_date = pd.to_datetime(df.test_date)
df.first_use_date = pd.to_datetime(df.first_use_date, errors='coerce')  # missing data stored as NaT (Not a Time)

# transform the categorical columns into Category datatypes
cat_col_names = ["test_type", "test_result", "test_class_id", "postcode_area", "make", "model", "colour", "fuel_type"]
for col_name in cat_col_names:
    df[col_name] = df[col_name].astype("category")

# calculate the summary statistics for the dataset
df_describe = df.describe(include='all')

# removing all but the test categorised as "normal tests" to prevent data leakage
df = df[df.test_type == "NT"]

# removing all but the test class used for cars (class 4)
df = df[df.test_class_id == 4]

# check for duplicates in fields that should be unique vals
num_duplicate_vehicle_ids = len(df) - df.vehicle_id.nunique()  # duplicates detected in vehicle ids
num_duplicate_tests = len(df) - df.test_id.nunique()  # no duplicates detected in test ids

# Note: there is a specific classification for Re-Tests, we exclude them from the dataset when we filter for NT, so it
# is odd that we see duplicated vehicle ids in the dataset
df_duplicated_vehicle_ids = df[df.vehicle_id.duplicated(keep=False)].sort_values(by="vehicle_id")

# Most of the duplicates occur twice in the dataframe, from visual inspection of the dataframe it's clear that this
# commonly occurs when a car failed its initial MOT but retakes the test without the classification of "re-test"
# e.g. df[df.vehicle_id == 298937]["test_date"]
df_duplicated_counts = df_duplicated_vehicle_ids.groupby("vehicle_id").count()

# Approach to managing these duplicates: given that these rows contain useful information about the probability of a car
# passing its MOT on its first attempt we'll keep the first attempt at the MOT for each car in the dataframe by sorting
# the dataframe df by test_date and then keeping only the first entry for each vehicle_id - i.e. the fail result for
# those cars that do have duplicate entries representing multiple attempts
df.sort_values(by="test_date").drop_duplicates(subset="vehicle_id", keep="first", inplace=True)

# FEATURE ENGINEERING

# need to be able to classify the remaining rows as passes and non-passes
df["test_result_passed"] = df.test_result.apply(lambda result_code: 1 if result_code == "P" else 0)

df.test_result_passed.mean()  # Probability of passing MOT on first attempt = 0.689

# create column for the age of the car at the date of the tests
df["vehicle_age"] = df["test_date"]-df["first_use_date"]

# add the make and model into a single column to allow it to be considered a combined value
df["make_model"] = df["make"].astype(str) + " " + df["model"].astype(str)   # make & model are categorical dtypes

# extract value counts from categorical columns
df_make_model_freq = df.make_model.value_counts()
df_make_freq = df.make.value_counts()
df_fuel_type_freq = df.fuel_type.value_counts()

# TODO: create dummy variables for the fuel type columns
# TODO: create dummy vars for only the main model / make combinations
# TODO: visualisations / summary tables?

# quick visual inspection of the 
df_make_model_freq.head(75).sort_values().plot(kind='hist', bins=20, figsize=(10,5))
plt.savefig('plot_test.png')

print(df_make_model_freq.iloc[:100].values.sum()/len(df))

