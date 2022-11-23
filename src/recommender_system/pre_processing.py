# Data pre-processing
import pandas as pd

from numpy import unique


# PREVIEW
def preProcessing(ratings):
    ratings.info()  # Checking the info of the dataset
    #There are 100,004 observations and 4 columns in the data

    #Drop columns that are not numeric
    # Get numerical columns only from the dataframe
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    ratings = ratings.select_dtypes(include=numerics)

    #zero-variance predictors
    counts = ratings.nunique()
    to_del = [i for i,v in enumerate(counts) if v == 1]
    print(to_del)
    # drop useless columns
    ratings.drop(to_del, axis=1, inplace=True)   

    # counts = ratings.nunique()
    # #Few-value columns
    # # record columns to delete
    # to_del = [i for i,v in enumerate(counts) if (float(v)/ratings.shape[0]*100) < 1] 
    # print("before")
    # print(to_del)
    # print("after")
    # # drop useless columns
    # if(len(to_del)>0):
    #     ratings.drop(to_del, axis=1, inplace=True)   

    #data deduplication
    # delete duplicate rows 
    # ratings.drop_duplicates(inplace=True) 
    
    #Missing values 
    #checking for null values
    # print(ratings.isnull().sum().sum())  # zero null values
    sum_of_null_values = ratings.isnull().sum().sum()
  
    if(sum_of_null_values>1):
        ratings.replace("", np.nan, regex=False, inplace=True)  # replace the dashes with Nan

    # Dropping the timestamp column
    ratings.drop(['timestamp'], axis=1, inplace=True)
    return ratings
    # ratings.info()
 