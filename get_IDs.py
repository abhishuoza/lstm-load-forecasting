import psutil
import pandas as pd
import csv

# Filtering from original data "CD_INTERVAL_READING_ALL_NO_QUOTES.csv" for a specific id and between given time period, saved as "id_<id>.csv"

svmem = psutil.virtual_memory()
print (svmem.available)
PATH = r"sgsc-cthanplug-readings.csv"
df_sample = pd.read_csv(PATH, nrows=10)
df_sample_size = df_sample.memory_usage(index=True).sum()
print (df_sample_size)
print (df_sample)
# define a chunksize that would occupy a maximum of 1Gb
# we divide by 10 because we have selected 10 lines in our df_sample
my_chunk = (1000000000 / df_sample_size)/10
my_chunk = int(my_chunk//1) # we get the integer part
print (my_chunk)
# create the iterator
iter_csv = pd.read_csv(
    PATH,
    iterator=True,
    chunksize=my_chunk)
# concatenate according to a filter to our result dataframe
print("HEYO")
df_result = pd.concat(
    [chunk[(chunk[" PLUG_NAME"] == "Hot Water System") & (chunk["READING_TIME"] >= '2013-06-01') & (chunk["READING_TIME"] <= '2013-08-31')]
    for chunk in iter_csv])
# df_result = pd.concat(
#     [chunk[(chunk["timestamp"] >= '2018-06-01') & (chunk["timestamp"] <= '2018-08-31')]
#     for chunk in iter_csv])
print (df_result)

df_result.to_csv('C:/Users/abhis/Desktop/AU/UGRP/pythonProject1/plug_readings_ids.csv')

print("done")

IDs = df_result["CUSTOMER_ID"].unique()







# Filtering from original data "CD_INTERVAL_READING_ALL_NO_QUOTES.csv" for a specific id and between given time period, saved as "id_<id>.csv"

svmem = psutil.virtual_memory()
print (svmem.available)
PATH = r"D:/CD_INTERVAL_READING_ALL_NO_QUOTES.csv"
df_sample = pd.read_csv(PATH, nrows=10)
df_sample_size = df_sample.memory_usage(index=True).sum()
print (df_sample_size)
print (df_sample)
# define a chunksize that would occupy a maximum of 1Gb
# we divide by 10 because we have selected 10 lines in our df_sample
my_chunk = (1000000000 / df_sample_size)/10
my_chunk = int(my_chunk//1) # we get the integer part
print (my_chunk)
# create the iterator
iter_csv = pd.read_csv(
    PATH,
    iterator=True,
    chunksize=my_chunk)
# concatenate according to a filter to our result dataframe
print("HEYO")
df_result = pd.concat(
    [chunk[(chunk["CUSTOMER_ID"] in IDs) & (chunk["READING_DATETIME"] >= '2013-06-01') & (chunk["READING_DATETIME"] <= '2013-08-31')]
    for chunk in iter_csv])
# df_result = pd.concat(
#     [chunk[(chunk["timestamp"] >= '2018-06-01') & (chunk["timestamp"] <= '2018-08-31')]
#     for chunk in iter_csv])
print (df_result)

df_result.to_csv('C:/Users/abhis/Desktop/AU/UGRP/pythonProject1/All_IDs_filtered.csv')
print("done")

