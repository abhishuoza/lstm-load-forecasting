import psutil
import pandas as pd
import csv

# Filtering from original data "CD_INTERVAL_READING_ALL_NO_QUOTES.csv" for IDs and between given time period, saved as "id_<id>.csv"

svmem = psutil.virtual_memory()
print(svmem.available)
PATH = r"D:/CD_INTERVAL_READING_ALL_NO_QUOTES.csv"
df_sample = pd.read_csv(PATH, nrows=10)
df_sample_size = df_sample.memory_usage(index=True).sum()
print(df_sample_size)
print(df_sample)
# define a chunksize that would occupy a maximum of 1Gb
# we divide by 10 because we have selected 10 lines in our df_sample
my_chunk = (1000000000 / df_sample_size) / 10
my_chunk = int(my_chunk // 1)  # we get the integer part
print(my_chunk)
# create the iterator
iter_csv = pd.read_csv(
    PATH,
    iterator=True,
    chunksize=my_chunk)
# concatenate according to a filter to our result dataframe
print("HEYO")
list = [10006414, 22222]
df_result = pd.concat(
    [chunk[((chunk["CUSTOMER_ID"] == 10509861) |
            (chunk["CUSTOMER_ID"] == 10595596) |
            (chunk["CUSTOMER_ID"] == 10598990) |
            (chunk["CUSTOMER_ID"] == 10692972) |
            (chunk["CUSTOMER_ID"] == 10702066) |
            (chunk["CUSTOMER_ID"] == 11081920) |
            (chunk["CUSTOMER_ID"] == 11462018) |
            (chunk["CUSTOMER_ID"] == 8145135) |
            (chunk["CUSTOMER_ID"] == 8147703) |
            (chunk["CUSTOMER_ID"] == 8149711) |
            (chunk["CUSTOMER_ID"] == 8156517) |
            (chunk["CUSTOMER_ID"] == 8176593) |
            (chunk["CUSTOMER_ID"] == 8181075) |
            (chunk["CUSTOMER_ID"] == 8184653) |
            (chunk["CUSTOMER_ID"] == 8196621) |
            (chunk["CUSTOMER_ID"] == 8196659) |
            (chunk["CUSTOMER_ID"] == 8196669) |
            (chunk["CUSTOMER_ID"] == 8196671) |
            (chunk["CUSTOMER_ID"] == 8198267) |
            (chunk["CUSTOMER_ID"] == 8198319) |
            (chunk["CUSTOMER_ID"] == 8198345) |
            (chunk["CUSTOMER_ID"] == 8211599) |
            (chunk["CUSTOMER_ID"] == 8257034) |
            (chunk["CUSTOMER_ID"] == 8257054) |
            (chunk["CUSTOMER_ID"] == 8264534) |
            (chunk["CUSTOMER_ID"] == 8273230) |
            (chunk["CUSTOMER_ID"] == 8273592) |
            (chunk["CUSTOMER_ID"] == 8282282) |
            (chunk["CUSTOMER_ID"] == 8291712) |
            (chunk["CUSTOMER_ID"] == 8308588) |
            (chunk["CUSTOMER_ID"] == 8326944) |
            (chunk["CUSTOMER_ID"] == 8328122) |
            (chunk["CUSTOMER_ID"] == 8334780) |
            (chunk["CUSTOMER_ID"] == 8342852) |
            (chunk["CUSTOMER_ID"] == 8347238) |
            (chunk["CUSTOMER_ID"] == 8350006) |
            (chunk["CUSTOMER_ID"] == 8351602) |
            (chunk["CUSTOMER_ID"] == 8376656) |
            (chunk["CUSTOMER_ID"] == 8419708) |
            (chunk["CUSTOMER_ID"] == 8432046) |
            (chunk["CUSTOMER_ID"] == 8451629) |
            (chunk["CUSTOMER_ID"] == 8459427) |
            (chunk["CUSTOMER_ID"] == 8466525) |
            (chunk["CUSTOMER_ID"] == 8478501) |
            (chunk["CUSTOMER_ID"] == 8482121) |
            (chunk["CUSTOMER_ID"] == 8487285) |
            (chunk["CUSTOMER_ID"] == 8487297) |
            (chunk["CUSTOMER_ID"] == 8487461) |
            (chunk["CUSTOMER_ID"] == 8496980) |
            (chunk["CUSTOMER_ID"] == 8504552) |
            (chunk["CUSTOMER_ID"] == 8519102) |
            (chunk["CUSTOMER_ID"] == 8523058) |
            (chunk["CUSTOMER_ID"] == 8540084) |
            (chunk["CUSTOMER_ID"] == 8557605) |
            (chunk["CUSTOMER_ID"] == 8566459) |
            (chunk["CUSTOMER_ID"] == 8568209) |
            (chunk["CUSTOMER_ID"] == 8617151) |
            (chunk["CUSTOMER_ID"] == 8618165) |
            (chunk["CUSTOMER_ID"] == 8655993) |
            (chunk["CUSTOMER_ID"] == 8661542) |
            (chunk["CUSTOMER_ID"] == 8673172) |
            (chunk["CUSTOMER_ID"] == 8679346) |
            (chunk["CUSTOMER_ID"] == 8680284) |
            (chunk["CUSTOMER_ID"] == 8685932) |
            (chunk["CUSTOMER_ID"] == 8687500) |
            (chunk["CUSTOMER_ID"] == 8733828) |
            (chunk["CUSTOMER_ID"] == 8804804) |
            (chunk["CUSTOMER_ID"] == 9012348) |
            (chunk["CUSTOMER_ID"] == 9393680)) & (chunk["READING_DATETIME"] >= '2013-06-01') & (
                       chunk["READING_DATETIME"] <= '2013-08-31')]
     for chunk in iter_csv])
# df_result = pd.concat(
#     [chunk[(chunk["timestamp"] >= '2018-06-01') & (chunk["timestamp"] <= '2018-08-31')]
#     for chunk in iter_csv])
print(df_result)

df_result.to_csv('C:/Users/abhis/Desktop/AU/UGRP/pythonProject1/8282282_filtered.csv')
print("done")
