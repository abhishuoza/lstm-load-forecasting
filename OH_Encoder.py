import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# After Filtering, before one hot encoding, I removed unnecessary columns, added TIME_STEP and DAY_OF_WEEK columns, saved as "customer_<id>.csv"

# One hot encoded the columns in "customer_<id>.csv", saved as "OH_customer_<id>.csv"

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
X_raw = pd.read_csv("IDs_with_timestep_dayofweek.csv")


# X_raw = X_raw[['TIME_STEP', 'DAY_OF_WEEK']]
# print(X_raw)
OH_cols = pd.DataFrame(OH_encoder.fit_transform(X_raw[['TIME_STEP', 'DAY_OF_WEEK']]))

# One-hot encoding removed index; put it back
OH_cols.index = X_raw.index

# Remove categorical columns (will replace with one-hot encoding)
num_X_raw = X_raw.drop(['TIME_STEP', 'DAY_OF_WEEK'], axis=1)

# Add one-hot encoded columns to numerical features
OH_X = pd.concat([num_X_raw, OH_cols], axis=1)
OH_X = OH_X.reset_index(drop = True)
print(OH_X)
OH_X.to_csv('C:/Users/abhis/Desktop/AU/UGRP/pythonProject1/OH_SGSC.csv')
