import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import neighbors

# Import Datasets
j17 = pd.read_csv('conditions_j17.csv', dtype=str, skiprows = None)
comal = pd.read_csv('conditions_comal.csv', dtype=str, skiprows = None)

# Convert to DateTime
j17['DateTime_j17'] = pd.to_datetime(j17['DateTime_j17'])
comal['DateTime_Com'] = pd.to_datetime(comal['DateTime_Com'])

#Merge datasets into one dataset
merged = j17.merge(comal, left_on=['DateTime_j17'], right_on=['DateTime_Com'], how='left', suffixes=['_J17', '_Comal'])

# Drop NaN values
merged.dropna(axis=0,how='any', inplace=True)

print(merged.head())
print(merged.DateTime_Com.dtype)
print(merged.shape)


fix,ax = plt.subplots()

ax.plot(merged.iloc[0:,1:],merged.iloc[2:,3:])


# j17_train = merged.iloc[0:,1:]
# comal_train = merged.iloc[2:,3:]
#
#
#
# # train = train_test_split(merged.iloc[0], merged['MaxLevel_Comal'], random_state=33)
# lr = LinearRegression(normalize=True)
# lr.fit(merged['MaxLevel_J17'], merged['MaxLevel_Comal'])
# knn = neighbors.KNeighborsClassifier(n_neighbors=5)
