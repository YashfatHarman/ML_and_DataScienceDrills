import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from scipy.stats import norm, skew

color = sns.color_palette()
sns.set_style("darkgrid")

data_foldername = "Data"

filename = "train.csv"
df_train = pd.read_csv(data_foldername + "/" + filename)
print("size of train data:",df_train.shape)

#get a peek at the train data
#print(df_train.head(10).iloc[:, 0:20])

#print(df_train.describe().iloc[:,:20] )
#print(df_train.describe().iloc[:,20:] )

#print("train columns:", df_train.columns)
    #Index(['Id', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
    #       'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
    #       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
    #       'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
    #       'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
    #       'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
    #       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
    #       'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',
    #       'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',
    #       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
    #       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
    #       'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',
    #       'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',
    #       'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
    #       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',
    #       'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',
    #       'SaleCondition', 'SalePrice'],
    #      dtype='object')

filename = "test.csv"
df_test = pd.read_csv(data_foldername + "/" + filename)
print("size of test data:",df_test.shape)
#print("test columns:", df_test.columns)
    #same as train columns, excpet SalePrice.



'''
#drop the id column because we don't need it currently
'''

#but save before dropping
train_id = df_train["Id"]
test_id = df_test["Id"]

df_train.drop("Id", axis = 1, inplace = True)
df_test.drop("Id", axis = 1, inplace = True)
print("size of train data after dropping ID:",df_train.shape)
print("size of test data after dropping ID:",df_test.shape)

'''
# Dealing with outliers.
'''

#apparently there are a few outliers in GrLivArea.
#locate and remove those.

#plt.scatter(x = df_train["GrLivArea"], y = df_train["SalePrice"])
#plt.ylabel("SalePrice", fontsize = 13)
#plt.xlabel("GrLivArea", fontsize = 13)
#plt.show()

#there are a few points in the graph where the GrLivArea > 4000 but price < 200000.
#seems odd. remove these.

df_outliers = df_train[ (df_train["GrLivArea"] > 4000) & (df_train["SalePrice"] < 200000) ]
#print(df_outliers)
    #two records comes out

df_train = df_train.drop( df_outliers.index )
    #indices that dropped are [523, 1298]
print("size of train data after dropping outliers:",df_train.shape)


'''
# Correlation analysis
'''
#most correlated features
corrmat = df_train.corr()
    #there are only 38 columns that has numeric values. So this will return a 38*38 matrix.

#plotting the complete correlation matrix
#plt.figure(figsize = (10,10))
#g = sns.heatmap(df_train.corr())

#this is messy. Just get the top 10 features correlation-wise
top_corr_features = corrmat[abs(corrmat["SalePrice"]) > 0.5]["SalePrice"]
print("top correlated features:")
print(top_corr_features)
    #    
    #OverallQual     0.795774
    #YearBuilt       0.523608
    #YearRemodAdd    0.507717
    #TotalBsmtSF     0.651153
    #1stFlrSF        0.631530
    #GrLivArea       0.734968
    #FullBath        0.562165
    #TotRmsAbvGrd    0.537769
    #GarageCars      0.641047
    #GarageArea      0.629217
    #SalePrice       1.000000
    
#instead of feature names, just get their indices
top_corr_features_indices = corrmat.index[abs(corrmat["SalePrice"]) > 0.5]
print(top_corr_features_indices)

#plt.figure(figsize = (10,10))
#g = sns.heatmap(df_train[top_corr_features_indices].corr(), annot = True, cmap = "RdYlGn")
#plt.show()
    #here, we find that the four most correlated features with SalePrice are [OverallQual, GrLivArea, TotalBsmtSF, GarageCars]    
    #also, (GarageArea, GarageCars) and (TotalBsmtSF, 1stFlrSF) are heavily correlated, which makes sense.
    
#lets vizualize the correlations

#sns.barplot(df_train["OverallQual"],df_train["SalePrice"])
#plt.show()

#sns.barplot(df_train["GrLivArea"],df_train["SalePrice"])
#plt.show()

#sns.barplot(df_train["TotalBsmtSF"],df_train["SalePrice"])
#plt.show()

#TODO: for GrLivArea and TotalBsmtSF, it might be a better visualization to get ten buckets each (histogram),
#and then plot average SalePrice for each element in the bucket 

#sns.barplot(df_train["GarageCars"],df_train["SalePrice"])
#plt.show()
    
#scatterplot between "salePrice" and correlated variables
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
#sns.pairplot(df_train[cols], size = 2)
#plt.show()

#relationship between TotalBsmtSF and GrLiveArea
#sns.scatterplot(df_train.GrLiveArea, df_train.TotalBsmtSF)
#plt.scatter(x = df_train["GrLivArea"], y = df_train["TotalBsmtSF"])
    #so basement area can be equal or smaller than living area, but not larger. Common sense.
#plt.show()
'''
#target variable transform.
#Saleprice is in hundred thousnds, year_built is in thousads. Other columns have different range.
#Keeping them as is would mean some column gets more weight than others. So need to normalize them.
'''

#lets check the skewness of data
def check_skewness(col):
    sns.distplot(df_train[col], fit = norm)
    fig = plt.figure()
    res = stats.probplot(df_train[col], plot = plt)
    (mu,sigma) = norm.fit(df_train[col])
    print("\nmu = {:.2f} and sigma = {:.2f}".format(mu,sigma))
    plt.show()

def plot_histogram(col, bins = 100):
    #this can show an easier eyeballing of the skewness
    plt.hist(x = df_train[col], bins = bins)
    plt.xlabel(col)
    plt.ylabel("frequency")
    plt.show()
    
    
#check_skewness("SalePrice")
#plot_histogram("SalePrice")

#if we take log(1+x) conversion for SalePrice, the distribution becomes somewaht normal
#df_train["modified_saleprice"] = np.log1p(df_train["SalePrice"])
#check_skewness("modified_saleprice")
#plot_histogram("modified_saleprice")

#check_skewness("OverallQual")
#plot_histogram("OverallQual")
#check_skewness("GrLivArea")
#plot_histogram("GrLivArea")
#check_skewness("GarageCars")
#plot_histogram("GarageCars")
#check_skewness("TotalBsmtSF")
#plot_histogram("TotalBsmtSF")
#check_skewness("FullBath")
#plot_histogram("FullBath")
#check_skewness("YearBuilt")
#plot_histogram("YearBuilt")

'''
#Feature Engineering.
'''
#concatenate both train and test values
ntrain = df_train.shape[0]
ntest = df_test.shape[0]
y_train = df_train.SalePrice.values
print(y_train.shape)
all_data = pd.concat((df_train, df_test)).reset_index(drop = True)
print(all_data.shape)
all_data.drop(["SalePrice"], axis = 1, inplace = True)
print("finally, all data size is: {}".format(all_data.shape))

'''
#missing data
'''
all_data_na = ( all_data.isnull().sum() / len(all_data)) * 100
    # percent of missing value in each column
#print(all_data_na[0:40])
#print(all_data_na[40:])

#all_data_na = all_data_na.drop( all_data_na[all_data_na == 0].index ).sort_values(ascending = False) 
all_data_na = all_data_na[all_data_na > 0].sort_values(ascending = False)
    #top columns with missing values
print(all_data_na.shape)
print(all_data_na)

missing_data = pd.DataFrame(all_data_na)
missing_data.columns = ["Missing Ratio"]
print(missing_data.shape)
print(missing_data)

#plot it
f, ax = plt.subplots(figsize = (12,8))
plt.xticks(rotation = "90")
sns.barplot(x = all_data_na.index, y = all_data_na)
plt.xlabel("Features", fontsize = 15)
plt.ylabel("Percent of missing values", fontsize = 15)
plt.title("Percent of missing data by feature", fontsize= 15)

#columns with most missing value (%)
    #PoolQC            99.691464
    #MiscFeature       96.400411
    #Alley             93.212204
    #Fence             80.425094
    #FireplaceQu       48.680151
    #LotFrontage       16.660953

#only records where PoolQC has some data
PoolQC_not_null = all_data.PoolQC[all_data.PoolQC.notnull()]
print("only records where PoolQC has some data")
print(PoolQC_not_null)

'''
#handle missing data. Fill-up with probabble values.
'''
LotFrontage_null = all_data[ all_data["LotFrontage"].isnull() ][["LotFrontage","Neighborhood"]]
print("LotFrontage_null shape:", LotFrontage_null.shape)
print("first few:")
print(LotFrontage_null.head(10))

grouped_df = all_data.groupby("Neighborhood")["LotFrontage"]
temp_LotFrontage_medians_by_neighborhood = {}
for key,item in grouped_df:
    print(key)
    print(grouped_df.get_group(key).median())
    temp_LotFrontage_medians_by_neighborhood[key] = grouped_df.get_group(key).median()
    
#group by neighborhood and fill in missing ones with the median LotFrontage for the corresponding neighborhood
#all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median))

def tester(row):
    if pd.isnull(row["LotFrontage"]):
        return temp_LotFrontage_medians_by_neighborhood[row["Neighborhood"]]
    else:
        return row["LotFrontage"]

#temp = all_data[all_data["LotFrontage"].isnull()]["Neighborhood"].apply(lambda x: temp_LotFrontage_medians_by_neighborhood[x])
#print(temp.shape)
#print(temp)

all_data["test"] = all_data.apply(tester,axis = 1)

print("after fill-up:")
print(all_data.iloc[0:10][["LotFrontage", "Neighborhood","test"]])



























