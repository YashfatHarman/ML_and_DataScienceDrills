import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.graphics.mosaicplot import mosaic
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import VotingClassifier

import xgboost as xgb
#import lightgbm as lgb #light gbm can nly be used in 64 bit python systems.

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix

print("Hello world!")

train = pd.read_csv("Data/train.csv")
test = pd.read_csv("Data/test.csv")

print("train shape: ", train.shape)
print("test shape: ", test.shape)

combine = pd.concat([train.drop("Survived",1), test])
print("combine shape: ", combine.shape)

#printing first 10 rows to train to get an idea
print(train.head(10))

print(train.describe())

#missing values
print("\n\nAmount of missing values:")
print(train.isnull().sum(axis=0))

print(test.info())

#separate the ones who survived and who didn't
surv = train[train["Survived"]==1]
nosurv = train[train["Survived"]==0]
surv_col = "blue"
nosurv_col = "red"

print("Survived: {} ({:.2f} percent); Not survived: {} ({:.2f} percent); Total: {}".format(len(surv), len(surv)/len(train)*100, len(nosurv), len(nosurv)/len(train)*100, len(train)))

#create barplot for categorical features
#create histograms for scaled features
#plt.figure(figsize=[12, 10])

#plt.subplot(331)
#sns.distplot(surv["Age"].dropna().values, bins = range(0,81,1), kde = False, color = surv_col)
#sns.distplot(nosurv["Age"].dropna().values, bins = range(0,81,1), kde = False, color = nosurv_col, axlabel = "Age")

#plt.subplot(332)
#sns.barplot("Sex","Survived", data=train)

#plt.subplot(333)
#sns.barplot("Pclass","Survived", data=train)

#plt.subplot(334)
#sns.barplot("Embarked","Survived", data=train)

#plt.subplot(335)
#sns.barplot("SibSp","Survived", data=train)

#plt.subplot(336)
#sns.barplot("Parch","Survived", data=train)

#plt.subplot(337)
#sns.distplot(np.log10(surv["Fare"].dropna().values + 1), kde = False, color = surv_col)
#sns.distplot(np.log10(nosurv["Fare"].dropna().values + 1), kde = False, color = nosurv_col, axlabel = "Fare")

#plt.subplots_adjust(top = 0.92, bottom = 0.08, left = 0.10, right = 0.95, hspace = 0.25, wspace = 0.35)

#plt.show() 


tab = pd.crosstab(train["SibSp"], train["Survived"])
print(tab)

#cabin numbers. How many do we know?
print("We know {} of {} cabin numbers in the train data set.".format(len(train["Cabin"].dropna()), len(train)))
print("We know {} of {} cabin numbers in the test data set.".format(len(test["Cabin"].dropna()), len(test)))

#print some sample cabin numbers
#print(train.loc[:,["Survived", "Cabin"]].dropna().head(50))

#Just testing. These two means the same. The number of people who survived and their cabin no are known.
#print(len(surv["Cabin"].dropna()))
#print(len(train[train["Survived"] == 1]["Cabin"].dropna()))

#let's find out how many unique ticket numbers are there
print("There are {} unique ticket numbers among the {} tickets.".format(train["Ticket"].nunique(), train["Ticket"].count()))

#so, sharing ticket number was not uncommon. Let's dig that up a bit.
#grouped = train.groupby("Ticket")
#k = 0
#for name, group in grouped:
#    if(len(grouped.get_group(name)) > 1):
#        print(group.loc[:, ["Survived", "Name", "Fare", "Age"]])
#        k += 1
        
'Check the correlation between features by using overview plot.'
#plt.figure(figsize = (14,12))
#sns.heatmap(train.drop("PassengerId", axis = 1).corr(), vmin = -1.0, vmax = 1.0, square = True, annot = True)
#plt.show()        

'Also, check correlation between features using pairplot'
#cols = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"]
#g = sns.pairplot(data = train.dropna(), vars = cols, size = 1.5, hue = "Survived", palette = [nosurv_col, surv_col])
#g.set(xticklabels = [])
#plt.show()
        

'Age vs survived distribution plot. Shows that for young adults, being female increases your survival chance.'
msurv = train[(train["Survived"] == 1) & (train["Sex"] == "male")]
fsurv = train[(train["Survived"] == 1) & (train["Sex"] == "female")]
mnosurv = train[(train["Survived"] == 0) & (train["Sex"] == "male")]
fnosurv = train[(train["Survived"] == 0) & (train["Sex"] == "female")]

#plt.figure(figsize = [13,5])
#plt.subplot(121)
#sns.distplot(fsurv["Age"].dropna().values, bins = range(0,81,1), kde = False, color = surv_col)
#sns.distplot(fnosurv["Age"].dropna().values, bins = range(0,81,1), kde = False, color = nosurv_col, axlabel = "female Age")

#plt.subplot(122)
#sns.distplot(msurv["Age"].dropna().values, bins = range(0,81,1), kde = False, color = surv_col)
#sns.distplot(mnosurv["Age"].dropna().values, bins = range(0,81,1), kde = False, color = nosurv_col, axlabel = "male Age")
#plt.show()

'violin plot to show the correlation between pclass and age'
#sns.violinplot(x = "Pclass", y = "Age", hue = "Survived", data = train, split = True)
#plt.hlines([0,10], xmin = -1, xmax = 3, linestyles = "dotted")
#plt.show()

'mosaic plot for Pclass vs Sex'
#dummy = mosaic(train, ['Survived', 'Sex', 'Pclass'])
#plt.show()

'Jumping a bit.'

'---------------------------------------------------------------------'

'Lets fill up the missing values.'
#Too many cabin numbers are missing, so no way to fill those up.

#Two passangers does not have embarkation port data. Lets fill that up.
#print(train[ train["Embarked"].isnull() ] ) #two woman, travelling alone, first class

#print(combine.where((combine["Embarked"] != "Q") & (combine["Pclass"] < 1.5) & (combine["Sex"] == "female")).groupby(["Embarked","Pclass", "Sex", "Parch", "SibSp"]).size())

#results show that 30 single female in first class in C, vs 20 similars in S.
#so we select "C" as our filler data value.
train["Embarked"].iloc[61] = "C"
train["Embarked"].iloc[829] = "C"
 
#the passanger without a ticket fare on record
print(test[test["Fare"].isnull()])
# a 60 year old single third class passanger. 
# We will just assign the median of third class fares to him.
test["Fare"].iloc[152] = combine["Fare"][combine["Pclass"] == 3].dropna().median() 
print(test["Fare"].iloc[152])


'---------------------------------------------------------------------'
'Lets work with engineered features.'

'List all new features that we can think of in one place.'

combine = pd.concat([train.drop("Survived", 1), test])
survived = train["Survived"]

#print(combine.shape)
#print(survived.shape)

combine["Child"] = combine["Age"] <= 10 #just saving the indices, not creating new dataframe at this point
combine["Cabin_known"] = combine["Cabin"].isnull() == False
combine["Age_known"] = combine["Age"].isnull() == False
combine["Family"] = combine["SibSp"] + combine["Parch"]
combine["Alone"] = (combine["SibSp"] + combine["Parch"]) == 0
combine["Large_Family"] = (combine["SibSp"] > 2) | (combine['Parch'] > 3)
combine["Deck"] = combine["Cabin"].str[0]
combine["Deck"] = combine["Deck"].fillna(value = "U")
combine["Ttype"] = combine["Ticket"].str[0]
combine["Title"] = combine["Name"].str.split(", ", expand = True)[1].str.split(".", expand = True)[0]   #str.split() belongs to pandas, works the same as string split. Expand = True returns the splitted dataframe instead of just index.
combine["Fare_cat"] = pd.DataFrame(np.floor(np.log10(combine["Fare"] + 1))).astype("int")
combine["Bad_ticket"] = combine["Ttype"].isin(["3","4","5","6","7","8","A","L","W"])
    #why are they bad, again? Check, there is an explanation i'm forgetting
    #Answer: these ticket types have less than 25% survival rate. (Though it could just be statistical fluctuations conidering the overall survival rate was around 38% )
combine["Young"] = (combine["Age"] < 30) | (combine["Title"].isin(["Master","Miss","Mlle"]))
combine["Shared_ticket"] = np.where(combine.groupby("Ticket")["Name"].transform("count") > 1, 1, 0)
combine["Ticket_group"] = combine.groupby("Ticket")["Name"].transform("count")
combine["Fare_eff"] = combine["Fare"]/combine["Ticket_group"]
combine["Fare_eff_cat"] = np.where(combine["Fare_eff"]>16.0, 2, 1)
combine["Fare_eff_cat"] = np.where(combine["Fare_eff"] < 8.5, 0, combine["Fare_eff_cat"])
test = combine.iloc[len(train):]
train = combine.iloc[:len(train)]
train.loc[:,"Survived"] = survived

surv = train[train["Survived"] == 1]
nosurv = train[train["Survived"] == 0]

'Now lets check how important these derived features are.'

'1. First, check the importantce of: Child'

#tab = pd.crosstab(train["Child"], train["Pclass"])
#print(tab)

#tab = pd.crosstab(train["Child"], train["Sex"])
#print(tab)

#g = sns.factorplot(x = "Sex", y = "Survived", hue = "Child", col = "Pclass", data = train, aspect = 0.9, size = 3.5, ci = 95.0)
#plt.show()

'We learn: Male children appear to have a survival advantage in 2nd and 3rd class'

'2. Then check the importance of: Cabin_known'
#cab = pd.crosstab(train["Cabin_known"], train["Survived"])
#print(cab)

#dummy = cab.div(cab.sum(1).astype(float), axis = 0).plot(kind = "bar", stacked = True)
#dummy = plt.xlabel("Cabin known")
#dummy = plt.ylabel("Percentage")
#plt.show()

#g = sns.factorplot(x = "Sex",y = "Survived", hue = "Cabin_known", col = "Pclass", data = train, aspect = 0.9, size = 3.5, ci = 95.0)
#plt.show()

'We learn: it is more likely to know the cabin of a passenger who survived. Can be useful'


'3. Then check the importance of: Deck'

#tab = pd.crosstab(train["Deck"], train["Survived"])
#print(tab)
#dummy = tab.div(tab.sum(1).astype(float), axis = 0).plot(kind="bar", stacked = True)
#dummy = plt.xlabel("Deck")
#dummy = plt.ylabel("Percentage")
#plt.show()

'We learn: if we know the deck number, percentage of survival is much higher than fthose for whom we do not know the deck.'
'Among the known decks, B, D and E have best suurvival chance at 66%. C and F have 60%. A and G at 50%.'

'''
For B and C, the distribution is like this
Survived    0    1
Deck              
A           8    7
B          12   35
C          24   35
Is the difference significant? Do a binomial test.
'''
#prob = stats.binom_test(x = 12, n = 12+35, p = 24/(24+35))
'''
    #    Here, x = number of occurance, n = number of trial, p = exected probability 
    #    As per my understanding, here we are saying that if we the case of C as the regular case of
    #    probability (that is 24 out of 24+35 try), and then if the number of occurance drops to 12
    #    out of (12+35) try, can that happen just by chance? Or is it a significant drop?
    #    If the test result is les than 5%, means there is less that 5% chance that this will happen
    #    by chance, we accept that as significant. 
'''
#print(prob)
'The result is 0.0374155274012 < 5%, so just about formally significant.'

#g = sns.factorplot(x = "Deck", y = "Survived", hue = "Sex", col = "Pclass", data = train, aspect = 0.9, size = 3.5, ci = 95.0 )
#plt.show()
'We learn: there are some variations across cabins for male passengers in first class, but doesnt look very significant.'


'4. Then check the importance of: Ttype and Bad_ticket'
#print(train["Ttype"].unique())
#print(test["Ttype"].unique())
#tab = pd.crosstab(train["Ttype"], train["Survived"])
#print(tab)
#sns.barplot(x = "Ttype", y = "Survived", data = train, ci = 95.0, color = "green")
#plt.show()
'We learn: The plot shows significant difference between Ttype 1,2,3. This can be useful.'

#tab = pd.crosstab(train["Bad_ticket"], train["Survived"])
#print(tab)
#g = sns.factorplot(x = "Bad_ticket", y = "Survived", hue = "Sex", col = "Pclass", data = train, aspect = 0.9, size = 3.5, ci = 95.0)
#plt.show()
'We learn: bad_tickets are bad for males and third class passengers.'

#tab = pd.crosstab(train["Deck"], train["Bad_ticket"])
#print(tab)
#dummy = tab.div(tab.sum(1).astype(float), axis = 0).plot(kind = "bar", stacked = True)
#dummy = plt.xlabel("Deck")
#dummy = plt.ylabel("Percentage")
#plt.show()
'Looks like there are more bad_tickets in decks D, F,G and U. But the correlations desnt look strong enough.'

'5. Then check the importance of: Age_known'
#tab = pd.crosstab(train["Age_known"], train["Survived"])
#print(tab)
#dummy = tab.div(tab.sum(1).astype(float), axis = 0).plot(kind = "bar", stacked = True)
#dummy = plt.xlabel("Age known")
#dummy = plt.ylabel("Percentage")
#plt.show()
'Looks like there is 10% more chance of survival if the age is known. Is it significant?'
#prob = stats.binom_test(x = 424, n = 424+290, p = 125 / (125+52))
#print(prob)
'prob is 1.56232645422e-10 << 5%, so it is indeed significant.'
'However, is the effect already captured by age or sex?'
#g = sns.factorplot(x = "Sex", y = "Age_known", col = "Pclass", data = train, aspect = 0.9, size = 3.5, ci = 95.0 )
#plt.show()
'We learn: we are mre likely to know the age of females or higher class passangers, which are already good survival predictors. The causality might be reversed though, but the main thing is the feature Age_known might be redundant.'


'6. Then check the importance of: Family'
#tab = pd.crosstab(train["Family"], train["Survived"])
#print(tab)
#dummy = tab.div(tab.sum(1).astype(float), axis = 0).plot(kind = "bar", stacked = True)
#dummy = plt.xlabel("Family Members")
#dummy = plt.ylabel("Percentage")
#plt.show()
'We learn: Having 1-3 family members works best for survival'


'7. Then check the importance of: Alone'
#tab = pd.crosstab(train["Alone"], train["Survived"])
#print(tab)
#sns.barplot('Alone', 'Survived', data = train)
#plt.show()
'People Travelling alone seems to be 20% more likely to be killed. Seems significant.'
#g = sns.factorplot(x = "Sex", y = "Alone", col = "Pclass", data = train, aspect = 0.9, size = 3.5, ci = 95.0)
#plt.show()
'But as we can see from the factorplot, more men were travelling alone than the women, so in the presence of sex this feature may not be very sgnificant.'
'We learn: play with this feature carefully in the modelling step.'

'8. Then check the importance of: Large_family'
#tab = pd.crosstab(train["Large_Family"], train["Survived"])
#print(tab)
#sns.barplot('Large_Family', 'Survived', data = train)
#plt.show()
'Looks like having a lage_family reduces our chance of survival.'
#g = sns.factorplot(x = "Sex", y = "Large_Family", col = "Pclass", data = train, aspect = 0.9, size = 3.5, ci = 95.0)
#plt.show()
'More large families were travelling in the third class. Also, 3rd class women were mroe likely to be part of a large family then 3rd class males. We already saw earlier that third class males were mostly travelling alone.'

'9. Then check the importance of: Shared_ticket'
#tab = pd.crosstab(train["Shared_ticket"], train["Survived"])
#print(tab)
#sns.barplot('Shared_ticket', 'Survived', data = train)
#plt.show()
'sharing a ticket appears good for survival'
#tab = pd.crosstab(train["Shared_ticket"], train["Sex"])
#print(tab)
#g = sns.factorplot(x = "Sex", y = "Shared_ticket", col = "Pclass", data = train, aspect = 0.9, size = 3.5, ci = 95.0)
#plt.show()
'But again, sharing ticket is more common for first class passengers and females.'

'We learn: Several of these derived features are strongly correlated with Sex and Pclass.'
'Whether there is any actual signal that can be used to imporove the learning accuracy needs to be investigated.'

'10. Lets have a closer look at title and see if they can be used to deduce missing age values.'
#print(combine["Age"].groupby(combine["Title"]).count())
#print(combine["Age"].groupby(combine["Title"]).mean())
#print(combine["Age"].groupby(combine["Title"]).median())
#print("There are {} unique titles in total.".format(len(combine["Title"].unique())))
#'There are 18 unique titles, but the most frequent ones are:  Mr (581), Miss (210), Mrs (170), and Master (53)'
#dummy = combine[combine["Title"].isin(["Mr","Miss", "Mrs", "Master"])]
#foo = dummy["Age"].hist(by=dummy["Title"], bins = np.arange(0,81,1))
#plt.show()
'So Master is very clearly capturing male children. Miss has a sizeable overlap with Mrs, but it is still more likely to indcate to younger women/teenagers.'
'So we will combine Age and Title to identify young population.'
'Anyone with Age < 30, or having a title of Master, Miss and Mlle are classified as Young.'
#tab = pd.crosstab(train["Young"], train["Survived"])
#print(tab)
#sns.barplot("Young", "Survived", data=train)
#plt.show()

#tab = pd.crosstab(train['Young'], train['Pclass'])
#print(tab)
#g = sns.factorplot(x="Sex", y="Young", col="Pclass", data=train, aspect=0.9, size=3.5, ci=95.0)
#plt.show()
'Because of classifying Miss as young, more women were classified as Young than males.'

'11. Lets look into Fare_Category'
'First, check again the distribution of Fare with respect to Pclass'
#plt.figure(figsize = [6,5])

#plt.subplot(311)
#ax1 = sns.distplot(   np.log10(surv["Fare"][surv["Pclass"] == 1].dropna().values+1), kde = False, color = surv_col )
#ax1 = sns.distplot(   np.log10(nosurv["Fare"][nosurv["Pclass"] == 1].dropna().values+1), kde = False, color = nosurv_col )
#ax1.set_xlim(0, np.max(np.log10(train["Fare"].dropna().values + 1)))

#plt.subplot(312)
#ax1 = sns.distplot(   np.log10(surv["Fare"][surv["Pclass"] == 2].dropna().values+1), kde = False, color = surv_col )
#ax1 = sns.distplot(   np.log10(nosurv["Fare"][nosurv["Pclass"] == 2].dropna().values+1), kde = False, color = nosurv_col )
#ax1.set_xlim(0, np.max(np.log10(train["Fare"].dropna().values + 1)))

#plt.subplot(313)
#ax1 = sns.distplot(   np.log10(surv["Fare"][surv["Pclass"] == 3].dropna().values+1), kde = False, color = surv_col )
#ax1 = sns.distplot(   np.log10(nosurv["Fare"][nosurv["Pclass"] == 3].dropna().values+1), kde = False, color = nosurv_col )
#ax1.set_xlim(0, np.max(np.log10(train["Fare"].dropna().values + 1)))

#plt.subplots_adjust(top = 0.92, bottom = 0.08, left = 0.10, right = 0.95, hspace = 0.25, wspace = 0.35)
#plt.show()

'The distribution is pretty broad. To simplify, we can classify fares into three categories: 0-10, 10-100, 100+'
'We defined the feature Fare_cat this way.'
#tab = pd.crosstab(train["Fare_cat"], train["Survived"])
#print(tab)
#sns.barplot("Fare_cat", "Survived", data=train)
#plt.show()
'clearly, rich people has more chance of survival'
#g = sns.factorplot(x = "Sex", y = "Fare_cat", col = "Pclass", data = train, aspect = 0.9, ci = 95.0)
#plt.show()
'Interestingly, women are more likely to have a more expensive ticket categry than men. Is it because women were more likely to need a cabin?'

'12. Checking out Fare_eff category. '
'first check if all people sharing a ticket is paying the same price'
#combine.groupby("Ticket")["Fare"].transform("std").hist()
#plt.show()
#num = np.sum(combine.groupby("Ticket")["Fare"].transform("std") > 0)
#print("number of people sharing tickets but have different fares: ", num)
##there are only 2 such people. Lets check them out.
#dummy = combine.iloc[np.where(combine.groupby("Ticket")["Fare"].transform("std") > 0)]
#print(dummy)
#fares are close enough to ignore

'Lets assume that the fares are total fares for the cabin. So per person cost will be total ticket cost divided by thenumber of people sharing them. We created a feature called Fare_eff this way. Lets see how that changes the fare distriution.'

#plt.figure(figsize = [6,5])

#plt.subplot(311)
#ax1 = sns.distplot(   np.log10(surv["Fare_eff"][surv["Pclass"] == 1].dropna().values+1), kde = False, color = surv_col )
#ax1 = sns.distplot(   np.log10(nosurv["Fare_eff"][nosurv["Pclass"] == 1].dropna().values+1), kde = False, color = nosurv_col )
#ax1.set_xlim(0, np.max(np.log10(train["Fare_eff"].dropna().values + 1)))

#plt.subplot(312)
#ax1 = sns.distplot(   np.log10(surv["Fare_eff"][surv["Pclass"] == 2].dropna().values+1), kde = False, color = surv_col )
#ax1 = sns.distplot(   np.log10(nosurv["Fare_eff"][nosurv["Pclass"] == 2].dropna().values+1), kde = False, color = nosurv_col )
#ax1.set_xlim(0, np.max(np.log10(train["Fare_eff"].dropna().values + 1)))

#plt.subplot(313)
#ax1 = sns.distplot(   np.log10(surv["Fare_eff"][surv["Pclass"] == 3].dropna().values+1), kde = False, color = surv_col )
#ax1 = sns.distplot(   np.log10(nosurv["Fare_eff"][nosurv["Pclass"] == 3].dropna().values+1), kde = False, color = nosurv_col )
#ax1.set_xlim(0, np.max(np.log10(train["Fare_eff"].dropna().values + 1)))

#plt.subplots_adjust(top = 0.92, bottom = 0.08, left = 0.10, right = 0.95, hspace = 0.25, wspace = 0.35)
#plt.show()
'This makes the fare distribution much narrower, and almost no overlaping. So this seems like the plausible case.'

'Compare the standard deviations of values in Fare and Fare_eff.'
#print( combine[combine["Fare"]>1].groupby("Pclass")["Fare"].std() )
#print( combine[combine["Fare_eff"]>1].groupby("Pclass")["Fare_eff"].std() )

'Also we can investigate the outliers as seen in the distplot.'
#print(combine[(combine["Pclass"] == 1) & (combine["Fare_eff"] > 0) & (combine["Fare_eff"] < 10) ])

#print(combine[(combine["Pclass"] == 3) & (np.log10(combine["Fare_eff"]) > 1.2) ])

'both of the cases looks like data entry error. Probably not a big deal.'

'However, for prediction Fare or Fare_eff will probablyw ont help much, as most of the information is captured in Pclass anyway. This graphs shows it clearly.'
#ax = sns.boxplot(x = "Pclass", y = "Fare_eff", hue = "Survived", data = train )
#ax.set_yscale("log")
#ax.hlines([8.5,16], -1, 4, linestyles = "dashed")
#plt.show()

'from the graph, it seems like the horizontal lines 8.5 and 16 cleanly separates the Fare_eff values into the three corresponding classes. So we used these values to define the feature Fare_eff_cat.'
#tab = pd.crosstab(train["Fare_eff_cat"], train["Survived"])
#print(tab)
#sns.barplot("Fare_eff_cat", "Survived", data= train)
#plt.show()

#g = sns.factorplot(x = "Sex", y = "Fare_eff_cat", col = "Pclass", data = train, aspect = 0.9, size = 3.5, ci = 95.0)
#plt.show()

'''
Done with investigation. Next start modeling.
------------------------------------------------------------------------
'''

'6. Preparing for modelling'

'Modify the categorical string column types to integers.'
combine = pd.concat([train.drop("Survived",1), test])   #drop "Survive" colun along axis 1, then concat to test
survived = train["Survived"]

combine["Sex"] = combine["Sex"].astype("category")
combine["Sex"].cat.categories = [0,1]
combine["Sex"] = combine["Sex"].astype("int")

combine["Embarked"] = combine["Embarked"].astype("category")
combine["Embarked"].cat.categories = [0,1,2]
combine["Embarked"] = combine["Embarked"].astype("int")

combine["Deck"] = combine["Deck"].astype("category")
combine["Deck"].cat.categories = [0,1,2,3,4,5,6,7,8]
combine["Deck"] = combine["Deck"].astype("int")

test = combine.iloc[len(train):]
train = combine.iloc[:len(train)]
train.loc[:,"Survived"] = survived

#print(train.loc[:,["Sex","Embarked"]].head())

'A final correlation matrix before modelling starts'
#ax = plt.subplots(figsize = (12,10))
#foo = sns.heatmap(train.drop("PassengerId", axis = 1).corr(), vmax = 1.0, square = True, annot = True)
#plt.show()
'The heatmap shows the new features have heavy correlation witht he features they were created from. Kinda no brainer. Will need to keep only the ones that carry the strongest signals.'

'train-test split'
training, testing = train_test_split(train, test_size = 0.2, random_state = 0)
print("Total sample: {}, train: {}, test: {}".format(train.shape[0], training.shape[0], testing.shape[0]))

'Try regression'
cols = ["Sex", "Pclass", "Cabin_known", "Large_Family", "Parch", "SibSp", "Young", "Alone", "Shared_ticket", "Child"]   #some arbitrarily selected columns. A lot of them should be redundant.

#cols = ["Sex", "Pclass", "Large_Family", "Young", "Cabin_known"]   #some arbitrarily selected columns. A lot of them should be redundant.

'''
Costs before standardization:
#all 10 cols:
#Train score: 0.816011235955
#Test score: 0.815642458101

# -SibSp, -Parch
#Train score: 0.814606741573
#Test score: 0.810055865922

# only: "Sex", "Pclass", "Large_Family", "Young"
#Train score: 0.806179775281
#Test score: 0.810055865922

# Only Sex and Pclass:
#Train score: 0.786516853933
#Test score: 0.787709497207

# "Sex", "Pclass", "Parch", "SibSp", "Young" 
#Train score: 0.801966292135
#Test score: 0.810055865922

#["Sex", "Pclass", "Large_Family", "Young", "Child"]
#Train score: 0.813202247191
#Test score: 0.798882681564

#["Sex", "Pclass", "Large_Family", "Child"]
#Train score: 0.813202247191
#Test score: 0.798882681564

#"Sex", "Pclass", "Large_Family", "Child", "Young", "Alone"
#Train score: 0.813202247191
#Test score: 0.798882681564

#"Sex", "Pclass", "Large_Family", "Young", "Cabin_known"
#Train score: 0.807584269663
#Test score: 0.815642458101
#Also same with Shared_ticket

'''

tcols = np.append(["Survived"], cols)

df = training.loc[:,tcols].dropna()
#print(df.shape)
X_training = df.loc[:, cols]
#print(X_training.shape)
y_training = np.ravel(df.loc[:,["Survived"]])
#y = df.loc[:,["Survived"]]
#print(y_training.shape)

df_testing = testing.loc[:, tcols].dropna()
#print(df_testing.shape)

X_testing = df_testing.loc[:, cols]
#print(X_testing.shape)
y_testing = np.ravel(df_testing.loc[:, ["Survived"]])
#print(y_testing.shape) 


'Standardize the features. Given the selected columns, it probably doesnt matter much. But no harm in doing I guess.'
stdsc = StandardScaler()
X_training_std = stdsc.fit_transform(X_training)
X_testing_std = stdsc.transform(X_testing)

'Logistic Regression'
print("\nLogistic Regression:")

clf_log = LogisticRegression()
clf_log = clf_log.fit(X_training_std,y_training)

score_log = clf_log.score(X_training_std,y_training)
print("Train score:, logistic regression:", score_log)

'Let check testing score'

score_log2 = clf_log.score(X_testing_std,y_testing)
print("Test score, logistic regression:", score_log2)

'Checking the coeeficients, just because.'
print(pd.DataFrame( list( zip(X_training.columns, np.transpose(clf_log.coef_)) ) ))

'Perceptron'
print("\nPerceptron:")

clf_pctr = Perceptron(class_weight = "balanced")
clf_pctr = clf_pctr.fit(X_training_std, y_training)

score_pctr = cross_val_score(clf_pctr, X_training_std, y_training, cv = 5).mean()
print("Train score, perceptron:", score_pctr)

score_pctr2 = clf_pctr.score(X_testing_std, y_testing)
print("Test score, perceptron:", score_pctr2)

'K-Nearest Neighbor'
print("\n K-Nearest Neighbor:")
clf_knn = KNeighborsClassifier(n_neighbors = 10, weights = "distance")
clf_knn = clf_knn.fit(X_training_std, y_training)
score_knn = cross_val_score(clf_knn, X_training_std, y_training, cv = 5).mean()
print("Train score, KNN:", score_knn)

score_knn2 = clf_knn.score(X_testing_std, y_testing)
print("Test score, KNN:", score_knn2)
 
'Support Vector Machines'
print("\n SVM:")
clf_svm = svm.SVC(class_weight = "balanced")
clf_svm = clf_svm.fit(X_training_std, y_training)
score_svm = cross_val_score(clf_svm, X_training_std, y_training, cv = 5).mean()
print("Train score, SVM:", score_svm)

score_svm2 = clf_svm.score(X_testing_std, y_testing)
print("Test score, SVM:", score_svm2)
 
'Bagging classifier'
print("\nBagging Clasifier:")
bagging = BaggingClassifier(KNeighborsClassifier(n_neighbors = 5, weights = "distance"), oob_score = True, max_samples = 0.5, max_features = 1.0)
clf_bag = bagging.fit(X_training_std, y_training)
score_bag = clf_bag.oob_score_
print("Train score, Bagging:", score_bag)

score_bag2 = clf_bag.score(X_testing_std, y_testing)
print("Test score, Bagging:", score_bag2)


'Decision Tree'
print("\n Decision Tree:")
clf_tree = DecisionTreeClassifier(max_depth = 3, class_weight = "balanced", min_weight_fraction_leaf = 0.01)
clf_tree = clf_tree.fit(X_training_std, y_training)
score_tree = cross_val_score(clf_tree, X_training_std, y_training, cv = 5).mean()
print("Train score, Decision Tree:", score_tree)

score_tree2 = clf_tree.score(X_testing_std, y_testing)
print("Test score, Decision Tree:", score_tree2)

'Random Forest'
print("\nRandom Forest:")
clf_rf = RandomForestClassifier(n_estimators = 100, max_depth = None, min_samples_split = 10)
clf_rf= clf_rf.fit(X_training_std, y_training)
score_rf = cross_val_score(clf_rf, X_training_std, y_training, cv = 5).mean()
print("Train score, Random Forest:", score_rf)

score_rf2 = clf_rf.score(X_testing_std, y_testing)
print("Test score, Random Forest:", score_rf2)

'Extremely Randomized Trees'
print("\nExtremelt Randomized Trees: ")
clf_ext = ExtraTreesClassifier(max_features = "auto", bootstrap = True, oob_score = True, n_estimators = 100, max_depth = None, min_samples_split = 10)
clf_ext= clf_ext.fit(X_training_std, y_training)
score_ext = cross_val_score(clf_ext, X_training_std, y_training, cv = 5).mean()
print("Train score, Extremely Randomized Trees:", score_ext)

score_ext2 = clf_ext.score(X_testing_std, y_testing)
print("Test score, Extremely Randomized Trees:", score_ext2)

'Graient Boosting'
print("\nGradient Boosting: ")
clf_gb = GradientBoostingClassifier(
            #loss='exponential',
            n_estimators=1000,
            learning_rate=0.1,
            max_depth=3,
            subsample=0.5,
            random_state=0)
clf_gb= clf_gb.fit(X_training_std, y_training)
score_gb = cross_val_score(clf_gb, X_training_std, y_training, cv = 5).mean()
print("Train score, Gradient Boosting:", score_gb)

score_gb2 = clf_gb.score(X_testing_std, y_testing)
print("Test score, Gradient Boosting:", score_gb2)
            
            
'Ada Boost'
print("\nAda Boost: ")
clf_ada = AdaBoostClassifier(
            n_estimators=400,
            learning_rate=0.1)
clf_ada= clf_ada.fit(X_training_std, y_training)
score_ada = cross_val_score(clf_ada, X_training_std, y_training, cv = 5).mean()
print("Train score, Ada Boost:", score_ada)

score_ada2 = clf_ada.score(X_testing_std, y_testing)
print("Test score, Ada Boost:", score_ada2)
            
'Extreme Gradient Boosting: XGBoost'
print("\nXGBoost: ")
clf_xgb = xgb.XGBClassifier(
            max_depth = 2,
            n_estimators=500,
            subsample = 0.5,
            learning_rate=0.1)
clf_xgb= clf_xgb.fit(X_training_std, y_training)
score_xgb = cross_val_score(clf_xgb, X_training_std, y_training, cv = 5).mean()
print("Train score, XGBoost:", score_xgb)

score_xgb2 = clf_xgb.score(X_testing_std, y_testing)
print("Test score, XGBoost:", score_xgb2)

'LightGBM'
'Only works in 64 bit python system'
#print("\nLightGBM: ")
#clf_lgb = lgb.LGBMClassifier(
#            max_depth = 2,
#            n_estimators=500,
#            subsample = 0.5,
#            learning_rate=0.1)
#clf_lgb= clf_lgb.fit(X_training_std, y_training)
#score_lgb = cross_val_score(clf_lgb, X_training_std, y_training, cv = 5).mean()
#print("Train score, LGBM:", score_lgb)

#score_lgb2 = clf_lgb.score(X_testing_std, y_testing)
#print("Test score, LGBM:", score_lgb2)


'Optimizing one classifier in more detail'
'Extremely Randomized Trees were chosen for demonstation, but any other clasifier will do. We ll try one later.'

'First do a grid search.'

clf_ext = ExtraTreesClassifier(max_features = "auto", bootstrap = True, oob_score = True)
param_grid = {  "criterion" : ["gini", "entropy"],
                "min_samples_leaf" : [1,5,10],
                "min_samples_split" : [8,10,12],
                "n_estimators" : [20,50,100] }
gs = GridSearchCV(estimator = clf_ext, param_grid = param_grid, scoring = "accuracy", cv = 3)
gs = gs.fit(X_training_std,y_training)
print("Grid Serach. Best scores:",gs.best_score_)
print("Grid Serach. Best params:",gs.best_params_)

'Now lets use the optimized parameter values from grid search to find out the feature importances'
clf_ext = ExtraTreesClassifier(max_features = "auto", bootstrap = True, oob_score = True, criterion = "gini", min_samples_leaf = 1, min_samples_split = 12, n_estimators = 100)
clf_ext= clf_ext.fit(X_training_std, y_training)
score_ext = clf_ext.score(X_training_std,y_training)
print("Train score, Extremely Randomized Trees:", score_ext)

score_ext2 = clf_ext.score(X_testing_std, y_testing)
print("Test score, Extremely Randomized Trees:", score_ext2)

print(pd.DataFrame( list( zip(X_training.columns, np.transpose(clf_ext.feature_importances_)) ) ).sort_values(1, ascending = False))

'Lets make a confusion matrix to better visualize the data'
y_pred = gs.predict(X_testing_std)
confmat = confusion_matrix(y_true = y_testing, y_pred = y_pred)
print(confmat)

'plot confusion matrix'
#fig,ax = plt.subplots(figsize = (4, 4))
#ax.matshow(confmat, cmap = plt.cm.Blues, alpha = 0.3)
#for i in range(confmat.shape[0]):
#    for j in range(confmat.shape[1]):
#        ax.text(x = j, y = i, s = confmat[i,j], va = "center", ha = "center")
#plt.xlabel("predicted label")
#plt.ylabel("true label")
#plt.show()

'cross validation, again. just because we are having a deeper look'
scores = cross_val_score(clf_ext, X_training_std, y_training, cv = 5)
print("CV score, training: ",scores)
print("Mean score: {:.3f}, std deviation: {:.3f}".format(np.mean(scores), np.std(scores)) )

'ranking model by their score'
'by training score'
print("Rank by training score:")
models = pd.DataFrame( { 
            'Model' : ['Logistic Regression', 'Perceptron', 'K-Nearest Neighbor', 'Bagging KNN', 'Support Vector Machine', 'Random Forest', 'Extremely Randomized Trees', 'Gradient Boosting', 'Ada Boost', 'XGBoost'],
            'score' : [score_log, score_pctr, score_knn, score_bag, score_svm, score_rf, score_ext, score_gb, score_ada, score_xgb] }  )
            
print(models.sort_values(by = 'score', ascending = False))

'by testing score'
print("Rank by testing score:")
models = pd.DataFrame( { 
            'Model' : ['Logistic Regression', 'Perceptron', 'K-Nearest Neighbor', 'Bagging KNN', 'Support Vector Machine', 'Random Forest', 'Extremely Randomized Trees', 'Gradient Boosting', 'Ada Boost', 'XGBoost'],
            'score' : [score_log2, score_pctr2, score_knn2, score_bag2, score_svm2, score_rf2, score_ext2, score_gb2, score_ada2, score_xgb2] }  )
            
print(models.sort_values(by = 'score', ascending = False))


'compare feature importances'
print("Comparing feature importance ...")
summary = pd.DataFrame( list( zip(X_training.columns, 
                                    np.transpose(clf_rf.feature_importances_),
                                    np.transpose(clf_ext.feature_importances_),
                                    np.transpose(clf_gb.feature_importances_),
                                    np.transpose(clf_ada.feature_importances_),
                                    np.transpose(clf_xgb.feature_importances_) 
                                    ) ), columns = ['Feature','RF','XRT','GB','AB','XGB'])
                                    
summary['Median'] = summary.median(1)
print(summary.sort_values('Median', ascending = False))


'Now make some Ensemble methods'
clf_vote = VotingClassifier(
            estimators = [
                ('lr',clf_log), 
                ('pctr',clf_pctr), 
                ('knn',clf_knn),
                ('bag',clf_bag), 
                ('svm',clf_svm), 
                ('rf',clf_rf), 
                ('ext',clf_ext), 
                ('gb',clf_gb), 
                ('ada',clf_ada), 
                ('xgb',clf_xgb)
                ],
             weights = [2,1,2,2,2,2,3,2,2,3],
             voting = 'hard'   
            )   #weights are selected arbitrarily in this case
clf_vote.fit(X_training_std,y_training)
scores = cross_val_score(clf_vote, X_training_std, y_training, cv = 5, scoring = 'accuracy')
print("Voting: Accuracy: {:.2f} +- {:.2f}".format(scores.mean(), scores.std()))


'Skipping out-of-fold prediction for the time being'

'Can we check a heatmap among the classifiers to see their correlation?'
#print("trying to find correlation among different predictors:")
rf_predictions = clf_rf.predict(X_testing_std)
ext_predictions = clf_ext.predict(X_testing_std)
gb_predictions = clf_gb.predict(X_testing_std)
ada_predictions = clf_ada.predict(X_testing_std)
xgb_predictions = clf_xgb.predict(X_testing_std)
svm_predictions = clf_svm.predict(X_testing_std)

#joint_pred = np.concatenate( (rf_predictions.reshape(rf_predictions.shape[0],1), ext_predictions.reshape(ext_predictions.shape[0],1), gb_predictions.reshape(gb_predictions.shape[0],1), ada_predictions.reshape(ada_predictions.shape[0],1), xgb_predictions.reshape(xgb_predictions.shape[0],1),svm_predictions.reshape(svm_predictions.shape[0],1) ), axis = 1)

joint_pred = pd.DataFrame( { "rf":rf_predictions.ravel(),
                             "ext":ext_predictions.ravel(),
                             "gb":gb_predictions.ravel(),
                             "ada":ada_predictions.ravel(),
                             "xgb":xgb_predictions.ravel(),
                             "svm":svm_predictions.ravel()     } )

#print(joint_pred.shape)

#print(joint_pred[:5, :])

#plt.figure(figsize = (12,10))
#foo = sns.heatmap(joint_pred.corr(), vmax = 1.0, square = True, annot = True)
#plt.show()


'Finally, prepare submission file. And submit.'
print("preparing for submission ...")
clf = clf_vote
df2 = test.loc[:,cols].fillna(method = "pad")
df2_std = stdsc.transform(df2)

surv_pred = clf.predict(df2_std)
print("surv_pred shape:", surv_pred.shape)

submit = pd.DataFrame({ "PassengerId": test.loc[:,"PassengerId"], "Survived" : surv_pred.T })
print("submit shape:",submit.shape)
print(submit.iloc[:5])

submit.to_csv("Data/submit.csv",index = False)
print("Done.")
