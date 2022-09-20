# EC401 Assignment 1
## Compiled by Divya 21227707087
### Koop and Tobias 2004 Labour Market Experience Data is a panel of 2,178 individuals with a total of 17,919 observations.
### Time Trend Takes values from 0 to 14. I have used fixed time trend at 14, recent most year. This leaves us with 1499 Observations.
### The cleansed data has been stored as "kt.xlsx"
#
#IMPORTING LIBRARIES
## For Linear Algebra
import numpy as np
## For dealing with Data Frames
import pandas as pd
## For Visualization
import matplotlib.pyplot as plt
## For OLS
import statsmodels.api as sm
## For Statistical tests
import scipy.stats as stats
#
# DATA
## Importing  data stored in local derive (as an excel)
kt = pd.read_excel("C:/Users/Divya Bhardwaj/Assignments/401_Assignments/kt.xlsx")
## Changing Default index from 0 to 1
kt.index=kt.index+1
## Displaying first 20 entries of data
kt.head(10)
## Displays details about the data of each Variable
kt.info()
## adding a columns of 1 in kt
kt['CONSTANT']= 1
##Descriptive Statistics, have ignored PERSON ID as its descriptive stats has no useful economic interpretation
kt.loc[:, kt.columns != 'PERSONID'].describe()
#
# DEFINING X1 
## Assigning columns to X1 from kt
X1=kt[["CONSTANT","EDUC","POTEXPER","ABILITY"]]
X1.info()
# DEFINING X2
## Assigning columns to X2 from kt
X2=kt[["MOTHERED","FATHERED","SIBLINGS"]]
X2.info()
#DEFINING Y
## Assigning columns to Y from kt
Y=pd.DataFrame()
Y["LOGWAGE"]=kt.LOGWAGE
Y.info()
# Merging X1 and X2
X=pd.DataFrame()
X=pd.DataFrame.join(X1,X2)
X.info()
#
#QUESTION 1 : Scatter plot Y and X1
## Scatter Plot Y and Constant
plt.plot( X1["CONSTANT"], Y["LOGWAGE"],linestyle="None",marker="o", markerfacecolor="red")
plt.xlabel("CONSTANT")
plt.ylabel("LOGWAGE")
plt.title("SCATTER PLOT B/W CONSTANT AND LOGWAGE")
## Scatter Plot Y and Educ
plt.plot( X1["EDUC"], Y["LOGWAGE"], linestyle="None",marker="o", markerfacecolor="red")
plt.xlabel("EDUC")
plt.ylabel("LOGWAGE")
plt.title("SCATTER PLOT B/W EDUC AND LOGWAGE")
## Scatter Plot Y and Potexper
plt.plot( X1["POTEXPER"],Y["LOGWAGE"], linestyle="None",marker="o", markerfacecolor="red")
plt.xlabel("POTEXPER")
plt.ylabel("LOGWAGE")
plt.title("SCATTER PLOT B/W POTEXPER AND LOGWAGE")
## Scatter Plot Y and Ability
plt.plot( X1["ABILITY"],Y["LOGWAGE"], linestyle="None",marker="o", markerfacecolor="red")
plt.xlabel("ABILITY")
plt.ylabel("LOGWAGE")
plt.title("SCATTER PLOT B/W ABILITY AND LOGWAGE")
#
# QUESTION 2 : Correlation of each variable X and Y
##Calculating Correlation of Y with X variables
kt_corr=X.apply(Y["LOGWAGE"].corr)
kt_corr
### NaN to be interpreted as no correlation with constant, which is true as variable "CONSTANT" is a constant
#
# QUESTION 3 
##Estimated regression Formula : LOGWAGE= beta_0*1 +beta_1*EDUC + beta_2*POTEXPER +beta_3*ABILITY +beta_4*MOTHERED + beta_5*FATHERED + beta_6*SIBLINGS + e
#
# QUESTION 4 : Without using in built OLS function
## USing Matrix Multiplication t calculate beta_hat =(X'X)^(-1)(X'Y)
beta_hat=np.matmul(np.linalg.inv(np.matmul(np.transpose(X),X)),(np.matmul(np.transpose(X),Y)))
beta_hat
#
# QUESTION 5 : Using in built
## Regressing using the formula
lm_formula = sm.OLS.from_formula( 'LOGWAGE ~ CONSTANT + EDUC + POTEXPER + ABILITY + MOTHERED + FATHERED + SIBLINGS -1', kt).fit()
lm_formula.summary()
##Regressing Y on X1 and X2
lm_builtin=sm.OLS(Y,X).fit()
lm_builtin.summary()
#
# QUESTION 6 :Regress each variable of X2 on X1, computing residuals and storing them in X2_asterisk
##Regressing Mother Education on X1
lm_mothered=sm.OLS(X2.MOTHERED,X1).fit()
lm_mothered.summary()
##Regressing Father Education on X1
lm_fathered=sm.OLS(X2.FATHERED,X1).fit()
lm_fathered.summary()
##Regressing Siblings on X1
lm_siblings=sm.OLS(X2.SIBLINGS,X1).fit()
lm_siblings.summary()
## Residuals in each case
lm_mothered.resid
lm_fathered.resid
lm_siblings.resid
#Storing these values in X2*
##Only _ Undersscore is allowed in variable names
X2_asterisk=pd.DataFrame()
X2_asterisk["mothered_resid"]=lm_mothered.resid
X2_asterisk["fathered_resid"]=lm_fathered.resid
X2_asterisk["siblings_resid"]=lm_siblings.resid
X2_asterisk.describe()
##Is the mean statistically different from Zero ?
stats.ttest_1samp(a=X2_asterisk.mothered_resid, popmean=0)
## Large p value fail to reject null, mean of lm_mothered_resid is not significantly differentn0
stats.ttest_1samp(a=X2_asterisk.fathered_resid, popmean=0)
## Large p value fail to reject null, mean of lm_fathered_resid is not significantly different from 0
stats.ttest_1samp(a=X2_asterisk.siblings_resid, popmean=0)
## Large p value fail to reject null, mean of lm_siblings_resid is not significantly different 0
#
# QUESTION 7 :Regressing Y on X1
lm_y_on_x1=sm.OLS(Y,X1).fit()
lm_y_on_x1.summary()
# 
#QUESTION 8 : regressing Y on X1 and X2
##Regressing Y on X1 and X2
lm_y_on_x1_and_x2=sm.OLS.from_formula('Y~X1+X2-1',kt).fit()
lm_y_on_x1_and_x2.summary()
## This is same as regressing Y on X 
lm_y_on_x=sm.OLS(Y,X).fit()
lm_y_on_x.summary()
#
# QUESTION 9 : Regressing Y onn X1 and X2_asterisk
#Regressing Y on X1 and X2_asterisk
lm_y_on_x1_and_x2_asterisk=sm.OLS.from_formula('LOGWAGE ~  CONSTANT + EDUC + POTEXPER + ABILITY + X2_asterisk.mothered_resid + X2_asterisk.fathered_resid +X2_asterisk.siblings_resid - 1',kt).fit()
lm_y_on_x1_and_x2_asterisk.summary()

#Done
