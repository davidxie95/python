# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 19:13:59 2018

@author: jahan

from risk engineering website
https://risk-engineering.org/linear-regression-analysis/
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import scipy.stats
import seaborn as sns
#matplotlib inline
#config InlineBackend.figure_formats=['svg']

df = pd.DataFrame({'cigarettes' : [0,10,20,30], 'CVD' : [572,802,892,1025], 'lung' : [14, 105, 208, 355]})
df.head()

plt.plot(df.cigarettes, df.CVD, "o")
plt.margins(0.1)
plt.title(u"Deaths for different smoking intensities", weight='bold')
plt.xlabel(u"Cigarettes smoked per day")
plt.ylabel(u"CVD deaths");

# regression modeling using stat models library
import statsmodels.api as sm
import statsmodels.formula.api as smf

lm = smf.ols("CVD ~ cigarettes", data=df).fit()
xmin = df.cigarettes.min()
xmax = df.cigarettes.max()
X = np.linspace(xmin, xmax, 100)
# params[0] is the intercept (beta0)
# params[1] is the slope (beta1)
Y = lm.params[0] + lm.params[1] * X
plt.plot(df.cigarettes, df.CVD, "o")
plt.plot(X, Y, color="darkgreen")
plt.xlabel("Cigarettes smoked per day")
plt.ylabel("Deaths from cardiovascular disease")
plt.margins(0.1)

# make similar plot for lung cancer and cigarettes
lm = smf.ols("lung ~ cigarettes", data=df).fit()
xmin = df.cigarettes.min()
xmax = df.cigarettes.max()
X = np.linspace(xmin, xmax, 100)
X = sm.add_contant(X)
# params[0] is the intercept (beta0)
# params[1] is the slope (beta1)
Y = lm.params[0] + lm.params[1] * X
plt.plot(df.cigarettes, df.lung, "o")
plt.plot(X, Y, color="darkgreen")
plt.xlabel("Cigarettes smoked per day")
plt.ylabel("Lung cancer deaths")
plt.margins(0.1)


lm.summary()
"""
In particular, the  R2R2  value of 0.987 is very high, indicating a good level of fit to our dataset. 
However, given the small size of our dataset (only 4 observations, even if each observation is based on a large population), 
the 95% confidence interval for our model parameters  β0β0  and  β1β1  is quite large.

The Seaborn package provides convenient functions for making plots of linear regression models. 
In particular, the regplot function generates a regression plot that includes 95% confidence intervals for the model parameters.

"""
sns.regplot(df.cigarettes, df.CVD);