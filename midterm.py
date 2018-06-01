import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
from pandas import set_option
from pandas import read_csv


filename = 'pima-indians-diabetes.data.csv'
names = ['1','2','3','4','5','6','7','8', '9', '10']
data = read_csv(filename)
data1 = data.drop(data.columns[0], axis=1) # when dataframe is read it generates a new index column, this will remove the extra column, check data in variable explorer
# save features as pandas dataframe for stepwise feature selection
X1 = data1.drop(data1.columns[8], axis = 1)
Y1 = data1.drop(data1.columns[0:8], axis = 1)
# separate features and response into two different arrays
array = data1.values
X = array[:,0:8]
Y = array[:,8]
# First perform exploratory data analysis using correlation and scatter plot
# look at the first 20 rows of data
peek = data1.head(20)
print(peek)

from sklearn.preprocessing import StandardScaler
StandardScaler(copy=True, with_mean=False, with_std=False)
scaler = StandardScaler()
print(scaler.fit(data))
print(scaler.mean_)
print(scaler.transform(data))


#from sklearn.preprocessing import Normalizer
#X_normalized = preprocessing.normalize(X, norm='l1')


# descriptive statistics: mean, max, min, count, 25 percentile, 50 percentile, 75 percentile
set_option('display.width', 100)
set_option('precision', 1)
description = data.describe()
print(description)

# we look at the distribution of data and its descriptive statistics
plt.figure() # new plot
data1.hist()
plt.show()


# correlation heat map, pay attention to correlation between all predicators/features and each predictor and the output
plt.figure() # new plot
corMat = data1.corr(method='pearson')
print(corMat)
# plot correlation matrix as a heat map
sns.heatmap(corMat, square=True)
plt.yticks(rotation=0)
plt.xticks(rotation=90)
plt.title("CORELATION MATTRIX USING HEAT MAP")
plt.show()

# scatter plot of all data
plt.figure()
scatter_matrix(data1)
plt.show()

# Determiniation of dominant features , Method one Recursive Model Elimination, 
# very similar idea to foreward selection but done recurssively. This method is gready
# which means it tries one feature at the time
NUM_FEATURES = 5 # this is kind of arbitrary but the idea should come by observing the scatter plots and correlation.
model = LinearRegression()
rfe = RFE(model, NUM_FEATURES)
fit = rfe.fit(X, Y)
print("Num Features: %d") % fit.n_features_
print("Selected Features: %s") % fit.support_
print("Feature Ranking: %s") % fit.ranking_
 

# stepwise forward-backward selection
# need to change the input types as X in this function needs to be a pandas
# dataframe

 
def stepwise_selection(X, y, 
                       initial_list=[], 
                       threshold_in=0.01, 
                       threshold_out = 0.05, 
                       verbose=True):
    """ Perform a forward-backward feature selection 
    based on p-value from statsmodels.api.OLS
    Arguments:
        X - pandas.DataFrame with candidate features
        y - list-like with the target
        initial_list - list of features to start with (column names of X)
        threshold_in - include a feature if its p-value < threshold_in
        threshold_out - exclude a feature if its p-value > threshold_out
        verbose - whether to print the sequence of inclusions and exclusions
    Returns: list of selected features 
    Always set threshold_in < threshold_out to avoid infinite looping.
    See https://en.wikipedia.org/wiki/Stepwise_regression for the details
    """
    included = list(initial_list)
    while True:
        changed=False
        # forward step
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.argmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        # backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.argmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included

result = stepwise_selection(X1, Y1)

print('resulting features:')
print(result)