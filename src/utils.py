""" 
This script stores functions
""" 

### ---------------------------------------------------------------------------
### Preliminaries.
### ---------------------------------------------------------------------------
import os 
import pandas as pd
import numpy as np
import sys, os
import scipy.stats as st
from itertools import tee, islice, chain

### ---------------------------------------------------------------------------
### Disable and enable printing.
### ---------------------------------------------------------------------------

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

### ---------------------------------------------------------------------------
### Iterations through dataframe.
### ---------------------------------------------------------------------------

def previous_and_next(some_iterable):
    '''
    Provides reference to previous and next items within loops.
    '''
    prevs, items, nexts = tee(some_iterable, 3)
    prevs = chain([None], prevs)
    nexts = chain(islice(nexts, 1, None), [None])
    return zip(prevs, items, nexts)

### ---------------------------------------------------------------------------
### Drop rows or clumns based on a list.
### ---------------------------------------------------------------------------

def droprows(df,target_column,to_drop):
    ''' 
    Drops rows based on a list.
    '''
    counts = df[target_column].value_counts(normalize = True)
    #print('Unique counts-ratio before: ', counts)

    df_drop = df[df[target_column].isin(to_drop)]
    df = df.drop(df_drop.index, axis=0)

    counts = df[target_column].value_counts(normalize = True) 
    #print('Unique counts-ratio after: ', counts)
    return df

### ---------------------------------------------------------------------------
### Map data.
### ---------------------------------------------------------------------------

def mapper(df,keys,values,target_column):
    ''' 
    Maps entries in target column to mapping list.
    '''
    map_values = dict(zip([keys], [values]))
    mapper = target_column.isin(map_values)
    df.loc[mapper, target_column] = df.loc[mapper, target_column].apply(lambda row: map_values[row])
    df.fillna('', inplace=True)
    return df

### ---------------------------------------------------------------------------
### Map and clean data.
### ---------------------------------------------------------------------------

def mapandclean(df, mapping, target_column):
    ''' 
    Maps entries in target column to mapping list and drops unmapped rows.
    '''
    mapp = mapping[target_column].unique()
    target = df[target_column]
    df["mapp"] = np.where(target.isin(mapp),"Mapped","not mapped")
    df = df[df.mapp != "not mapped"]
    df["mapp"].unique()
    df[target_column].unique()
    df.drop(["mapp"], axis = 1)
    return df

### ---------------------------------------------------------------------------
### Remove gaps.
### ---------------------------------------------------------------------------

#Replace gaps ("-") in dataframe by NaN.
def replacegaps(df):
    """
    Replaces nans based on list.
    """
    # Specifiy gaps.
    nan = ["", " ", "-", 0, "0", "nan", "NaN", "NaT"]
    df = df.replace(nan, np.nan)
    return df

### ---------------------------------------------------------------------------
### Convert columns formats.
### ---------------------------------------------------------------------------

#Change format of datetime columns.
dt_cols = ['reld','relp','cstp', 'ctff']

def convert_times(df, dt_cols):
    """
    Converts dt_cols into datetime64 formated columns.
    """
    for column in df[dt_cols]:
        try:
            #df[column] = df[column].astype('datetime64')
            df[column] = pd.to_datetime(df[column], errors='coerce')
        except:
            print('Non-datetime values in column : ', column)
            #print(df[column].unique())
    return df

### ---------------------------------------------------------------------------
### Unstack columns.
### ---------------------------------------------------------------------------

# Split all stacked columns in rows while duplicating unstacked column values.
def splitDataFrameList(df,target_column,separator):
    ''' 
    Unstacks target_column while into seperate rows, 
    where the values os other columns are duplicated into.
    '''
    def splitListToRows(row,row_accumulator,target_column,separator):
        split_row = row[target_column].split(separator)
        for s in split_row:
            new_row = row.to_dict()
            new_row[target_column] = s
            row_accumulator.append(new_row)
    new_rows = []
    df.apply(splitListToRows,axis=1,args = (new_rows,target_column,separator))
    new_df = pd.DataFrame(new_rows)
    return new_df

### ---------------------------------------------------------------------------
### Segment categorical columns.
### ---------------------------------------------------------------------------

def categorical_segment(df, column_name:str) -> 'grouped_dataframe':
    '''
    Creates a grouped dataframe for each categoriacal column, counting BDQVs.
    '''
    segmented_df = df[[column_name, 'bdqv_bool']]
    segmented_bdqv_df = segmented_df[segmented_df['bdqv_bool'] != 'No']
    grouped_df = segmented_bdqv_df.groupby(column_name).count().reset_index().rename(columns = {'bdqv_bool':'bdqved'})
    total_count_df = segmented_df.groupby(column_name).count().reset_index().rename(columns = {'bdqv_bool':'Total'})
    merged_df = pd.merge(grouped_df, total_count_df, how = 'inner', on = column_name)
    merged_df['Percent_bdqved'] = merged_df[['bdqved','Total']].apply(lambda x: (x[0] / x[1]) * 100, axis=1) 
    return merged_df

### ---------------------------------------------------------------------------
### Segment continous columns.
### ---------------------------------------------------------------------------

def continous_segment(df, column_name:str) -> 'segmented_df':
    segmented_df = df[[column_name, 'bdqv']]
    segmented_df = segmented_df.replace( {'bdqv': {'No':'Retained','Yes':'bdqved'} } )
    segmented_df['Customer'] = ''
    return segmented_df

### ---------------------------------------------------------------------------
### One hot encoding.
### ---------------------------------------------------------------------------

def onehotencoding(source, item):
    '''
    Uses a source list containing all possible values to 
    implement one hot encoding for all items in a dataset.
    '''
    number = source[source == item].index.values
    return(int(number))

### ---------------------------------------------------------------------------
### Create adjacency matrices.
### ---------------------------------------------------------------------------

def create_adjacency(df):
    '''
    Creates an adjacency matrix for ohe component part for one vehicle series.
    '''
    component_part = pd.concat([df["component"], df["part"]]).drop_duplicates()
    component_part = component_part.reset_index(drop = True)
    n = len(component_part)
    # Create adjacency matrix by counting occurences in the dataframe.
    adjacency = np.mat(np.zeros([n, n])).astype(int)
    adjacency[df["component"].astype(str).str.get_dummies(), df["part"].astype(str).str.get_dummies()]+=1
    adjacency[df["part"].astype(str).str.get_dummies(), df["component"].astype(str).str.get_dummies()]+=1
    return adjacency

### ---------------------------------------------------------------------------
### 2-sample Kolmogorov Smirnov test. 
### ---------------------------------------------------------------------------

def KS_cdf_diff(samp_a, samp_b):
    '''
    Computes the cdf and a difference vector for two samples.
    '''    
    # Concatenate and sort samples.
    samp_conc = np.sort(np.concatenate((samp_a, samp_b)))
    
    # Compute CDF of sample a and b.
    samp_a_cdf = [np.round(st.percentileofscore(samp_a, value)/100, 1) for value in samp_conc]
    samp_b_cdf = [np.round(st.percentileofscore(samp_b, value)/100, 1) for value in samp_conc]

    # Compute absolute difference and its maximum.
    samp_diff = np.abs(np.subtract(samp_a_cdf, samp_b_cdf))  
    
    print(samp_conc, samp_a_cdf, samp_b_cdf, samp_diff)
    
    return samp_conc, samp_a_cdf, samp_b_cdf, samp_diff


def KS_crit(arr1, arr2):
    '''
    Computes the critical value of two arrays for 2-sample KS-test.
    '''
    return 1.36*np.sqrt(len(arr1)**-1 + len(arr2)**-1)

### ---------------------------------------------------------------------------
### Orthogonal Procrustes Approach. 
### ---------------------------------------------------------------------------

def procrustes(X, Y, scaling=True, reflection='best'):
    """
               This function is the numpy implementation corresponding to the matlab version
       Outputs
       ------------
       d：the residual sum of squared errors, normalized according to a measure of the scale of X, ((X - X.mean(0))**2).sum()
       Z：the matrix of transformed Y-values
       tform：a dict specifying the rotation, translation and scaling that maps X --> Y
       """
    n, m = X.shape
    ny, my = Y.shape
    muX = X.mean(0)
    muY = Y.mean(0)
    X0 = X - muX
    Y0 = Y - muY
    ssX = (X0 ** 2.).sum()
    ssY = (Y0 ** 2.).sum()
    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)
    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY
    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m - my)), 0)
    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)
    if reflection is not 'best':
        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0
        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:, -1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)
    traceTA = s.sum()
    if scaling:
        # optimum scaling of Y
        b = traceTA * normX / normY
        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA ** 2
        # transformed coords
        Z = normX * traceTA * np.dot(Y0, T) + muX
    else:
        b = 1
        d = 1 + ssY / ssX - 2 * traceTA * normY / normX
        Z = normY * np.dot(Y0, T) + muX
    # transformation matrix
    if my < m:
        T = T[:my, :]
    c = muX - b * np.dot(muY, T)
    # transformation values
    tform = {'rotation': T, 'scale': b, 'translation': c}
    return d, Z, tform

### ---------------------------------------------------------------------------
### Association mining. 
### ---------------------------------------------------------------------------

def apriori(data, min_support=0.04, max_length = 2):
    # Collecting Required Library
    import numpy as np
    import pandas as pd
    from itertools import combinations
    # Step 1:
    # Creating a dictionary to stored support of an itemset.
    support = {} 
    L = list(data.columns)
    
    # Step 2: 
    #generating combination of items with len i in ith iteration
    for i in range(1, max_length+1):
        P = set(combinations(L,i))
        
    # Reset "L" for next ith iteration
        L =set()     
    # Step 3: 
        #iterate through each item in "c"
        for j in list(P):
            #print(j)
            sup = data.loc[:,j].sum().sum()/len(P) # The sum of all BDQVs per module divided by the count of unique modules.
            if sup > min_support:
                #print(sup, j)
                support[j] = sup
                
                # Appending frequent itemset in list "L", already reset list "L" 
                L = list(set(L) | set(j))
        
    # Step 4: data frame with cols "items", 'support'
    result = pd.DataFrame(list(support.items()), columns = ["Items", "Support"])
    mean_supp = result["Support"].mean()
    return(result, mean_supp)

def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def association_rule(df, min_threshold=0.5):
    import pandas as pd
    from itertools import permutations
    
    # STEP 1:
    #creating required varaible
    
    min_supp = np.min(df["Support"])
    max_supp = np.max(df["Support"])
    
    support = pd.Series(df.Support.values, index=df.Items).to_dict()

    data = []
    L= df.Items.values
    
    # Step 2:
    #generating rule using permutation
    p = list(permutations(L, 2))
        
    # Iterating through each rule
    for i in p:
        
        # If LHS(Antecedent) of rule is subset of RHS then valid rule.
        if set(i[0]).issubset(i[1]):
            conf = support[i[1]]/support[i[0]]
            # Normalize conf with max supp:
            conf = (conf - min_supp) / (max_supp - min_supp)
            #print(i, conf)
            if conf > min_threshold:
                #print(i, conf)
                j = i[1][not i[1].index(i[0][0])]
                #lift = support[i[1]]/(support[i[0]]* support[(j,)])
                #leverage = support[i[1]] - (support[i[0]]* support[(j,)])
                #convection = (1 - support[(j,)])/(1- conf)
                data.append([i[0], (j,), support[i[0]], support[(j,)], support[i[1]], conf])

         
    # STEP 3:
    result = pd.DataFrame(data, columns = ["antecedents", "consequents", "antecedent support", "consequent support",
                                        "support", "confidence"])
    return(result)

# Compute the correlation matrix and delete features with hig correlation.

### ---------------------------------------------------------------------------
### Multicollinearity.
### ---------------------------------------------------------------------------

#Feature selection class to eliminate multicollinearity
class MultiCollinearityEliminator():
    
    #Class Constructor
    def __init__(self, df, target, threshold):
        self.df = df
        self.target = target
        self.threshold = threshold

    #Method to create and return the feature correlation matrix dataframe
    def createCorrMatrix(self, include_target = False):
        #Checking we should include the target in the correlation matrix
        if (include_target == False):
            df_temp = self.df.drop([self.target], axis =1)
            
            #Setting method to Pearson to prevent issues in case the default method for df.corr() gets changed
            #Setting min_period to 30 for the sample size to be statistically significant (normal) according to 
            #central limit theorem
            corrMatrix = df_temp.corr(method='pearson', min_periods=30).abs()
        #Target is included for creating the series of feature to target correlation - Please refer the notes under the 
        #print statement to understand why we create the series of feature to target correlation
        elif (include_target == True):
            corrMatrix = self.df.corr(method='pearson', min_periods=30).abs()
        return corrMatrix

    #Method to create and return the feature to target correlation matrix dataframe
    def createCorrMatrixWithTarget(self):
        #After obtaining the list of correlated features, this method will help to view which variables 
        #(in the list of correlated features) are least correlated with the target
        #This way, out the list of correlated features, we can ensure to elimate the feature that is 
        #least correlated with the target
        #This not only helps to sustain the predictive power of the model but also helps in reducing model complexity
        
        #Obtaining the correlation matrix of the dataframe (along with the target)
        corrMatrix = self.createCorrMatrix(include_target = True)                           
        #Creating the required dataframe, then dropping the target row 
        #and sorting by the value of correlation with target (in asceding order)
        corrWithTarget = pd.DataFrame(corrMatrix.loc[:,self.target]).drop([self.target], axis = 0).sort_values(by = self.target)                    
        print(corrWithTarget, '\n')
        return corrWithTarget

    #Method to create and return the list of correlated features
    def createCorrelatedFeaturesList(self):
        #Obtaining the correlation matrix of the dataframe (without the target)
        corrMatrix = self.createCorrMatrix(include_target = False)                          
        colCorr = []
        #Iterating through the columns of the correlation matrix dataframe
        for column in corrMatrix.columns:
            #Iterating through the values (row wise) of the correlation matrix dataframe
            for idx, row in corrMatrix.iterrows():                                            
                if(row[column]>self.threshold) and (row[column]<1):
                    #Adding the features that are not already in the list of correlated features
                    if (idx not in colCorr):
                        colCorr.append(idx)
                    if (column not in colCorr):
                        colCorr.append(column)
        print(colCorr, '\n')
        return colCorr

    #Method to eliminate the least important features from the list of correlated features
    def deleteFeatures(self, colCorr):
        #Obtaining the feature to target correlation matrix dataframe
        corrWithTarget = self.createCorrMatrixWithTarget()                                  
        for idx, row in corrWithTarget.iterrows():
            print(idx, '\n')
            if (idx in colCorr):
                self.df = self.df.drop(idx, axis =1)
                break
        return self.df

    #Method to run automatically eliminate multicollinearity
    def autoEliminateMulticollinearity(self):
        #Obtaining the list of correlated features
        colCorr = self.createCorrelatedFeaturesList()                                       
        while colCorr != []:
            #Obtaining the dataframe after deleting the feature (from the list of correlated features) 
            #that is least correlated with the taregt
            self.df = self.deleteFeatures(colCorr)
            #Obtaining the list of correlated features
            colCorr = self.createCorrelatedFeaturesList()                                     
        return self.df

### ---------------------------------------------------------------------------
### Chi-squared test.
### ---------------------------------------------------------------------------

import scipy.stats as stats
from scipy.stats import chi2_contingency

class ChiSquare:
    def __init__(self, DataFrame):
        self.df = DataFrame
        self.p = None #P-Value
        self.chi2 = None #Chi Test Statistic
        self.dof = None
        
        self.dfObserved = None
        self.dfExpected = None
        
    def _print_chisquare_result(self, colX, alpha):
        result = ""
        if self.p<alpha:
            result="{0} is IMPORTANT for Prediction".format(colX)
        else:
            result="{0} is N0T an important predictor. (Discard {0} from model)".format(colX)

        print(result)
        
    def TestIndependence(self,colX,colY, alpha=0.01):
        X = self.df[colX].astype(str)
        Y = self.df[colY].astype(str)
        
        self.dfObserved = pd.crosstab(Y,X) 
        chi2, p, dof, expected = stats.chi2_contingency(self.dfObserved.values)
        self.p = p
        self.chi2 = chi2
        self.dof = dof 
        
        self.dfExpected = pd.DataFrame(expected, columns=self.dfObserved.columns, index = self.dfObserved.index)
        
        self._print_chisquare_result(colX,alpha)







