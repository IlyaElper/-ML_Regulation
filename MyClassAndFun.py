import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from IPython.display import display
from scipy import stats # statistical tests
import os
from ipywidgets import interact
# machine learning 
from scipy.stats import chi2_contingency
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV,learning_curve
from sklearn.metrics import confusion_matrix,roc_curve,accuracy_score,classification_report,make_scorer,f1_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.inspection import permutation_importance
from sklearn.base import clone
from itertools import combinations
from joblib import Parallel, delayed
class col_transformer():
    """
    This class provides methods for headling and transform features in d DataFrame.
    """
    def __init__(self,in_df:pd.DataFrame):
        self.df=in_df
        
#     def month_handling(self)->pd.DataFrame:
#         if 'month' not in self.df.columns:
#             raise ValueError('column "month" not found in the DataFrame!')
        
#         self.df['month']=self.df['month'].astype(str)
#         month_dummies=pd.get_dummies(self.df['month'],prefix='month',drop_first=True ,dtype=int)
#         self.df=pd.concat([self.df,month_dummies],axis=1)
#         self.df.drop('month',axis=1,inplace=True)
#         return self.df
    
    def occupation_handling(self)->pd.DataFrame:
        """
        Handle the 'occupation' column:
        - Replace specific values with NaN.
        - Fill missing values with the most frequent value for each customer.
        - Create dummy variables for 'occupation'.
        - Drop the original 'occupation' column.
        """
        if 'occupation' not in self.df.columns:
            raise ValueError ('column "occupation" not found in the DataFrame!')
            
        self.df.loc[(self.df.occupation=='_______'),'occupation']=np.nan
        self.df['occupation'] = self.df.groupby('customer_id')['occupation'].transform(lambda x: x.mode()[0])
        
        print('occupation done')
        return self.df
    
    def type_of_loan_handling(self) -> pd.DataFrame:
        """
        Handle the 'type_of_loan' column:
        - Remove 'and' and spaces from the values.
        - Split the values using commas.
        - Create new columns for each unique loan type.
        - Populate the new columns with 1 or 0 based on the presence of each loan type.
        - Drop the original 'type_of_loan' column.
        """
        if 'type_of_loan' not in self.df.columns:
            raise ValueError ('column "type_of_loan" not found in the DataFrame.')
            
         # Replace 'and' with an empty string and remove spaces in the 'Type_of_Loan' column
        self.df['type_of_loan'] = self.df['type_of_loan'].str.replace('and', '')
        self.df['type_of_loan'] = self.df['type_of_loan'].str.replace(' ', '')
        
         # Split the values in the 'Type_of_Loan' column using commas and create a new DataFrame 'Type_of_Loan_df'
        Type_of_Loan_df = self.df['type_of_loan'].str.split(',').apply(pd.Series)
        
        # Initialize variables
        i = 0
        my_df = pd.DataFrame()
        
        # Iterate through the columns of 'Type_of_Loan_df'
        while i < len(Type_of_Loan_df.columns):
            # Create a new DataFrame 'df' containing the data from the current column
            df = pd.DataFrame({'col': Type_of_Loan_df[i]})
        
            # Concatenate 'df' with 'my_df' while ignoring index to create a single DataFrame
            my_df = pd.concat([my_df, df], ignore_index=True)
            i += 1
        
        # Drop rows with NaN values
        my_df = my_df.dropna()
        
        # Create a list of unique loan types from the 'my_df' DataFrame
        Type_of_Loan_list = my_df['col'].unique()
        
        # Convert the 'Type_of_Loan' column in 'train_df' to string data type
        self.df['type_of_loan'] = self.df['type_of_loan'].astype(str)
        
        # Iterate through the unique loan types and create new columns in 'train_df' with 0 values
        for loan_type in Type_of_Loan_list:
            self.df[loan_type] = 0
        
        # Use the 'apply' function to populate the newly created columns with 1 or 0
        # indicating whether each loan type is present in the 'Type_of_Loan' column
        for loan_type in Type_of_Loan_list:
            self.df[loan_type] = self.df['type_of_loan'].apply(lambda x: 1 if loan_type in x else 0)
        
        # Remove the original 'Type_of_Loan' column 
        del self.df['type_of_loan']
        print('type_of_loan done')
        return self.df
    
    def credit_mix_handling(self) -> pd.DataFrame:
        if 'credit_mix' not in self.df.columns:
            raise ValueError ('column "credit_mix" not found in the DataFrame!')
        
        # Deleting strange values like "_"
        self.df['credit_mix'] = self.df['credit_mix'].replace("_", pd.NA)
        
        # Replacing missing values in each row, associated with a specific customer_id, with the most frequent value.
        self.df['credit_mix']= self.df.groupby('customer_id')['credit_mix'].transform(lambda x: x.mode()[0])
        self.df['credit_mix'].astype(str)
        
        
        print('credit_mix done')
        return self.df
        
    def payment_of_min_amount_handling(self) -> pd.DataFrame:
        if 'payment_of_min_amount' not in self.df.columns:
            raise ValueError ('column "payment_of_min_amount" not found in the DataFrame!')
        
        # Deleting strange values like "NM"
        self.df['payment_of_min_amount'] = self.df['payment_of_min_amount'].replace("NM", pd.NA)
        
        # Replacing missing values in each row, associated with a specific customer_id, with the most frequent value.
        self.df['payment_of_min_amount']= self.df.groupby('customer_id')['payment_of_min_amount'].transform(lambda x: x.mode()[0])
        self.df['payment_of_min_amount'].astype(str)
        
        
        print('payment_of_min_amount done')
        return self.df
    
    
    def payment_behaviour_handling(self) -> pd.DataFrame:
        if 'payment_behaviour' not in self.df.columns:
            raise ValueError ('column "payment_behaviour" not found in the DataFrame!')
        
        self.df['payment_behaviour'] = self.df['payment_behaviour'].replace('!@9#%8', pd.NA)
        self.df['payment_behaviour']= self.df.groupby('customer_id')['payment_behaviour'].transform(lambda x: x.mode()[0])
        spent=self.df.payment_behaviour.str.extract('([a-zA-Z]+)_').astype(str)
        self.df = pd.concat([self.df, spent], axis=1)
        self.df.rename(columns={0: 'spent'}, inplace=True)
        
        # After executing this method values will be Poor==0, Good==1, Standard==2               
        mapping = {"High":1, "Low":0}
        self.df['spent'] = self.df['spent'].replace(mapping)
        
         #Since this column is the target variable, I expect it to be located at the end of the dataset.
        spent_column = self.df['spent']
        self.df = self.df.drop(columns=['spent'])
        self.df['spent'] = spent_column

        value_payments=self.df.payment_behaviour.str.extract('spent_([a-zA-Z]+)_').astype(str)
        self.df = pd.concat([self.df, value_payments], axis=1)
        self.df.rename(columns={0: 'value_payments'}, inplace=True)
        
        # After executing this method values will be Poor==0, Good==1, Standard==2               
        mapping = {"Medium":1, "Small":0, "Large":2}
        self.df['value_payments'] = self.df['value_payments'].replace(mapping)
        
         #Since this column is the target variable, I expect it to be located at the end of the dataset.
        value_payments_column = self.df['value_payments']
        self.df = self.df.drop(columns=['value_payments'])
        self.df['value_payments'] = value_payments_column
        
        self.df.drop(['payment_behaviour'], axis=1, inplace=True)
        print('payment_behaviour done')
        return self.df
    
    def ssn_handling(self) -> pd.DataFrame:
        if 'ssn' not in self.df.columns:
            raise ValueError ('column "ssn" not found in the DataFrame!')
        
        self.df.loc[(self.df.ssn=='#F%$D@*&8'),'ssn']=np.nan
        self.df['ssn_aaa']=self.df.ssn.str.extract('([0-9]+)').astype(str)
        self.df['ssn_aaa']=self.df.groupby('customer_id')['ssn_aaa'].transform(lambda x: x.mode()[0])
        self.df.drop('ssn', axis=1, inplace=True)
        print('ssn done')
        return self.df
    
        
    
    
    def floats_handling(self) -> pd.DataFrame:
        """
        Handle float columns:
        - Replace specific strange values with NA.
        - Clean data points from non-numeric characters.
        - Fill missing values with the mode for common columns or median for others.
        - Convert the columns to float and round to 4 decimal places.
        """
        float_columns = ['credit_utilization_ratio',#
                         'annual_income',#
                         'monthly_inhand_salary', 
                         'changed_credit_limit', #
                         'outstanding_debt', #
                         'total_emi_per_month',#
                         'amount_invested_monthly', #
                         'monthly_balance']#
        
        for column in float_columns:
            if column not in self.df.columns:
                raise ValueError(f"Column '{column}' not found in the DataFrame.")
        # Deleting specific strange values
        mapping = {'__-333333333333333333333333333__':pd.NA, 
                  '__10000__':pd.NA}
        self.df.replace(mapping, inplace=True)
        
        # Cleaning the datapoint from non-numeric characters.
        common_to_cost_id=['annual_income', 'changed_credit_limit','amount_invested_monthly','monthly_balance']
        for column in float_columns:
            self.df[column] = self.df[column].astype(str) # All columns will be changed as str to impelement replace
            self.df[column] = self.df[column].str.replace(r'[^0-9.]', '', regex=True) # All characters gone except 0 to 9 and dot
            self.df[column] = self.df[column].replace('', pd.NA) # After deleting characters some rows could be emtpy, they're NA now
            if column in common_to_cost_id:
                self.df[column] = self.df.groupby('customer_id')[column].transform(lambda x: x.mode()[0])
            else:
                self.df[column] = self.df.groupby('customer_id')[column].transform(
                    lambda x: x.fillna(x.median()) if not x.isnull().all() else x)
            
        self.df['total_emi_per_month'] = self.df['total_emi_per_month'].replace(0, pd.NA)
        self.df[column] = self.df[column].astype(float).round(4)
        print('floats done')
        return self.df
    
    def integers_handling(self) -> pd.DataFrame:
        """
        Handle integer columns:
        - Replace specific characters with empty string and NA.
        - Clean data points from non-numeric characters.
        - Fill missing values with the mode for common columns or median for others.
        - Convert the columns to integers.
        """
        integer_columns=['age',                   
                          'num_credit_card',
                          'interest_rate',
                          'delay_from_due_date',
                          'num_of_delayed_payment',
                          'num_of_loan',
                          'num_bank_accounts',
                          'num_credit_inquiries',]
        for column in integer_columns:
            if column not in self.df.columns:
                raise ValueError(f"Column '{column}' not found in the DataFrame.")
        common_to_cost_id=['num_credit_inquiries','age','num_bank_accounts','num_credit_card','interest_rate','num_of_loan']
        for column in integer_columns:
            self.df[column] = self.df[column].astype(str) # All columns will be changed as str to ne able to impelement replace
            self.df[column] = self.df[column].str.replace(r'[^0-9]', '', regex=True) # All characters gone except 0 to 9 
            self.df[column] = self.df[column].replace('', pd.NA) #After deleting characters some rows could be emtpy, they're NA now    
            if column in common_to_cost_id:
                self.df[column] = self.df.groupby('customer_id')[column].transform(lambda x: x.mode()[0])
            else :
                self.df[column] = self.df.groupby('customer_id')[column].transform(
                    lambda x: x.fillna(x.median()) if not x.isnull().all() else x)
                
            self.df[column] = self.df[column].astype(int) 
            print(column)
        print('int done')
    
        return self.df   
            
    def transform_credit_history_age(self) -> pd.DataFrame:
        """
        Transform the 'credit_history_age' column:
        - Extract years and months.
        - Calculate the total months.
        - Fill missing values with forward fill.
        - Rename the column to 'credit_history_months'.
        """
        if 'credit_history_age' not in self.df.columns:
            raise ValueError ('column "credit_history_age" not found in the DataFrame!')
        
        
        years_months =  self.df["credit_history_age"].str.extract(r'(\d+)\s*Years?\s*and\s*(\d+)\s*Months?')
        years = pd.to_numeric(years_months[0], errors='coerce')#.fillna(0)
        months = pd.to_numeric(years_months[1], errors='coerce')#.fillna(0)
        credit_history_monts=(years * 12 + months)
        self.df = pd.concat([self.df, credit_history_monts], axis=1)
        self.df.rename(columns={0: 'credit_history_monts'}, inplace=True)
        self.df['credit_history_monts'] = self.df.groupby('customer_id')['credit_history_monts'].transform(
            lambda x: x.fillna(x.ffill()) if not x.isnull().all() else x)
        self.df.drop('credit_history_age', axis=1, inplace=True)
        print('credit_history_age done')

        return self.df 
    
    
    def credit_score_handling(self) -> pd.DataFrame:
        if 'credit_score' not in self.df.columns:
            raise ValueError ('"credit_score" not found in the DataFrame.')
            
        # After executing this method values will be Poor==0, Good==1, Standard==2               
        mapping = {"Standard":1, "Poor":0, "Good":2}
        self.df['credit_score'] = self.df['credit_score'].replace(mapping)
        
         #Since this column is the target variable, I expect it to be located at the end of the dataset.
        credit_score_column = self.df['credit_score']
        self.df = self.df.drop(columns=['credit_score'])
        self.df['credit_score'] = credit_score_column
        
        print('credit_score done')
        return self.df
    
    def credit_mix_handling(self) -> pd.DataFrame:
        if 'credit_mix' not in self.df.columns:
            raise ValueError ('"credit_mix" not found in the DataFrame.')
        # Deleting strange values like "-"
        self.df['credit_mix'] = self.df['credit_mix'].replace("_", pd.NA)
        
        # Replacing missing values in each row, associated with a specific customer_id, with the most frequent value.
        self.df['credit_mix']= self.df.groupby('customer_id')['credit_mix'].transform(lambda x: x.mode()[0])
        # After executing this method values will be Poor==0, Good==1, Standard==2               
        mapping = {"Standard":1, "Bad":0, "Good":2}
        self.df['credit_mix'] = self.df['credit_mix'].replace(mapping)
        
         #Since this column is the target variable, I expect it to be located at the end of the dataset.
        credit_score_column = self.df['credit_mix']
        self.df = self.df.drop(columns=['credit_mix'])
        self.df['credit_mix'] = credit_score_column
        
        print('credit_mix done')
        return self.df
        
    def knn_impuler(self)->pd.DataFrame:
        missing_data = self.df.isnull().sum()
        missing_variables = missing_data[missing_data > 0].index.tolist()
        if len(missing_variables) != 0:
            imputer = KNNImputer(n_neighbors=5)
            for variabl in missing_variables:
                if self.df[variabl].dtype == int or self.df[variabl].dtype == float: 
                    column_array = self.df[variabl].values.reshape(-1, 1)
                    imputed_column_array = imputer.fit_transform(column_array)
                    self.df[variabl] = imputed_column_array
                    print('Missing values in the %s variable were filled using the knn method'%(variabl))
                else:
                    print('variable %s cannot be replenished using KNN inpuler'%(variabl))
        return self.df
    
class Preprocessor():
    '''
    A utility class to perform transformations on a DataFrame.
    '''
    def __init__(self, dataframe:pd.DataFrame):
        '''
        Constructor method.
        '''
        # Creating copy of original dataframe
        self.df = dataframe.copy()
         
        # Creating instances of transformers.    
        self.col_transformer = col_transformer(in_df=self.df)

        
    def transform(self) -> pd.DataFrame:
        '''
        Transform the DataFrame using col_transformer methods.
        '''

        # Numeric transformation
        self.df = self.col_transformer.credit_mix_handling()
        self.df = self.col_transformer.type_of_loan_handling()
        self.df = self.col_transformer.floats_handling()
        self.df = self.col_transformer.integers_handling()
        self.df = self.col_transformer.occupation_handling()
        self.df = self.col_transformer.payment_behaviour_handling()
        self.df = self.col_transformer.payment_of_min_amount_handling()
        self.df = self.col_transformer.ssn_handling()
        self.df = self.col_transformer.transform_credit_history_age()
        self.df = self.col_transformer.credit_score_handling()
        self.df = self.col_transformer.credit_mix_handling()
        self.df = self.col_transformer.knn_impuler()

        return self.df
    
    def transform_and_gen_dummies(self)-> pd.DataFrame:
#         self.transform()
        
        occupation_dummies = pd.get_dummies(self.df['occupation'],  prefix='occupation', dtype=int)
        self.df = pd.concat([self.df, occupation_dummies], axis=1)
        self.df.drop('occupation', axis=1, inplace=True)
        
       
       # credit_mix_dummies = pd.get_dummies(self.df['credit_mix'], drop_first=True, prefix='credit_mix', dtype=int)
       # self.df = pd.concat([self.df, credit_mix_dummies], axis=1)
       # self.df.drop('credit_mix', axis=1, inplace=True)
        
        # Binary encoding
        payment_of_min_amount_dummies = pd.get_dummies(self.df['payment_of_min_amount'],drop_first=True ,prefix='payment_of_min_amount', dtype=int)
        self.df = pd.concat([self.df, payment_of_min_amount_dummies], axis=1)
        self.df.drop('payment_of_min_amount', axis=1, inplace=True)
        
        # Binary encoding spent
       # spent_dummies = pd.get_dummies( self.df['spent'],drop_first=True , dtype=int)
       # self.df = pd.concat([self.df, spent_dummies], axis=1)
        #self.df.drop('spent', axis=1, inplace=True)
        # Binary encoding value_payments
        #value_payments_dummies = pd.get_dummies(self.df['value_payments'],drop_first=True , dtype=int)
        #self.df = pd.concat([self.df, value_payments_dummies], axis=1)
        #self.df.drop('value_payments', axis=1, inplace=True)
        return self.df
    
    
    
def plot_missing(df):
    '''
    This function shows the distribution of missing values for variables with missing data in the dataset as a graph.
    '''
    missing_data = df.isnull().sum()
    missing_variables = missing_data[missing_data > 0].index.tolist()

    if len(missing_variables) == 0:
        print("No missing data in the variables.")
    else:
        plt.figure(figsize=(10, 4), dpi=80)
        sns.heatmap(df[missing_variables].isnull().transpose(), xticklabels=False, cbar=False, cmap='viridis')
        plt.title('Distribution of Missing Values among Variables', fontsize=20)
        plt.show()

        print("Missing observations in %:" )
        display(missing_data[missing_variables] / len(df) * 100)
        print("Number of missing observations:")
        display(missing_data[missing_variables])
        
        
def display_dataframe_summary(df,show_rows=2):
    '''
    The function is designed to provide a comprehensive summary of a DataFrame, including its shape, data types, and key statistical characteristics

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    print('First %s rows:'%(str(show_rows)))
    display(df.head(show_rows))
    print('_' * 50)
    print('Last %s rows:'%(str(show_rows)))
    display(df.tail(show_rows))
    print('_' * 50)
    print('Shape (number of rows and columns):')
    display(df.shape)
    print('_' * 50)
    print('DataFrame information:')
    display(df.info())
    print('_' * 50)
    print('Summary of key statistical characteristics for numeric columns:')
    display(df.describe())
    print('_' * 50)
    print('Summary of statistics for columns with data type "object":')
    display(df.describe(include="object"))


def targe_plot(df,target):
    fig, ax = plt.subplots(1, 3,figsize=(18,6), dpi=100)

    sns.histplot(x=target, data=df, kde=True, element="step", stat="density", ax=ax[0])

    ## Plotting the QQ_Plot. 
    stats.probplot(df.loc[:,target], plot = ax[1])

    sns.boxplot(x=target, data=df,  ax=ax[2])
    ax[0].set_title(f"Skewness: {df[target].skew()}  \n Kurtosis: {df[target].kurt()}", fontsize=14, )

    plt.suptitle(target, fontsize=18)
    fig.tight_layout()
    plt.show()
  
def perform_tests(df):
    # Непрерывные переменные
    transform_var_list = ['age', 'annual_income', 'interest_rate', 'delay_from_due_date', 'changed_credit_limit', 'outstanding_debt', 'credit_utilization_ratio', 'credit_history_monts', 'num_of_delayed_payment', 'total_emi_per_month', 'amount_invested_monthly', 'monthly_balance']

    summary = []

    # Хи-квадрат тест для категориальных переменных
    for col in df.columns[:-1]:
        if col in transform_var_list:
            t_stat, pvalue = f_classif(df[[col]], df["credit_score"])
            summary.append([col, t_stat[0], pvalue[0]])
        else :
            cross = pd.crosstab(index=df[col], columns=df["credit_score"])
            t_stat, pvalue, *_ = chi2_contingency(cross)
            summary.append([col, t_stat, pvalue])

    return pd.DataFrame(
        data=summary,
        columns=["column", 't-statistic', "p-value"]
    )


def ROC_curve(model,X_test,Y_test):
    sns.set(font_scale=1.5)
    sns.set_color_codes("muted")
    plt.figure(figsize=(10, 8))
    fpr, tpr, thresholds = roc_curve(Y_test, model.predict_proba(X_test)[:,1], pos_label=1)
    lw = 2
    plt.plot(fpr, tpr, lw=lw, label='ROC curve ')
    plt.plot([0, 1], [0, 1])
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.savefig("ROC.png")
    plt.show()
    
    
def Perm_import(model,X_test,Y_test,random_state=42):
    # Calculation of permutation importance of features
    result = permutation_importance(model, X_test, Y_test, n_repeats=10, random_state=random_state)
    # Getting Feature Importance
    feature_importance = result.importances_mean
    # Feature Importance Visualization
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(feature_importance)), feature_importance, align='center')
    plt.xticks(range(len(feature_importance)), X_test.columns, rotation='vertical')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Permutational importance of features')
    plt.show()
    
def plot_learning_curve(estimator, X_train, y_train, X_test, y_test, train_sizes):
    train_sizes, train_scores, test_scores = learning_curve(estimator, X_train, y_train, train_sizes=train_sizes, cv=5)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.title('Learning Curve')
    plt.xlabel('Training Set Size')
    plt.ylabel('Score')
    plt.grid(True)
    
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color='b')
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color='r')
    
    plt.plot(train_sizes, train_scores_mean, 'o-', color='b', label='Training Score')
    plt.plot(train_sizes, test_scores_mean, 'o-', color='r', label='Test Score')
    
    plt.legend(loc='best')
    plt.show()


class SBS():
    def __init__(self, estimator, k_features, scoring=accuracy_score, test_size=0.25, random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state

    def from_dataframe(self, df, target_column):
        X = df.drop(columns=[target_column]).values
        y = df[target_column].values
        return X, y

    def to_dataframe(self, X, df):
        new_df = df.copy()
        new_df['New_Column'] = X  # Replace 'New_Column' with the name you desire
        return new_df
    def fit(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(X_train, y_train, X_test, y_test, self.indices_)
        self.scores_ = [score]

        while dim > self.k_features:
            print(dim)
            scores = []
            subsets = []

            def calculate_score(p):
                return self._calc_score(X_train, y_train, X_test, y_test, p)

            results = Parallel(n_jobs=-1)(delayed(calculate_score)(p) for p in combinations(self.indices_, r=dim - 1))
            
            for result, p in zip(results, combinations(self.indices_, r=dim - 1)):
                scores.append(result)
                subsets.append(p)

            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1
            self.scores_.append(scores[best])

        self.k_scores_ = self.scores_[-1]
        return self
    def transform(self,X):
        return X[:,self.indices_]
    def _calc_score(self,X_train,y_train,X_test,y_test,indices):
        self.estimator.fit(X_train[:, indices],y_train)
        y_pred=self.estimator.predict(X_test[:,indices])
        score=self.scoring(y_test,y_pred)
        return score
