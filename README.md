# What is it?
EvolutionaryFS module helps in identifying combination of features that gives best result. Process of searching best combination is called 'feature selection'. This library uses population based methods for performing feature selection, such as genetic algorithm. 

# Input parameters
  - **Machine Learning Parameters**
    
    `model` : Model object. It should have .fit and .predict attribute

    `data_dict` : X and Y training and test data provided in dictionary format. Below is example of 5 fold cross validation data with keys.
        {0:{'x_train':x_train_dataframe,'y_train':y_train_array,'x_test':x_test_dataframe,'y_test':y_test_array},
        1:{'x_train':x_train_dataframe,'y_train':y_train_array,'x_test':x_test_dataframe,'y_test':y_test_array},
        2:{'x_train':x_train_dataframe,'y_train':y_train_array,'x_test':x_test_dataframe,'y_test':y_test_array},
        3:{'x_train':x_train_dataframe,'y_train':y_train_array,'x_test':x_test_dataframe,'y_test':y_test_array},
        4:{'x_train':x_train_dataframe,'y_train':y_train_array,'x_test':x_test_dataframe,'y_test':y_test_array}}

        If you only have train and test data and do not wish to do cross validation, use above dictionary format, with only one key.

    `cost_function` : Cost function for finding cost between actual and predicted values, depending on regression or classification problem.
        cost function should accept 'actual' and 'predicted' as arrays and return cost for the both.
    
    `average` : Averaging to be used. This is useful for clasification metrics such as 'f1_score', 'jaccard_score', 'fbeta_score', 'precision_score',
        'recall_score' and 'roc_auc_score' when dependent variable is multi-class
    
    `cost_function_improvement` : Objective is to whether increase or decrease the cost during subsequent iterations.
        For regression it should be 'decrease' and for classification it should be 'increase'
    
    `columns_list` : Column names present in x_train_dataframe and x_test which will be used as input list for searching best list of features.
    
  - **Genetic Algorithm Parameters**
    
    `generations` : Number of generations to run genetic algorithm. 100 as deafult
    
    `population` : Number of individual chromosomes. 50 as default. It should be kept as low number if number of possible permutation and combination of feature sets are small.
    
    `prob_crossover` : Probability of crossover. 0.9 as default
    
    `prob_mutation` : Probability of mutation. 0.1 as default
        
    `run_time` : Number of minutes to run the algorithm. This is checked in between generations.
        At start of each generation it is checked if runtime has exceeded than alloted time.
        If case run time did exceeds provided limit, best result from generations executed so far is given as output.
        Default is 2 hours. i.e. 120 minutes.

# Output
  - **best_columns** : List object with list of column names which gives best performance for the model. These features can be used for training and saving models separately by the user.

# How to use is it?
```python
from EvolutionaryFS import GeneticAlgorithmFS

### Regression
>>> from sklearn.ensemble import RandomForestRegressor
>>> from sklearn.metrics import mean_squared_error

>>> model_object=RandomForestRegressor(n_jobs=-1,random_state=1,n_estimators=1000)
>>> evoObj=GeneticAlgorithmFS(model=model_object,data_dict=data_dict,cost_function=mean_squared_error,average='',cost_function_improvement='decrease',columns_list=['column1','column2'],generations=20,population=30,prob_crossover=0.9,prob_mutation=0.1,run_time=60000)
>>> best_columns=evoObj.GetBestFeatures()

### Classification
>>> from sklearn.linear_model import LogisticRegression
>>> from sklearn.metrics import f1_score

>>> model_object=LogisticRegression(n_jobs=-1,random_state=1)
>>> evoObj=GeneticAlgorithmFS(model=model_object,data_dict=data_dict,cost_function=f1_score,average='micro',cost_function_improvement='increase',columns_list=['column1','column2'],generations=20,population=30,prob_crossover=0.9,prob_mutation=0.1,run_time=60000)
>>> best_columns=evoObj.GetBestFeatures()
```
# Where to get it?
`pip install EvolutionaryFS`

# Dependencies
 - [numpy](https://numpy.org/)
