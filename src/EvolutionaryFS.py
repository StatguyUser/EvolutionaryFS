import numpy as np
np.random.seed(1)

import time
import warnings
import sys
import os
if not sys.warnoptions:    
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = 'ignore'

class GeneticAlgorithmFS:
    '''
    Machine Learning Parameters
    ----------
    
    model : Model object. It should have .fit and .predict attribute
        
    data_dict : X and Y training and test data provided in dictionary format. Below is example of 5 fold cross validation data with keys.
        {0:{'x_train':x_train_dataframe,'y_train':y_train_array,'x_test':x_test_dataframe,'y_test':y_test_array},
        1:{'x_train':x_train_dataframe,'y_train':y_train_array,'x_test':x_test_dataframe,'y_test':y_test_array},
        2:{'x_train':x_train_dataframe,'y_train':y_train_array,'x_test':x_test_dataframe,'y_test':y_test_array},
        3:{'x_train':x_train_dataframe,'y_train':y_train_array,'x_test':x_test_dataframe,'y_test':y_test_array},
        4:{'x_train':x_train_dataframe,'y_train':y_train_array,'x_test':x_test_dataframe,'y_test':y_test_array}}

        If you only have train and test data and do not wish to do cross validation, use above dictionary format, with only one key.

    cost_function : Cost function for finding cost between actual and predicted values, depending on regression or classification problem.
        cost function should accept 'actual' and 'predicted' as arrays and return cost for the both.
    
    average : Averaging to be used. This is useful for clasification metrics such as 'f1_score', 'jaccard_score', 'fbeta_score', 'precision_score',
        'recall_score' and 'roc_auc_score' when dependent variable is multi-class
    
    cost_function_improvement : Objective is to whether increase or decrease the cost during subsequent iterations.
        For regression it should be 'decrease' and for classification it should be 'increase'
    
    columns_list : Column names present in x_train_dataframe and x_test which will be used as input list for searching best list of features.
    
    Genetic Algorithm Parameters
    ----------
    
    generations : Number of generations to run genetic algorithm. 100 as deafult
    
    population : Number of individual chromosomes. 50 as default. It should be kept as low number if number of possible permutation and combination of feature sets are small.
    
    prob_crossover : Probability of crossover. 0.9 as default
    
    prob_mutation : Probability of mutation. 0.1 as default
        
    run_time : Number of minutes to run the algorithm. This is checked in between generations.
        At start of each generation it is checked if runtime has exceeded than alloted time.
        If case run time did exceeds provided limit, best result from generations executed so far is given as output.
        Default is 2 hours. i.e. 120 minutes.


    Output
    ----------
    best_columns : List object with list of column names which gives best performance for the model. These features can be used for training and saving models separately by the user.
    
    
    '''
    def __init__(self,model,data_dict,cost_function,average=None,cost_function_improvement='increase',columns_list=[],generations=50,population=40,prob_crossover=0.9,prob_mutation=0.1,run_time=600):
        self.model=model
        self.data_dict=data_dict
        self.cost_function=cost_function
        self.average=average
        self.cost_function_improvement=cost_function_improvement
        self.generations=generations
        self.population=population
        self.prob_crossover=prob_crossover
        self.prob_mutation=prob_mutation
        self.run_time=run_time
        self.columns_list=columns_list


    def _get_feature_index(self,features):
        t=0
        index_list=[]
        
        for feat in features:
            if feat==1:
                index_list.append(t)
            t+=1
        return index_list
    
    def _getCost(self,population_array):
        
        columns_list=list(map(list(self.columns_list).__getitem__,self._get_feature_index(population_array)))

        fold_cost=[]
        
        for i in self.data_dict.keys():
            
            x_train=self.data_dict[i]['x_train'][columns_list]
            y_train=self.data_dict[i]['y_train']
            
            x_test=self.data_dict[i]['x_test'][columns_list]
            y_test=self.data_dict[i]['y_test']
                        
            self.model.fit(x_train,y_train)
            y_test_predict=self.model.predict(x_test)
            
            if self.average:
                fold_cost.append(self.cost_function(y_test,y_test_predict,average=self.average))
            else:
                fold_cost.append(self.cost_function(y_test,y_test_predict))

        return np.mean(fold_cost)

    
    def _check_unmatchedrows(self,population_matrix,population_array):
        pop_check=0
        
        #in each row of population matrix
        for pop_so_far in range(population_matrix.shape[0]):
            #check duplicate
            if sum(population_matrix[pop_so_far]!=population_array)==population_array.shape[0]:
                #assign 1 for duplicate
                pop_check=1
                break
        return pop_check
    def _get_population(self,population_matrix,population_array):
        iterate=0
        ## append until population and no duplicate chromosome
        while population_matrix.shape[0]<self.population:
            #prepare population matrix
            np.random.shuffle(population_array)
            
            #check if it is first iteration, if yes then append
            if iterate==0:
                population_matrix=np.vstack((population_matrix,population_array))
                iterate+=1
            #if second iteration and once chromosome already, check if it is duplicate
            elif iterate==1 and sum(population_matrix[0]==population_array)!=population_array.shape[0]:
                population_matrix=np.vstack((population_matrix,population_array))
                iterate+=1
            elif iterate>1 and self._check_unmatchedrows(population_matrix,population_array)==0:
                population_matrix=np.vstack((population_matrix,population_array))
        
        return population_matrix
    
    def _get_parents(self,population_array,population_matrix):
        
        #keep space for best chromosome
        parents = np.empty((0,population_array.shape[0]))
        
        #get 6 unique index to fetch from population
        indexes=np.random.randint(0,population_matrix.shape[0],6)
        
        while len(np.unique(indexes))<6:
            indexes=np.random.randint(0,population_matrix.shape[0],6)
            
        #mandatory run twice as per GA algorithm
        for run_range in range(2):
            
            #get 3 unique index to fetch from population
            if run_range==0:
                index_run=indexes[0:3]
            #if second run then from half till end
            else:
                index_run=indexes[3:]
                
            ## gene pool 1
            gene_1=population_matrix[index_run[0]]
            ## cost of gene 1
            fold_cost1=self._getCost(population_array=gene_1)
            #print('fold_cost1',fold_cost1)
            ## gene pool 2
            gene_2=population_matrix[index_run[1]]
            ## cost of gene 2
            fold_cost2=self._getCost(population_array=gene_2)
            #print('fold_cost2',fold_cost2)
            
            ## gene pool 3
            gene_3=population_matrix[index_run[2]]
            ## cost of gene 2
            fold_cost3=self._getCost(population_array=gene_3)
            #print('fold_cost3',fold_cost3)
            
            #get best chromosome from 3 and assign best chromosome
            if fold_cost1==max(fold_cost1,fold_cost2,fold_cost3):
                parents=np.vstack((parents,gene_1))
            elif fold_cost2==max(fold_cost1,fold_cost2,fold_cost3):
                parents=np.vstack((parents,gene_2))
            elif fold_cost3==max(fold_cost1,fold_cost2,fold_cost3):
                parents=np.vstack((parents,gene_3))
        
        return parents[0],parents[1]
    
    def _crossover(self,parent1,parent2):
        
        #placeholder for child chromosome
        child1=np.empty((0,len(parent1)))
        child2=np.empty((0,len(parent2)))
        
        crsvr_rand_prob=np.random.rand()
        
        if crsvr_rand_prob < self.prob_crossover:
            index1=np.random.randint(0,len(parent1))
            index2=np.random.randint(0,len(parent2))
            
            #get different indices to make sure crossover happens
            while index1 == index2:
                index2=np.random.randint(0,len(parent1))
                
            index_parent1=min(index1,index2)
            index_parent2=max(index1,index2)
            
            #parent1
            #first segment
            first_seg_parent1=parent1[:index_parent1]
            #second segment
            mid_seg_parent1=parent1[index_parent1:index_parent2+1]
            #third segment
            last_seg_parent1=parent1[index_parent2+1:]
            child1=np.concatenate((first_seg_parent1,mid_seg_parent1,last_seg_parent1))
            
            #parent2
            #first segment
            first_seg_parent2=parent2[:index_parent1]
            #second segment
            mid_seg_parent2=parent2[index_parent1:index_parent2+1]
            #third segment
            last_seg_parent2=parent2[index_parent2+1:]
            child2=np.concatenate((first_seg_parent2,mid_seg_parent2,last_seg_parent2))
            
            return child1,child2
        else:
            return parent1,parent2
        
    def _mutation(self,child):
        #mutated child 1 placeholder
        mutated_child=np.empty((0,len(child)))
        
        #get random probability at each index of chromosome and start with 0
        t=0
        
        for cld1 in child:
            rand_prob_mutation = np.random.rand()
            if rand_prob_mutation<self.prob_mutation:
                #swap value
                if child[t]==0:
                    child[t]=1
                else:
                    child[t]=0
                
                mutated_child=child
            #if probability is less
            else:
                mutated_child=child
            t+=1
            
        return mutated_child
    
    def _getpopulationMatrix(self,total_columns):
        #generate chromosome based on number of features in base model and hyperparameter
        population_array=np.random.randint(0,2,total_columns)
        
        #shuffle after concatenating 0 abd 1
        np.random.shuffle(population_array)
        
        #create blank population matrix to append all individual chrososomes
        
        population_matrix=np.empty((0,total_columns))
        
        #get population matrix
        population_matrix=self._get_population(population_matrix,population_array)
        
        #best solution for each generation
        best_of_a_generation = np.empty((0,len(population_array)+1))
        
        return population_array,population_matrix,best_of_a_generation
    
    def GetBestFeatures(self):
        #record time
        start=time.time()
        
        
        if 0 in self.data_dict.keys():
#            total_columns=len(self.data_dict[0]['x_train'].shape[1]
            total_columns=len(self.columns_list)
            print('total_columns',total_columns)

        ##get population array to begin
        population_array,population_matrix,best_of_a_generation=self._getpopulationMatrix(total_columns=total_columns)
            

        for genrtn in range(self.generations):
            #if time exceeds, break loop
            if (time.time()-start)//60>self.run_time:
                print('==========================================Run time exceeded allocated time. Producing best solution generated so far.==========================================')
                break
            
            #placeholder for saving new generation
            new_population = np.empty((0,len(population_array)))
            
            #placeholder for saving new generation
            new_population_with_obj_val = np.empty((0,len(population_array)+1))
            
            #placeholder for saving best solution for each generation
            sorted_best = np.empty((0,len(population_array)+1))


            #doing it half population size will mean getting matrix of population size equal to original matrix
            for family in range(int(self.population/2)):

                parent1=[]
                parent2=[]
                
                while len(parent1)==0 and len(parent2)==0:
                    parent1,parent2=self._get_parents(population_array=population_array,population_matrix=population_matrix)

                #crossover
                child1=[]
                child2=[]
                while len(child1)==0 and len(child2)==0:
                    child1,child2=self._crossover(parent1=parent1,parent2=parent2)

                #mutation
                mutated_child1 = []
                mutated_child2 = []
                while len(mutated_child1)==0 and len(mutated_child2)==0:
                    mutated_child1=self._mutation(child=child1)
                    mutated_child2=self._mutation(child=child2)
                

                #get cost function for 2 mutated child and print for generation, family and child                
                fold_cost1=self._getCost(population_array=mutated_child1)
                fold_cost2=self._getCost(population_array=mutated_child2)
                
                #create population for next generation
                new_population=np.vstack((new_population,mutated_child1,mutated_child2))
                
                #save cost and child
                mutant1_with_obj_val=np.hstack((fold_cost1,mutated_child1))
                mutant2_with_obj_val=np.hstack((fold_cost2,mutated_child2))
                
                #stack both chromosome of the family
                new_population_with_obj_val = np.vstack((new_population_with_obj_val,mutant1_with_obj_val,mutant2_with_obj_val))
                
            #at end of generation, change population as the stacked chromosome set from previous generation
            population_matrix = new_population
            
            if self.cost_function_improvement=='increase':
                #find the best solution for generation based on objective function and stack
                sorted_best = np.array(sorted(new_population_with_obj_val,key=lambda x:x[0],reverse=True))
            elif self.cost_function_improvement=='decrease':
                #find the best solution for generation based on objective function and stack
                sorted_best = np.array(sorted(new_population_with_obj_val,key=lambda x:x[0],reverse=False))

            print('==========================================Best performance for genetaion ',genrtn,':',sorted_best[0][0],'==========================================')
            best_of_a_generation=np.vstack((best_of_a_generation,sorted_best[0]))

        if self.cost_function_improvement=='increase':
            #sort by metric
            best_metric_chromosome_pair=np.array(sorted(best_of_a_generation,key=lambda x:x[0],reverse=True))[0]
        elif self.cost_function_improvement=='decrease':
            best_metric_chromosome_pair=np.array(sorted(best_of_a_generation,key=lambda x:x[0],reverse=False))[0]

        #best chromosome, metric and vocabulary
        best_chromosome = best_metric_chromosome_pair[1:]

#        columns_list=list(map(list(self.data_dict[0]['x_train'].columns).__getitem__,self._get_feature_index(best_chromosome)))
        columns_list=list(map(list(self.columns_list).__getitem__,self._get_feature_index(best_chromosome)))

        print('========================================== Best result: ',best_metric_chromosome_pair[0],'==========================================')
        print('==========================================Execution time in minutes:',(time.time()-start)//60,'==========================================')
        return columns_list
        
if __name__=="__main__":
    print('Evolutionary Algorithm for feature selection')