import numpy as np

import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier

from sklearn import preprocessing

from sklearn import model_selection

# ----------------------------------------------------------------------------

def differential_evolution(fobj, 
                           bounds, 
                           mut=0.2, 
                           crossp=0.7, 
                           popsize=20, 
                           maxiter=100,
                           verbose = True):
    '''
    This generator function yields the best solution x found so far and 
    its corresponding value of fobj(x) at each iteration. In order to obtain 
    the last solution,  we only need to consume the iterator, or convert it 
    to a list and obtain the last value with list(differential_evolution(...))[-1]    
    
    
    @params
        fobj: function to minimize. Can be a function defined with a def 
            or a lambda expression.
        bounds: a list of pairs (lower_bound, upper_bound) for each 
                dimension of the input space of fobj.
        mut: mutation factor
        crossp: crossover probability
        popsize: population size
        maxiter: maximum number of iterations
        verbose: display information if True    
    '''
    #Dimension of the input space of 'fobj'
    n_dimensions = len(bounds)
    
    #This generates our initial population by a random method
    #Each component pop[i] is between [0, 1] which is normalized.
    pop = np.random.rand(popsize, n_dimensions) #
    
    #We will use the bounds to denormalize each component only for evaluating them with fobj. 
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    pop_denorm = min_b + pop * diff


    #Evaluate denormalized population and save each memeber's effiency in fitness array
    fitness = np.asarray([fobj(ind) for ind in pop_denorm])

    #The memeber of initial population which has teh best cost is saved in 'best' variable
    best_idx = np.argmin(fitness)
    best = pop_denorm[best_idx]
    
    if verbose:
        print('** Lowest cost in initial population = {} '.format(fitness[best_idx]))   

    #Create 'maxiter' numbers of generations       
    for i in range(maxiter):
        if verbose:
            print('** Starting generation {}, '.format(i))  
        #Create 'popsize' numbers of members in each generation 
        #For each individual 'j' in the population do:          
        for j in range(popsize):
                #Pick three distinct individuals a, b and c from the current population at random. 
                #The individuals a, b and c must be distinct from j as well.
            idxs = [idx for idx in range(popsize) if idx != j]
            a, b, c = pop[np.random.choice(idxs, 3, replace = False)]

            #Create a mutant vector <a + mut * (b – c)> where mut is a constant (passed as an argument to
            #the differential_evolution function)
            #Clip the mutant entries to the interval [0, 1]
            mutant = np.clip(a + mut * (b - c), 0, 1)

                        #Create a trial vector by assigning to trial[k] with probability crossp the value mutant[k]
                        #and probability 1-crossp the value of j[k]
            cross_points = np.random.rand(n_dimensions) < crossp
            if not np.any(cross_points):
                cross_points[np.random.randint(0, n_dimensions)] = True
            trial = np.where(cross_points, mutant, pop[j])
            trial_denorm = min_b + trial * diff

            #Evaluate the cost of the trial vector. If the trial vector is better than j, then replace j by the
                        #trial vector
            f = fobj(trial_denorm)
            if f < fitness[j]:
                fitness[j] = f
                pop[j] = trial
                if f < fitness[best_idx]:
                    best_idx = j
                    best = trial_denorm
        yield best, fitness[best_idx]

# ----------------------------------------------------------------------------

def task_1():
    '''
    Our goal is to fit a curve (defined by a polynomial) to the set of points 
    that we generate randomly. 

    '''

    # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
    def fmodel(x,w):

        '''
        Compute and return the value y of the polynomial with coefficient 
        vector w at x.  
        For example, if w is of length 5, this function should return
        w[0] + w[1]*x + w[2] * x**2 + w[3] * x**3 + w[4] * x**4 
        The argument x can be a scalar or a numpy array.
        The shape and type of x and y are the same (scalar or ndarray).

        @params
                x: x can be a scalar or a numpy array.
                w: coefficient vector w
        '''

        if isinstance(x, float) or isinstance(x, int):
            y = 0
        else:
                #Python evaluates the accompanying 'type(x) is np.ndarray', which is hopefully true. 
                #If the expression is false, Python raises an AssertionError exception. 
                #If the assertion fails, Python uses ArgumentExpression as the argument for the AssertionError
            assert type(x) is np.ndarray
            y = np.zeros_like(x)

        for i in reversed(range(0,len(w))):
            y = w[i] + y*x
        return y
    # -------------
    # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        
    def rmse(w):
        '''
        Compute and return the root mean squared error (RMSE) of the 
        polynomial defined by the weight vector w. 
        The RMSE is is evaluated on the training set (X,Y) where X and Y
        are the numpy arrays defined in the context of function 'task_1'.        
        '''
        y_pred = fmodel(X, w)
        return np.sqrt(sum((Y - y_pred)**2) / len(Y))
        
    # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .  
    
    # Create the training set
    X = np.linspace(-5, 5, 500)
    Y = np.cos(X) + np.random.normal(0, 0.2, len(X))
    
    # Create the DE generator
    de_gen = differential_evolution(rmse, [(-5, 5)] * 6, mut=1, maxiter=2000)
    
    # We'll stop the search as soon as we found a solution with a smaller
    # cost than the target cost
    target_cost = 0.5
    
    # Loop on the DE generator
    for i , p in enumerate(de_gen):
        w, c_w = p
        # w : best solution so far
        # c_w : cost of w        
        # Stop when solution cost is less than the target cost
        if c_w< target_cost:
            break
        
    # Print the search result
    print('Stopped search after {} generation. Best cost found is {}'.format(i,c_w))
    #    result = list(differential_evolution(rmse, [(-5, 5)] * 6, maxiter=1000))    
    #    w = result[-1][0]
        
    # Plot the approximating polynomial
    plt.scatter(X, Y, s=2)
    plt.plot(X, np.cos(X), 'r-',label='cos(x)')
    plt.plot(X, fmodel(X, w), 'g-',label='model')
    plt.legend()
    plt.title('Polynomial fit using DE')
    plt.show()    
    
# ----------------------------------------------------------------------------

def task_2():
    '''
    Goal : find hyperparameters for a MLP
    
       w = [nh1, nh2, alpha, learning_rate_init]
    '''
    
    
    # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .  
    def eval_hyper(w):
        '''
        Return the negative of the accuracy of a MLP with trained 
        with the hyperparameter vector w
        
        alpha : float, optional, default 0.0001
                L2 penalty (regularization term) parameter.
        '''
        
        nh1, nh2, alpha, learning_rate_init  = (
                int(1+w[0]), # nh1
                int(1+w[1]), # nh2
                10**w[2], # alpha on a log scale
                10**w[3]  # learning_rate_init  on a log scale
                )


        clf = MLPClassifier(hidden_layer_sizes=(nh1, nh2), 
                            max_iter=100, 
                            alpha=alpha, #1e-4
                            learning_rate_init=learning_rate_init, #.001
                            solver='sgd', verbose=10, tol=1e-4, random_state=1
                            )
        
        clf.fit(X_train_transformed, y_train)
        # compute the accurary on the test set
        mean_accuracy = clf.score( X_test_transformed,y_test)
 
        return -mean_accuracy
    
    # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .  

    # Load the dataset
    input_file = input("Please input the address of your dataset_inputs ")
    target_file=input("Please input the address of your dataset_targets ")
    X_all = np.loadtxt(input_file, dtype=np.uint8)[:1000]
    y_all = np.loadtxt(target_file,dtype=np.uint8)[:1000]    
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X_all, y_all, test_size=0.4, random_state=42)
       
    # Preprocess the inputs with 'preprocessing.StandardScaler'
    
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_transformed = scaler.transform(X_train)
    X_test_transformed = scaler.transform(X_test)


    
    bounds = [(1,100),(1,100),(-6,2),(-6,1)]  # bounds for hyperparameters
    
    de_gen = differential_evolution(
            eval_hyper, 
            bounds, 
            mut = 1,
            popsize=10, 
            maxiter=20,
            verbose=True)
    
    for i, p in enumerate(de_gen):
        w, c_w =p
        print('Generation {},  best cost {}'.format(i,abs(c_w)))
        # Stop if the accuracy is above 90%
        if abs(c_w)>0.90:
            break
 
    # Print the search result
    print('Stopped search after {} generation. Best accuracy reached is {}'.format(i,abs(c_w)))   
    print('Hyperparameters found:')
    print('nh1 = {}, nh2 = {}'.format(int(1+w[0]), int(1+w[1])))          
    print('alpha = {}, learning_rate_init = {}'.format(10**w[2],10**w[3]))
# ----------------------------------------------------------------------------    
def task_3(number_of_runs):
    '''
        Goal : to run experiments to compare cost and performance of the following
        (population_size, max_iter) allocations in the list [(5,40), (10,20),(20,10),(40,5)].
        
        @params
                number_of_runs: run each allocation 'number_of_runs' times to get the best, worst and mean accuracy

    '''
        # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
    def eval_hyper(w):
        '''
        Return the negative of the accuracy of a MLP with trained 
        with the hyperparameter vector w
        
        alpha : float, optional, default 0.0001
                L2 penalty (regularization term) parameter.
        '''
        
        nh1, nh2, alpha, learning_rate_init  = (
                int(1+w[0]), # nh1
                int(1+w[1]), # nh2
                10**w[2], # alpha on a log scale
                10**w[3]  # learning_rate_init  on a log scale
                )


        clf = MLPClassifier(hidden_layer_sizes=(nh1, nh2), 
                            max_iter=100, 
                            alpha=alpha, #1e-4
                            learning_rate_init=learning_rate_init, #.001
                            solver='sgd', verbose=10, tol=1e-4, random_state=1
                            )
        
        clf.fit(X_train_transformed, y_train)
        # compute the accurary on the test set
        mean_accuracy = clf.score( X_test_transformed,y_test)
 
        return -mean_accuracy
    
    # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
    
    population_size, max_iter= np.asarray([(5,40), (10,20),(20,10),(40,5)]).T
    # Load the dataset
    input_file = input("Please input the address of your dataset_inputs ")
    target_file=input("Please input the address of your dataset_targets ")
    X_all = np.loadtxt(input_file, dtype=np.uint8)[:1000]
    y_all = np.loadtxt(target_file,dtype=np.uint8)[:1000] 
    
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X_all, y_all, test_size=0.4, random_state=42)
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_transformed = scaler.transform(X_train)
    X_test_transformed = scaler.transform(X_test)
    bounds = [(1,100),(1,100),(-6,2),(-6,1)] 
    
    for i in range(len(population_size)):
        #allocate new population_size and max_iter to evaluate their performance
        population=population_size[i]
        iteration=max_iter[i]

        #Print used population_size and max_iter
        print('*******************************************************************')
        print('POPULATION SIZE {},  MAX ITERATION {}'.format(population,iteration))
        print('*******************************************************************')

        
        de_gen = differential_evolution(
            eval_hyper, 
            bounds, 
            mut = 1,
            popsize=population, 
            maxiter=iteration,
            verbose=True)
    
        for i, p in enumerate(de_gen):
            w, c_w =p
            print('Generation {},  best cost {}'.format(i,abs(c_w)))
        # Stop if the accuracy is above 90%
            if abs(c_w)>0.90:
                break
 
        # Print the search result
        print('Stopped search after {} generation. Best accuracy reached is {}'.format(i,abs(c_w)))   
        print('Hyperparameters found:')
        print('nh1 = {}, nh2 = {}'.format(int(1+w[0]), int(1+w[1])))          
        print('alpha = {}, learning_rate_init = {}'.format(10**w[2],10**w[3]))
    


    
     # bounds for hyperparameters
    
    
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    pass
    task_1()    
#    task_2()    
#    task_3()    
