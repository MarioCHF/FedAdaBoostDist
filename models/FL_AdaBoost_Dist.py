import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,recall_score,precision_score
import math
from sklearn.preprocessing import OneHotEncoder
from flex.data import FedDataDistribution, FedDatasetConfig,Dataset
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import f1_score
from models.AdaBoostClassifier2 import AdaBoostClassifier2

#For reproducibility, on top of a random_state, fix also a numpy seed (for dirichlet weights)

class FLEnsembleDist():
    def __init__(self,data,targets,public_data,clients_classifier = 'DecisionTreeClassifier',max_depth=None,
                  max_leaf_nodes=None, server_classifier = 'DecisionTreeClassifier',
                 Nclients=5,public_data_prediction='majority_voting',clients_weight_adj = 'server_error',
                 server_alpha_weight_adj='common_abs',prediction_weights = 'only_server',
                 T=10,data_distribution ='iid',distribution_param=None, alpha_counter = 3,random_state = 0,
                 adapt_client_weight = None, balanced_target_client_weight=False):
        '''Simulated federated learning ensemble algorithm. Parameters: 
        
        data: A 2-d array representing all the clients data.
        
        targets: 1-d array representing predictions associated with data.
        
        public_data: 2-d array of dimensions data.shape. It represents the public data used to train the server model.
        
        clients_classifier: The learning algorithm used to train clients models. It can take the values: DecisionTreeClasifier, ...

        server_classifier: The learning algorithm used to train the server's models. It can take the values: DecisionTreeClasifier, ...

        Nclients: Integer representing the number of clients

        public_data_prediction: It can take the values: 'majority_voting', 'weighted_majority_voting'. It is the way public data is labeled
        based on each clients model prediction. 

        clients_weight_adj: It is the way clients data weights are actualized in every iteration.
        It can take the values: 
        'server_error': Weight gets actualized according to the error of the server model in the clients data
        

        server_alpha_weight_adj: It can take the values: 'common_abs','common_weighted','own','avg_abs','avg_weighted'. It is the way server's models 
        weights are adjusted. 
        'common_abs' == It uses the mean of all clients error.
        'common_weighted' == It uses the weighted mean (over data instances) of all clients errors.
        'own' == Each client has its own weight associated to the server calculated with its own error. 
        'avg_abs' == mean between 'own' and 'common_abs'
        'avg_weighted' == mean between 'own' and 'common_weighted'
        
        prediction_weights: It represents which prediction models are used. It can take the values:
        'only_server': The prediction for each client only uses server models with its weights to predict
        'server_and_clients': The prediction for each client uses the server models and its own models each with its corresponding weights 

        
        adapt_client_weight: It is a 1-d numpy array that takes values in (0,1). It represents the scalar that will be multiplied to 
        the clients weights if prediction_weights is 'server_and_clients'. If no value is passed it is set to (d_i/sum(d_i))_n by default,
        where di is the number of data of client i. 

        balanced_target_client_weight: Bool. If True, it weights the adaptative client weights to unbalanced classes. This way, clients
        with datasets with mostly one class label will not vote much into the FL prediction. More info in memoir. 

        T: Integer value. It represents the number of communications between clients and server.

        data_distribution: The way data is distributed between clients. It can take the values: 
        'iid': data is distributed i.i.d. between clients.
        'niid_dirichlet_label_skew': Following a Dirichlet distribution per target class. Draws a dirichlet distribution p_n for each clas n so that
        client j has p_jn proportion of class n.
        'niid_label_quantity_skew': Each client has only data_parameter of the total labels of the target.
        'niid_quantity_skew': Data is partitioned through clients following a Dir_n(data_param) distribution.  

        data_parameter: The parameter of the distribution followed. Default None. 
        'iid': no need of parameter
        'niid_dirichlet_label_skew': beta parameter of dirichlet distribution, Dir_n(beta). As beta tends to 0 it leads 
        to a more unbalanced partitioning. Must be greater than 0. 
        'niid_label_quantity_skew': The amounts of labels per client. Must be greater than 0.
        'niid_quantity_skew': Data distribution according to Dir(beta). Must be greater than 0.

        alpha_counter: It represents how many negative server weight in a row stands before the algorithm stops

        random_state: For reproducibility
        ...


        '''
        self.data = data
        self.targets = targets
        self.Nclients =Nclients 
        self.clients_classifier = clients_classifier
        self.server_classifier = server_classifier
        self.public_data = public_data
        self.public_data_prediction = public_data_prediction
        self.clients_weight_adj = clients_weight_adj
        self.server_alpha_weight_adj = server_alpha_weight_adj
        self.prediction_weights = prediction_weights
        self.T = T
        self.data_distribution = data_distribution
        self.distribution_param = distribution_param
        self.alpha_counter = alpha_counter
        self.random_state = random_state
        self.target_values,self.target_count = np.unique(targets,return_counts=True,axis=0)
        self.balanced_target_client_weight = balanced_target_client_weight
        self.domY = len(self.target_values) #Axis=0 works for OneHot and for 1d array 
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.initialize_data()
        if prediction_weights == 'server_and_clients':
            if adapt_client_weight is None:
                self.adapt_client_weight = self.number_data_clients/self.number_total_train_data
            else:
                self.adapt_client_weight = adapt_client_weight
            if self.balanced_target_client_weight:
                self.adapt_client_weight = self.adapt_client_weight*self.balanced_data_constant

    def initialize_data(self):
        self.data = np.insert(self.data,self.data.shape[1],np.ones(self.data.shape[0]),axis=1)
        clients_names = range(self.Nclients)
        train_clients_data = {} 
        test_clients_data = {}
        number_data_clients = np.zeros(self.Nclients)
        balanced_data_constant = np.zeros(self.Nclients)
        flex_train_data = Dataset.from_array(self.data,self.targets) #Transforms the numpy to flex dataset to do the partition
        if self.data_distribution == 'iid':
            federated_data = FedDataDistribution.iid_distribution(centralized_data = flex_train_data,
                                                                  n_nodes = self.Nclients) #It is always distributed the same way, no need for seed
        if self.data_distribution == 'niid_dirichlet_label_skew': 
            weights_per_label = np.random.dirichlet(np.repeat(self.distribution_param,self.Nclients), self.domY).transpose() #each target class is distributed following
                                                                                                        #a dirichlet distribution
            #Checking that each client has at least 10 instances (need the amount of instances of each class and multiply it to the client weights for each class and sum it) Just this does not work beacuse it sometimes gets divided in just 1 class
            #Instead, we impose that at least two classes has 10 or more instances. 
            condition = (weights_per_label*self.target_count).sum(axis=1)
            #condition = (weights_per_label*self.target_count) #At least 10 instances per class (so that train and test)
            #condition = ((weights_per_label*self.target_count) < 10).sum(axis=1) #A 1d array of lenght clients representing how many labels with more than 10 instances has each client
            while (condition < 10).any():
                weights_per_label = np.random.dirichlet(np.repeat(self.distribution_param,self.Nclients), self.domY).transpose()
                #condition = ((weights_per_label*self.target_count) < 10).sum(axis=1)
                condition = (weights_per_label*self.target_count).sum(axis=1)
            self.distribution_weights = weights_per_label
            config_niid = FedDatasetConfig(seed=self.random_state, n_nodes=self.Nclients, 
                               replacement=False, weights_per_label=weights_per_label)
            federated_data = FedDataDistribution.from_config(centralized_data = flex_train_data,config = config_niid)
        if self.data_distribution == 'niid_label_quantity_skew': 
            #each client gets labels_per_node<domY classes of the target values.
            config_niid = FedDatasetConfig(seed=self.random_state, n_nodes=self.Nclients, 
                               replacement=False, labels_per_node=self.distribution_param)
            federated_data = FedDataDistribution.from_config(centralized_data = flex_train_data,config = config_niid)
        if self.data_distribution == 'niid_quantity_skew': #Variable amount of data per client (following a dirichlet distrib)
            weights = np.random.dirichlet(np.repeat(self.distribution_param,self.Nclients))
            min_weight_value = 10/self.data.shape[0] #In order to have at least 10 instances per client. As number of instances grow, it will be easier to be satisfied
            while (weights < min_weight_value).any(): #algo arriesgado, si se pone un beta muy pequeño puede que no se cumpla mucho. Hay que ponerlo porque si no puede generar conjuntos vacios de datos
                weights = np.random.dirichlet(np.repeat(self.distribution_param,self.Nclients))
            self.distribution_weights = weights
            config_niid = FedDatasetConfig(seed=self.random_state, n_nodes=self.Nclients, 
                               replacement=False, weights=weights)
            federated_data = FedDataDistribution.from_config(centralized_data = flex_train_data,config = config_niid)
            


        
        for i in clients_names: #Separating data for clients and storing data distribution
            data,targets = federated_data[i].to_numpy()
            seed = self.random_state + i
            X_train,X_test,y_train,y_test = train_test_split(data,targets,test_size=0.1,train_size=0.9,random_state=seed) 
            train_clients_data[i] = (X_train,y_train)
            test_clients_data[i] = (X_test,y_test)
            number_data_clients[i] = len(y_train)
            _,counts = np.unique(y_train,return_counts=True)
            sorted_count = np.sort(counts)[-2:]
            balanced_data_constant[i] =  (sorted_count[-1] - (sorted_count[-1]-sorted_count[0]))/sorted_count[-1]
            #Temporal change
        self.train_clients_data = train_clients_data
        self.test_clients_data = test_clients_data
        self.number_data_clients = number_data_clients
        self.number_total_train_data = (number_data_clients.sum())
        self.balanced_data_constant = balanced_data_constant
        #OneHotEncoder for weighted predictions
        transform = OneHotEncoder(sparse_output=False)
        transform.fit(self.targets.reshape(-1,1)) #Estoy usando el test? Importa? Se supone que debe tener las mismas clases
        self.transform = transform

    def public_data_predict(self,arr):
        '''It takes a 2-d array with Nclients rows and as many columns as instances has public_data. It returns a 
        1-d array representing the labels asigned to public_data based on clients predictions stored in the input'''
        average_public_data_predict = np.zeros(arr.shape[1])
        
        if self.public_data_prediction == 'majority_voting':
            for i in range(self.Nclients):
                unique,counts = np.unique(arr[:,i],return_counts=True) #arr is an array with each row representing each client prediction
                average_public_data_predict[i] = unique[np.argmax(counts)] #If two prediction get voted the same number of times it takes the one with smaller index
        
        if self.public_data_prediction == 'weighted_majority_voting':
            weighted_sum = np.zeros((arr.shape[1],self.domY))
            for i in range(self.Nclients):
                OneHotprediction = self.transform.transform(arr[i].reshape(-1,1))
                weighted_sum = weighted_sum + (OneHotprediction*self.number_data_clients[i])/self.number_total_train_data
            predicted_indices = weighted_sum.argmax(axis=1)
            predicted_labels = np.zeros((arr.shape[1],self.domY))
            predicted_labels[np.arange(arr.shape[1]),predicted_indices] = 1
            average_public_data_predict = self.transform.inverse_transform(predicted_labels).flatten()
        return average_public_data_predict
            

    def client_weight_adjustment(self,i,j):
        '''It takes the connection time and the number of the client.
        It returns the weighted error of the server model at time i associated with the training data of the client j and the
        training data of the clients with weights actualized according to the error'''
        X_train, y_train = self.train_clients_data[j]
        serverModel = self.models_dict['server'][i]
        clientModel = self.models_dict[j][i]
        X_train_no_weights = X_train[:,:-1]
        server_predicted_data = serverModel.predict(X_train_no_weights)
        client_predicted_data = clientModel.predict(X_train_no_weights)
        server_filter = (y_train == server_predicted_data) 
        sample_weight = X_train[:,-1]
        client_err = 1 - accuracy_score(y_train,client_predicted_data,sample_weight = sample_weight)
        server_err = 1 - accuracy_score(y_train,server_predicted_data,sample_weight = sample_weight)
        if (server_err != 1): #When error is 1 we simply set alpha to 0
            if (server_err == 0):
                alpha = ((math.log(((self.domY -1)*(1-(server_err+0.005)))/(server_err+0.005)))/2) #Original AdaBoost stops when error is 0. Since we have multiple clients
                                                                                                   # we can't do that. To avoid dividing by 0 we soften the error summing 0.005, 
                                                                                                    #that way weight associated to the model is high. 
            else:
                alpha = ((math.log(((self.domY -1)*(1-server_err))/server_err))/2)
        else:
            alpha = 0
        if (server_err < 1-(1/self.domY)) and (server_err !=0): #If err is greater than that then alpha changes signs and messes the weight actualization
            X_train[server_filter,-1] = X_train[server_filter,-1]*math.exp(-alpha)  #When it predicts correctly, weight is decreased
            X_train[~server_filter,-1] = X_train[~server_filter,-1]*math.exp(alpha) #When it predicts wrongly, weight is increased
        return server_err,client_err,(X_train,y_train),alpha

    def server_alpha_weight_adjustment(self,err_arr):
        '''
        It takes a 1-d array of lenght Nclients storing each clients errors and returns
        the weight of the server model corresponding to those errors.
        '''
        #alpha = np.zeros(self.Nclients)
        if ((self.server_alpha_weight_adj == 'common_abs') or (self.server_alpha_weight_adj == 'avg_abs')):
            server_err = (err_arr.sum())/self.Nclients
            if (server_err != 1): #When error is 1 we simply set alpha to 0
                if (server_err == 0):
                    alpha = ((math.log(((self.domY -1)*(1-(server_err+0.005)))/(server_err+0.005)))/2) #Original AdaBoost stops when error is 0. Since we have multiple clients
                                                                                                   # we can't do that. To avoid dividing by 0 we soften the error summing 0.005, 
                                                                                                    #that way weight associated to the model is high. 
                else:
                    alpha = ((math.log(((self.domY -1)*(1-server_err))/server_err))/2)
            else:
                alpha = 0
            #alpha = ((math.log(((self.domY -1)*(1-server_err))/server_err))/2)
        elif ((self.server_alpha_weight_adj == 'common_weighted') or (self.server_alpha_weight_adj == 'avg_weighted') or (self.server_alpha_weight_adj == 'own')): 
            #Even though own doesn't need these weights they are calculated to have a stopping criteria.

            server_err = ((err_arr*self.number_data_clients).sum())/(self.number_total_train_data)
            if (server_err != 1): #When error is 1 we simply set alpha to 0
                if (server_err == 0):
                    alpha = ((math.log(((self.domY -1)*(1-(server_err+0.005)))/(server_err+0.005)))/2) #Original AdaBoost stops when error is 0. Since we have multiple clients
                                                                                                   # we can't do that. To avoid dividing by 0 we soften the error summing 0.005, 
                                                                                                    #that way weight associated to the model is high. 
                else:
                    alpha = ((math.log(((self.domY -1)*(1-server_err))/server_err))/2)
            else:
                alpha = 0
            #alpha = ((math.log(((self.domY -1)*(1-server_err))/server_err))/2)
        else:
            alpha = 0
        return alpha

    def fitmodel(self):
        '''A method that simulates the training (similar to fit)'''
        #self.initialize_data()
        public_data = self.public_data
        Nclients = self.Nclients
        predicted_public_data = np.zeros((Nclients,public_data.shape[0])) #a numpy array that stores public data predictions of each clients (per row)
        #self.clients_alphas = np.zeros((self.T,self.Nclients)) #A 2-d array that stores alphas of the clients at each timestep in position (timestep x client)
        self.models_dict = {}
        counter = 0 #A counter to check how many times in a row server's alpha weight is zero. When it gets to 3 iteration stops. 
        for i in range(Nclients): 
            self.models_dict[i] = {}
        self.models_dict['server'] = {}
        self.server_models_weights = np.zeros(self.T)
        self.own_server_model_weights = np.zeros((self.T,self.Nclients)) #stores in (i,j) each alpha of client j 
                                                                    #calculated with the error of the server model at timestep i
        self.clients_model_weights = np.zeros((self.T,self.Nclients))

        for i in range(self.T):
            for j in range(Nclients):
                seed = self.random_state + j
                model = DecisionTreeClassifier(random_state=seed,max_depth=self.max_depth,max_leaf_nodes=self.max_leaf_nodes)
                X_train,y_train = self.train_clients_data[j]
                X_train_no_weight = X_train[:,:-1] 
                weights = X_train[:,-1]
                if i != 0:
                    d = int(self.number_data_clients[j])
                    prob = weights/weights.sum()
                    weighted_choices = np.random.choice(np.arange(d),d,p=prob)
                    X_train_no_weight = X_train_no_weight[weighted_choices,:]            
                model.fit(X_train_no_weight,y_train)
                predicted_public_data[j,:] = model.predict(public_data)
                self.models_dict[j][i] = model
                
            voted_public_data_labels = self.public_data_predict(predicted_public_data)
            server_model = DecisionTreeClassifier(random_state=(seed+self.Nclients),max_depth=self.max_depth,max_leaf_nodes=self.max_leaf_nodes)
            server_model.fit(public_data,voted_public_data_labels)
            self.models_dict['server'][i] = server_model
            server_err_arr = np.zeros(Nclients)
            clients_err_arr = np.zeros(Nclients)
            for j in range(Nclients):
               server_err,client_err, self.train_clients_data[j],own_client_alpha = self.client_weight_adjustment(i,j)
               server_err_arr[j] = server_err
               clients_err_arr[j] = client_err
               self.own_server_model_weights[i,j] = own_client_alpha #clients own alpha with its data and client model 
            
            alpha = self.server_alpha_weight_adjustment(server_err_arr) #shared alpha weights for all clients (mean of errors of server model predicting client data)
            
            clients_err_arr[clients_err_arr == 0] = 0.005 #To avoid division by 0 and get a high value of alpha.
            clients_err_arr[clients_err_arr == 1] = 1-(1/self.domY) #To avoid log(0) and to not count that model (it predicts badly). Setting it to this makes the alpha <=0 hence, 0.
            self.clients_model_weights[i,:] = ((np.log(((self.domY -1)*(1-clients_err_arr))/clients_err_arr))/2) #abs alphas associated with each client

            if (alpha <= 0): #Checks if alphas are negative and sets them to zero. If more than 2 alphas are negative in a row, the iteration stops.
                if counter < (self.alpha_counter):
                    counter += 1
                else:
                    break
            else:
                counter = 0
            self.server_models_weights[i] = max(alpha,0)
                
        self.own_server_model_weights[self.own_server_model_weights < 0] = 0 #All the clients server weights that were negative are set to zero.
        self.clients_model_weights[self.clients_model_weights < 0] = 0 #All the clients models weights that were negative are set to zero.

    def fit_local_clients_models(self):
        self.local_clients_models_dict = {}
        for i in range(self.Nclients):
            X_train,y_train = self.train_clients_data[i]
            seed = self.random_state + i
            #The number of estimators is 2T so that the clients on local and with server have the same amount of models.
            #model = AdaBoostClassifier(DecisionTreeClassifier(random_state=seed),n_estimators=self.T*2,algorithm='SAMME') #Using SAMME.R bc I think SAMME has a bug
            model = AdaBoostClassifier2(n_estimators=self.T*2,random_state=seed,max_depth=self.max_depth)
            model.fit(X_train[:,:-1],y_train)
            self.local_clients_models_dict[i]= model

    def predict_data(self,data,server_weights,i=None):
        '''A method that given a 2-d array, a 1-d array and an int representing data, weights associated with models and client number respectively 
           returns a 1-d array with the prediction of the client'''
        server_models = self.models_dict['server']
        if self.prediction_weights == 'only_server':
            weighted_sum = np.zeros((data.shape[0],self.domY)) #array of row size the number of data and column size targets size
            for key,model in server_models.items():
                prediction = model.predict(data)
                OneHotprediction = self.transform.transform(prediction.reshape(-1,1))
                weighted_sum = weighted_sum + OneHotprediction*server_weights[key]

        elif self.prediction_weights == 'server_and_clients':
            weighted_sum = np.zeros((data.shape[0],self.domY)) #array of row size the number of data and column size targets size
            client_models = self.models_dict[i]
            for key,model in server_models.items():
                server_prediction = model.predict(data)
                client_prediction = client_models[key].predict(data)
                server_OneHotprediction = self.transform.transform(server_prediction.reshape(-1,1))
                client_OneHotprediction = self.transform.transform(client_prediction.reshape(-1,1))
                weighted_sum = weighted_sum + server_OneHotprediction*server_weights[key]
                weighted_sum = weighted_sum + client_OneHotprediction*(self.clients_model_weights[key,i]*self.adapt_client_weight[i])
                                                                        #Problema, datos totales pasados a los clientes. Se le mete ruido al numero de datos totales?
        predicted_indices = weighted_sum.argmax(axis=1)
        predicted_labels = np.zeros((data.shape[0],self.domY))
        predicted_labels[np.arange(data.shape[0]),predicted_indices] = 1
        return self.transform.inverse_transform(predicted_labels).flatten()
    
    def client_predict_data(self,data,i):
        '''A method that predicts the labels of data associated with client i'''
        if (self.server_alpha_weight_adj == 'common_abs') or (self.server_alpha_weight_adj == 'common_weighted'):
            if self.prediction_weights == 'only_server': #This distinction is done to avoid a for loop when the clients models are not used
                predicted_data = self.predict_data(data,self.server_models_weights)
            elif self.prediction_weights == 'server_and_clients':
                predicted_data = self.predict_data(data,self.server_models_weights,i)
        elif (self.server_alpha_weight_adj == 'own'):
                weights = self.own_server_model_weights[:,i]
                predicted_data = self.predict_data(data,weights,i)
        elif (self.server_alpha_weight_adj == 'avg_abs') or (self.server_alpha_weight_adj == 'avg_weighted'):
                weights = (self.own_server_model_weights[:,i] + self.server_models_weights)/2
                predicted_data = self.predict_data(data,weights,i)
        return predicted_data

    def global_predict_data(self,data):
        '''A method that given a 2-d array returns the prediction of the Ensemble FL model (a 2-d array where row i is the prediction of client i).'''
        all_predicted_labels = np.zeros((self.Nclients,data.shape[0]))
        if (self.server_alpha_weight_adj == 'common_abs') or (self.server_alpha_weight_adj == 'common_weighted'):
            if self.prediction_weights == 'only_server': #This distinction is done to avoid a for loop when the clients models are not used
                predicted_data = self.predict_data(data,self.server_models_weights)
                all_predicted_labels[:,:] = predicted_data
            elif self.prediction_weights == 'server_and_clients':
                for i in range(self.Nclients):
                    predicted_data = self.predict_data(data,self.server_models_weights,i)
                    all_predicted_labels[i,:] = predicted_data
        elif (self.server_alpha_weight_adj == 'own'):
            for i in range(self.Nclients):
                weights = self.own_server_model_weights[:,i]
                predicted_data = self.predict_data(data,weights,i)
                all_predicted_labels[i,:] = predicted_data
        elif (self.server_alpha_weight_adj == 'avg_abs') or (self.server_alpha_weight_adj == 'avg_weighted'):
            for i in range(self.Nclients):
                weights = (self.own_server_model_weights[:,i] + self.server_models_weights)/2
                predicted_data = self.predict_data(data,weights,i)
                all_predicted_labels[i,:] = predicted_data
        return all_predicted_labels
    
    def overall_acc_score(self,global_Xtest_data,global_ytest_data):
        '''A method that given data and its labels returns a Dataframe with a row for each client and columns: data of each client,
          accuracy of the federated model with each client's data, accuracy of the federated model with the data given as input, 
          accuracy of the local Adaboost trained model (trained only with its data) with the client's data and accuracy of the AdaBoost
          trained model with the data given as input'''
        FL_acc_own_data = np.zeros(self.Nclients)
        FL_acc_global_data = np.zeros(self.Nclients)
        local_acc_own_data = np.zeros(self.Nclients)
        local_acc_global_data = np.zeros(self.Nclients)
        global_difference = np.zeros(self.Nclients)
        local_difference = np.zeros(self.Nclients)
        for i in range(self.Nclients):
            own_data_Xtest,own_data_ytest = self.test_clients_data[i]
            local_model = self.local_clients_models_dict[i]
            FL_acc_own_data[i] = accuracy_score(self.client_predict_data(own_data_Xtest[:,:-1],i),own_data_ytest)*100
            FL_acc_global_data[i] = accuracy_score(self.client_predict_data(global_Xtest_data,i),global_ytest_data)*100
            local_acc_own_data[i] = accuracy_score(local_model.predict(own_data_Xtest[:,:-1]),own_data_ytest)*100
            local_acc_global_data[i] = accuracy_score(local_model.predict(global_Xtest_data),global_ytest_data)*100
            global_difference[i] = FL_acc_global_data[i] - local_acc_global_data[i]
            local_difference[i] = FL_acc_own_data[i] - local_acc_own_data[i]
        return pd.DataFrame({'data_distrib':self.number_data_clients, 'FL_acc_own_data':FL_acc_own_data,'FL_acc_global_data':FL_acc_global_data,
                             'local_acc_own_data':local_acc_own_data,'local_acc_global_data':local_acc_global_data,
                             'local_difference':local_difference,'global_difference':global_difference})
    
    def overall_F1_score(self,global_Xtest_data,global_ytest_data):
        FL_acc_own_data = np.zeros(self.Nclients)
        FL_acc_global_data = np.zeros(self.Nclients)
        local_acc_own_data = np.zeros(self.Nclients)
        local_acc_global_data = np.zeros(self.Nclients)
        global_difference = np.zeros(self.Nclients)
        local_difference = np.zeros(self.Nclients)
        for i in range(self.Nclients):
            own_data_Xtest,own_data_ytest = self.test_clients_data[i]
            local_model = self.local_clients_models_dict[i]
            FL_acc_own_data[i] = f1_score(self.client_predict_data(own_data_Xtest[:,:-1],i),own_data_ytest,
                                          labels=np.unique(own_data_ytest),average='weighted',zero_division=0.0)*100
            FL_acc_global_data[i] = f1_score(self.client_predict_data(global_Xtest_data,i),global_ytest_data,
                                             labels=np.unique(global_ytest_data),average='weighted',zero_division=0.0)*100
            local_acc_own_data[i] = f1_score(local_model.predict(own_data_Xtest[:,:-1]),own_data_ytest,
                                             labels=np.unique(own_data_ytest),average='weighted',zero_division=0.0)*100
            local_acc_global_data[i] = f1_score(local_model.predict(global_Xtest_data),global_ytest_data,
                                                labels=np.unique(global_ytest_data),average='weighted',zero_division=0.0)*100
            global_difference[i] = FL_acc_global_data[i] - local_acc_global_data[i]
            local_difference[i] = FL_acc_own_data[i] - local_acc_own_data[i]
        return pd.DataFrame({'data_distrib':self.number_data_clients, 'FL_acc_own_data':FL_acc_own_data,'FL_acc_global_data':FL_acc_global_data,
                             'local_acc_own_data':local_acc_own_data,'local_acc_global_data':local_acc_global_data,
                             'local_difference':local_difference,'global_difference':global_difference})
    
    def clients_score_dataframes(self,scheme,global_Xtest,targets):
        score_models= [None]*self.Nclients
        #score_local_models = []
        labels,counts = np.unique(targets,return_counts=True)
        counts = counts.reshape(-1,1)
        total_counts = np.ones((2,1))*len(targets)
        for i in range(self.Nclients):
            if scheme == 'FL':
                predictions = self.client_predict_data(global_Xtest,i)
            elif scheme == 'local':
                local_model = self.local_clients_models_dict[i]
                predictions = local_model.predict(global_Xtest)
            dataframe = pd.DataFrame(np.nan, index= (labels.tolist() + ['Accuracy','macro_avg','weighted_avg']), columns = ['precision','recall','f1_score','count']) #Creating the scores dataframe
            #lc_dataframe = pd.DataFrame(np.nan, index= (labels.tolist() + ['Accuracy','macro_avg','weighted_avg']), columns = ['precision','recall','f1_score','count'])
            #FL_prediction = self.client_predict_data(global_Xtest,i)
            #local_prediction = local_model.predict(global_Xtest)
            
            #Scores for each label in test set
            f1score = f1_score(predictions,targets,labels=labels,average=None).reshape(-1,1)
            recall = recall_score(predictions,targets,labels=labels,average=None).reshape(-1,1)
            precision = precision_score(predictions,targets,labels=labels,average=None).reshape(-1,1)
            labels_scores = np.concatenate([precision,recall,f1score,counts],axis=1)

            dataframe.loc[labels] = labels_scores #Storing the scores of FL 
            #lc_dataframe.loc[labels] = FL_labels_scores #Storing the scores of local model

            #Avg scores for all labels 
            macrof1score = f1_score(predictions,targets,labels=labels,average='macro')
            weightedf1score = f1_score(predictions,targets,labels=labels,average='weighted')
            macrorecall = recall_score(predictions,targets,labels=labels,average='macro')
            weightedrecall = recall_score(predictions,targets,labels=labels,average='weighted')
            macroprecision = precision_score(predictions,targets,labels=labels,average='macro')
            weightedprecision = precision_score(predictions,targets,labels=labels,average='weighted')

            avg_FL_f1score = np.array([macrof1score,weightedf1score]).reshape(-1,1)
            avg_FL_recall = np.array((macrorecall,weightedrecall)).reshape(-1,1)
            avg_FL_precision = np.array((macroprecision,weightedprecision)).reshape(-1,1)
            overall_scores = np.concatenate([avg_FL_precision,avg_FL_recall,avg_FL_f1score,total_counts],axis=1)

            dataframe.iloc[-2:] = overall_scores

            #Accuracy score
            dataframe.loc['Accuracy',['f1_score','count']] = [accuracy_score(predictions,targets),len(targets)]
            dataframe = dataframe.rename_axis(f'Client {i}',axis=1) 
            score_models[i] = dataframe
        return score_models

    
    def score_model(self):
        '''A method that returns a dictionary with keys the number of the clients and values
        a tuple (clients accuracy,server accuracy) on each clients test'''
        scores_dict = {}
        for i in range(self.Nclients):
            X_test,y_test = self.test_clients_data[i]
            X_test = X_test[:,:-1]
            client_prediction = self.local_clients_models_dict[i].predict(X_test) #Predicction with the first decision Tree
            server_prediction = self.client_predict_data(X_test,i)
            #server_prediction = self.models_dict['server'][0].predict(X_test)
            client_acc = accuracy_score(y_test,client_prediction)
            server_acc = accuracy_score(y_test,server_prediction)
            scores_dict[i] = (client_acc,server_acc)
        return scores_dict

    
    def sumar_overall_score(self,global_Xtest_data,global_ytest_data):
        FL_acc_own_dataf1 = np.zeros(self.Nclients)
        FL_acc_global_dataf1 = np.zeros(self.Nclients)
        FL_acc_own_dataacc = np.zeros(self.Nclients)
        FL_acc_global_dataacc = np.zeros(self.Nclients)
        for i in range(self.Nclients):
            own_data_Xtest,own_data_ytest = self.test_clients_data[i]
            FL_acc_own_dataf1[i] = f1_score(self.client_predict_data(own_data_Xtest[:,:-1],i),own_data_ytest,
                                          labels=np.unique(own_data_ytest),average='weighted',zero_division=0.0)*100
            FL_acc_global_dataf1[i] = f1_score(self.client_predict_data(global_Xtest_data,i),global_ytest_data,
                                             labels=np.unique(global_ytest_data),average='weighted',zero_division=0.0)*100
            FL_acc_global_dataacc[i] = accuracy_score(self.client_predict_data(global_Xtest_data,i),global_ytest_data)*100
            FL_acc_own_dataacc[i] = accuracy_score(self.client_predict_data(own_data_Xtest[:,:-1],i),own_data_ytest)*100
        acc_global = FL_acc_global_dataacc.mean()
        acc_local = FL_acc_own_dataacc.mean()
        f1_global = FL_acc_global_dataf1.mean()
        f1_local = FL_acc_own_dataf1.mean()
        return acc_global,f1_global,acc_local,f1_local