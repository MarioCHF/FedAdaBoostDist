import math
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

class AdaBoostClassifier2():
    def __init__(self,n_estimators,random_state,max_depth = None):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.max_depth = max_depth
        

    def fit(self,X_data,y_data):
        X_data = np.insert(X_data,X_data.shape[1],np.ones(X_data.shape[0]),axis=1)
        self.model_weights = np.zeros(self.n_estimators)
        self.models_dict = {}
        self.domY = len(np.unique(y_data))
        transform = OneHotEncoder(sparse_output=False)
        transform.fit(y_data.reshape(-1,1))
        self.transform = transform
        for i in range(self.n_estimators):
            model = DecisionTreeClassifier(random_state=self.random_state,max_depth=self.max_depth)
            X_train_no_weight = X_data[:,:-1]
            weights = X_data[:,-1]
            if i!=0:
                d = int(X_data.shape[0])
                prob = weights/weights.sum()
                weighted_choices = np.random.choice(np.arange(d),d,p=prob)
                X_train_no_weight = X_train_no_weight[weighted_choices,:]
            model.fit(X_train_no_weight,y_data)
            self.models_dict[i] = model
            #let's actualize the weights and calculate the model's weight.
            predicted_data = model.predict(X_train_no_weight)  
            filter = (y_data == predicted_data)
            err = 1 - accuracy_score(y_data,predicted_data,sample_weight = weights)
            if (err < 1-(1/self.domY)): 
                if (err == 0):
                    alpha = ((math.log(((self.domY -1)*(1-(err+0.005)))/(err+0.005)))/2)
                    self.model_weights[i] = alpha
                    break
                else:
                    alpha = ((math.log(((self.domY -1)*(1-err))/err))/2)
            else:
                break #if the error is too high stop iteration
            self.model_weights[i] = alpha
            X_data[filter,-1] = X_data[filter,-1]*math.exp(-alpha)  #When it predicts correctly, weight is decreased
            X_data[~filter,-1] = X_data[~filter,-1]*math.exp(alpha) #When it predicts wrongly, weight is increased

    
    def predict(self,data):
        weighted_sum = np.zeros((data.shape[0],self.domY))
        for key, model in self.models_dict.items():
            prediction = model.predict(data)
            OneHotprediction = self.transform.transform(prediction.reshape(-1,1))
            weighted_sum = weighted_sum + OneHotprediction*self.model_weights[key]
        predicted_indices = weighted_sum.argmax(axis=1)
        predicted_labels = np.zeros((data.shape[0],self.domY))
        predicted_labels[np.arange(data.shape[0]),predicted_indices] = 1
        return self.transform.inverse_transform(predicted_labels).flatten()
    
