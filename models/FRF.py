from flex.data import FedDataDistribution, FedDatasetConfig,Dataset
from flex.pool import FlexPool

from flextrees.pool import (
    deploy_server_config_rf,
    deploy_server_model_rf,
    aggregate_trees_from_rf,
    train_rf,
    collect_clients_trees_rf,
    set_aggregated_trees_rf,
)

from flex.pool.decorators import (
    evaluate_server_model,
    init_server_model,
)
import random
from flextrees.utils import GlobalRandomForest
import numpy as np
from flex.model import FlexModel

max_depth = 8
n_estimators = 10
    #let's define the modifications to flextrees functions

@init_server_model
def init_server_model_rf2(config=None, *args, **kwargs):
    """Function to initialize the server model

    Args:
        dataset_features (list): List that contains the name of the features
        config (dict, optional): Dict that contains the configuration of the
        server model. Defaults to None.
    """
    

    server_flex_model = FlexModel()

    if config is None:
        config = {
            'server_params': {
                'max_depth': max_depth,
                'n_estimators': n_estimators,
            },
            'clients_params': {
                'max_depth': max_depth,
                'n_estimators': n_estimators,
            }
        }

    server_flex_model['model'] = GlobalRandomForest(max_depth=config['server_params']['max_depth'], 
                                                    n_estimators=config['server_params']['n_estimators'])

    server_flex_model.update(config)

    return server_flex_model

@init_server_model
def init_server_model_rf_theirs(config=None, *args, **kwargs):
    """Function to initialize the server model

    Args:
        dataset_features (list): List that contains the name of the features
        config (dict, optional): Dict that contains the configuration of the
        server model. Defaults to None.
    """
    

    server_flex_model = FlexModel()

    if config is None:
        config = {
            'server_params': {
                'max_depth': 8,
                'n_estimators': 100,
            },
            'clients_params': {
                'max_depth': 8,
                'n_estimators': 100,
            }
        }

    server_flex_model['model'] = GlobalRandomForest(max_depth=config['server_params']['max_depth'], 
                                                    n_estimators=config['server_params']['n_estimators'])

    server_flex_model.update(config)

    return server_flex_model


@init_server_model
def init_server_model_rf_no_pruning(config=None, *args, **kwargs):
    """Function to initialize the server model

    Args:
        dataset_features (list): List that contains the name of the features
        config (dict, optional): Dict that contains the configuration of the
        server model. Defaults to None.
    """
    

    server_flex_model = FlexModel()

    if config is None:
        config = {
            'server_params': {
                'max_depth': None,
                'n_estimators': 100,
            },
            'clients_params': {
                'max_depth': None,
                'n_estimators': 100,
            }
        }

    server_flex_model['model'] = GlobalRandomForest(max_depth=config['server_params']['max_depth'], 
                                                    n_estimators=config['server_params']['n_estimators'])

    server_flex_model.update(config)

    return server_flex_model

@evaluate_server_model
def evaluate_global_rf_model2(server_flex_model, test_data, *args, **kwargs):
    """Evaluate global model on the server with a global test set.

    Args:
        server_flex_model (FlexModel): Server Flex Model.
        X (ArrayLike): Array with the data to evaluate.
        y (ArrayLike): Labels of the data to evaluate.
    """
    test_dict,X_global_test,y_global_test = test_data
    from sklearn.metrics import accuracy_score, f1_score
    preds_rf = server_flex_model['model'].predict(X_global_test)
    acc_global, f1_global= accuracy_score(y_global_test, preds_rf)*100, f1_score(y_global_test, preds_rf,labels=np.unique(y_global_test), average='weighted',zero_division=0.0)*100
    n_clients = len(test_dict)
    f1_scores = np.zeros(n_clients)
    acc_scores = np.zeros(n_clients)
    for i,(X_test_local,y_test_local) in enumerate(test_dict.values()):
        y_pred = server_flex_model['model'].predict(X_test_local)
        acc_score = accuracy_score(y_test_local,y_pred)
        f1_score1 = f1_score(y_test_local,y_pred,labels=np.unique(y_test_local),average='weighted',zero_division=0.0)
        acc_scores[i] = acc_score
        f1_scores[i] = f1_score1
    
    acc_local = acc_scores.mean()*100
    f1_local = f1_scores.mean()*100

    return acc_global,f1_global,acc_local,f1_local

def FRF_eval(train_dict,test_dict,X_global_test,y_global_test,hyperparameters="ours"):
    n_clients = len(train_dict)
    federated_data = {}
    for key,(data,targets) in train_dict.items():
        flex_data = Dataset.from_array(data,targets)
        federated_data[key] = flex_data
    # Set server config
    if hyperparameters == "ours":
        pool = FlexPool.client_server_pool(federated_data, init_server_model_rf2)
        total_estimators = 10
    elif hyperparameters == "theirs":
        pool = FlexPool.client_server_pool(federated_data, init_server_model_rf_theirs)
        total_estimators = 100
    elif hyperparameters == "no_pruning":
        pool = FlexPool.client_server_pool(federated_data, init_server_model_rf_no_pruning)
        total_estimators = 100
    clients = pool.clients
    aggregator = pool.aggregators
    server = pool.servers
    # Total number of estimators
    
    # Number of estimators per client
    nr_estimators = total_estimators // n_clients

    # Deploy clients config
    server.map(func=deploy_server_config_rf, dst_pool=pool.clients)
    clients.map(func=train_rf)
    #clients.map(func=evaluate_local_rf_model_at_clients)
    aggregator.map(func=collect_clients_trees_rf, dst_pool=pool.clients, nr_estimators=nr_estimators)
    aggregator.map(func=aggregate_trees_from_rf)
    aggregator.map(func=set_aggregated_trees_rf, dst_pool=pool.servers)
    server.map(func=deploy_server_model_rf, dst_pool=pool.clients)
    results = server.map(func=evaluate_global_rf_model2, test_data=(test_dict,X_global_test,y_global_test))
    acc_global,f1_global,acc_local,f1_local = results[0]
    return acc_global,f1_global,acc_local,f1_local
