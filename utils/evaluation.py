
import numpy as np
import pandas as pd
from models.FL_AdaBoost_Dist import FLEnsembleDist
#from sklearn.ensemble import AdaBoostClassifier
#from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,f1_score
from models.AdaBoostClassifier2 import AdaBoostClassifier2
import math

def FLcross_val_score(data_splits:list,public_data,metric = 'accuracy',Nclients=5,public_data_prediction='majority_voting',
                      clients_weight_adj = 'server_error',server_alpha_weight_adj='common_abs',prediction_weights = 'only_server',
                 T=10,data_distribution ='iid',distribution_param=None, alpha_counter = 3,random_state = 0,adapt_client_weight=None,
                 balanced_target_client_weight=False,return_overall_scores=False):
    '''A function that evaluates the federated model given by the input parameters across all the datasplits and returns a 5-tuple
    consisting of the list of FLmodels, the list of centralized AdaBoost models (with all data), list of DataFrame of scores of each FLmodel,
    list of dataframes of stats of each FLmodel and a list of the scores of the centralized models respectively'''
    number_splits = len(data_splits)
    FLmodels  =[None]*number_splits
    accDataFrames = [None]*number_splits
    describeDataFrames = [None]*number_splits
    centralized_model_scores = [None]*number_splits
    centralized_models = [None]*number_splits
    overall_FL_scores = [None]*number_splits
    overall_local_scores = [None]*number_splits

    for i,data in enumerate(data_splits):
        np.random.seed(random_state)
        X_train,X_test,y_train,y_test = data
        FLmodel = FLEnsembleDist(data=X_train,targets=y_train,public_data = public_data, Nclients=Nclients,
                             public_data_prediction=public_data_prediction,clients_weight_adj = clients_weight_adj,
                             server_alpha_weight_adj=server_alpha_weight_adj,
                             prediction_weights=prediction_weights,T=T,data_distribution=data_distribution, 
                             distribution_param=distribution_param, alpha_counter = alpha_counter,random_state=random_state,
                             adapt_client_weight=adapt_client_weight,balanced_target_client_weight=balanced_target_client_weight)
        FLmodel.fitmodel()
        FLmodel.fit_local_clients_models()
        centralized_model = AdaBoostClassifier2(n_estimators=T*2,random_state=random_state+3*Nclients)
        centralized_model.fit(X_train,y_train)
        if metric == 'accuracy':
            DataFrame = FLmodel.overall_acc_score(X_test,y_test)
            centralized_model_scores[i] = accuracy_score(centralized_model.predict(X_test),y_test)*100
        elif metric == 'f1_score':
            DataFrame = FLmodel.overall_F1_score(X_test,y_test)
            centralized_model_scores[i] = f1_score(centralized_model.predict(X_test),y_test,labels=np.unique(y_train),average='macro')*100
        centralized_models[i] = centralized_model
        FLmodels[i] = FLmodel
        accDataFrames[i] = DataFrame
        describeDataFrames[i] = DataFrame.describe()

        if return_overall_scores:
            clients_FL_scores = FLmodel.clients_score_dataframes(scheme='FL',global_Xtest=X_test,targets=y_test)
            overall_FL_scores[i] = clients_FL_scores
    
            clients_local_scores = FLmodel.clients_score_dataframes(scheme='local',global_Xtest=X_test,targets=y_test)
            overall_local_scores[i] = clients_local_scores

    avgResult = describeDataFrames[0].loc['mean']
    avgResult.name = 0
    avgResult['centralized_scores'] = centralized_model_scores[0]
    avgResult = pd.Series.to_frame(avgResult)

    for i in range(len(data_splits)-1):
        a = describeDataFrames[i+1].loc['mean']
        a['centralized_scores'] = centralized_model_scores[i+1]
        avgResult[i+1] = a
    avgResult = avgResult.T.describe()
    if return_overall_scores:

        return FLmodels,centralized_models,accDataFrames,describeDataFrames,centralized_model_scores,avgResult,overall_FL_scores,overall_local_scores
    else:
        return FLmodels,centralized_models,accDataFrames,describeDataFrames,centralized_model_scores,avgResult



def FL_simple_cross_val_score(data_splits:list,public_data,Nclients=5,public_data_prediction='majority_voting',
                      clients_weight_adj = 'server_error',server_alpha_weight_adj='common_abs',prediction_weights = 'only_server',
                 T=10,data_distribution ='iid',distribution_param=None, alpha_counter = 3,random_state = 0,adapt_client_weight=None,
                 balanced_target_client_weight=False):
    
    number_splits = len(data_splits)
    FLmodels  =[None]*number_splits
    accDataFrames = [None]*number_splits
    F1DataFrames = [None]*number_splits
    acc_centralized_model_scores = [None]*number_splits
    f1_centralized_model_scores = [None]*number_splits
    centralized_models = [None]*number_splits

    for i,data in enumerate(data_splits):
        np.random.seed(random_state)
        X_train,X_test,y_train,y_test = data
        FLmodel = FLEnsembleDist(data=X_train,targets=y_train,public_data = public_data, Nclients=Nclients,
                             public_data_prediction=public_data_prediction,clients_weight_adj = clients_weight_adj,
                             server_alpha_weight_adj=server_alpha_weight_adj,
                             prediction_weights=prediction_weights,T=T,data_distribution=data_distribution, 
                             distribution_param=distribution_param, alpha_counter = alpha_counter,random_state=random_state,
                             adapt_client_weight=adapt_client_weight,balanced_target_client_weight=balanced_target_client_weight)
        FLmodel.fitmodel()
        FLmodel.fit_local_clients_models()
        centralized_model = AdaBoostClassifier2(n_estimators=T*2,random_state=random_state+3*Nclients)
        centralized_model.fit(X_train,y_train)
        DataFrameacc = FLmodel.overall_acc_score(X_test,y_test)
        acc_centralized_model_scores[i] = accuracy_score(centralized_model.predict(X_test),y_test)*100
        DataFramef1 = FLmodel.overall_F1_score(X_test,y_test)
        f1_centralized_model_scores[i] = f1_score(centralized_model.predict(X_test),y_test,labels=np.unique(y_train),average='weighted',zero_division=0.0)*100
        centralized_models[i] = centralized_model
        FLmodels[i] = FLmodel
        accDataFrames[i] = DataFrameacc
        F1DataFrames[i] = DataFramef1

    return FLmodels,centralized_models,accDataFrames,F1DataFrames,acc_centralized_model_scores,f1_centralized_model_scores
        
def wilconxonbest(dataset: pd.DataFrame, alpha: float = 0.05, verbose: bool = False):
    """
    Perform the Wilcoxon signed-rank test. This non-parametric test is used to compare two related samples, matched
    samples, or repeated measurements on a single sample to assess whether their population mean ranks differ. It is
    an alternative to the paired Student's t-test when the data is not normally distributed.

    Parameters
    ----------
    dataset : pandas.DataFrame
        A DataFrame with exactly two columns, each representing a different condition or time point for the same
        subjects.
    alpha : float, optional
        The significance level for the test. Default is 0.05.
    verbose : bool, optional
        If True, prints the detailed results table.

    Returns
    -------
    w_wilcoxon : float
        The Wilcoxon test statistic, which is the smallest of the sums of the positive and negative ranks.
    p_value : float or None
        The p-value for the hypothesis test (only for large sample sizes, otherwise None).
    cv_alpha_selected : float or None
        The critical value for the test at the specified alpha level (only for small sample sizes, otherwise None).
    hypothesis : str
        A string stating the conclusion of the test based on the test statistic, critical value, or p-value and alpha.

    Note
    ----
    The Wilcoxon signed-rank test makes fewer assumptions than the t-test and is appropriate when the data
    are not normally distributed. It ranks the absolute differences between pairs, then compares these ranks.
    The test is sensitive to ties and has different procedures for small and large sample sizes. For large samples,
    the test statistic is approximately normally distributed, allowing the use of normal approximation for p-value
    calculation.
    """
    if dataset.shape[1] != 2:
        raise "Error: The test only needs two samples"

    results_table = dataset.copy()
    columns = list(dataset.columns)
    differences_results = dataset[columns[0]] - dataset[columns[1]]
    absolute_dif = differences_results.abs()
    absolute_dif = absolute_dif.sort_values()
    results_wilconxon = {"index": [], "dif": [], "rank": [], "R": []}
    rank = 0.0
    tied_ranges = not (len(set(absolute_dif)) == absolute_dif.shape[0])
    for index in absolute_dif.index:
        if math.fabs(0 - absolute_dif[index]) < 1e-10:
            continue
        rank += 1.0
        results_wilconxon["index"] += [index]
        results_wilconxon["dif"] += [differences_results[index]]
        results_wilconxon["rank"] += [rank]
        results_wilconxon["R"] += ["+" if differences_results[index] > 0 else "-"]

    df = pd.DataFrame(results_wilconxon)
    df = df.set_index("index")
    df = df.sort_index()
    results_table = pd.concat([results_table, df], axis=1)

    tie_sum = 0

    if tied_ranges:
        vector = [abs(i) for i in results_table["dif"]]

        counts = {}
        for number in vector:
            try:
                counts[number] = counts[number] + 1
            except KeyError:
                counts[number] = 1

        ranks = results_table["rank"].to_numpy()
        for index, number in enumerate(vector):
            if counts[number] > 1:
                rank_sum = sum(ranks[i] for i, x in enumerate(vector) if x == number)
                average_rank = rank_sum / counts[number]
                for i, x in enumerate(vector):
                    if x == number:
                        ranks[i] = average_rank
        tie_sizes = np.array(list(counts.values()))
        tie_sum = (tie_sizes ** 3 - tie_sizes).sum()

    if verbose:
        print(results_table)

    r_plus = results_table[results_table.R == "+"]["rank"].sum()
    r_minus = results_table[results_table.R == "-"]["rank"].sum()

    #w_wilcoxon = min([r_plus, r_minus])
    #num_problems = results_table.shape[0] - (results_table.R.isna().sum())
    #mean_wilcoxon = (num_problems * (num_problems + 1)) / 4.0

    #std_wilcoxon = num_problems * (num_problems + 1) * ((2 * num_problems) + 1)
    #std_wilcoxon = math.sqrt(std_wilcoxon / 24.0 - (tie_sum / 48))
    #z_wilcoxon = (w_wilcoxon - mean_wilcoxon) / std_wilcoxon

    #cv_alpha_selected = stats.get_cv_willcoxon(num_problems, alpha)

    #p_value = 2 * stats.get_p_value_normal(z_wilcoxon)

    # if num_problems > 25:


    return r_plus,r_minus
