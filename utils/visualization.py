import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def difference_errorbar_plot(scoreFrames,labels,Nclients):

    fig= plt.figure(figsize=(12,6))
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)
    plt.setp((ax1,ax2),xticks=np.arange(0,10*Nclients,10),xticklabels = np.arange(0,Nclients))
    ax1.set_title('Avg local_difference of each client')
    ax2.set_title('Avg global_difference of each client')
    ax1.set_xlabel('clients')
    ax1.set_ylabel('(FL_local-own_local)')
    ax2.set_xlabel('clients')
    ax2.set_ylabel('(FL_global-own_global)')
    markers = ['.','+','v','s','d']
    x = np.linspace(-4.2,4.2,len(scoreFrames)) + 0.5
    x_data= np.zeros(Nclients)
    avg_lc_data = np.zeros(Nclients)
    std_lc_data = np.zeros(Nclients)
    avg_gc_data = np.zeros(Nclients)
    std_gc_data = np.zeros(Nclients)

    for i,data in enumerate(scoreFrames):
        dataframe = pd.concat(data,axis=0)
        for j in range(Nclients):
            client_data = dataframe[dataframe.index == j].describe()
            x_data[j] = x[i] + 10*j
            avg_lc_data[j] = client_data.loc['mean']['local_difference']
            std_lc_data[j] = client_data.loc['std']['local_difference']  
            avg_gc_data[j] = client_data.loc['mean']['global_difference']
            std_gc_data[j] = client_data.loc['std']['global_difference']
            ax1.axvline(10*j-5,color = 'lightgrey',linestyle='dotted')
            ax2.axvline(10*j-5,color = 'lightgrey',linestyle='dotted')
        ax1.errorbar(x_data,avg_lc_data,yerr=std_lc_data,fmt=markers[i],label = labels[i],linewidth = 1, capsize = 3, ecolor='darkgray')
        ax2.errorbar(x_data,avg_gc_data,yerr=std_gc_data,fmt=markers[i],label = labels[i],linewidth = 1, capsize = 3,ecolor='darkgray')
   

    ax1.legend()
    ax2.legend()


def splits_scores_bar_plot(describeFrames,AvgFrames,centralized_scores,labels):
    fig,axs = plt.subplots(3,2,figsize=(12,21))
    axs = axs.flatten()

    #xtick_shift = (len(labels)-1)/2

    x = np.arange(len(labels))  # the label locations
    width = 0.10  # the width of the bars
    multiplier = 0
    names = ['FL_own_score','FL_gl_score','lc_own_score','lc_gl_score']
    for i in range(len(describeFrames[0])):
        multiplier = 0
        dataframe = pd.Series.to_frame(describeFrames[0][i].loc['mean'][['FL_acc_own_data','FL_acc_global_data','local_acc_own_data','local_acc_global_data']])
        stddataframe = pd.Series.to_frame(describeFrames[0][i].loc['std'][['FL_acc_own_data','FL_acc_global_data','local_acc_own_data','local_acc_global_data']])
        for listOfFrames in describeFrames[1:]:
            Series = listOfFrames[i].loc['mean'][['FL_acc_own_data','FL_acc_global_data','local_acc_own_data','local_acc_global_data']]
            stdSeries = listOfFrames[i].loc['std'][['FL_acc_own_data','FL_acc_global_data','local_acc_own_data','local_acc_global_data']]
            dataframe = pd.concat((dataframe,Series),axis=1)
            stddataframe = pd.concat((stddataframe,stdSeries),axis=1)
        dataframe = dataframe.T.reset_index(drop=True)
        stddataframe = stddataframe.T.reset_index(drop=True)
        #dataframe

        for j,column in enumerate(dataframe.columns):
            offset = width * multiplier
            z = x + offset
            rects = axs[i].bar(z, dataframe[column].to_numpy(), width, label=names[j])
            rects = axs[i].errorbar(z, dataframe[column].to_numpy(),fmt='.',yerr=stddataframe[column].to_numpy(),linewidth = 1, capsize = 1,ecolor='k')
            #ax.bar_label(rects, padding=3)
            multiplier += 1
            offset = width*multiplier
        rects = axs[i].bar(x+offset,centralized_scores[i],width,label='centr_score')
        # Add some text for labels, title and custom x-axis tick labels, etc.
        axs[i].set_ylabel('Score')
        axs[i].set_title(f'split {i}')
    
        axs[i].set_xticks(x + 2*width, labels)
        axs[i].legend(loc='upper left')
        axs[i].set_ylim(0, 150)

    multiplier = 0
    dataframe = pd.Series.to_frame(AvgFrames[0].loc['mean'][['FL_acc_own_data','FL_acc_global_data','local_acc_own_data','local_acc_global_data']])
    stddataframe = pd.Series.to_frame(AvgFrames[0].loc['std'][['FL_acc_own_data','FL_acc_global_data','local_acc_own_data','local_acc_global_data']])
    for listOfFrames in AvgFrames[1:]:
        Series = listOfFrames.loc['mean'][['FL_acc_own_data','FL_acc_global_data','local_acc_own_data','local_acc_global_data']]
        stdSeries = listOfFrames.loc['std'][['FL_acc_own_data','FL_acc_global_data','local_acc_own_data','local_acc_global_data']]
        dataframe = pd.concat((dataframe,Series),axis=1)
    dataframe = dataframe.T.reset_index(drop=True)
    stddataframe = stddataframe.T.reset_index(drop=True)
    #dataframe

    for j,column in enumerate(stddataframe.columns):
        offset = width * multiplier
        z = x + offset
        rects = axs[-1].bar(z, dataframe[column].to_numpy(), width, label=names[j])
        rects = axs[-1].errorbar(z, dataframe[column].to_numpy(),fmt='.',yerr=stddataframe[column].to_numpy(),linewidth = 1, capsize = 1,ecolor='k')
        #ax.bar_label(rects, padding=3)
        multiplier += 1
        offset = width*multiplier
    rects = axs[-1].bar(x+offset,np.array(centralized_scores).mean(),width,label='avg_centr_score')
    rects = axs[-1].errorbar(x+offset, np.repeat(np.array(centralized_scores).mean(),len(x+offset)),fmt='.',
                         yerr=np.repeat(np.array(centralized_scores).std(),len(x+offset)),linewidth = 1, capsize = 1,ecolor='k')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    axs[-1].set_ylabel('Score')
    axs[-1].set_title('Avg scores in splits')
    axs[-1].set_xticks(x + 2*width, labels)
    axs[-1].legend(loc='upper left')
    axs[-1].set_ylim(0, 150)



def parameter_global_score_overview(splits,params):
    '''Takes a dictionary with keys 0,...,numberofsplits and values lists of dataframes where the pair
    i:[list of length len(param)] corresponds to a list of describeDataframes that where position j is the result of an FLmodel
     trained with split i dataset and parameter params[j]'''
    
    fig,axs = plt.subplots(3,2,figsize=(12,21))
    axs = axs.flatten()
    labels = ['FL_acc_global_data','local_acc_global_data']

    for i in splits:
        split0 = splits[i]
        merged_data = pd.concat(split0,axis=0)
        avg_stats = merged_data[merged_data.index == 'mean'][['FL_acc_global_data','local_acc_global_data']]
        #std_stats = merged_data[merged_data.index == 'std'][['FL_acc_global_data','local_acc_global_data']]
        for label in labels:
            acc = avg_stats[label].to_numpy()
            #std =std_stats[label].to_numpy()
            #y1 = acc - std
            #y2 = acc + std
            #axs[i].fill_between(data_params,y1,y2,alpha=.5,linewidth=0)
            axs[i].plot(params,acc,linewidth=2,label=label)

        axs[i].set_ylabel('Score')
        axs[i].set_title(f'split {i}')
        axs[i].legend(loc='upper left')
        axs[i].set_ylim(0, 150)
    plt.show()