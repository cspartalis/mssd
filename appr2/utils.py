import csv
import pandas as pd
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from numpy import unravel_index
from sklearn.neighbors import NearestNeighbors


##########################################################
##################   GOLDEN TRUTH DF   ###################
##########################################################

def answers_dataframe(data):
    """
    :arg data of the answers (predicted or ground truth) per user
    :return dataframe in the appropriate form for evaluation metrics
    """
    df = pd.DataFrame(data)
    df = df.iloc[:, 0:22]
    df = df.dropna()
    df.rename(columns={0: 'subject'}, inplace=True)

    # Extract the numeric information from answers 16, 18.
    df['original16'] = df[16].astype(str)
    df['original18'] = df[18].astype(str)
    df[16] = df[16].apply(lambda x: re.sub("[^0-3]", "", str(x))).astype(int)
    df[18] = df[18].apply(lambda x: re.sub("[^0-3]", "", str(x))).astype(int)

    df['overall_score'] = df.sum(axis=1)

    # minimal depression (depression levels 0-9)
    # mild depression (depression levels 10-18)
    # moderate depression (depression levels 19-29)
    # severe depression (depression levels 30-63)
    def dep_lvl(overall_score):
        if overall_score <= 9:
            return 0
        elif overall_score <= 18:
            return 1
        elif overall_score <= 29:
            return 2
        else:
            return 3

    df['depression_level'] = df['overall_score'].apply(lambda x: dep_lvl(x))

    df.set_index('subject', inplace=True)

    return df


##########################################################
####################   EVALUATION   ######################
##########################################################

def evaluation(file='predictions.txt'):
    ground_truth_filepath= '../Data/2021/Depression_Questionnaires_anon.txt'
    ground_truth_data = pd.read_csv(ground_truth_filepath, header=None, sep=' ')
    df_gt = answers_dataframe(ground_truth_data)
    
    # load predicted answers
    predictions_filepath = file
    predictions_data = pd.read_csv(predictions_filepath, header = None, delim_whitespace=True)
    df_pred = answers_dataframe(predictions_data)

    # average hit rate
    subjects = df_gt.index.tolist()

    count = 0
    for sub in subjects:
        for i in [x for x in range(1, 22) if (x != 16 and x != 18)]:
            if df_gt.loc[sub, i] == df_pred.loc[sub, i]:
                count = count + 1

        for col in ['original16', 'original18']:
            if df_gt.loc[sub, col] == df_pred.loc[sub, col]:
                count = count + 1

    ahr = count / (21 * len(subjects)) * 100
    ahr = round(ahr, 2)
    print("Average Hit Rate: " + str(ahr) + "%")

    # average closeness rate
    sum_cr = 0
    for sub in subjects:
        for i in range(1, 22):
            ad = abs(df_gt.loc[sub,i] - df_pred.loc[sub,i])
            cr = (3 - ad) / 3
            sum_cr = sum_cr + cr

    acr = sum_cr / (len(subjects)*21/100)
    acr = round(acr, 2)
    print("Average Closeness Rate: " + str(acr) + "%")

    # average diference between overall depression levels
    sum_dodl = 0
    for sub in subjects:
        ad = abs(df_gt.loc[sub, 'overall_score'] - df_pred.loc[sub, 'overall_score'])
        dodl = (63 - ad) / 63
        sum_dodl = sum_dodl + dodl

    adodl = sum_dodl / len(subjects) * 100
    adodl = round(adodl, 2)
    print("Average Difference between Overall Depression Levels: " + str(adodl) + "%")

    # depression hit rate
    count = 0
    for sub in subjects:
        if df_gt.loc[sub, 'depression_level'] == df_pred.loc[sub, 'depression_level']:
                count = count + 1

    dchr = count / len(subjects) * 100
    dchr = round(dchr, 2)
    print("Depression Category Hit Rate: " + str(dchr) + "%")
    
##########################################################
##################   KNN NEIGHBORS   #####################
##########################################################

def knn_center(subject_embeddings, query, k):
    """
    :arg subject's posts embeddings
    :arg query (e.g. keyword of the question)
    :arg # of neighbors
    :return center of KNNs
    """
    
    neigh = NearestNeighbors(n_neighbors=k , metric='cosine') # default metric is 'minkowski'
    neigh.fit(subject_embeddings)
      
    # load queries for each question (pessimism, sadness, etc)
    pickle_in = open('../Data/queries.pkl', 'rb')
    queries_dict = pickle.load(pickle_in)
    
    q_names = ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9', 'q10', 'q11',
               'q12', 'q13', 'q14', 'q15', 'q16', 'q17', 'q18', 'q19', 'q20', 'q21']
    
    # get the embeddings of the answers
    for i in range(0, 21):
        query_embedding = queries_dict[q_names[i]]
        query_embedding = np.array(query_embedding).reshape(1, -1)
        neighbors = neigh.kneighbors(query_embedding, return_distance=False)
        
#         ...to be continued

