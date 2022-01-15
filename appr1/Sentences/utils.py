import csv
import pandas as pd
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from numpy import unravel_index
from sklearn.neighbors import NearestNeighbors

def questions_16_18(predicted_answers):
    """
    :arg    predicted answers 0-6
    :return predicted answers 0,1a,1b,2a,2b,3a,3b
    """
    for i in [15, 17]:
        if predicted_answers[i] == 1:
            predicted_answers[i] = '1a'
        elif predicted_answers[i] == 2:
            predicted_answers[i] = '1b'
        elif predicted_answers[i] == 3:
            predicted_answers[i] = '2a'
        elif predicted_answers[i] == 4:
            predicted_answers[i] = '2b'
        elif predicted_answers[i] == 5:
            predicted_answers[i] = '3a'
        elif predicted_answers[i] == 6:
            predicted_answers[i] = '3b'

    return (predicted_answers)
    
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
###############   PREDICTIONS (SIMPLE)   #################
##########################################################

def predict_answers(subject_embedding):
    """
    :arg embedding representing the subject
    :return predicted_answers
    """
    # load dictionary w/ answers (q_dict['q1'] = [[arr_emb_ans0], ... , [arr_emb_ans3], ... maybe more]
    pickle_in = open('../Data/q_dict.pkl', 'rb')
    q_dict = pickle.load(pickle_in)
#     print(q_dict)
    
    q_names = ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9', 'q10', 'q11',
               'q12', 'q13', 'q14', 'q15', 'q16', 'q17', 'q18', 'q19', 'q20', 'q21']
    
    predicted_answers = []
    # get the embeddings of the answers
    for i in range(0, 21):
        embeddings_of_answers = q_dict[q_names[i]]

        # compute cosine similarity between all pairs
        similarities = []
        
        for ans_embedding in embeddings_of_answers:
            cos_sim = cosine_similarity(np.array(subject_embedding).reshape(1, -1), np.array(ans_embedding).reshape(1, -1))
            similarities.append(cos_sim)
        predicted_answers.append(similarities.index(max(similarities)))

    predicted_answers = questions_16_18(predicted_answers)

    return (predicted_answers)

##########################################################
####################   EVALUATION   ######################
##########################################################

def evaluation(filename='predictions.txt'):
    ground_truth_filepath= '../Data/2021/Depression_Questionnaires_anon.txt'
    ground_truth_data = pd.read_csv(ground_truth_filepath, header=None, sep=' ')
    df_gt = answers_dataframe(ground_truth_data)
    
    # load predicted answers
#     predictions_filepath = 'predictions.txt'
    predictions_data = pd.read_csv(filename, header = None, delim_whitespace=True)
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
###############   PREDICTIONS (KMEANS)   #################
##########################################################
    
def kmeans_answers(cluster_centers):
    """
    :arg kmeans cluster centers
    :return predicted_answers
    """
    # load dictionary w/ answers (q_dict['q1'] = [[arr_emb_ans0], ... , [arr_emb_ans3], ... maybe more]
    pickle_in = open('../Data/q_dict.pkl', 'rb')
    q_dict = pickle.load(pickle_in)
    
    q_names = ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9', 'q10', 'q11',
               'q12', 'q13', 'q14', 'q15', 'q16', 'q17', 'q18', 'q19', 'q20', 'q21']
    
    predicted_answers = []
    # get the embeddings of the answers
    for i in range(0, 21):
        embeddings_of_answers = q_dict[q_names[i]]

        answers_per_cluster_center = [] 
        for cluster_center in cluster_centers:
            # compute cosine similarity between all pairs
            similarities = []
            
            for ans_embedding in embeddings_of_answers:
                cos_sim = cosine_similarity(np.array(cluster_center).reshape(1, -1), np.array(ans_embedding).reshape(1, -1))
                similarities.append(float(cos_sim))
            answers_per_cluster_center.append(similarities)

        answers_per_cc = np.array(answers_per_cluster_center)
        indices = unravel_index(answers_per_cc.argmax(), answers_per_cc.shape)
        index_col = indices[1]
        predicted_answers.append(index_col)
            
    predicted_answers = questions_16_18(predicted_answers)

    return (predicted_answers)

##########################################################
#################   PREDICTIONS (KNN)   ##################
##########################################################

def knn_answers(subject_embeddings, k):
    """
    :arg NearestNeighbors class
    :return predicted answers
    """
    neigh = NearestNeighbors(n_neighbors=k , metric='cosine') # default metric is 'minkowski'
    neigh.fit(subject_embeddings)
    
    # load dictionary w/ answers (q_dict['q1'] = [[arr_emb_ans0], ... , [arr_emb_ans3], ... maybe more]
    pickle_in = open('../Data/q_dict.pkl', 'rb')
    questions_dict = pickle.load(pickle_in)
    
    # load queries for each question (pessimism, sadness, etc)
    pickle_in = open('../Data/queries.pkl', 'rb')    # watch out! queries.pkl is different from q_dict.pkl
    queries_dict = pickle.load(pickle_in)
    
    q_names = ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9', 'q10', 'q11',
               'q12', 'q13', 'q14', 'q15', 'q16', 'q17', 'q18', 'q19', 'q20', 'q21']
    
    predicted_answers = []
    # get the embeddings of the answers
    for i in range(0, 21):
        query_embedding = queries_dict[q_names[i]]
        query_embedding = np.array(query_embedding).reshape(1, -1)
        neighbors = neigh.kneighbors(query_embedding, return_distance=False)
        # calculate the neighborhood center
        sum_emb = 0
        for n in neighbors[0]:
            sum_emb = sum_emb + np.array(subject_embeddings[n])
        center_emb = sum_emb / k
    
        # compute cosine similarity between all pairs
        similarities = []
        embeddings_of_answers = questions_dict[q_names[i]]
        for ans_embedding in embeddings_of_answers:
            cos_sim = cosine_similarity(center_emb.reshape(1, -1), np.array(ans_embedding).reshape(1, -1))
            similarities.append(cos_sim)
        predicted_answers.append(similarities.index(max(similarities)))

    predicted_answers = questions_16_18(predicted_answers)

    return (predicted_answers)

##########################################################
############   PREDICTIONS (WEIGHTED KNN)   ##############
##########################################################

def weighted_knn_answers(subject_embeddings, k):
    """
    :arg NearestNeighbors class
    :return predicted answers
    """
    neigh = NearestNeighbors(n_neighbors=k , metric='cosine') # default metric is 'minkowski'
    neigh.fit(subject_embeddings)
    
    # load dictionary w/ answers (q_dict['q1'] = [[arr_emb_ans0], ... , [arr_emb_ans3], ... maybe more]
    pickle_in = open('../Data/q_dict.pkl', 'rb')
    questions_dict = pickle.load(pickle_in)
    
    # load queries for each question (pessimism, sadness, etc)
    pickle_in = open('../Data/queries.pkl', 'rb')    # watch out! queries.pkl is different from q_dict.pkl
    queries_dict = pickle.load(pickle_in)
    
    q_names = ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9', 'q10', 'q11',
               'q12', 'q13', 'q14', 'q15', 'q16', 'q17', 'q18', 'q19', 'q20', 'q21']
    
    predicted_answers = []
    # get the embeddings of the answers
    for i in range(0, 21):
        query_embedding = queries_dict[q_names[i]]
        query_embedding = np.array(query_embedding).reshape(1, -1)
        distances, neighbors = neigh.kneighbors(query_embedding, return_distance=True) # distances are in [0, 2]
        
        weights = 2 - distances[0]
        normalized_weights = weights / sum(weights)
        
        # calculate the neighborhood center
        center_emb, j = 0, 0
        for n in neighbors[0]:
            center_emb = center_emb + np.array(subject_embeddings[n]) * np.array(normalized_weights[j])
            j = j + 1
    
        # compute cosine similarity between all pairs
        similarities = []
        embeddings_of_answers = questions_dict[q_names[i]]
        for ans_embedding in embeddings_of_answers:
            cos_sim = cosine_similarity(center_emb.reshape(1, -1), np.array(ans_embedding).reshape(1, -1))
            similarities.append(cos_sim)
        predicted_answers.append(similarities.index(max(similarities)))

    predicted_answers = questions_16_18(predicted_answers)

    return (predicted_answers)

##########################################################
#############   PREDICTIONS (GRID SEARCH)   ##############
#########################################################

def grid_search_answers(subject_embeddings, k, lamda):
    """
    :arg NearestNeighbors class
    :return predicted answers
    """
    neigh = NearestNeighbors(n_neighbors=k , metric='cosine') # default metric is 'minkowski'
    neigh.fit(subject_embeddings)
    
    # load dictionary w/ answers (q_dict['q1'] = [[arr_emb_ans0], ... , [arr_emb_ans3], ... maybe more]
    pickle_in = open('../Data/q_dict.pkl', 'rb')
    questions_dict = pickle.load(pickle_in)
    
    # load queries for each question (pessimism, sadness, etc)
    pickle_in = open('../Data/queries.pkl', 'rb')    # watch out! queries.pkl is different from q_dict.pkl
    queries_dict = pickle.load(pickle_in)
    
    q_names = ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9', 'q10', 'q11',
               'q12', 'q13', 'q14', 'q15', 'q16', 'q17', 'q18', 'q19', 'q20', 'q21']
    
    predicted_answers = []
    # get the embeddings of the answers
    for i in range(0, 21):
        query_embedding = queries_dict[q_names[i]]
        query_embedding = np.array(query_embedding).reshape(1, -1)
        distances, neighbors = neigh.kneighbors(query_embedding, return_distance=True) # distances are in [0, 2]
        
        weights = 2 - distances[0]
        lamda_weights = lamda + (1-lamda) * weights
        normalized_lamda_weights = lamda_weights/sum(lamda_weights)
        
        # calculate the neighborhood center
        center_emb, j = 0, 0
        for n in neighbors[0]:
            center_emb = center_emb + np.array(subject_embeddings[n]) * np.array(normalized_lamda_weights[j])
            j = j + 1
    
        # compute cosine similarity between all pairs
        similarities = []
        embeddings_of_answers = questions_dict[q_names[i]]
        for ans_embedding in embeddings_of_answers:
            cos_sim = cosine_similarity(center_emb.reshape(1, -1), np.array(ans_embedding).reshape(1, -1))
            similarities.append(cos_sim)
        predicted_answers.append(similarities.index(max(similarities)))

    predicted_answers = questions_16_18(predicted_answers)

    return (predicted_answers)