import nltk
import sklearn 
import pandas as pd
import os
import requests
import json
from utils import preprocess_text, getDecisionModelsJsonData, getVectorizersJsonData, download_file, add_performance_to_specific_model
import joblib
from sklearn.metrics import accuracy_score, precision_score, f1_score


# Downloading required modules 

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')



# Downloading the actives models and calculating performance metrics




with open("./mlModel/versions/active_couple_of_model.json", 'r') as file1:
    models = json.load(file1)

# download models["vectorizer_modelID"]
with open("./mlModel/versions/models/vectorizers/models.json", 'r') as file3:
    vectorizers = json.load(file3)

vec_modelData = vectorizers[models["vectorizer_modelID"]]
print("Downloading Vectorizer model")
download_file(vec_modelData["directUri"], "./mlModel/versions/models/vectorizers", vec_modelData["name"])

# download models["decisionTree_modelId"]
with open("./mlModel/versions/models/decisionTree/models.json", 'r') as file2:
    decisionTrees = json.load(file2)

decision_tree_modelData = decisionTrees[models["decisionTree_modelId"]]
print("Downloading Decision Tree model")
download_file(decision_tree_modelData["directUri"], "./mlModel/versions/models/decisionTree", decision_tree_modelData["name"])

# Model performance if isAlreadyTested == False

if not decision_tree_modelData["isAlreadyTested"]:

    model = joblib.load('./mlModel/versions/models/decisionTree/{}'.format(decision_tree_modelData["name"]))
    tfidf_vectorizer = joblib.load('./mlModel/versions/models/vectorizers/{}'.format(vec_modelData["name"]))
    perf_dataset = pd.read_csv("./mlModel/data/twitter_performance_1.csv", names=['Tweet_ID', 'Subject', 'Sentiment', 'Tweet_content'])
    modelTrainDataVersion = 1

    sentimentMap = {
        "Irrelevant": 0, "Negative": 1, "Neutral" : 2, "Positive": 3
    }

    perf_dataset.dropna(axis=0, inplace=True)

    X = perf_dataset["Tweet_content"]
    y_true = perf_dataset['Sentiment'].map(sentimentMap)


    y_pred = []
    # Preprocess the input text
    for i in range(len(X)):
        text = X.iloc[i]
        processed_text = preprocess_text(text)
        # Transform the preprocessed text using the TfidfVectorizer
        text_vector = tfidf_vectorizer.transform([processed_text])
        y_pred.append(model.predict(text_vector)[0])


    modelMetrics = {
            "trainDataVersion":modelTrainDataVersion,
            "modelMetrics":{
                "Accuracy": accuracy_score(y_true, y_pred),
                "Precision": precision_score(y_true, y_pred, average='macro'),
                "F1_score": f1_score(y_true, y_pred, average='macro')
            }

        }

    # Adding performances
    add_performance_to_specific_model(decision_tree_modelData["modelId"], modelMetrics, './mlModel/versions/models/decisionTree/models.json')

