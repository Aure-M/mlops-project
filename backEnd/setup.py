import nltk
import sklearn 
import pandas as pd
import os
import requests
import json
# Downloading required modules 

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')



# Downloading the actives models and calculating performance metrics

def download_file(url, destination_folder, modelName):
    # Create the destination folder if it doesn't exist
    try:
        os.makedirs(destination_folder, exist_ok=True)

        # Extract filename from the URL
        filename = modelName

        # Download the file
        response = requests.get(url)
        if response.status_code == 200:
            # Write the downloaded file to the destination folder
            with open(os.path.join(destination_folder, filename), 'wb') as f:
                f.write(response.content)
            print(f"Downloaded {filename} to {destination_folder}")
        else:
            print(f"Failed to download {url}")
    except:
        print('An exception occurred while downloading a model')



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



# Model performance