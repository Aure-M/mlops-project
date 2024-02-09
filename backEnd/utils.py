import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import json
import os
import requests

english_stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()


def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove some characters (e.g., @)
    text = re.sub(r'(@)', ' ', text)
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove stopwords
    text = ' '.join([
                word
                for word in text.split()
                if word not in english_stop_words
            ])
    # Perform stemming
    text = ' '.join([stemmer.stem(word) for word in text.split()])
    return text



def getModelsJsonData():

    with open("./mlModel/versions/active_couple_of_model.json", 'r') as file1:
        models = json.load(file1)

    file1.close()

    return models

def getDecisionModelsJsonData():
    
    models = getModelsJsonData()

    with open("./mlModel/versions/models/decisionTree/models.json", 'r') as file2:
        decisionTrees = json.load(file2)

    decision_tree_modelData = decisionTrees[models["decisionTree_modelId"]]

    file2.close()

    return decision_tree_modelData

def getVectorizersJsonData():

    models = getModelsJsonData()

    with open("./mlModel/versions/models/vectorizers/models.json", 'r') as file3:
        vectorizers = json.load(file3)
    
    vec_modelData = vectorizers[models["vectorizer_modelID"]]
    file3.close()

    return vec_modelData


def add_performance_to_specific_model(target_id, new_value, json_file_path):
   
    with open(json_file_path, 'r') as json_file:
        objects_list = json.load(json_file)
    json_file.close()
    # Find the object with the specified ID
    for obj in objects_list:
        if obj.get('modelId') == target_id:
            # Add the value to the object
            obj['performanceMetrics'] = new_value
            obj['isAlreadyTested'] = True
            break
    else:
        # If no object with the specified ID is found, raise an error
        raise ValueError(f"No model found with modelId :  '{target_id}'")

    # Save the updated list of objects to a JSON file
    with open(json_file_path, 'w') as json_file:
        json.dump(objects_list, json_file, indent=4)
    json_file.close()



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