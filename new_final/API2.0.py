from flask import Flask, request, jsonify
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import joblib
import cv2
import pytesseract
import re

app = Flask(__name__)

contract_model_path = './contract_classification_model.h5'
tokenizer_path = './tokenizer.pkl'

contract_model = load_model(contract_model_path)

# Load the associated tokenizer
with open(tokenizer_path, 'rb') as handle:
    tokenizer = pickle.load(handle)

label_encoder = joblib.load('label_encoder.pkl')


def image_to_articles(image):
    """ 
    This function is designed to take an image as input and retrieve the articles
    contained in that image in a dictionary format.
    Dictionary format:
    articles_dict = {article number(int): article content}
    """
    try:
        img = cv2.imread(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        threshold_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        text = pytesseract.image_to_string(threshold_img)

        article_matches = re.finditer(r'\b(?:ARTICLE|Article|article)\b', text)
        indices = [match.start() for match in article_matches]

        # Dictionary to store articles
        articles_dict = {}

        for i in range(len(indices)):
            start_index = indices[i]
            end_index = indices[i + 1] if i + 1 < len(indices) else None
            article_content = text[start_index:end_index].strip()
            if article_content:
                articles_dict[len(articles_dict) + 1] = article_content

        return articles_dict

    except Exception as e:
        return {"error": str(e)}


def get_suggestions_for_article(article_text, suggestion_model, suggestion_tokenizer, dataset):
    # Tokenize and pad the new text for content suggestion
    new_text_sequence = suggestion_tokenizer.texts_to_sequences([article_text])
    new_text_padded = pad_sequences(new_text_sequence, maxlen=suggestion_model.input_shape[1])

    # Predict the content vector for the new text
    predicted_content_vector = suggestion_model.predict(new_text_padded)

    # get the needed dataset
    df = pd.read_csv(dataset)

    if not df.empty:
        # Tokenize and vectorize the text data
        suggestion_tokenizer = tf.keras.preprocessing.text.Tokenizer()
        suggestion_tokenizer.fit_on_texts(df['Content'])
        X = suggestion_tokenizer.texts_to_sequences(df['Content'])

        # Adjust to the expected length
        X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=suggestion_model.input_shape[1])

        new_text_sequence = suggestion_tokenizer.texts_to_sequences([article_text])
        new_text_padded = pad_sequences(new_text_sequence, maxlen=suggestion_model.input_shape[1])

        # Predict Content Vectors
        predicted_content_vector = suggestion_model.predict(new_text_padded)
        content_vectors = suggestion_model.predict(X)

        # Calculate Content Similarity
        content_similarity = np.dot(content_vectors, predicted_content_vector.T)
        content_similarity = content_similarity.reshape(-1)

        similarity_threshold = 0.7
        top_content_indices = np.where(content_similarity > similarity_threshold)[0]

        if top_content_indices.size > 0:
            top_content_indices = top_content_indices[top_content_indices != 0]

            if top_content_indices.size > 0:
                # Get recommended content from the DataFrame
                recommended_content = df['Content'].iloc[top_content_indices].tolist()

                # Take the top 5 after filtering
                recommended_content = recommended_content[:5]

                return {'article_content': article_text, 'recommended_content': recommended_content}

    return {'article_content': article_text, 'error': 'No data available for recommendation'}


@app.route('/predict_contract', methods=['POST'])
def predict_contract():
    try:
        image_file = request.files.get('image')

        # Ensure the filename is a string
        filename = str(image_file.filename)

        # Save the image file to a location
        image_path = os.path.join("upload_folder", filename)
        image_file.save(image_path)

        new_text = image_to_articles(image_path)

        # Predict the contract type only once based on the first article
        first_article_text = next(iter(new_text.values()))
        first_article_text_sequence = tokenizer.texts_to_sequences([first_article_text])
        first_article_text_padded = pad_sequences(first_article_text_sequence, maxlen=contract_model.input_shape[1])

        classes = ['Contrat de prestation de Service', 'cdd', 'cdi', 'contrat de vente et achat', 'freelance',
                   'location commerciale', 'nda']

        # Predict the contract class
        predicted_class = np.argmax(contract_model.predict(first_article_text_padded), axis=1)[0]
        predicted_class = classes[predicted_class]

        result_dict = {predicted_class: []}

        # Directory containing suggestion models and tokenizers
        models_dir = 'models_+tokenizer/'
        suggestion_models = {}
        suggestion_tokenizers = {}

        for filename in os.listdir(models_dir):
            if filename.endswith('.h5'):
                model_name = os.path.splitext(filename)[0]

                # Load suggestion model
                suggestion_model_path = os.path.join(models_dir, filename)
                suggestion_model = load_model(suggestion_model_path)
                suggestion_models[model_name] = suggestion_model

                # Load associated tokenizer
                tokenizer_name = f'{model_name.replace("model_", "")}.pickle'
                tokenizer_path = os.path.join(models_dir, tokenizer_name)
                with open(tokenizer_path, 'rb') as handle:
                    suggestion_tokenizer = pickle.load(handle)
                suggestion_tokenizers[model_name] = suggestion_tokenizer

        # Load the corresponding suggestion model and tokenizer
        suggestion_model = suggestion_models[predicted_class]
        suggestion_tokenizer = suggestion_tokenizers[predicted_class]

        dataset_mapping = {
            'Contrat de prestation de Service': './contracts/Contrat_de_prestation_de_Service/Contrat_de_prestation_de_Service.csv',
            'cdd': './contracts/cdd/cdd.csv',
            'cdi': './contracts/cdi/cdi.csv',
            'contrat de vente et achat': './contracts/contrat_de_vente_et_achat/contrat de vente et achat.csv',
            'freelance': './contracts/freelance/freelance.csv',
            'location commerciale': './contracts/location commerciale/location commerciale.csv',
            'nda': './contracts/nda/nda.csv'
        }

        for article_number, article_text in new_text.items():
            dataset = dataset_mapping[predicted_class]

            result_dict[predicted_class].append(get_suggestions_for_article(article_text,
                                                                           suggestion_model,
                                                                           suggestion_tokenizer,
                                                                           dataset))

        return jsonify(result_dict)

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
