from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import os
import pickle
import joblib
import cv2
import pytesseract
import re
from keras.models import load_model
import pandas as pd
import numpy as np
from collections import Counter
from heapq import nlargest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


app = Flask(__name__)

random_forest_model = joblib.load("random_forest_model.pkl")

# Load the associated label encoder
label_encoder = joblib.load("label_encoder.pkl")

# Initialize TF-IDF vectorizer
vectorizer = joblib.load("tfidf_vectorizer.pkl")

def image_to_articles(image):
    try:
        img_array = np.frombuffer(image.read(), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        threshold_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        text = pytesseract.image_to_string(threshold_img)

        # Default keyword
        keyword = "ARTICLE"

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

        # If the dictionary is empty, use anything after ":"
        if not articles_dict:
            colon_index = text.find(":")
            if colon_index != -1:
                keyword = text[:colon_index].strip()

        # Re-run with the determined keyword
        article_matches = re.finditer(fr'\b(?:{re.escape(keyword)})\b', text, re.IGNORECASE)
        indices = [match.start() for match in article_matches]

        # Reset dictionary
        articles_dict = {}

        for i in range(len(indices)):
            start_index = indices[i]
            end_index = indices[i + 1] if i + 1 < len(indices) else None
            article_content = text[start_index:end_index].strip()
            if article_content:
                articles_dict[len(articles_dict) + 1] = article_content

        return articles_dict
    except Exception as e:
        print(f"An error occurred: {e}")
        return {}



def load_suggestion_models(models_dir):
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

    return suggestion_models, suggestion_tokenizers

def get_suggestions_for_article(article_text, suggestion_model, suggestion_tokenizer, dataset):
    try:
        expected_input_shape = suggestion_model.input_shape[1]
        # Tokenize and pad the new text for content suggestion
        new_text_transformed = suggestion_tokenizer.texts_to_sequences([article_text])
        new_text_padded = tf.keras.preprocessing.sequence.pad_sequences(new_text_transformed, maxlen = expected_input_shape)

        # Predict the content vector for the new text
        predicted_content_vector = suggestion_model.predict(new_text_padded)

        # Load the dataset
        df = pd.read_csv(dataset)

        if not df.empty:
            # Tokenize and pad the existing content
            existing_content_transformed = suggestion_tokenizer.texts_to_sequences(df['Content'])
            existing_content_padded = tf.keras.preprocessing.sequence.pad_sequences(existing_content_transformed, maxlen=expected_input_shape)

            # Predict Content Vectors
            content_vectors = suggestion_model.predict(existing_content_padded)

            # Calculate Content Similarity
            content_similarity = np.dot(content_vectors.astype(np.float32), predicted_content_vector.astype(np.float32).T)
            content_similarity = content_similarity.reshape(-1)

            similarity_threshold = 0.7
            top_content_indices = np.where(content_similarity > similarity_threshold)[0]

            if top_content_indices.size > 0:
                # Get recommended content from the DataFrame
                recommended_content = df['Content'].iloc[top_content_indices].tolist()

                # Take the top 5 after filtering
                recommended_content = recommended_content[:5]

                return {'article_content': article_text, 'recommended_content': recommended_content}

        return {'article_content': article_text, 'error': 'No data available for recommendation'}

    except Exception as e:
        print("Error occurred:", e)
        import traceback
        traceback.print_exc()
        return {'error': str(e)}

def test_mem(articles):
    new_data = [article for article in articles.values()]

    # Preprocess the new data using the loaded vectorizer
    new_features = vectorizer.transform(new_data)

    # Make predictions using the loaded model
    predicted_class = random_forest_model.predict(new_features)
    
    # Decode the predicted labels using the label encoder
    predicted_labels = label_encoder.inverse_transform(predicted_class)

    # Find the most common word in the predicted labels
    most_common_word = Counter(predicted_labels).most_common(1)[0][0]

    print("Predicted Labels:", predicted_labels)
    print("Most Common Word:", most_common_word)

    return most_common_word

def calculate_similarity(input_articles, suggestion_text):
    # Combine input articles into a single string
    input_text = ' '.join(input_articles)

    if not input_text or not suggestion_text:
        return 0.0

    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Fit and transform the vectorizer on the combined text
    tfidf_matrix_input = vectorizer.fit_transform([input_text])
    tfidf_matrix_suggestion = vectorizer.transform([suggestion_text])

    # Calculate cosine similarity between input articles and suggestion text
    similarity_score = cosine_similarity(tfidf_matrix_input, tfidf_matrix_suggestion).flatten()[0]

    return similarity_score

def get_most_similar_suggestions(input_articles, all_suggestions):
    # Calculate similarity scores for each suggestion
    similarity_scores = [calculate_similarity(input_articles, suggestion) for suggestion in all_suggestions]

    # Combine suggestions with their corresponding similarity scores
    suggestions_with_scores = zip(all_suggestions, similarity_scores)

    # Keep only the best two suggestions based on similarity score
    best_suggestions = nlargest(2, suggestions_with_scores, key=lambda x: x[1])

    # Extract only the suggestion content
    best_suggestions_content = [suggestion[0] for suggestion in best_suggestions]

    return best_suggestions_content




@app.route('/')
def index():
    return render_template('./index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the uploaded text
        image_file = request.files.get('image')

        # Predict the contract type without saving the image
        articles = image_to_articles(image_file)
        predicted_class = test_mem(articles)
        
        result_dict = {'predicted_contract': predicted_class}
        result_dict['recommended_content'] = []
        # Directory containing suggestion models and tokenizers
        models_dir = 'models_+tokenizer/'

        # Load suggestion models and tokenizers
        suggestion_models, suggestion_tokenizers = load_suggestion_models(models_dir)

        # Load the corresponding suggestion model and tokenizer
        suggestion_model = suggestion_models[predicted_class]
        suggestion_tokenizer = suggestion_tokenizers[predicted_class]

        dataset_mapping = {
            'Contrat de prestation de service': './contracts/Contrat_de_prestation_de_Service/Contrat_de_prestation_de_Service.csv',
            'cdd': './contracts/cdd/cdd.csv',
            'cdi': './contracts/cdi/cdi.csv',
            'contrat de vente et achat': './contracts/contrat_de_vente_et_achat/contrat de vente et achat.csv',
            'freelance': './contracts/freelance/freelance.csv',
            'location commerciale': './contracts/location commerciale/location commerciale.csv',
            'nda': './contracts/nda/nda.csv',
            'partenariat': './contracts/partenariat/partenariat.csv',
            'Location_Habitation': './contracts/Location_Habitation/Location_Habitation.csv',
            'Location_courte_duree': './contracts/Location_courte_duree/Location_courte_duree.csv',
            'influencer': './contracts/influencer/influencer.csv',
            'Sous_location_contrat': './contracts/Sous_location_contrat/Sous_location_contrat.csv',
        }

        all_article_texts = list(articles.values())

        # Get suggestions for all articles
        all_suggestions = []
        for article_text in all_article_texts:
            dataset = dataset_mapping[predicted_class]
            suggestions = get_suggestions_for_article(article_text,
                                                      suggestion_model,
                                                      suggestion_tokenizer,
                                                      dataset)
            all_suggestions.extend(suggestions.get('recommended_content', []))

        print(all_suggestions)
        # Keep only the best two suggestions based on content similarity
        best_suggestions = get_most_similar_suggestions(list(articles.values()), all_suggestions)
        return jsonify({'recommended:': best_suggestions})

    except Exception as e:
        return jsonify({'error try this out': str(e)})



if __name__ == '__main__':
    app.run(debug=True, port=5000)
