from flask import Flask, render_template, request
import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from io import BytesIO
import pickle
from nltk.tokenize import word_tokenize 
from collections import Counter  
import string
import lime
from lime.lime_text import LimeTextExplainer
import shap
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from lime.lime_text import LimeTextExplainer
from sklearn.feature_extraction.text import TfidfVectorizer
import random

app = Flask(__name__)

# resampling data 
"""
X = data['message']
y = data['label']

from imblearn.over_sampling import RandomOverSampler
oversampler = RandomOverSampler(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = oversampler.fit_resample(X.values.reshape(-1, 1), y)

resampled_data = pd.DataFrame({'message': X_resampled.squeeze(), 'label': y_resampled})

"""

# Load and preprocess the dataset
data = pd.read_csv('spam.csv', encoding='ISO-8859-1')
data = data.rename(columns={'v1': 'message_type', 'v2': 'message'})
data = data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis='columns')
label_mapping = {'ham': 0, 'spam': 1}
data['label'] = data['message_type'].map(label_mapping)

# more preprocessing
"""
#Removing Duplicates
data = data.drop_duplicates()

#Reindexing
data = data.reset_index(drop=True)

text = re.sub(r'[^a-zA-Z]', ' ', text)

# Convert text to lowercase
text = text.lower()

# Tokenize the text
words = text.split()

# Remove stopwords
words = [word for word in words if word not in set(stopwords.words('english'))]

# Join the cleaned words back into a sentence
cleaned_text = ' '.join(words)

"""

X = data['message']

# Create a CountVectorizer for BoW
vectorizer = CountVectorizer()
X_bow = vectorizer.fit_transform(X)
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X_bow, y, test_size=0.2, random_state=42)

models = {
    'KNN': KNeighborsClassifier(n_neighbors=100),
    'Naive Bayes': MultinomialNB(),
    'SVM Linear': SVC(kernel='linear'),
    'SVM RBF': SVC(kernel='rbf'),
    'Logistic Regression': LogisticRegression()
}

results = {}

for model_name, model in models.items():
    pipeline = make_pipeline(model)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[model_name] = accuracy

import pickle

# Saving models in pickle files
c_drive_directory = 'C:/Users/Vamsi Krishna/Downloads/'

models = {
    'KNN': KNeighborsClassifier(n_neighbors=100),
    'Naive Bayes': MultinomialNB(),
    'SVM Linear': SVC(kernel='linear'),
    'SVM RBF': SVC(kernel='rbf'),
    'Logistic Regression': LogisticRegression()
}

for model_name, model in models.items():
    pickle_file_path = c_drive_directory + model_name + '_model.sav'
    
    with open(pickle_file_path, 'wb') as pickle_file:
        pickle.dump(model, pickle_file)


from sklearn.feature_extraction.text import CountVectorizer

# Create a CountVectorizer for BoW
vectorizer_bow = CountVectorizer()
X_bow = vectorizer_bow.fit_transform(data['message'])

from sklearn.feature_extraction.text import TfidfVectorizer

# Create a TfidfVectorizer for TF-IDF
vectorizer_tfidf = TfidfVectorizer()
X_tfidf = vectorizer_tfidf.fit_transform(data['message'])

@app.route('/')
def index():
    return render_template('index.html', result=None,data=data)

@app.route('/bow')
def bow():
    bow_data = X_bow
    return render_template('bow.html', bow_vectors=bow_data)

@app.route('/tfidf')
def tfidf():
    tfidf_data = X_tfidf
    return render_template('tf-idf.html', tfidf_vectors=tfidf_data)

@app.route('/classify', methods=['POST'])
def classify():
    text_input = request.form['text_input']
    model_classifiers = {
        'KNN': KNeighborsClassifier(n_neighbors=2),
        'Naive Bayes': MultinomialNB(),
        'SVM Linear': SVC(kernel='linear'),
        'SVM RBF': SVC(kernel='rbf'),
        'Logistic Regression': LogisticRegression()
    }

    results = {}

    for model_name, model_classifier in model_classifiers.items():
        pipeline = make_pipeline(model_classifier)
        pipeline.fit(X_train, y_train)
        X_input = vectorizer.transform([text_input])
        prediction = pipeline.predict(X_input)[0]
        result = "spam" if prediction == 1 else "ham"

        # Calculate accuracy
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        results[model_name] = {
            'result': result,
            'accuracy': accuracy
        }

    return render_template('index.html', results=results, data=data)


from wordcloud import WordCloud
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from textblob import TextBlob

# VISUALIZATIONS 

@app.route('/visualizations')
def visualizations():
    # word cloud (both spam and ham)
    spam_text = " ".join(data[data['label'] == 1]['message'])  # Word cloud from spam messages
    ham_text = " ".join(data[data['label'] == 0]['message'])  # Word cloud from ham messages
    spam_wordcloud = WordCloud(width=800, height=400, background_color='black').generate(spam_text)
    spam_wordcloud.to_file("static/spam_wordcloud.png")
    ham_wordcloud = WordCloud(width=800, height=400, background_color='black').generate(ham_text)
    ham_wordcloud.to_file("static/ham_wordcloud.png")
    # Sentiment analysis for ham messages
    ham_text = " ".join(data[data['label'] == 0]['message'])
    ham_sentiment = TextBlob(ham_text)
    ham_sentiment = "Ham" if ham_sentiment.sentiment.polarity > 0 else "Ham"
    # Sentiment analysis for spam messages
    spam_text = " ".join(data[data['label'] == 1]['message'])
    spam_sentiment = TextBlob(spam_text)
    spam_sentiment = "Spam" if spam_sentiment.sentiment.polarity > 0 else "Spam"

    # Generate the pie chart
    ham_count = len(data[data['label'] == 0])
    spam_count = len(data[data['label'] == 1])
    labels = [ham_sentiment, spam_sentiment]
    sizes = [ham_count, spam_count]
    colors = ['#3E6B48', '#E64B35']
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.savefig("static/pie_chart.png")

    # Generate the confusion matrix
    confusion = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 5))
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', cbar=False, linewidths=0.5)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig("static/confusion_matrix.png")

    wordcloud_img_url = "static/wordcloud.png"
    pie_chart_img_url = "static/pie_chart.png"
    accuracy_img_url = "static/accuracy_bar_chart.png"

    ham_messages = " ".join(data[data['message_type'] == 'ham']['message'])
    spam_messages = " ".join(data[data['message_type'] == 'spam']['message'])

    def preprocess_message(message):
        tokens = word_tokenize(message)
        tokens = [word.lower() for word in tokens if word.isalnum()]
        return tokens
    ham_tokens = preprocess_message(ham_messages)
    spam_tokens = preprocess_message(spam_messages)
    ham_word_freq = Counter(ham_tokens)
    spam_word_freq = Counter(spam_tokens)

    # Top 10 frequent words 
    top_words_ham, top_freqs_ham = zip(*ham_word_freq.most_common(10))
    top_words_spam, top_freqs_spam = zip(*spam_word_freq.most_common(10))

    sentiment_results = {
        'ham_sentiment': ham_sentiment,
        'spam_sentiment': spam_sentiment
    }
    ham_message_lengths = [len(message) for message in ham_messages.split('\n') if message.strip()]
    spam_message_lengths = [len(message) for message in spam_messages.split('\n') if message.strip()]

    # Create histograms for message length distribution

    plt.figure(figsize=(10, 6))
    sns.histplot(ham_message_lengths, bins=50, color='green', label='Ham')
    sns.histplot(spam_message_lengths, bins=50, color='red', label='Spam')
    plt.title('Message Length Distribution')
    plt.xlabel('Message Length')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(axis='y')

    return render_template('visualizations.html',
                        sentiment_results=sentiment_results,
                        top_words_ham=top_words_ham, top_freqs_ham=top_freqs_ham,
                        top_words_spam=top_words_spam, top_freqs_spam=top_freqs_spam)


# LIME AND SHAP FUNCTION
"""
@app.route('/lime_shap', methods=['GET'])
def lime_shap():
    
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X_tfidf = tfidf_vectorizer.fit_transform(data['message'])

    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, data['label'], test_size=0.2, random_state=42)

    logistic_model = LogisticRegression()
    logistic_model.fit(X_train, y_train)

    explainer = LimeTextExplainer(class_names=['ham', 'spam'])

    def predict_proba(text_instances):
        text_instances_tfidf = tfidf_vectorizer.transform(text_instances)
        probabilities = logistic_model.predict_proba(text_instances_tfidf)
        return probabilities

    num_instances, _ = X_test.shape
    
    random_instance_index = random.randint(0, num_instances - 1)

    text_instance = X_test[random_instance_index]
    true_label = y_test[random_instance_index]

    text_instance_str = " ".join(tfidf_vectorizer.get_feature_names_out()[text_instance.indices])

    text_instance = [text_instance_str]

    explanation = explainer.explain_instance(text_instance[0], predict_proba, num_features=10)

    lime_explanation_image = explanation.as_pyplot_figure()
    lime_explanation_image.savefig('static/lime_plot.png', format='png')

    shap_values = explainer.shap_values(text_instance)
    shap_plot_filename = 'shap_plot.png'
    shap.summary_plot(shap_values, text_instance, feature_names=tfidf_vectorizer.get_feature_names_out(), show=False)
    plt.tight_layout()
    plt.savefig(shap_plot_filename)

    return render_template('lime_shap.html', lime_plot='lime_plot.png')"""

@app.route('/lime_shap', methods=['GET'])
def lime_shap():
    lime_plot1 = 'lime_plot1.png'
    lime_plot2 = 'lime_plot2.png'
    shap_plot = 'shap_plot.png'

    return render_template('lime_shap.html', lime_plot1=lime_plot1, lime_plot2=lime_plot2, shap_plot=shap_plot)

@app.route('/spam_messages')
def spam_messages():
    spam_messages = data[data['label'] == 1]['message'].tolist()
    return render_template('spam_messages.html', spam_messages=spam_messages)

if __name__ == '__main__':
    app.run(debug=True)



