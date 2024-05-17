import re
import nltk
from nltk.stem import WordNetLemmatizer
from collections import Counter
import math
nltk.download('wordnet')
# //////////////////////
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
# /////////////
from sklearn.decomposition import PCA  # PCA helps in reducing dimensions for visualization
from sklearn.cluster import KMeans  # KMeans for clustering
from sklearn.metrics import silhouette_score, adjusted_rand_score  # Evaluation metrics
from collections import Counter  # For counting occurrences of items
import matplotlib.pyplot as plt  # For visualization

# ///////PREPROCESSING
def preprocess(text):
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    special_chars = r'[^\w\s]'
    number_pattern = r'\b\d+\b'
    lemmatizer = WordNetLemmatizer()

    # Read stopwords from file
    with open("data\Stopword-List.txt", 'r') as f:
        stop_words = set(word.strip() for word in f)

    text = re.sub(url_pattern, '', text)
    text = re.sub(special_chars, '', text)
    text = re.sub(number_pattern, '', text)
    tokens = text.lower().split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 1]

    return tokens



def preprocess_query(query):
    # Define patterns and objects for preprocessing
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    special_chars = r'[^\w\s]'
    number_pattern = r'\b\d+\b'
    lemmatizer = WordNetLemmatizer()

    # Read stopwords from file
    with open("data\Stopword-List.txt", 'r') as f:
        stop_words = set(word.strip() for word in f)


    query = re.sub(url_pattern, '', query)   # Remove URLs
    query = re.sub(special_chars, '', query)# Remove special characters
    query = re.sub(number_pattern, '', query)   # Remove numbers
    tokens = query.lower().split() # Convert to lowercase and tokenize
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]    # Remove stopwords and lemmatize
    return " ".join(tokens)



# /////////DATA PARSING

#parsing through each document and storing it in dict
dict = {}
docid=[]
for i in range(1, 27):
    try:
        with open(f"data/{i}.txt", 'r', encoding='cp1252') as f:
            docid.append(i + 1)
            content = f.read()
            tokens = preprocess(content)
            dict[i] = tokens
    except FileNotFoundError:
        print(f"File {i} is missing!")


# print(dict)
# print(docid)

# Define label_mapping
label_mapping = {
    1: "Explainable Artificial Intelligence",
    2: "Explainable Artificial Intelligence",
    3: "Explainable Artificial Intelligence",
    7: "Explainable Artificial Intelligence",
    8: "Heart Failure",
    9: "Heart Failure",
    11: "Heart Failure",
    12: "Time Series Forecasting",
    13: "Time Series Forecasting",
    14: "Time Series Forecasting",
    15: "Time Series Forecasting",
    16: "Time Series Forecasting",
    17: "Transformer Model",
    18: "Transformer Model",
    21: "Transformer Model",
    22: "Feature Selection",
    23: "Feature Selection",
    24: "Feature Selection",
    25: "Feature Selection",
    26: "Feature Selection"
}










# ////////////////////////////// TEXT CLASSIFICATION


def train_knn_classifier(documents, labels, test_size=0.3, random_state=42, k=3):
    # Convert tokens into documents
    documents_text = [" ".join(tokens) for tokens in documents.values()]
    y = []

    for doc_id in documents.keys():
        label = labels.get(doc_id, "Unknown")
        y.append(label)

    # Convert documents to TF-IDF vectors
    count_vectorizer = CountVectorizer()
    tfidf_transformer = TfidfTransformer()
    X_counts = count_vectorizer.fit_transform(documents_text)
    X_tfidf = tfidf_transformer.fit_transform(X_counts)

    # Split the data into training and testing sets with stratification
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=test_size, random_state=random_state, stratify=y)

    # Train the k-NN classifier
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(X_train, y_train)

    # Predict labels for the test data
    y_pred = knn_classifier.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return knn_classifier, count_vectorizer, tfidf_transformer,accuracy, report

def classify_new_query(query, knn_classifier, count_vectorizer, tfidf_transformer):
    # Preprocess the query
    preprocessed_query = preprocess_query(query)

    # Convert preprocessed query to TF-IDF vector
    query_counts = count_vectorizer.transform([preprocessed_query])
    query_tfidf = tfidf_transformer.transform(query_counts)

    # Predict label for the query
    predicted_label = knn_classifier.predict(query_tfidf)

    return predicted_label[0]

# Train the KNN classifier
knn_classifier, count_vectorizer, tfidf_transformer,accuracy, report = train_knn_classifier(dict, label_mapping)

# Input query from the user
query = input("Enter your query: ")


# Classify the query
predicted_label = classify_new_query(query, knn_classifier, count_vectorizer, tfidf_transformer)
print("Predicted Label for the Query:", predicted_label)
print("Accuracy:", accuracy)
print("Classification Report:")
print(report)







# /////////////////////////////TEXXT CLASSIFICATIPN


# Function to evaluate clustering performance
def evaluate_clustering(documents, k=3, visualize=True):
    # Convert tokens into documents
    documents = [" ".join(tokens) for tokens in documents.values()]
    y = []

    # Fetch labels from a mapping
    for doc_id in dict.keys():
        label = label_mapping.get(doc_id, "Unknown")
        y.append(label)

    # Convert documents to TF-IDF vectors
    count_vectorizer = CountVectorizer()
    tfidf_transformer = TfidfTransformer()
    X_counts = count_vectorizer.fit_transform(documents)
    X_tfidf = tfidf_transformer.fit_transform(X_counts)

    # Fitting KMeans clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_tfidf)
    labels_pred = kmeans.labels_

    # Calculate purity
    cluster_labels = set(labels_pred)
    purity = 0
    for cluster_label in cluster_labels:
        true_labels_in_cluster = [y[i] for i, label in enumerate(labels_pred) if label == cluster_label]
        label_counts = Counter(true_labels_in_cluster)
        majority_class_count = max(label_counts.values())
        purity += majority_class_count
    purity /= len(y)

    # Calculate silhouette score
    silhouette = silhouette_score(X_tfidf, labels_pred)

    # Calculate adjusted Rand index
    rand_index = adjusted_rand_score(y, labels_pred)

    # Visualization
    if visualize:
        # Reduce dimensions for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_tfidf.toarray())

        # Plot clusters
        plt.figure(figsize=(8, 6))
        for i in range(k):
            plt.scatter(X_pca[labels_pred == i, 0], X_pca[labels_pred == i, 1], label=f'Cluster {i}')
        plt.title('KMeans Clustering')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()
        plt.show()

    return purity, silhouette, rand_index

# Example usage:
purity, silhouette, rand_index = evaluate_clustering(dict)
print("Purity:", purity) # Indicates how well clusters represent a single class
print("Silhouette Score:", silhouette) # Measures how compact and well-separated the clusters are
print("Random Index:", rand_index) # Indicates the similarity between two different clusterings

