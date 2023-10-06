from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



genre_names = ['Action', 'Adventure', 'Comedy', 'Drama', 'Horror', 'Sci-Fi', 'Thriller']

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Compute TF-IDF matrix
tfidf_matrix = vectorizer.fit_transform(genre_names)

# Compute cosine similarity matrix
cosine_similarities = cosine_similarity(tfidf_matrix)

# Print the similarity matrix
for i in range(len(genre_names)):
    for j in range(i + 1, len(genre_names)):
        similarity = cosine_similarities[i, j]
        print(f"Similarity between '{genre_names[i]}' and '{genre_names[j]}': {similarity}")