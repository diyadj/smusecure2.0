import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict

# Read dara
df = pd.read_csv('processed_data.csv')
sentences = df['clean_text'].apply(lambda text: text.split()).tolist()

# Train word2vec model
w2v_model = Word2Vec(sentences, 
                    vector_size=50,
                    window=3,
                    min_count=1,
                    workers=1)

# Get vocabulary and vectors
words = list(w2v_model.wv.key_to_index.keys())
word_vectors = w2v_model.wv[words]

# 1. K-means Clustering
n_clusters = 5  # Adjust based on your needs
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(word_vectors)

# 2. PCA for visualization
pca = PCA(n_components=2)
word_vectors_2d = pca.fit_transform(word_vectors)

# Create DataFrame with all information
analysis_df = pd.DataFrame({
    'word': words,
    'x': word_vectors_2d[:, 0],
    'y': word_vectors_2d[:, 1],
    'cluster': clusters
})

# 3. Create cluster visualization
fig_clusters = px.scatter(analysis_df, 
                         x='x', 
                         y='y', 
                         color='cluster',
                         hover_data=['word'],
                         title='Word Clusters')
fig_clusters.show()

# 4. Analyze clusters
def analyze_clusters(df):
    cluster_analysis = defaultdict(list)
    for cluster in range(n_clusters):
        cluster_words = df[df['cluster'] == cluster]['word'].tolist()
        cluster_analysis[f'Cluster {cluster}'] = cluster_words
    return cluster_analysis

cluster_contents = analyze_clusters(analysis_df)

# Print cluster analysis
print("\nCluster Analysis:")
for cluster, words in cluster_contents.items():
    print(f"\n{cluster}: {', '.join(words[:10])}{'...' if len(words) > 10 else ''}")

# 5. Cosine Similarity Analysis
def find_similar_words(word, n=5):
    """Find n most similar words for a given word"""
    try:
        similar_words = w2v_model.wv.most_similar(word, topn=n)
        return similar_words
    except KeyError:
        return []

# 6. Function to analyze document similarity
def get_document_vector(text, model):
    """Convert document to vector by averaging word vectors"""
    words = text.split()
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    if not word_vectors:
        return None
    return np.mean(word_vectors, axis=0)

# Calculate document similarities
doc_vectors = []
valid_indices = []

for idx, text in enumerate(df['clean_text']):
    vec = get_document_vector(text, w2v_model)
    if vec is not None:
        doc_vectors.append(vec)
        valid_indices.append(idx)

doc_vectors = np.array(doc_vectors)
similarity_matrix = cosine_similarity(doc_vectors)

# Create similarity heatmap
fig_similarity = go.Figure(data=go.Heatmap(
    z=similarity_matrix,
    colorscale='Viridis'
))
fig_similarity.update_layout(
    title='Document Similarity Heatmap',
    xaxis_title='Document Index',
    yaxis_title='Document Index'
)
fig_similarity.show()

# Function to find most similar documents
def find_similar_documents(doc_idx, n=5):
    """Find n most similar documents to the given document"""
    similarities = similarity_matrix[doc_idx]
    most_similar = np.argsort(similarities)[::-1][1:n+1]  # exclude self
    return [(valid_indices[idx], similarities[idx]) for idx in most_similar]

# Example usage of functions
print("\nExample Analysis:")
example_word = words[0]
print(f"\nSimilar words to '{example_word}':")
similar_words = find_similar_words(example_word)
for word, score in similar_words:
    print(f"{word}: {score:.3f}")

print("\nExample document similarity:")
example_doc_idx = 0
similar_docs = find_similar_documents(example_doc_idx)
print(f"Documents most similar to document {valid_indices[example_doc_idx]}:")
for doc_idx, similarity in similar_docs:
    print(f"Document {doc_idx}: Similarity = {similarity:.3f}")