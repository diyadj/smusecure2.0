import spacy
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from textblob import TextBlob
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF, PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go

# Load spacy for NER
nlp = spacy.load('en_core_web_sm')

def get_topics(df, text_column, n_topics=5):
    """Topic modeling with NMF and TF-IDF"""
    tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
    doc_term_matrix = tfidf.fit_transform(df[text_column])
    
    nmf = NMF(n_components=n_topics, random_state=42)
    nmf_output = nmf.fit_transform(doc_term_matrix)
    
    feature_names = tfidf.get_feature_names_out()
    topics = {}
    for topic_idx, topic in enumerate(nmf.components_):
        top_terms = [feature_names[i] for i in topic.argsort()[:-10:-1]]
        topics[f"Topic {topic_idx + 1}"] = top_terms
    
    return topics, nmf_output

def analyze_sentiment(df, text_column):
    """Sentiment Analysis using TextBlob"""
    sentiments = []
    for text in df[text_column]:
        blob = TextBlob(text)
        sentiments.append({
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        })
    return pd.DataFrame(sentiments)

def extract_entities(df, text_column):
    """Named Entity Recognition using spaCy"""
    entities = defaultdict(Counter)
    for text in df[text_column]:
        doc = nlp(text)
        for ent in doc.ents:
            entities[ent.label_].update([ent.text])
    return entities

def get_text_stats(df, text_column):
    """Calculate text statistics"""
    stats = {
        'avg_word_count': df[text_column].str.split().str.len().mean(),
        'avg_sentence_count': df[text_column].apply(lambda x: len(TextBlob(x).sentences)).mean(),
        'vocabulary_size': len(set(' '.join(df[text_column]).split()))
    }
    return stats

def analyze_time_patterns(df):
    """Analyze time-based patterns if date column exists"""
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        time_patterns = {
            'articles_by_day': df.groupby(df['date'].dt.date).size(),
            'articles_by_month': df.groupby(df['date'].dt.to_period('M')).size(),
            'articles_by_weekday': df.groupby(df['date'].dt.day_name()).size()
        }
        return time_patterns
    return None

def create_word2vec_model(df, text_column):
    """Create and train Word2Vec model"""
    # Prepare sentences
    sentences = [text.split() for text in df[text_column]]
    
    # # Initialize model
    # w2v_model = Word2Vec(
    #     vector_size=100,
    #     window=5,
    #     min_count=5,
    #     workers=4
    # )
    
    # # Build vocabulary
    # w2v_model.build_vocab(sentences)
    
    # # Train model
    # w2v_model.train(
    #     sentences,
    #     total_examples=len(sentences),
    #     epochs=30
    # )
    try:
    # Create and train the model in one step
        w2v_model = Word2Vec(
            sentences=sentences,  # Pass sentences directly here
            vector_size=100,
            window=5,
            min_count=5,
            workers=4,
            epochs=30
        )
    
        return w2v_model

    except Exception as e:
        print(f"Error in Word2Vec model creation: {str(e)}")
        raise
    
def get_document_embeddings(df, text_column, w2v_model):
    """Generate document embeddings using Word2Vec"""
    def get_doc_vector(text):
        words = text.split()
        word_vecs = [w2v_model.wv[word] for word in words if word in w2v_model.wv]
        if not word_vecs:
            return np.zeros(w2v_model.vector_size)
        return np.mean(word_vecs, axis=0)
    
    doc_vectors = np.array([get_doc_vector(text) for text in df[text_column]])
    return doc_vectors

def perform_clustering(doc_vectors, n_clusters=20):
    """Perform KMeans clustering on document vectors"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(doc_vectors)
    
    cluster_centers = kmeans.cluster_centers_
    cluster_contents = defaultdict(list)
    
    for idx, cluster in enumerate(clusters):
        cluster_contents[f'Cluster {cluster}'].append(idx)
        
    return clusters, cluster_centers, cluster_contents

def analyze_similarity(doc_vectors):
    """Analyze document similarity using cosine similarity"""
    similarity_matrix = cosine_similarity(doc_vectors)
    
    similar_articles = []
    for i in range(len(similarity_matrix)):
        similar_indices = similarity_matrix[i].argsort()[-6:-1][::-1]
        similarities = similarity_matrix[i][similar_indices]
        similar_articles.append(list(zip(similar_indices, similarities)))
        
    return similarity_matrix, similar_articles

def create_visualizations(df, text_column, doc_vectors, clusters, similarity_matrix, topics, results):
    """Create all visualizations"""
    # PCA for dimensionality reduction
    pca = PCA(n_components=2)
    doc_vectors_2d = pca.fit_transform(doc_vectors)
    
    cluster_df = pd.DataFrame({
        'x': doc_vectors_2d[:, 0],
        'y': doc_vectors_2d[:, 1],
        'cluster': clusters
    })
    
    # Cluster visualization
    fig_clusters = px.scatter(
        cluster_df,
        x='x',
        y='y',
        color='cluster',
        title='Article Clusters Visualization'
    )
    
    # Similarity heatmap
    fig_similarity = go.Figure(data=go.Heatmap(
        z=similarity_matrix,
        colorscale='Viridis'
    ))
    fig_similarity.update_layout(
        title='Article Similarity Heatmap',
        xaxis_title='Article Index',
        yaxis_title='Article Index'
    )
    
    # Topic visualization
    topic_viz = px.bar(
        x=[f"Topic {i+1}" for i in range(len(topics))],
        y=[len(results['document_topics'][results['document_topics'] == f'Topic_{i+1}']) 
           for i in range(len(topics))],
        title='Distribution of Dominant Topics'
    )
    
    # Sentiment visualization
    sentiment_viz = px.scatter(
        results['sentiments'],
        x='polarity',
        y='subjectivity',
        title='Sentiment Analysis Distribution'
    )
    
    # Entity visualization
    top_entities = {etype: dict(counter.most_common(5)) 
                   for etype, counter in results['entities'].items()}
    entity_viz = go.Figure()
    for etype, ents in top_entities.items():
        entity_viz.add_trace(go.Bar(
            name=etype,
            x=list(ents.keys()),
            y=list(ents.values())
        ))
    entity_viz.update_layout(title='Top Named Entities by Type')
    
    # Time visualization
    time_viz = None
    if 'time_patterns' in results:
        time_viz = px.line(
            x=results['time_patterns']['articles_by_day'].index,
            y=results['time_patterns']['articles_by_day'].values,
            title='Articles Published Over Time'
        )
    
    return {
        'topic_distribution': topic_viz,
        'sentiment_distribution': sentiment_viz,
        'entity_distribution': entity_viz,
        'cluster_distribution': fig_clusters,
        'similarity_heatmap': fig_similarity,
        'time_distribution': time_viz if time_viz else None
    }

def analyze_news_articles(df, text_column='clean_text', n_clusters=10, n_topics=10):
    """Main analysis function"""
    print("Starting comprehensive news article analysis...")
    results = {}
    
    try:
        # 1. Topic Modeling
        print("Performing topic modeling...")
        topics, topic_distribution = get_topics(df, text_column, n_topics)
        results['topics'] = topics
        topic_df = pd.DataFrame(
            topic_distribution,
            columns=[f'Topic_{i+1}' for i in range(topic_distribution.shape[1])]
        )
        results['document_topics'] = topic_df.idxmax(axis=1)
        
        # 2. Sentiment Analysis
        print("Analyzing sentiment...")
        results['sentiments'] = analyze_sentiment(df, text_column)
        
        # 3. Named Entity Recognition
        print("Extracting entities...")
        results['entities'] = extract_entities(df, text_column)
        
        # 4. Text Statistics
        print("Calculating text statistics...")
        results['text_stats'] = get_text_stats(df, text_column)
        
        # 5. Time Analysis
        print("Analyzing time patterns...")
        time_patterns = analyze_time_patterns(df)
        if time_patterns:
            results['time_patterns'] = time_patterns
        
        # 6. Word2Vec
        print("Creating Word2Vec model...")
        w2v_model = create_word2vec_model(df, text_column)
        
        # 7. Document Embeddings
        print("Generating document embeddings...")
        doc_vectors = get_document_embeddings(df, text_column, w2v_model)
        
        # 8. Clustering
        print("Performing clustering...")
        clusters, cluster_centers, cluster_contents = perform_clustering(doc_vectors, n_clusters)
        results['clusters'] = clusters
        results['cluster_contents'] = cluster_contents
        
        # 9. Similarity Analysis
        print("Analyzing document similarity...")
        similarity_matrix, similar_articles = analyze_similarity(doc_vectors)
        results['similarity_matrix'] = similarity_matrix
        results['similar_articles'] = similar_articles
        
        # 10. Create Visualizations
        print("Creating visualizations...")
        results['visualizations'] = create_visualizations(
            df, text_column, doc_vectors, clusters, 
            similarity_matrix, topics, results
        )
        
        # 11. Cross-analysis metrics
        print("Calculating cross-analysis metrics...")
        topic_cluster_matrix = pd.crosstab(
            results['document_topics'],
            pd.Series(clusters, name='Cluster')
        )
        results['topic_cluster_relationship'] = topic_cluster_matrix
        
        cluster_sentiments = pd.DataFrame({
            'cluster': clusters,
            'sentiment': results['sentiments']['polarity']
        }).groupby('cluster').mean()
        results['cluster_sentiments'] = cluster_sentiments
        
        cluster_entities = defaultdict(lambda: defaultdict(Counter))
        for idx, cluster in enumerate(clusters):
            doc = nlp(df.iloc[idx][text_column])
            for ent in doc.ents:
                cluster_entities[cluster][ent.label_][ent.text] += 1
        results['cluster_entities'] = dict(cluster_entities)
        
        print("Analysis complete!")
        return results
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise

def display_analysis(results):
    """Display and interpret the analysis results"""
    print("\n=== Topic Analysis ===")
    for topic, terms in results['topics'].items():
        print(f"\n{topic}: {', '.join(terms)}")
    
    print("\n=== Cluster Analysis ===")
    for cluster in sorted(results['cluster_contents'].keys()):
        print(f"\n{cluster}: {len(results['cluster_contents'][cluster])} articles")
        print(f"Average sentiment: {results['cluster_sentiments'].loc[int(cluster.split()[-1])]['sentiment']:.3f}")
    
    print("\n=== Text Statistics ===")
    for stat, value in results['text_stats'].items():
        print(f"{stat}: {value:.2f}")
    
    print("\n=== Top Entities by Cluster ===")
    for cluster, entity_types in results['cluster_entities'].items():
        print(f"\nCluster {cluster}:")
        for entity_type, entities in entity_types.items():
            top_entities = entities.most_common(3)
            if top_entities:
                print(f"  {entity_type}: {', '.join(f'{e[0]} ({e[1]})' for e in top_entities)}")
    
    print("\n=== Topic-Cluster Relationship ===")
    print(results['topic_cluster_relationship'])
    
    # Display visualizations
    for viz_name, fig in results['visualizations'].items():
        if fig is not None:
            fig.show()

# Main execution
if __name__ == "__main__":
    try:
        # Load your data
        df = pd.read_csv('processed_data.csv')
        
        # Run analysis
        results = analyze_news_articles(df)
        
        # Display results
        display_analysis(results)
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")