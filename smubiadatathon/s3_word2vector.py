import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import plotly.graph_objects as go


df = pd.read_csv('processed_data.csv')
sentences = df['clean_text'].apply(lambda text: text.split()).tolist()

# Train word2vec model with optimised parameters
# Change Parameters accordingly
w2v_model = Word2Vec(sentences, 
                    vector_size=50,
                    window=3,
                    min_count=1,  # Increased to reduce vocabulary size
                    workers=1,
                    compute_loss=False)  # Disable loss computation for speed

# Get vocabulary and vectors (limit to top N most frequent words)
max_words = 10  # Adjust this number based on your needs
words = sorted(w2v_model.wv.key_to_index.keys(), 
              key=lambda x: w2v_model.wv.get_vecattr(x, "count"), 
              reverse=True)[:max_words]
word_vectors = w2v_model.wv[words]

# Apply PCA
pca = PCA(n_components=2)
word_vectors_2d = pca.fit_transform(word_vectors)

# Create DataFrame for visualization
pca_df = pd.DataFrame(word_vectors_2d, columns=['x', 'y'])
pca_df['word'] = words

# Create interactive scatter plot with optimized settings
fig = go.Figure(data=go.Scattergl(  # Using Scattergl for better performance
    x=pca_df['x'],
    y=pca_df['y'],
    mode='markers',  # Removed text mode for initial view
    marker=dict(
        color=np.random.randn(len(words)),
        colorscale='Viridis',
        line_width=1,
        size=5  # Smaller markers for better performance
    ),
    text=pca_df['word'],
    hoverinfo='text'  # Show text only on hover
))

# Optimise layout
fig.update_layout(
    title="Word2Vec Embeddings Visualization (Top {} Words)".format(max_words),
    xaxis_title="First Principal Component",
    yaxis_title="Second Principal Component",
    showlegend=False,
    uirevision=True  # Maintain zoom level on updates
)

# Add updatemenus for interactivity
fig.update_layout(
    updatemenus=[
        dict(
            type="buttons",
            showactive=False,
            buttons=[
                dict(
                    label="Show Labels",
                    method="update",
                    args=[{"textposition": "bottom center", "mode": "markers+text"}]
                ),
                dict(
                    label="Hide Labels",
                    method="update",
                    args=[{"textposition": None, "mode": "markers"}]
                )
            ]
        )
    ]
)

fig.show()