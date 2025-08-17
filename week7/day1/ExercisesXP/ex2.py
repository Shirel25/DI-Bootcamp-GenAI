import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from gensim.models import Word2Vec

# 1) Import df depuis ex1.py (il contient la colonne Clean_Review)
from ex1 import df

# ================================
# Step 1: Train Word2Vec
# ================================
# Use the tokenized cleaned reviews
sentences = df["Clean_Review"].tolist()

# Train Word2Vec model
w2v = Word2Vec(
    sentences=sentences,   # tokenized sentences
    vector_size=50,        # size of word embeddings
    window=3,              # context window
    min_count=1,           # keep all words
    workers=4,             # number of CPU cores
    sg=1,                  # skip-gram (better for small data)
    epochs=200,            # training epochs
    seed=42
)

print(f"Vocabulary size: {len(w2v.wv.key_to_index)}")
print(f"Vector dimension: {w2v.wv.vector_size}")

# ================================
# Step 2: Plot embeddings
# ================================
def plot_word_embeddings(model, top_n=50):
    """
    Reduce word embeddings to 2D with PCA and plot them with labels.
    """
    words = list(model.wv.key_to_index.keys())[:top_n]
    vectors = [model.wv[w] for w in words]

    # Reduce dimensions (vector_size -> 2D)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(vectors)

    # Scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(coords[:, 0], coords[:, 1])

    # Annotate words
    for (x, y), word in zip(coords, words):
        plt.annotate(word, (x, y), xytext=(3, 3), textcoords="offset points")

    plt.title("Word2Vec Embeddings (PCA 2D)")
    plt.show()

# Call the plotting function
plot_word_embeddings(w2v)


# ==============================================================================
# I trained a Word2Vec model (Skip-gram, 50 dimensions, window size 3, 200 epochs) on the preprocessed reviews.
# Then, I applied PCA to reduce the embeddings to 2D for visualization.
# The scatter plot shows that semantically related words (e.g., pizza – restaurant – staff, or delicious – dessert) are located close to each other.
# This indicates that Word2Vec successfully captured some of the semantic relationships between words in the dataset.