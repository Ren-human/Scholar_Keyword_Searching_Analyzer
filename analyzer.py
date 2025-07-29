import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer
from collections import defaultdict, Counter

# Load keywords from CSV
df = pd.read_csv("scholar_keywords.csv")

# Parse keywords per document
documents_keywords = [
    [kw.strip() for kw in row["Top Keywords"].split(",") if kw.strip()]
    for _, row in df.iterrows()
]

# Flatten and track document origins
all_phrases = []
phrase_to_doc = defaultdict(list)
for i, phrases in enumerate(documents_keywords):
    for phrase in phrases:
        all_phrases.append(phrase)
        phrase_to_doc[phrase].append(i)

# Encode using SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(all_phrases)

# Cluster phrases semantically
n_clusters = 10
clusterer = AgglomerativeClustering(n_clusters=n_clusters)
labels = clusterer.fit_predict(embeddings)

# Organize phrases into clusters
cluster_to_phrases = defaultdict(list)
for label, phrase in zip(labels, all_phrases):
    cluster_to_phrases[label].append(phrase)

# Calculate document coverage per cluster
cluster_summary = []
for cluster, phrases in cluster_to_phrases.items():
    top_phrase = Counter(phrases).most_common(1)[0][0]
    doc_coverage = len(set(doc for phrase in phrases for doc in phrase_to_doc[phrase]))
    cluster_summary.append((cluster, top_phrase, doc_coverage, len(phrases)))

# Create summary table
df_summary = pd.DataFrame(cluster_summary, columns=["Cluster ID", "Representative Phrase", "Doc Coverage", "# Phrases"])
df_summary.sort_values(by="Doc Coverage", ascending=False, inplace=True)

df_summary.to_csv('summary_result.csv')