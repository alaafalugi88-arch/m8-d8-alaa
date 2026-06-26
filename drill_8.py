"""Module 8 — Core Skills Drill: RAG Basics.

Three operational primitives for RAG: embed a sentence, verify a Weaviate
connection, ingest a small set of objects with externally-supplied vectors.
"""

import os
import numpy as np
import weaviate
import uuid
from sentence_transformers import SentenceTransformer

# Global cache for the embedding model (loaded only once)
_model = None


def get_model():
    """Load the model once and cache it."""
    global _model
    if _model is None:
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model


def embed_text(text: str) -> np.ndarray:
    """Return a 384-dim float32 numpy vector for the input string."""
    model = get_model()
    # encode returns numpy by default
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding.astype(np.float32)


def weaviate_ready(url: str = "http://localhost:8080") -> bool:
    """Return True if Weaviate at `url` is reachable and ready, else False."""
    try:
        # Support environment variable for CI
        if not url.startswith("http"):
            url = os.environ.get("WEAVIATE_URL", url)
        
        client = weaviate.Client(url)
        return client.is_ready()
    except Exception:
        return False


def ingest_corpus(client: weaviate.Client, class_name: str, items: list[dict]) -> int:
    """Ingest items into the named class. Return the count of ingested objects."""
    
    # 1. Create schema if class doesn't exist
    schema = {
        "class": class_name,
        "vectorizer": "none",
        "properties": [
            {
                "name": "title",
                "dataType": ["text"]
            },
            {
                "name": "text",
                "dataType": ["text"],
                "tokenization": "word"   # for BM25
            }
        ]
    }

    try:
        # Check if class exists
        existing = client.schema.get(class_name)
    except:
        existing = None

    if not existing:
        client.schema.create_class(schema)
        print(f"✅ Created Weaviate class: {class_name}")

    # 2. Batch ingest
    with client.batch as batch:
        batch.batch_size = 20
        for item in items:
            data_object = {
                "title": item["title"],
                "text": item["text"]
            }
            # Ensure vector is Python list
            vector = item["vector"]
            if isinstance(vector, np.ndarray):
                vector = vector.tolist()

            batch.add_data_object(
                data_object=data_object,
                class_name=class_name,
                vector=vector,
                uuid=str(uuid.uuid4())
            )

    # 3. Get final count
    result = (
        client.query
        .aggregate(class_name)
        .with_meta_count()
        .do()
    )
    
    count = result.get("data", {}).get("Aggregate", {}).get(class_name, [{}])[0].get("meta", {}).get("count", 0)
    return int(count)