import math

from codebase_archaeologist.embeddings import HashEmbedder


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_hash_embedder_is_deterministic_and_normalised():
    e = HashEmbedder(dim=32)
    a, b = e.embed_documents(["train the lstm model", "train the lstm model"])
    assert a == b
    assert math.isclose(sum(x * x for x in a), 1.0)
    assert e.dim == 32 and len(a) == 32


def test_hash_embedder_similarity_tracks_shared_words():
    e = HashEmbedder()
    q = e.embed_query("flask route")
    related, unrelated = e.embed_documents(["define a flask route here", "matrix algebra notes"])
    assert cosine(q, related) > cosine(q, unrelated)
