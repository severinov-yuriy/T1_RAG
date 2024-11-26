from core.vectorization import TextVectorizer


def test_text_vectorizer():
    vectorizer = TextVectorizer(model_name="all-MiniLM-L6-v2")
    texts = ["Hello world", "This is a test"]
    vectors = vectorizer.vectorize(texts)

    assert len(vectors) == 2
    assert len(vectors[0]) == 384  # Размерность векторов модели
