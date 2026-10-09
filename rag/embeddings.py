from FlagEmbedding import BGEM3FlagModel


MODEL_NAME = "BAAI/bge-m3"


def load_embedding_model():
    print(f"Loading embedding model: {MODEL_NAME}")

    model = BGEM3FlagModel(
        MODEL_NAME,
        use_fp16=False,
    )

    return model


if __name__ == "__main__":
    model = load_embedding_model()

    sentences = [
        "What causes heat dissipation failure?",
        "Heat dissipation failure can be related to temperature and rotational speed.",
        "Preventive maintenance replaces components according to scheduled intervals.",
    ]

    output = model.encode(
        sentences,
        batch_size=2,
        max_length=512,
    )

    dense_vectors = output["dense_vecs"]

    print("\nBGE-M3 TEST SUCCESS")
    print("Embedding shape:", dense_vectors.shape)
    print("First vector first 5 values:")
    print(dense_vectors[0][:5])