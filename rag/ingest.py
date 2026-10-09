from FlagEmbedding import BGEM3FlagModel
from qdrant_client import models

from rag.document_loader import load_all_documents
from rag.chunker import create_chunks
from rag.vector_store import (
    COLLECTION_NAME,
    create_collection,
    get_qdrant_client,
)


MODEL_NAME = "BAAI/bge-m3"

BATCH_SIZE = 4
MAX_LENGTH = 512
UPLOAD_BATCH_SIZE = 64


def main():
    print("Loading documents...")
    documents = load_all_documents()

    print("Creating chunks...")
    chunks = create_chunks(documents)

    print(f"Chunks: {len(chunks)}")

    print(
        f"\nLoading embedding model: "
        f"{MODEL_NAME}"
    )

    model = BGEM3FlagModel(
        MODEL_NAME,
        use_fp16=False,
    )

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    print("\nEmbedding chunks...")

    output = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
    )

    vectors = output["dense_vecs"]

    print(
        f"Embedding shape: "
        f"{vectors.shape}"
    )

    print("\nOpening Qdrant...")

    client = get_qdrant_client()

    # Ingestion 할 때는 기존 collection을
    # 새 데이터로 다시 생성
    create_collection(
        client,
        recreate=True,
    )

    print("\nUploading points...")

    for start in range(
        0,
        len(chunks),
        UPLOAD_BATCH_SIZE,
    ):
        end = min(
            start + UPLOAD_BATCH_SIZE,
            len(chunks),
        )

        points = []

        for index in range(start, end):
            chunk = chunks[index]

            points.append(
                models.PointStruct(
                    id=index,
                    vector=vectors[index].tolist(),
                    payload={
                        "chunk_id": chunk["chunk_id"],
                        "source": chunk["source"],
                        "page": chunk["page"],
                        "section": chunk["section"],
                        "text": chunk["text"],
                        "char_count": chunk["char_count"],
                    },
                )
            )

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True,
        )

        print(
            f"Uploaded: "
            f"{end}/{len(chunks)}"
        )

    count = client.count(
        collection_name=COLLECTION_NAME,
        exact=True,
    )

    print("\nINGEST SUCCESS")
    print(
        f"Stored points: "
        f"{count.count}"
    )

    client.close()


if __name__ == "__main__":
    main()