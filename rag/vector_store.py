from pathlib import Path

from qdrant_client import QdrantClient, models


PROJECT_ROOT = Path(__file__).resolve().parents[1]

QDRANT_PATH = PROJECT_ROOT / "database" / "qdrant"

COLLECTION_NAME = "manufacturing_knowledge"

VECTOR_SIZE = 1024


def get_qdrant_client() -> QdrantClient:
    QDRANT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    client = QdrantClient(
        path=str(QDRANT_PATH)
    )

    return client


def create_collection(
    client: QdrantClient,
    recreate: bool = False,
):
    exists = client.collection_exists(
        COLLECTION_NAME
    )

    if recreate and exists:
        print(
            f"Deleting existing collection: "
            f"{COLLECTION_NAME}"
        )

        client.delete_collection(
            COLLECTION_NAME
        )

        exists = False

    if not exists:
        print(
            f"Creating collection: "
            f"{COLLECTION_NAME}"
        )

        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE,
            ),
        )

    else:
        print(
            f"Collection already exists: "
            f"{COLLECTION_NAME}"
        )


if __name__ == "__main__":
    client = get_qdrant_client()

    create_collection(
        client,
        recreate=False,
    )

    print("Qdrant local storage OK")

    client.close()