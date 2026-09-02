from pymilvus import CollectionSchema, DataType, FieldSchema, Function, FunctionType

import pytest

from yuxi.knowledge.base import KBOperationError
from yuxi.knowledge.implementations.milvus import (
    CONTENT_ANALYZER_PARAMS,
    CONTENT_SPARSE_FIELD,
    MilvusEmbeddingError,
    MilvusKB,
    MilvusSearchError,
    VECTOR_METRIC_TYPE,
)


class FakeHit:
    def __init__(self, content: str, distance: float):
        self.distance = distance
        self.entity = {
            "content": content,
            "source": "demo.md",
            "chunk_id": "chunk-1",
            "file_id": "file-1",
            "chunk_index": 0,
        }


class FakeCollection:
    def __init__(self, distance: float = 0.8):
        self.search_calls = []
        self.hybrid_calls = []
        self.distance = distance

    def search(self, **kwargs):
        self.search_calls.append(kwargs)
        return [[FakeHit("BM25 result", self.distance)]]

    def hybrid_search(self, **kwargs):
        self.hybrid_calls.append(kwargs)
        return [[FakeHit("Hybrid result", self.distance)]]


def make_kb(collection: FakeCollection) -> MilvusKB:
    kb = MilvusKB.__new__(MilvusKB)
    kb.databases_meta = {"db": {"embed_info": {}}}
    kb._get_query_params = lambda db_id: {}
    kb._get_embedding_function = lambda embed_info: lambda texts: [[0.1, 0.2] for _ in texts]

    async def get_collection(db_id: str):
        return collection

    kb._get_milvus_collection = get_collection
    return kb


async def test_keyword_mode_uses_milvus_bm25_search():
    collection = FakeCollection()
    kb = make_kb(collection)

    chunks = await kb.aquery(
        "alpha beta",
        "db",
        search_mode="keyword",
        bm25_top_k=7,
        bm25_drop_ratio_search=0.2,
    )

    assert chunks[0]["content"] == "BM25 result"
    assert chunks[0]["bm25_score"] == 0.8
    search_call = collection.search_calls[0]
    assert search_call["data"] == ["alpha beta"]
    assert search_call["anns_field"] == CONTENT_SPARSE_FIELD
    assert search_call["param"] == {
        "metric_type": "BM25",
        "params": {"drop_ratio_search": 0.2},
    }
    assert search_call["limit"] == 7


async def test_vector_mode_ignores_metric_type_override():
    collection = FakeCollection()
    kb = make_kb(collection)

    chunks = await kb.aquery("vector query", "db", search_mode="vector", metric_type="L2")

    assert chunks[0]["content"] == "BM25 result"
    search_call = collection.search_calls[0]
    assert search_call["anns_field"] == "embedding"
    assert search_call["param"]["metric_type"] == VECTOR_METRIC_TYPE


async def test_hybrid_mode_uses_milvus_native_hybrid_search():
    collection = FakeCollection()
    kb = make_kb(collection)

    chunks = await kb.aquery(
        "hybrid query",
        "db",
        search_mode="hybrid",
        final_top_k=3,
        bm25_top_k=8,
        vector_weight=0.6,
        bm25_weight=0.4,
    )

    assert chunks[0]["content"] == "Hybrid result"
    assert chunks[0]["hybrid_score"] == 0.8
    hybrid_call = collection.hybrid_calls[0]
    assert hybrid_call["limit"] == 3
    assert hybrid_call["rerank"]._weights == [0.6, 0.4]

    vector_request, bm25_request = hybrid_call["reqs"]
    assert vector_request.anns_field == "embedding"
    assert vector_request.data == [[0.1, 0.2]]
    assert vector_request.param["metric_type"] == VECTOR_METRIC_TYPE
    assert bm25_request.anns_field == CONTENT_SPARSE_FIELD
    assert bm25_request.data == ["hybrid query"]
    assert bm25_request.limit == 8
    assert bm25_request.param["metric_type"] == "BM25"


async def test_hybrid_mode_filters_scores_below_similarity_threshold():
    collection = FakeCollection(distance=0.1)
    kb = make_kb(collection)

    chunks = await kb.aquery(
        "hybrid query",
        "db",
        search_mode="hybrid",
        final_top_k=3,
        similarity_threshold=0.2,
    )

    assert chunks == []


def test_query_params_config_uses_bm25_parameters():
    kb = MilvusKB.__new__(MilvusKB)

    config = kb.get_query_params_config("db")

    option_keys = {option["key"] for option in config["options"]}
    assert "keyword_top_k" not in option_keys
    assert "metric_type" not in option_keys
    assert {
        "bm25_top_k",
        "vector_weight",
        "bm25_weight",
        "bm25_drop_ratio_search",
    } <= option_keys

    search_mode = next(option for option in config["options"] if option["key"] == "search_mode")
    descriptions = {option["value"]: option["description"] for option in search_mode["options"]}
    assert "BM25" in descriptions["keyword"]
    assert "BM25" in descriptions["hybrid"]


def test_collection_supports_bm25_requires_analyzed_content_sparse_field_and_function():
    kb = MilvusKB.__new__(MilvusKB)
    schema = CollectionSchema(
        fields=[
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
            FieldSchema(
                name="content",
                dtype=DataType.VARCHAR,
                max_length=65535,
                enable_analyzer=True,
                analyzer_params=CONTENT_ANALYZER_PARAMS,
            ),
            FieldSchema(name=CONTENT_SPARSE_FIELD, dtype=DataType.SPARSE_FLOAT_VECTOR),
        ],
        functions=[
            Function(
                name="content_bm25",
                input_field_names=["content"],
                output_field_names=[CONTENT_SPARSE_FIELD],
                function_type=FunctionType.BM25,
            )
        ],
    )

    collection = type("Collection", (), {"schema": schema})()

    assert kb._collection_supports_bm25(collection)


async def test_vector_mode_can_use_existing_async_embedding_path():
    collection = FakeCollection()
    kb = make_kb(collection)
    calls: list[list[str]] = []

    async def async_encode(texts):
        calls.append(texts)
        return [[0.3, 0.4] for _ in texts]

    kb._get_async_embedding_function = lambda embed_info: async_encode

    chunks = await kb.aquery(
        "vector query",
        "db",
        search_mode="vector",
        use_async_embedding=True,
    )

    assert calls == [["vector query"]]
    assert chunks[0]["content"] == "BM25 result"
    assert collection.search_calls[0]["data"] == [[0.3, 0.4]]


async def test_raise_on_error_distinguishes_embedding_failure_from_empty_recall():
    collection = FakeCollection()
    kb = make_kb(collection)

    async def failing_encode(_texts):
        raise RuntimeError("embedding unavailable")

    kb._get_async_embedding_function = lambda embed_info: failing_encode

    with pytest.raises(MilvusEmbeddingError) as exc_info:
        await kb.aquery(
            "vector query",
            "db",
            search_mode="vector",
            use_async_embedding=True,
            raise_on_error=True,
        )

    assert isinstance(exc_info.value.__cause__, RuntimeError)


async def test_raise_on_error_distinguishes_milvus_search_failure():
    class FailingCollection(FakeCollection):
        def search(self, **kwargs):
            del kwargs
            raise RuntimeError("milvus unavailable")

    kb = make_kb(FailingCollection())

    with pytest.raises(MilvusSearchError) as exc_info:
        await kb.aquery("vector query", "db", search_mode="vector", raise_on_error=True)

    assert isinstance(exc_info.value.__cause__, RuntimeError)


async def test_default_error_behavior_remains_empty_list_for_existing_callers():
    class FailingCollection(FakeCollection):
        def search(self, **kwargs):
            del kwargs
            raise RuntimeError("milvus unavailable")

    kb = make_kb(FailingCollection())

    assert await kb.aquery("vector query", "db", search_mode="vector") == []


async def test_get_file_content_does_not_hide_milvus_read_failure():
    class FailingCollection:
        def query_iterator(self, **_kwargs):
            raise RuntimeError("milvus unavailable")

    kb = MilvusKB.__new__(MilvusKB)
    kb.files_meta = {"file-1": {"database_id": "db"}}

    async def get_collection(_db_id: str):
        return FailingCollection()

    kb._get_milvus_collection = get_collection

    with pytest.raises(KBOperationError, match="Milvus"):
        await kb.get_file_content("db", "file-1")


async def test_get_file_content_streams_every_milvus_page_and_closes_iterator():
    class FakeIterator:
        def __init__(self):
            self.batches = iter(
                [
                    [{"chunk_id": "chunk-2", "chunk_index": 1, "content": "二"}],
                    [{"chunk_id": "chunk-1", "chunk_index": 0, "content": "一"}],
                ]
            )
            self.closed = False

        def next(self):
            return next(self.batches, [])

        def close(self):
            self.closed = True

    class PaginatedCollection:
        def __init__(self):
            self.iterator = FakeIterator()
            self.arguments = None

        def query_iterator(self, **kwargs):
            self.arguments = kwargs
            return self.iterator

    collection = PaginatedCollection()
    kb = MilvusKB.__new__(MilvusKB)
    kb.files_meta = {"file-1": {"database_id": "db"}}

    async def get_collection(_db_id: str):
        return collection

    kb._get_milvus_collection = get_collection

    result = await kb.get_file_content("db", "file-1")

    assert [value["id"] for value in result["lines"]] == [
        "chunk-1",
        "chunk-2",
    ]
    assert collection.arguments == {
        "batch_size": 1000,
        "expr": 'file_id == "file-1"',
        "output_fields": ["content", "chunk_id", "chunk_index"],
    }
    assert collection.iterator.closed is True


async def test_vector_mode_reuses_precomputed_embedding_and_exact_file_ids():
    collection = FakeCollection()
    kb = make_kb(collection)

    def unexpected_embedding(_embed_info):
        raise AssertionError("embedding must not run")

    kb._get_embedding_function = unexpected_embedding
    chunks = await kb.aquery(
        "vector query",
        "db",
        search_mode="vector",
        query_embedding=[0.3, 0.4],
        filter_file_ids=["file-1", 'file"2', "file-1"],
        raise_on_error=True,
    )

    assert chunks[0]["content"] == "BM25 result"
    search_call = collection.search_calls[0]
    assert search_call["data"] == [[0.3, 0.4]]
    assert search_call["expr"] == 'file_id in ["file-1", "file\\"2"]'


async def test_file_id_and_file_name_filters_are_combined():
    collection = FakeCollection()
    kb = make_kb(collection)

    await kb.aquery(
        "vector query",
        "db",
        file_name="共识",
        filter_file_ids=["file-1"],
        raise_on_error=True,
    )

    assert collection.search_calls[0]["expr"] == ('source like "%共识%" and file_id in ["file-1"]')


async def test_empty_file_id_filter_does_not_fall_back_to_global_search():
    collection = FakeCollection()
    kb = make_kb(collection)

    with pytest.raises(ValueError, match="filter_file_ids"):
        await kb.aquery(
            "vector query",
            "db",
            filter_file_ids=[],
            raise_on_error=True,
        )

    assert collection.search_calls == []


async def test_aembed_texts_batches_and_validates_vectors():
    collection = FakeCollection()
    kb = make_kb(collection)
    kb.databases_meta["db"]["embed_info"] = {"dimension": "2"}
    calls: list[list[str]] = []

    async def encode(texts):
        calls.append(texts)
        return [[0.1, 0.2] for _ in texts]

    kb._get_async_embedding_function = lambda _embed_info: encode

    result = await kb.aembed_texts("db", ["query", "route query"])

    assert calls == [["query", "route query"]]
    assert result == [[0.1, 0.2], [0.1, 0.2]]


@pytest.mark.parametrize("embedding", [[0.1], [0.1, float("inf")], []])
async def test_precomputed_embedding_rejects_invalid_vectors_without_search(embedding):
    collection = FakeCollection()
    kb = make_kb(collection)
    kb.databases_meta["db"]["embed_info"] = {"dimension": 2}

    with pytest.raises(ValueError, match="query_embedding"):
        await kb.aquery(
            "vector query",
            "db",
            search_mode="vector",
            query_embedding=embedding,
            raise_on_error=True,
        )

    assert collection.search_calls == []
