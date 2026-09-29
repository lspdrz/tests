"""A Milvus server played by a local gRPC service that keeps rows, for an instance booted on Milvus.

`serving_milvus()` yields a `FakeMilvus` speaking the part of Milvus's gRPC API that pymilvus's
`MilvusClient` uses for Open WebUI's store: connect, collections (create, describe, list, drop,
load and its progress), indexes (create, describe), insert, upsert, delete, query (with the
paging a query iterator does) and vector search with cosine scores worked out in Python. Filter
expressions are the conjunctions Open WebUI writes: `field == value`, `field in [...]` and
`metadata['key'] == value`, joined by `and`.

`hold_collection(name, rows)` fills a collection with rows written as an admin's own data (a string
`id`, float lists as vectors and dicts as JSON columns, named as the admin likes), the way external
knowledge finds it, and `searches` lists every vector search as `(collection, vector field, limit,
output fields)`. A search on a column the collection lacks or with a vector of the wrong size is
refused, as Milvus does.

`index_requests` lists every `(collection, field, index type)` the server was asked to build;
an empty type is a request that leaves the choice to the server. `index_settings` holds the
parameters of the last index asked for on each `(collection, field)` (`index_type`,
`metric_type` and the JSON `params`), and `call_metadata` the gRPC metadata of every call, where
the client's credentials (`authorization`) and database (`dbname`) travel. Setting
`refuse_untyped_scalar_index` makes the server refuse a scalar index without a type, the way a
Milvus Lite or an older standalone server does. `milvus_env(fake, multitenancy)` is the
environment of an instance that stores its vectors there.
"""

from __future__ import annotations

import ast
import contextlib
import json
import math
import operator
import re
import threading
from concurrent import futures
from dataclasses import dataclass, field
from typing import Iterator

import grpc
from pymilvus.grpc_gen import common_pb2, milvus_pb2, milvus_pb2_grpc, schema_pb2

VECTOR_TYPES = {schema_pb2.FloatVector}
ORDERINGS = {">": operator.gt, "<": operator.lt, ">=": operator.ge, "<=": operator.le}
CLAUSE = re.compile(
    r"^\s*(?P<field>\w+)(?:\[\s*['\"](?P<key>[^'\"]+)['\"]\s*\])?\s*"
    r"(?P<op>==|!=|>=|<=|>|<|\bin\b)\s*(?P<value>.+?)\s*$",
    re.DOTALL,
)


def _ok() -> common_pb2.Status:
    return common_pb2.Status(error_code=common_pb2.Success, code=0)


def _failed(reason: str, code: int = 1100) -> common_pb2.Status:
    return common_pb2.Status(error_code=common_pb2.UnexpectedError, code=code, reason=reason)


def _not_found(name: str) -> common_pb2.Status:
    return common_pb2.Status(
        error_code=common_pb2.CollectionNotExists, code=100, reason=f"collection not found[{name}]"
    )


def _split_conjunction(expression: str) -> list[str]:
    """Split on top-level ` and `, leaving quoted strings and brackets alone."""
    parts, current, quote, depth = [], "", None, 0
    index = 0
    while index < len(expression):
        char = expression[index]
        if quote:
            current += char
            if char == "\\" and index + 1 < len(expression):
                current += expression[index + 1]
                index += 1
            elif char == quote:
                quote = None
        elif char in "'\"":
            quote = char
            current += char
        elif char == "[":
            depth += 1
            current += char
        elif char == "]":
            depth -= 1
            current += char
        elif depth == 0 and expression[index : index + 5].lower() == " and ":
            parts.append(current)
            current = ""
            index += 4
        else:
            current += char
        index += 1
    parts.append(current)
    return [part.strip().strip("()").strip() for part in parts if part.strip()]


def _literal(text: str):
    text = re.sub(r"\btrue\b", "True", re.sub(r"\bfalse\b", "False", text))
    return ast.literal_eval(text)


def matches(row: dict, expression: str) -> bool:
    for clause in _split_conjunction(expression or ""):
        parsed = CLAUSE.match(clause)
        if not parsed:
            raise ValueError(f"cannot parse filter clause {clause!r}")
        value = row.get(parsed["field"])
        if parsed["key"] is not None:
            value = (value or {}).get(parsed["key"]) if isinstance(value, dict) else None
        expected = _literal(parsed["value"])
        comparison = parsed["op"]
        if comparison == "in":
            if value not in expected:
                return False
        elif comparison == "==" and value != expected:
            return False
        elif comparison == "!=" and value == expected:
            return False
        elif comparison in ORDERINGS:
            if value is None or not ORDERINGS[comparison](value, expected):
                return False
    return True


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    norms = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return dot / norms if norms else 0.0


@dataclass
class Collection:
    schema: schema_pb2.CollectionSchema
    rows: dict[str, dict] = field(default_factory=dict)
    indexes: dict[str, str] = field(default_factory=dict)

    @property
    def primary(self) -> str:
        return next(field.name for field in self.schema.fields if field.is_primary_key)

    def field_type(self, name: str) -> int:
        return next(field.data_type for field in self.schema.fields if field.name == name)


def _rows_from(fields_data, num_rows: int) -> list[dict]:
    rows = [{} for _ in range(num_rows)]
    for column in fields_data:
        if column.type == schema_pb2.FloatVector:
            dim = column.vectors.dim
            values = list(column.vectors.float_vector.data)
            cells = [values[index * dim : (index + 1) * dim] for index in range(num_rows)]
        elif column.type == schema_pb2.JSON:
            cells = [json.loads(raw) for raw in column.scalars.json_data.data]
        elif column.type in (schema_pb2.VarChar, schema_pb2.String):
            cells = list(column.scalars.string_data.data)
        elif column.type in (schema_pb2.Int64, schema_pb2.Int32):
            cells = list(column.scalars.long_data.data or column.scalars.int_data.data)
        else:
            raise ValueError(f"field type {column.type} is not played by the fake")
        for row, cell in zip(rows, cells):
            row[column.field_name] = cell
    return rows


def _column(collection: Collection, name: str, rows: list[dict]) -> schema_pb2.FieldData:
    kind = collection.field_type(name)
    column = schema_pb2.FieldData(type=kind, field_name=name)
    cells = [row.get(name) for row in rows]
    if kind == schema_pb2.FloatVector:
        dim = len(cells[0]) if cells else 0
        column.vectors.dim = dim
        column.vectors.float_vector.data.extend(value for cell in cells for value in cell)
    elif kind == schema_pb2.JSON:
        column.scalars.json_data.data.extend(json.dumps(cell).encode() for cell in cells)
    elif kind in (schema_pb2.VarChar, schema_pb2.String):
        column.scalars.string_data.data.extend(cells)
    else:
        column.scalars.long_data.data.extend(cells)
    return column


def _params(pairs) -> dict[str, str]:
    return {pair.key: pair.value for pair in pairs}


class FakeMilvus(milvus_pb2_grpc.MilvusServiceServicer):
    def __init__(self) -> None:
        self.collections: dict[str, Collection] = {}
        self.index_requests: list[tuple[str, str, str]] = []
        self.index_settings: dict[tuple[str, str], dict[str, str]] = {}
        self.call_metadata: list[dict[str, str]] = []
        self.searches: list[tuple[str, str, int, list[str]]] = []
        self.refuse_untyped_scalar_index = False
        self.address = ""
        self.lock = threading.Lock()

    def hold_collection(self, name: str, rows: list[dict]) -> None:
        """Hold `name` with `rows`: a string `id`, float lists for vectors and dicts for JSON."""
        kinds = {str: schema_pb2.VarChar, list: schema_pb2.FloatVector, dict: schema_pb2.JSON}
        fields = [
            schema_pb2.FieldSchema(
                name=column, data_type=kinds[type(cell)], is_primary_key=column == "id"
            )
            for column, cell in rows[0].items()
        ]
        collection = Collection(schema=schema_pb2.CollectionSchema(name=name, fields=fields))
        collection.rows = {row["id"]: row for row in rows}
        with self.lock:
            self.collections[name] = collection

    # --- connection ------------------------------------------------------------------------

    def Connect(self, request, context):
        info = common_pb2.ServerInfo(build_tags="v2.6.0-fake")
        return milvus_pb2.ConnectResponse(status=_ok(), server_info=info, identifier=1)

    def GetVersion(self, request, context):
        return milvus_pb2.GetVersionResponse(status=_ok(), version="v2.6.0")

    def AllocTimestamp(self, request, context):
        return milvus_pb2.AllocTimestampResponse(status=_ok(), timestamp=1)

    # --- collections -----------------------------------------------------------------------

    def CreateCollection(self, request, context):
        schema = schema_pb2.CollectionSchema()
        schema.ParseFromString(request.schema)
        with self.lock:
            self.collections[request.collection_name] = Collection(schema=schema)
        return _ok()

    def DescribeCollection(self, request, context):
        with self.lock:
            collection = self.collections.get(request.collection_name)
        if collection is None:
            return milvus_pb2.DescribeCollectionResponse(status=_not_found(request.collection_name))
        return milvus_pb2.DescribeCollectionResponse(
            status=_ok(),
            schema=collection.schema,
            collectionID=abs(hash(request.collection_name)) % 10**9,
            collection_name=request.collection_name,
            consistency_level=common_pb2.Bounded,
            shards_num=1,
        )

    def HasCollection(self, request, context):
        with self.lock:
            exists = request.collection_name in self.collections
        return milvus_pb2.BoolResponse(status=_ok(), value=exists)

    def ShowCollections(self, request, context):
        with self.lock:
            names = list(self.collections)
        return milvus_pb2.ShowCollectionsResponse(status=_ok(), collection_names=names)

    def DropCollection(self, request, context):
        with self.lock:
            self.collections.pop(request.collection_name, None)
        return _ok()

    def LoadCollection(self, request, context):
        return _ok()

    def GetLoadingProgress(self, request, context):
        return milvus_pb2.GetLoadingProgressResponse(status=_ok(), progress=100)

    def GetLoadState(self, request, context):
        return milvus_pb2.GetLoadStateResponse(status=_ok(), state=common_pb2.LoadStateLoaded)

    def Flush(self, request, context):
        return milvus_pb2.FlushResponse(status=_ok())

    # --- indexes ---------------------------------------------------------------------------

    def CreateIndex(self, request, context):
        settings = _params(request.extra_params)
        index_type = settings.get("index_type", "")
        with self.lock:
            collection = self.collections.get(request.collection_name)
            self.index_requests.append((request.collection_name, request.field_name, index_type))
            self.index_settings[(request.collection_name, request.field_name)] = settings
        if collection is None:
            return _not_found(request.collection_name)
        is_scalar = collection.field_type(request.field_name) not in VECTOR_TYPES
        if self.refuse_untyped_scalar_index and is_scalar and index_type in ("", "AUTOINDEX"):
            return _failed(f"index type not supported for scalar field {request.field_name}")
        with self.lock:
            collection.indexes[request.field_name] = index_type or "AUTOINDEX"
        return _ok()

    def DescribeIndex(self, request, context):
        with self.lock:
            collection = self.collections.get(request.collection_name)
        if collection is None:
            return milvus_pb2.DescribeIndexResponse(status=_not_found(request.collection_name))
        descriptions = [
            milvus_pb2.IndexDescription(
                index_name=name,
                field_name=name,
                state=common_pb2.Finished,
                params=[common_pb2.KeyValuePair(key="index_type", value=index_type)],
            )
            for name, index_type in collection.indexes.items()
            if not request.index_name or request.index_name == name
        ]
        if not descriptions:
            return milvus_pb2.DescribeIndexResponse(
                status=common_pb2.Status(
                    error_code=common_pb2.IndexNotExist, code=700, reason="index not found"
                )
            )
        return milvus_pb2.DescribeIndexResponse(status=_ok(), index_descriptions=descriptions)

    def GetIndexState(self, request, context):
        return milvus_pb2.GetIndexStateResponse(status=_ok(), state=common_pb2.Finished)

    # --- rows ------------------------------------------------------------------------------

    def _write(self, request, context):
        with self.lock:
            collection = self.collections.get(request.collection_name)
        if collection is None:
            return None, [], milvus_pb2.MutationResult(status=_not_found(request.collection_name))
        rows = _rows_from(request.fields_data, request.num_rows)
        with self.lock:
            for row in rows:
                collection.rows[row[collection.primary]] = row
        ids = schema_pb2.IDs(
            str_id=schema_pb2.StringArray(data=[row[collection.primary] for row in rows])
        )
        return collection, rows, ids

    def Insert(self, request, context):
        collection, rows, ids = self._write(request, context)
        if collection is None:
            return ids
        return milvus_pb2.MutationResult(status=_ok(), IDs=ids, insert_cnt=len(rows))

    def Upsert(self, request, context):
        collection, rows, ids = self._write(request, context)
        if collection is None:
            return ids
        return milvus_pb2.MutationResult(status=_ok(), IDs=ids, upsert_cnt=len(rows))

    def Delete(self, request, context):
        with self.lock:
            collection = self.collections.get(request.collection_name)
            if collection is None:
                return milvus_pb2.MutationResult(status=_not_found(request.collection_name))
            doomed = [key for key, row in collection.rows.items() if matches(row, request.expr)]
            for key in doomed:
                del collection.rows[key]
        return milvus_pb2.MutationResult(status=_ok(), delete_cnt=len(doomed))

    def Query(self, request, context):
        with self.lock:
            collection = self.collections.get(request.collection_name)
            rows = list(collection.rows.values()) if collection else []
        if collection is None:
            return milvus_pb2.QueryResults(status=_not_found(request.collection_name))
        params = _params(request.query_params)
        found = [row for row in rows if matches(row, request.expr)]
        found.sort(key=lambda row: row[collection.primary])
        if params.get("iterator") == "True" or "iterator" in params:
            # an iterator pages by primary key, strictly after the last one it saw
            last = re.search(r"\w+\s*>\s*['\"](.*?)['\"]", request.expr or "")
            if last:
                found = [row for row in found if row[collection.primary] > last.group(1)]
        offset = int(params.get("offset", 0) or 0)
        limit = int(params.get("limit", 0) or 0)
        found = found[offset : offset + limit] if limit > 0 else found[offset:]
        names = [name for name in request.output_fields if name != "count(*)"] or [
            collection.primary
        ]
        if collection.primary not in names:
            names.append(collection.primary)
        return milvus_pb2.QueryResults(
            status=_ok(),
            collection_name=request.collection_name,
            output_fields=names,
            fields_data=[_column(collection, name, found) for name in names],
            primary_field_name=collection.primary,
        )

    def Search(self, request, context):
        with self.lock:
            collection = self.collections.get(request.collection_name)
            rows = list(collection.rows.values()) if collection else []
        if collection is None:
            return milvus_pb2.SearchResults(status=_not_found(request.collection_name))
        placeholders = common_pb2.PlaceholderGroup()
        placeholders.ParseFromString(request.placeholder_group)
        vectors = [list(memoryview(raw).cast("f")) for raw in placeholders.placeholders[0].values]
        params = _params(request.search_params)
        top_k = int(params.get("topk", params.get("limit", 10)))
        vector_field = params.get("anns_field") or next(
            field.name for field in collection.schema.fields if field.data_type in VECTOR_TYPES
        )
        columns = {column.name: column for column in collection.schema.fields}
        missing = [name for name in [vector_field, *request.output_fields] if name not in columns]
        if missing:
            return milvus_pb2.SearchResults(status=_failed(f"field not found: {missing[0]}"))
        dims = {len(row[vector_field]) for row in rows}
        if any(len(vector) not in dims for vector in vectors) and dims:
            return milvus_pb2.SearchResults(status=_failed("vector dimension mismatch"))
        candidates = [row for row in rows if matches(row, request.dsl)]
        with self.lock:
            self.searches.append(
                (request.collection_name, vector_field, top_k, list(request.output_fields))
            )
        names = [name for name in request.output_fields] or [collection.primary]
        result = schema_pb2.SearchResultData(num_queries=len(vectors), top_k=top_k)
        hits_all: list[dict] = []
        for vector in vectors:
            ranked = sorted(
                candidates, key=lambda row: _cosine(vector, row[vector_field]), reverse=True
            )[:top_k]
            result.topks.append(len(ranked))
            result.scores.extend(_cosine(vector, row[vector_field]) for row in ranked)
            result.ids.str_id.data.extend(row[collection.primary] for row in ranked)
            hits_all.extend(ranked)
        result.output_fields.extend(names)
        result.fields_data.extend(_column(collection, name, hits_all) for name in names)
        return milvus_pb2.SearchResults(
            status=_ok(), results=result, collection_name=request.collection_name
        )


class _MetadataRecorder(grpc.ServerInterceptor):
    def __init__(self, fake: FakeMilvus) -> None:
        self.fake = fake

    def intercept_service(self, continuation, handler_call_details):
        with self.fake.lock:
            self.fake.call_metadata.append(dict(handler_call_details.invocation_metadata))
        return continuation(handler_call_details)


@contextlib.contextmanager
def serving_milvus() -> Iterator[FakeMilvus]:
    fake = FakeMilvus()
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=16), interceptors=[_MetadataRecorder(fake)]
    )
    milvus_pb2_grpc.add_MilvusServiceServicer_to_server(fake, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    fake.address = f"http://127.0.0.1:{port}"
    try:
        yield fake
    finally:
        server.stop(grace=None)


def milvus_env(fake: FakeMilvus, multitenancy: bool) -> dict[str, str]:
    """The environment of an instance whose vectors live in `fake`."""
    return {
        "VECTOR_DB": "milvus",
        "MILVUS_URI": fake.address,
        "ENABLE_MILVUS_MULTITENANCY_MODE": "true" if multitenancy else "false",
    }
