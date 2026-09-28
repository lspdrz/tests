# Per-dependency contract tests (`unit/deps/`)

One `test_<dep>.py` per third-party dependency, pinning **exactly the API
surface and behaviour Open WebUI relies on** from that package. Run inside an
environment with a bumped version installed, these catch the "new release
removed / renamed / changed an API we use" class of breakage — the gap our
other suites can't see, because they run against whatever is already installed.

## The contract (so files compose and authors don't collide)

- **One file owns one dependency.** Never edit another dep's file, `conftest.py`,
  or `pyproject.toml`.
- **Shared machinery is fixtures, not imports.** Use the `depcheck` fixture
  (see `conftest.py`) and `open_webui_backend` (from `../conftest.py`). No
  cross-file imports — that keeps dozens of independently-authored modules from
  fighting over sys.path.
- **Marker:** `pytestmark = pytest.mark.depcheck`.
- **Skip, don't fail, when absent:** `depcheck.load(import_name)` skips the test
  if the package isn't importable, so the suite runs anywhere.
- **Deterministic + offline.** No real network, DB, redis, or model downloads —
  use in-memory fakes / mock transports, or assert the API surface instead.

## What a good file contains

1. **Symbol existence** — `depcheck.assert_symbols(mod, [...])` for every symbol
   the backend references (dotted paths like `"hazmat.primitives.hashes.SHA256"`).
2. **Signatures** — `depcheck.assert_params(fn, [...])` for functions called with
   specific kwargs (this is what catches the #24560-class "kwarg dropped" bug).
3. **Behavioural contracts** — exercise the *actual* usage offline (sign+verify a
   token, `chardet.detect()` a known byte string, a MockTransport HTTP roundtrip).

`test_cryptography.py` is a reference exemplar.

## `depcheck` API

`load` / `try_load` / `has` / `resolve` / `assert_symbols` / `assert_callable` /
`assert_params` / `dist_version`. See `conftest.py`.

## Gotchas

- Don't `hasattr()` an object whose attribute is a property that executes —
  use `dir(instance)` / class introspection (a blank object's getter can raise).
- Rust-backed classes (cryptography AEAD, etc.) have no introspectable
  signature — pin them behaviourally, not with `assert_params`.

## Running

```bash
pytest unit/deps/                    # all dependency contracts
pytest unit/deps/test_fastapi.py     # one dependency
pytest -m depcheck                   # the whole class, anywhere
```

## Feature smoke tests (`integration/deps/`)

The contracts here pin a library's API. `integration/deps/` drives the same
libraries through the Open WebUI feature that uses them, over HTTP on a
scratch instance, so a bump that keeps the API but changes the behaviour still
fails. Both carry the `depcheck` marker; run both after a bump:

```bash
OPEN_WEBUI_SOURCE_DIR=../open-webui/backend pytest -m depcheck unit/deps integration/deps
```

| Library | Feature smoke test |
|---|---|
| pypdf, docx2txt, unstructured with python-pptx, pandas on pyarrow, openpyxl, xlrd and msoffcrypto, pypandoc, beautifulsoup4, chardet, ftfy | `integration/deps/test_document_extraction.py` (one upload per format, legacy Word, PowerPoint and Excel files, an Outlook message, PDF pages, labels, metadata and locks, pandoc missing, and a password-protected workbook) and `e2e/retrieval/test_attached_document_warnings.py` |
| azure-ai-documentintelligence | `integration/deps/test_document_extraction.py` (a PDF read by a local stand-in of the analyze API) |
| rapidocr with onnxruntime, OpenCV and Pillow | `integration/deps/test_document_extraction.py` (PDF image OCR) |
| pandas, openpyxl, xlrd and python-pptx without unstructured | `integration/deps/test_document_extraction.py` (an instance installed without the extra) |
| langchain-text-splitters, langchain-core, langchain-classic, tiktoken, transformers (`AutoTokenizer`), rank-bm25 | `integration/deps/test_chunking_and_search.py` |
| transformers, datasets, sentencepiece, soundfile (local text-to-speech on a tiny voice) | `integration/deps/test_local_text_to_speech.py`, `e2e/audio/test_local_read_aloud.py` |
| langchain-core | `integration/deps/test_tool_specs.py` (a workspace tool's spec) |
| Pillow | `integration/deps/test_image_validation.py` (model backgrounds, linked images, image edits) |
| bcrypt, argon2-cffi, PyJWT, pytz, authlib, itsdangerous, cryptography | `integration/deps/test_auth_stack.py` (and the back-channel logout token in `integration/auth/test_sso_account_sync.py`) |
| starlette-compress, Brotli, zstandard, Markdown, beautifulsoup4, brotlicffi, python-socketio (websocket and long polling), pycrdt | `integration/deps/test_transport_stack.py`, `e2e/chat/test_long_polling_chat.py` (python-socketio's event calls, disconnects and rooms also in `integration/chat/test_socket_runtime.py` and `integration/security/test_revoked_access_leaves_live_rooms.py`) |
| python-mimeparse, aiofiles, pydub, av, faster-whisper (local Whisper on a tiny model), soundfile (local SpeechT5 speech) | `integration/deps/test_audio_stack.py`, `e2e/audio/test_local_whisper_dictation.py` |
| pydantic, python-multipart | `integration/deps/test_request_bodies.py` |
| mcp, httpx, validators, black, beautifulsoup4, opentelemetry, requests, googleapis-common-protos | `integration/deps/test_outbound_stack.py` |
| opentelemetry (traces, metrics, logs over OTLP/HTTP and gRPC), psutil | `integration/deps/test_telemetry_export.py` |
| playwright (a remote browser server) | `integration/deps/test_browser_page_loader.py` |
| loguru | `integration/deps/test_logging.py` (the server log and the audit log) |
| ldap3 | `integration/auth/test_ldap_sign_in.py` (LDAP and LDAPS sign-in) |
| boto3 (S3), azure-storage-blob, azure-identity, google-cloud-storage | `integration/deps/test_object_storage.py` (uploads kept in a bucket or container) |
| azure-identity, azure-search-documents | `integration/deps/test_azure_services.py` (Entra ID sign-in to Azure OpenAI, Azure AI Search) |
| chromadb (embedded and server), pgvector with psycopg2, opensearch-py, pinecone, pymilvus (both layouts), qdrant-client (REST and gRPC, both storage modes), boto3 (S3 Vectors) | `integration/deps/test_vector_stores.py` |
| sentence-transformers (local embedding, tokenizer and `CrossEncoder` reranking) | `integration/deps/test_local_models.py` |
| sentencepiece | `integration/deps/test_chunking_and_search.py` (the transformers splitter on a `spiece.model`) |
| pyxlsb | `integration/deps/test_document_extraction.py` (a binary workbook) |
| weaviate-client | `integration/deps/test_weaviate_store.py` |
| elasticsearch | `integration/deps/test_elasticsearch_store.py` |
| ddgs, fake-useragent | `integration/deps/test_web_search_stack.py`, `e2e/retrieval/test_duckduckgo_web_search.py` |
| fastapi | `integration/deps/test_web_framework.py` and every instance boot and request |
| redis (with python-socketio's Redis manager) | `integration/deps/test_redis_stack.py` (two instances on one Redis, readiness, TLS), `integration/chat/test_redis_task_commands.py` (pub/sub, restart, Cluster), `integration/chat/test_socket_cleanup_locks.py`, `integration/resilience/test_redis_sentinel_failover.py` |
| aiocache | `integration/security/test_cache_key_builder.py` (a repeat model listing never reaches the provider) and `integration/deps/test_connection_stack.py` (until the TTL runs out) |
| aiohttp, aiodns | `integration/deps/test_connection_stack.py` (the model list timeout, a provider reached by name through c-ares) |
| aiosqlite, psycopg, psycopg2 | `integration/deps/test_database_stack.py` |
| alembic | `integration/migrations/test_lifecycle.py` |
| requests | `integration/retrieval/test_web_loaders.py` (a linked document fetched through the SSRF-guarded session), `integration/security/test_remote_file_download_limits.py` (streamed under the size limit), `integration/retrieval/test_v0114_source_text_and_docling.py` (Tika) |
| sqlalchemy | `integration/deps/test_database_stack.py` (JSON columns, search, counts and pages, SQLite PRAGMAs) and every request |
| starsessions | `integration/deps/test_server_sessions.py`, `e2e/security/test_sso_sign_in_on_redis_sessions.py` (SSO sign-in state in Redis) |
| uvicorn | `integration/deps/test_server_launch.py` (`open-webui serve` and `dev`), `integration/config/test_start_sh.py` |
| youtube-transcript-api | `integration/deps/test_video_transcripts.py`, `integration/retrieval/test_youtube_transcripts.py` |
| httpx | every instance boot and request |

The .rst, .epub and .odt uploads skip without a `pandoc` binary (the test of
the missing-pandoc message skips with one), legacy .doc and .ppt without
LibreOffice (`soffice`), the pydub tests skip without ffmpeg (the conversion
test also without ffprobe), and the token splitter skips when its tiktoken BPE
file is not cached, since none of them may be downloaded during a run. The
local Whisper tests build their model with torch, transformers and CTranslate2
and skip without them. Libraries for services the integration suite has no
local stand-in for (Oracle) keep their unit contracts only, as do libraries no
Open WebUI feature calls (accelerate, the anthropic SDK). The openai SDK is
pinned but Open WebUI never calls it (every provider request is its own), so
it has no contract; nor has PyMySQL, which nothing imports and whose MySQL URL
the async database engine cannot use (Open WebUI supports SQLite and
Postgres). pyarrow is reached through pandas, PyJWT keeps its sweep over the
source and python-multipart its pinned security floor. A contract kept next to
its feature smoke test says in its docstring which part no request reaches.
The Postgres cases need `pgserver`, whose vector extension predates pgvector's
`halfvec` and whose server has no SSL, so those two paths keep a unit contract
as well.
