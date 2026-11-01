Execution and ownership
=======================

Lifetime and concurrency
------------------------

Construct owners with ``Database.create`` or ``Database.open`` and close them
explicitly or through a context manager. A collection stores an owner and its
collection identity. Closing the owner releases its native database even when
collections remain; their later calls raise ``ClosedError``. Previously
materialized results remain usable and can be mutated without changing stored
data.

Native work releases the GIL. Input validation and Python result construction
hold it. Each owner admits one operation at a time; a concurrent or reentrant
call, including close, raises ``BusyError`` instead of queuing. A caught native
panic hides its payload and poisons the owner; only safe closing is then
allowed. These controls do not make arbitrary host extensions safe.

Fork inheritance is refused before inherited native locks are touched.
Finalization of an inherited owner does not unlock its parent's writer. A
child may create a fresh owner, subject to the actual file locks; it cannot
bypass a live parent writer. Prefer spawn or close before fork when practical.

Configuration and inputs
------------------------

Paths require built-in UTF-8-encodable str, with an explicit ``str(path)`` for a
Path object. Configurations and records require built-in dict. Unknown schema
keys are rejected. Batches and vectors require built-in list/tuple; subclasses,
generators and arbitrary conversion protocols are not silently invoked.

``Database.create(path, config=None)`` and
``Database.open(path, options=None)`` treat the whole mapping as omitted.
For fields inside a provided mapping, omission and explicit ``None`` follow
the individual field's rules. A create config permits nullable
``cache_budget_mb`` only; open options permit nullable
``resource_profile``, ``cache_budget_mb``, ``background_compaction`` and
``durability`` overrides. ``io_mode`` cannot be None. Collection tokenizer
omission selects the complete core default identity; an explicit tokenizer
must be that complete legal object, not None. Zero cache and explicit false
remain explicit values.

With no runtime overrides, create/open use ``resource_profile="balanced"``,
``cache_budget_mb=None`` (the selected preset's budget),
``background_compaction=False``, ``durability="commit"`` and ``io_mode="auto"``.
Nullable open overrides use these runtime defaults; they do not change stored
collection configuration. An explicit cache budget of zero stays zero.

For ``create_collection``, omitted ``embedding_model`` means ``None`` (no
explicit model setting). The core still derives and enforces the effective
model binding from committed embeddings. ``vectors_normalized`` defaults to
``False`` and ``vector_index`` to ``"exact"``. Omitted ``tokenizer`` selects the complete
``default_en`` identity; an explicit tokenizer must provide that whole legal
object. Only ``embedding_model`` accepts
``None`` among these optional collection fields.

Atomicity, signals and progress
-------------------------------

``write_batch`` fixes and validates all three input groups before a single
atomic commit. It never splits an oversized call; an empty batch does not
write. Manifest import validates its fixed inputs before bounded commits and
does not promise whole-import atomicity. Export uses one visible collection
view, its core safeguards and a reimportable manifest.

There is no forced cancellation or per-call timeout. A pending Python signal
may be raised only after native work has finished or committed. Known commit
facts are captured before Python result allocation and signal handling;
``KeyboardInterrupt`` preserves an original core error as its cause when both
occur. A write failure's ``operation_outcome`` distinguishes not_started,
not_committed, partially_committed, committed and unknown. Only confirmed
fields are present; counts are core records and may include collection and
tombstone records. Unknown does not mean rolled back. Automatic write replay
is not safe.

Resource and retrieval boundaries
---------------------------------

One call uses one core aggregate conversion/execution/materialization budget.
The existing business JSON size/depth limits do not become an extra cap on a
full text, an entire batch or a full result. Working-capacity rejection remains
possible for finite inputs. Python objects retained by the host after handoff,
Python heap, native allocations and process RSS are separate observations;
cache quota is not an RSS guarantee.

Search uses the shared core's defaults, filtering, exact vector scan,
normalization, weighted fusion, ordering and stable explain. External vector
semantics and user-selected weights affect application quality; no fixed
application recall is guaranteed. A component score is neither a relevance
probability nor a causal recall contribution. Increasing top_k may change
candidate normalization and ranking.

One complete content read obtains the chunk and its document from one view.
Search and a subsequent get are separate calls, so no cross-call transaction
is implied. Readers use their core fixed view. The SDK does not add a snapshot
or transaction API, encryption, provider service, telemetry or secure-erasure
guarantee.
