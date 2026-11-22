Native API
==========

File checks
-----------

.. autofunction:: cofferdb.check

A completed scan returns a :class:`cofferdb.types.CheckReport`, including
any issues and the reason for skipped index checks. The scan releases the
GIL and opens the existing file read-only. If it cannot establish a view,
inspect the structured :class:`cofferdb.CofferError` before retrying.
Counters above 2**53-1 use exact decimal strings.

Database
--------

.. autoclass:: cofferdb.Database
   :members:
   :undoc-members:
   :special-members: __enter__, __exit__

Collection
----------

.. autoclass:: cofferdb.Collection
   :members:
   :undoc-members:

Errors
------

.. autoexception:: cofferdb.CofferError

Core errors preserve ``code``, ``message``, ``detail``, ``recoverable`` and
nullable ``suggested_action``. Their namespace is separate from SDK lifecycle
and conversion failures. Python type/value errors and real ``MemoryError``
remain Python exceptions rather than invented engine error codes.

.. autoexception:: cofferdb.SdkError
.. autoexception:: cofferdb.ClosedError
.. autoexception:: cofferdb.BusyError
.. autoexception:: cofferdb.ForkedHandleError
.. autoexception:: cofferdb.PoisonedHandleError
.. autoexception:: cofferdb.EnginePanicError
.. autoexception:: cofferdb.ResultMaterializationError

Known write progress is attached as ``operation_outcome`` to a raised exception
when available. It is not an attribute guaranteed on every arbitrary Python
exception. See :class:`cofferdb.types.OperationOutcome` and :doc:`contracts`.


Writer maintenance
------------------

Database.snapshot returns a SnapshotReport for a new standalone file.
Database.rebuild_index reports the actual scope and persisted checkpoint.
Database.compact reports file sizes and index handling after replacement.
A signal or conversion error after native work can carry snapshot_outcome
or maintenance_outcome. Verify the named backup by size, SHA-256 and commit.
After an uncertain replacement, reopen the main path and check its contents.
