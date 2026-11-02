Native API
==========

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

