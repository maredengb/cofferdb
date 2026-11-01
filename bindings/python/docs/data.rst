Input and result types
======================

The exported aliases and ``TypedDict`` classes describe accepted inputs and
owned core projections. They do not construct validated native records or
coerce an arbitrary mapping/iterator. ``bool`` is rejected for numeric inputs
even though Python's static type system treats it as a subtype of ``int``.

``Profile`` has only ``expanded``, ``balanced`` and ``constrained``.
``Conflict`` has only ``error``, ``skip`` and ``replace``. ``BatchSeq[T]`` is a
built-in list or tuple, and ``Vector`` contains finite built-in int/float
values that can enter the core's f32 vector representation. Ordinary JSON
integers must enter canonical JSON without silent rounding.

.. py:type:: JsonValue

   The recursive JSON union exported as ``cofferdb.types.JsonValue``: None,
   bool, finite int/float, str, a built-in list or tuple of JSON values, or a
   str-keyed dict of JSON values. Tuples are accepted as input arrays; returned
   arrays are independently owned lists. This is a type alias, not a validated
   native object constructor.

.. automodule:: cofferdb.types
   :members:
   :undoc-members:
   :member-order: bysource

Numerical results retain the core projection. ``Number`` permits int or float;
for example an integral floating score may be materialized as a Python int.
This does not round scores. Source byte offsets use their dedicated exact
int/decimal-string representation through the full u64 range.

