CofferDB Python reference
=========================

This reference imports the installed native package. Its SDK version is shown
in the page title; its actually linked Rust engine version is |engine_version|.
A generated reference does not imply that every platform or release acceptance
check has passed.

The synchronous extension runs the shared Rust engine in the Python process.
There is no CLI subprocess, server, model provider or automatic vector
generation. Public inputs are built-in Python objects; returned dictionaries
and lists have independent ownership.

Package identity
----------------

.. autodata:: cofferdb.__version__
.. autodata:: cofferdb.engine_version

.. toctree::
   :maxdepth: 2

   api
   data
   contracts

Build and offline use
---------------------

Install the wheel whose API you intend to document, together with the pinned
documentation dependencies, then run from the repository root::

   python -m sphinx -W --keep-going -b html bindings/python/docs /tmp/cofferdb-python-reference

The build never inserts a source package into ``sys.path`` to mask a missing
native extension. Open the output ``index.html`` locally. Distributing the whole
output directory preserves its relative navigation, search assets and styles.
The source includes ``__init__.pyi`` and ``py.typed`` for static tooling; the
reference signatures and native docstrings below come from the installed API.

