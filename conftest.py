"""Root workspace conftest — deliberately holds no fixtures or hooks.

Its only job is to make ``<root>/conftest.py`` part of pytest's import
anchor set. With ``--import-mode=importlib`` (root ``pyproject.toml``
addopts) pytest derives a test module's import name from its path
relative to the anchor closest to it. That keeps the two packages'
identically-named modules (``tests/conftest.py``, and the many
``tests/test_*.py`` names both suites carry) in separate import
namespaces: ``packages.sr2.tests.*`` versus ``packages.spctr.tests.*``.

Without an anchor between the root and the packages, both
``packages/*/tests/conftest.py`` resolve to the same import name
``tests.conftest`` (each package ships ``tests/__init__.py``) and the
second registration aborts the session with "Plugin already registered
under a different name".

Do not add fixtures here: a root conftest applies to every test in both
packages.
"""
