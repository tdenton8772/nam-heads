"""Head implementations, one package per ``<slot>/<name>``.

Each holds ``head.py`` with one ``EncoderHead`` subclass and a literal
``IMPL_META``, and is registered under the ``nam.heads`` entry-point group in
pyproject.toml. A host runs one only when its slot selects it.
"""
