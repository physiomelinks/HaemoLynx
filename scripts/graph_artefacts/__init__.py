"""Rebuild a run's graph step by step on one or more code versions and measure
what each step leaves in a segmented vessel that is not a vessel.

See ``scripts/graph_artefacts.py`` for the command line. ``report`` holds the
measuring and the tables (unit-tested); ``build_side`` is the script each code
version is built by, in a process of its own.
"""
