#!/usr/bin/env bash
pip install -e .
pip install coverage pytest
python -m coverage run -m pytest && python -m coverage report