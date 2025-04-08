#!/usr/bin/env bash
pip install -e .
pip install jupyterlab
jupyter lab --NotebookApp.token='idinn-pass' --no-browser --port=8888 --allow-root --ip='0.0.0.0'