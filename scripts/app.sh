#!/usr/bin/env bash
pip install -e .
pip install -r app/requirements.txt
streamlit run app/app.py --server.port=8501 --server.address=0.0.0.0