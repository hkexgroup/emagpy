#!/bin/bash
# build the python package as well
python setup.py sdist bdist_wheel
#python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*
#python -m twine upload dist/*

