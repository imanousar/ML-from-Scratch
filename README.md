## Configs for libraries

```python
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt
pip install -e .
```

## Configs for docs

```
pip install mkdocs
pip install mkdocs-material
pip install mkdocstrings[python]
cd mlfromscratch
```

<b>Test locally</b>

```
mkdocs serve
```

<b>Deploy</b>

```
mkdocs gh-deploy
```

Test docs

```
pytest mlfromscratch/tests/test_docs.py --verbose"
```
