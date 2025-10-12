# installation of utils
```bash
pip install -e .
```
Dependencies are not properly declared, so you will have to install them yourselves if some import failes

## to create HTML:
```bash
cd notebooks
jupyter nbconvert --to html jax_mysteries.ipynb
```