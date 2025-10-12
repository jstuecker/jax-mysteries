# This repository
Is aimed to clarify some low-level aspects of jax's just-in-time compilation `jax.jit`. This is only of interest for you if you have already a good understanding of `jax` and `jax.jit` and want to get the most out of it.

# Installation of utils
```bash
pip install -e .
```
Dependencies are not properly declared, so you will have to install them yourself if some import failes

## to create HTML or markdown:
```bash
cd notebooks
jupyter nbconvert --to html jax_mysteries.ipynb
jupyter nbconvert --to markdown jax_mysteries.ipynb
```