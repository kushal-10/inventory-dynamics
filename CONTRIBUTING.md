Thank you for considering contributing to `idinn`! Whether you're here to report a bug, suggest an improvement, or contribute code, we appreciate your support.

## Seek Support or Report Issues

If you encounter any issues while using `idinn` or have questions that aren't covered by the documentation, feel free to open an issue in the [Issues section](https://gitlab.com/ComputationalScience/idinn/-/issues).

Please include as much relevant information as possible to help us reproduce and diagnose the problem. This might include:

- Operating system and platform details
- GPU usage (if relevant)
- Software versions (e.g., Python, dependencies)
- Error messages or logs
- Minimal code snippets

You’re also welcome to use the Issues section for general questions or feedback.

## Contribute Code

Want to contribute a new feature or fix a bug? Follow these steps:

1. **Fork** the [`idinn`](https://gitlab.com/ComputationalScience/idinn) repository on GitLab.
2. **Clone** your fork locally:
   ```bash
   git clone https://gitlab.com/<your-gitlab-username>/idinn
   cd idinn
   ```

3. **Set up the environment** using [`uv`](https://github.com/astral-sh/uv):

   ```bash
   uv sync --group dev --python 3.12
   ```
4. **Create a new branch** for your changes:

   ```bash
   git switch -c your-branch
   ```
5. **Implement your changes**, and write tests (if applicable) under the `tests/` directory.
6. **Run tests** using `pytest`:

   ```bash
   uv run pytest tests/
   ```
7. **Run type checks** using `mypy` under the `tests/` directory.:

   ```bash
   uv run mypy src/
   ```
8. **Commit and push** your changes:

   ```bash
   git add .
   git commit -m "Describe your changes"
   git push
   ```
9. **Open a Merge Request (MR)** to the `idinn` repository.

## Style Guide

### Docstrings

Use [NumPy-style docstrings](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard) for documenting functions, classes, and modules.

### Code Formatting

The code should follow [PEP 8](https://peps.python.org/pep-0008/) guidelines. Use [`ruff`](https://docs.astral.sh/ruff/) (with default settings) for linting and formatting:

```bash
uv run ruff check .
uv run ruff format .
```

## Documentation

Project documentation is located in the `docs/` directory. It is written in reStructuredText (`.rst`) and built with [Sphinx](https://www.sphinx-doc.org/en/master/). Documentation is hosted on [Read the Docs](https://about.readthedocs.com).

To build the docs locally:

```bash
cd docs
make html
```
