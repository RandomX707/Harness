# Identity

This agent writes Python code in `src/` and `tests/` only.

# Constraints

- Never touch `pyproject.toml`, `.harness/`, or any dotfile.
- Never run `pip install` outside a virtual environment.
- Never use `os.system()`.

# Tool scope

- `read_file`: unrestricted within project.
- `write_file`: `src/` and `tests/` only.
- `run_code`: sandboxed subprocess with 10s timeout.

# Verification required

Run `python -m pytest tests/` before marking task done. Retry up to 3 times on failure. Surface the error and stop if still failing after 3.

# Known failure patterns

- Always use `pathlib.Path` not `os.path`.
- Never import from `src/legacy/` which does not exist.
