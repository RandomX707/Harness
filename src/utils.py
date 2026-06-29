"""Small string utilities created by the harness demo."""

def normalize_whitespace(value: str) -> str:
    """Collapse repeated whitespace and trim the result."""
    return " ".join(value.split())
