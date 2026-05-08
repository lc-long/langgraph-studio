"""Minimal Vercel Python handler test."""


def handler(request, context=None):
    """Return a simple JSON response."""
    return {"message": "Hello from Python on Vercel!"}
