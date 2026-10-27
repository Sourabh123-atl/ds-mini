"""FLUX — Infectious Disease Intelligence Platform.
Root ASGI entrypoint for Vercel, Uvicorn, and standard cloud hosts.
"""
from api import app

# Export for ASGI servers and cloud platform discoverers (Vercel, Render, AWS Lambda)
__all__ = ["app"]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
