"""Unified comparison facade. Optional models fail gracefully when their weights are unavailable."""
from services.detector import compare_documents
__all__ = ["compare_documents"]
