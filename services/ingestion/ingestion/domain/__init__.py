"""Domain layer for the ingestion service.

Connectors and canonical dataset builders live here. They are pure Python
(no FastAPI, no storage side effects) so they can be tested in isolation and
reused by the HTTP endpoints.
"""
