from .base import Connector, SourceDoc
from .local_dir import LocalDirConnector
from .http_page import HttpPageConnector

__all__ = ["Connector", "SourceDoc", "LocalDirConnector", "HttpPageConnector"]
