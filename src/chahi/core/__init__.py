"""Core layer — Domain logic thuần túy, không phụ thuộc infrastructure.

Re-exports entities và interfaces cho convenient imports.
"""

from chahi.core.entities import (
    AnalysisContext,
    Article,
    LLMSettings,
    SourceCategory,
    SourceConfig,
)
from chahi.core.interfaces import IConfigReader, ILLMClient, INewsFetcher

__all__ = [
    "AnalysisContext",
    "Article",
    "IConfigReader",
    "ILLMClient",
    "INewsFetcher",
    "LLMSettings",
    "SourceCategory",
    "SourceConfig",
]
