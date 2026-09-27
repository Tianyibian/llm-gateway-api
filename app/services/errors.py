class LLMConfigurationError(RuntimeError):
    """Raised when the selected LLM provider cannot be configured."""


class EntityClarificationRequired(RuntimeError):
    """An adapter found multiple database candidates; ask rather than guess."""


class ConversationNotFoundError(LookupError):
    """Raised when a conversation does not exist or belongs to another user."""


class InvalidImageError(ValueError):
    """Raised when an uploaded image is unsupported or cannot be decoded."""


class EmbeddingServiceError(RuntimeError):
    """Raised when an embedding provider fails or returns invalid vectors."""


class KnowledgeBaseNotReadyError(RuntimeError):
    """Raised when retrieval is attempted before knowledge ingestion."""
