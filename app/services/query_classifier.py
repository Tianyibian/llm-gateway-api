from __future__ import annotations

from typing import Any

from app.models.schemas import QueryClassification
from app.services.errors import LLMConfigurationError


class QueryClassifier:
    """Classify a user query before any route-specific search is executed."""

    SYSTEM_PROMPT = """
You are a routing classifier for an e-commerce customer-support system.
Select exactly one route for the user's query:

- return_search: The answer requires return or refund policy information,
  including eligibility, return windows, return labels, packaging, return
  status, refund timing, or the return process. Choose this route even when a
  product name also appears in the query.
- product_search: The answer requires product catalog information, including
  specifications, features, compatibility, price, availability, category, or
  comparison between products.
- general_search: Greetings, general conversation, or questions that do not
  require return-policy or product-catalog information.

Classify only the user's intent. Do not answer the question and do not perform
the search. Give a concise reason and a confidence score from 0.0 to 1.0.
""".strip()

    def __init__(
        self,
        *,
        chain: Any,
        provider: str,
        model: str,
    ) -> None:
        self._chain = chain
        self.provider = provider
        self.model = model

    @classmethod
    def from_model(
        cls,
        *,
        model_client: Any,
        provider: str,
        model: str,
    ) -> QueryClassifier:
        """Build a Prompt | StructuredModel LCEL classifier pipeline."""
        try:
            from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        except ImportError as exc:
            raise LLMConfigurationError(
                "Query classification requires LangChain. Use Python 3.10+ and run "
                "'python -m pip install -r requirements-langchain.txt'."
            ) from exc

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", cls.SYSTEM_PROMPT),
                MessagesPlaceholder("history", optional=True),
                ("human", "Classify this query:\n{query}"),
            ]
        )
        structured_model = model_client.with_structured_output(QueryClassification)
        return cls(
            chain=prompt | structured_model,
            provider=provider,
            model=model,
        )

    async def classify(
        self,
        query: str,
        *,
        history: list[tuple[str, str]] | None = None,
    ) -> QueryClassification:
        result = await self._chain.ainvoke(
            {"query": query, "history": history or []}
        )
        if isinstance(result, QueryClassification):
            return result
        return QueryClassification.model_validate(result)
