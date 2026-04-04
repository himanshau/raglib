"""Reflection module for deciding next retrieval action."""

from __future__ import annotations

import logging
from typing import List

from raglib.components.evaluator import Evaluator
from raglib.schemas import Document

logger = logging.getLogger(__name__)

ACTION_SUFFICIENT: str = "sufficient"
ACTION_RETRY: str = "retry"
ACTION_WEB_FALLBACK: str = "web_fallback"
ACTION_GIVE_UP: str = "give_up"


class ReflectionModule:
    """Inspects retrieval quality and recommends next action."""

    def __init__(self, evaluator: Evaluator):
        """Initialize reflection with an evaluator component."""

        self.evaluator = evaluator

    def reflect(self, query: str, documents: List[Document]) -> str:
        """Return the next action based on current retrieval quality."""

        if not documents:
            logger.info("ReflectionModule found no documents; suggesting web fallback")
            return ACTION_WEB_FALLBACK

        scores = [self.evaluator.score_document(query, doc) for doc in documents]
        average = sum(scores) / len(scores)
        best = max(scores)

        if average >= self.evaluator.relevance_threshold + 0.15:
            action = ACTION_SUFFICIENT
        elif best >= self.evaluator.relevance_threshold:
            action = ACTION_RETRY
        elif len(documents) >= 3:
            action = ACTION_WEB_FALLBACK
        else:
            action = ACTION_GIVE_UP

        logger.info("ReflectionModule action=%s avg=%.3f best=%.3f", action, average, best)
        return action
