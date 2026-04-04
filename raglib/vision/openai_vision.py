"""OpenAI vision client implementation via LangChain."""

from __future__ import annotations

import base64
import logging
from typing import Any, Optional

from raglib.llm.constants import PROVIDER_DEFAULT_VISION_MODEL
from raglib.vision.base_vision import BaseVisionClient

logger = logging.getLogger(__name__)


class OpenAIVisionClient(BaseVisionClient):
    """Use OpenAI vision models to extract text from page images."""

    def __init__(self, api_key: Optional[str], model_name: Optional[str] = None):
        """Initialize an OpenAI vision-capable chat model through LangChain."""

        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise ImportError("pip install langchain-openai") from exc

        resolved_model = model_name or PROVIDER_DEFAULT_VISION_MODEL["openai"]
        self._model = ChatOpenAI(
            model=resolved_model,
            api_key=api_key,
            max_tokens=4096,
            temperature=0.0,
        )
        self._model_name = resolved_model
        logger.info("Using OpenAI vision model='%s'", resolved_model)

    def extract_text_from_image(self, image_path: str) -> str:
        """Extract plain text from an image by prompting a vision-capable model."""

        from langchain_core.messages import HumanMessage

        logger.debug("OpenAIVisionClient extracting text from image_path=%s", image_path)
        with open(image_path, "rb") as handle:
            encoded = base64.b64encode(handle.read()).decode("utf-8")

        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": (
                        "Extract all readable text from this image. "
                        "Return only the extracted text with no additional commentary."
                    ),
                },
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
            ]
        )

        response = self._model.invoke([message])
        return self._coerce_response(getattr(response, "content", response))

    def _coerce_response(self, content: Any) -> str:
        """Normalize model response payloads into plain text."""

        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and item.get("text"):
                    parts.append(str(item["text"]))
            return "\n".join(parts).strip()
        if isinstance(content, dict) and content.get("text"):
            return str(content["text"])
        return str(content)
