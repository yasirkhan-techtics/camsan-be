import base64
import io
from typing import Literal

from typing import Type, TypeVar

from fastapi import Depends, HTTPException
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from PIL import Image
from pydantic import BaseModel

from config import get_settings
from schemas.llm_schemas import (
    BatchVerificationResponse,
    LegendBBoxLLMResponse,
    LegendBBoxVerificationResponse,
    LegendExtractionLLMResponse,
)

T = TypeVar("T", bound=BaseModel)

Provider = Literal["openai", "gemini"]


class LLMService:
    """LLM abstraction using LangChain to support Gemini and OpenAI."""

    def __init__(self):
        self.settings = get_settings()
        self.provider: Provider = "gemini"
        self._gemini_model: ChatGoogleGenerativeAI | None = None
        self._openai_model: ChatOpenAI | None = None

    def set_provider(self, provider: Provider):
        self.provider = provider

    def extract_legends(
        self, instructions: str, image_path: str
    ) -> LegendExtractionLLMResponse:
        return self._invoke_with_structured_output(
            instructions, image_path, LegendExtractionLLMResponse
        )

    def verify_batch(
        self, instructions: str, table_image_path: str
    ) -> BatchVerificationResponse:
        return self._invoke_with_structured_output(
            instructions, table_image_path, BatchVerificationResponse
        )

    def detect_legend_bbox(
        self, instructions: str, image_path: str
    ) -> LegendBBoxLLMResponse:
        return self._invoke_with_structured_output(
            instructions, image_path, LegendBBoxLLMResponse
        )
    
    def verify_legend_bbox(
        self, instructions: str, image_with_bbox_path: str
    ) -> LegendBBoxVerificationResponse:
        """Verify if the drawn bounding box correctly captures the legend table."""
        return self._invoke_with_structured_output(
            instructions, image_with_bbox_path, LegendBBoxVerificationResponse
        )
    
    def _invoke_with_structured_output(
        self, instructions: str, image_path: str, response_model: Type[T]
    ) -> T:
        """Invoke LLM with structured output using Pydantic model."""
        messages = self._build_multimodal_messages(instructions, image_path, response_model)
        model = self._get_model()
        structured_model = model.with_structured_output(response_model)
        return structured_model.invoke(messages)

    def _invoke_with_template_comparison(
        self, instructions: str, template_image_path: str, detected_image_path: str, response_model: Type[T]
    ) -> T:
        """
        Invoke LLM with two images for template comparison.
        
        Args:
            instructions: The prompt/instructions for the LLM
            template_image_path: Path to the template image
            detected_image_path: Path to the detected/cropped image
            response_model: Pydantic model for structured output
            
        Returns:
            Structured response from LLM
        """
        messages = self._build_template_comparison_messages(
            instructions, template_image_path, detected_image_path, response_model
        )
        model = self._get_model()
        structured_model = model.with_structured_output(response_model)
        return structured_model.invoke(messages)

    def _build_template_comparison_messages(
        self, instructions: str, template_image_path: str, detected_image_path: str, response_model: Type[BaseModel]
    ) -> list[BaseMessage]:
        """Build messages for template comparison with two images."""
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a vision assistant that compares symbols in electrical drawings. "
                    "You will be shown a TEMPLATE image and a DETECTED image. "
                    "Follow the instructions carefully and return data in the specified format.",
                ),
                (
                    "human",
                    [
                        {"type": "text", "text": "TEMPLATE IMAGE:"},
                        self._image_content(template_image_path),
                        {"type": "text", "text": "DETECTED IMAGE:"},
                        self._image_content(detected_image_path),
                        {"type": "text", "text": "{instructions}"},
                    ],
                ),
            ]
        )
        return prompt.format_messages(instructions=instructions)

    def _build_multimodal_messages(
        self, instructions: str, image_path: str, response_model: Type[BaseModel]
    ) -> list[BaseMessage]:
        # with_structured_output() automatically handles schema formatting
        # so we just need to pass the instructions
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a vision assistant that analyzes images and returns structured data. "
                    "Follow the instructions carefully and return data in the specified format.",
                ),
                (
                    "human",
                    [
                        {
                            "type": "text",
                            "text": "{instructions}",
                        },
                        self._image_content(image_path),
                    ],
                ),
            ]
        )
        return prompt.format_messages(instructions=instructions)


    def _get_model(self):
        if self.provider == "gemini":
            if not self.settings.gemini_api_key:
                raise HTTPException(
                    status_code=500, detail="GEMINI_API_KEY is not configured."
                )
            if not self._gemini_model:
                self._gemini_model = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    temperature=0,
                    google_api_key=self.settings.gemini_api_key,
                )
            return self._gemini_model

        if not self.settings.openai_api_key:
            raise HTTPException(
                status_code=500, detail="OPENAI_API_KEY is not configured."
            )
        if not self._openai_model:
            self._openai_model = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0,
                api_key=self.settings.openai_api_key,
            )
        return self._openai_model

    @staticmethod
    def _image_content(image_path: str, max_dimension: int = 2048) -> dict:
        """
        Load image, resize if needed to reduce API quota usage, and return base64 encoded.
        
        Args:
            image_path: Path to the image file
            max_dimension: Maximum width or height (default 2048px)
        """
        img = Image.open(image_path)
        original_size = img.size
        
        # Resize if image is too large
        if max(img.size) > max_dimension:
            ratio = max_dimension / max(img.size)
            new_size = (int(img.width * ratio), int(img.height * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            print(f"Resized image from {original_size} to {img.size}")
        
        # Convert to bytes
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        encoded_size_kb = len(buffer.getvalue()) / 1024
        print(f"Sending {encoded_size_kb:.1f} KB to LLM")
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        # Close file handle to avoid Windows lock
        img.close()
        
        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        }


def get_llm_service() -> LLMService:
    return LLMService()


