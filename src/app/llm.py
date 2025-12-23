from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Dict, Optional, TYPE_CHECKING

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from .config import Settings

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI
    from langchain_ollama import ChatOllama


@dataclass
class LLMConfig:
    name: str
    constructor: Callable[[], BaseChatModel]
    max_context_tokens: int = 30000  # Safe default


class LLMRegistry:
    """
    Simple registry to offer at least three model choices.
    Extend with Groq, Gemini, etc. as needed.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._registry: Dict[str, LLMConfig] = {}
        self._build_defaults()

    def _build_defaults(self) -> None:
        if self.settings.openai_api_key:
            def _make_openai_gpt4():
                from langchain_openai import ChatOpenAI
                return ChatOpenAI(
                    api_key=self.settings.openai_api_key,
                    model_name="gpt-4o-mini",
                    temperature=0.2,
                )
            def _make_openai_gpt35():
                from langchain_openai import ChatOpenAI
                return ChatOpenAI(
                    api_key=self.settings.openai_api_key,
                    model_name="gpt-3.5-turbo",
                    temperature=0.3,
                )
            self._registry["openai-gpt4"] = LLMConfig(
                name="openai-gpt4",
                constructor=_make_openai_gpt4,
                max_context_tokens=120000,
            )
            self._registry["openai-gpt35"] = LLMConfig(
                name="openai-gpt35",
                constructor=_make_openai_gpt35,
                max_context_tokens=15000,
            )
        # Fixed set of Ollama models (edit here if you want different tags)
        ollama_models = ["llama2", "phi3:mini", "mistral"]
        for model_name in ollama_models:
            def _make_ollama(model=model_name):
                from langchain_ollama import ChatOllama
                return ChatOllama(model=model, temperature=0.2)
            self._registry[f"ollama-{model_name}"] = LLMConfig(
                name=f"ollama-{model_name}",
                constructor=_make_ollama,
                max_context_tokens=4000,
            )
        if self.settings.huggingface_token:
            def _make_huggingface():
                return ChatHuggingFace(
                    llm=HuggingFaceEndpoint(
                        repo_id=os.getenv("HF_MODEL", "HuggingFaceH4/zephyr-7b-alpha"),
                        huggingfacehub_api_token=self.settings.huggingface_token,
                        temperature=0.2,
                        max_new_tokens=256,  # Reduced to leave more room for context
                        task="text-generation",
                        model_kwargs={
                            "max_length": 32000,  # Hard limit
                        }
                    )
                )
            self._registry["huggingface"] = LLMConfig(
                name="huggingface-endpoint",
                constructor=_make_huggingface,
                max_context_tokens=25000,  # Conservative limit for 32K models
            )
            
    def options(self) -> Dict[str, LLMConfig]:
        return self._registry

    def get(self, key: str) -> BaseChatModel:
        if not self._registry:
            raise RuntimeError(
                "No LLM backends registered. Set OPENAI_API_KEY, OLLAMA_MODEL, or HUGGINGFACEHUB_API_TOKEN."
            )
        if key not in self._registry:
            raise KeyError(f"Model '{key}' not registered")
        return self._registry[key].constructor()
    
    def get_config(self, key: str) -> LLMConfig:
        if key not in self._registry:
            raise KeyError(f"Model '{key}' not registered")
        return self._registry[key]


def truncate_context(context: str, max_tokens: int = 20000) -> str:
    """
    Truncate context to approximate token limit.
    Rough estimate: 1 token ≈ 4 characters for English text.
    """
    max_chars = max_tokens * 4
    if len(context) <= max_chars:
        return context
    
    # Truncate and add indicator
    truncated = context[:max_chars]
    # Try to cut at a sentence boundary
    last_period = truncated.rfind('.')
    if last_period > max_chars * 0.8:  # If we can find a period in the last 20%
        truncated = truncated[:last_period + 1]
    
    return truncated + "\n\n[Context truncated due to length...]"


def build_prompt(context: str, persona: str, task: str, question: str) -> ChatPromptTemplate:
    template = """You are {persona}.
Task: {task}

Context:
{context}

User question: {question}
Answer using only the context. If insufficient, say you cannot find the answer."""
    return ChatPromptTemplate.from_template(template)


def run_llm(
    model: BaseChatModel,
    context: str,
    persona: str,
    task: str,
    question: str,
    max_context_tokens: int = 20000,  # Very conservative default
) -> str:
    # ALWAYS truncate context to prevent token limit errors
    truncated_context = truncate_context(context, max_context_tokens)
    
    prompt = build_prompt(
        context=truncated_context, 
        persona=persona, 
        task=task, 
        question=question
    )
    chain = prompt | model
    result: AIMessage = chain.invoke({
        "context": truncated_context, 
        "persona": persona, 
        "task": task, 
        "question": question
    })
    return result.content
