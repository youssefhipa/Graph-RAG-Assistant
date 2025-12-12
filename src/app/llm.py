from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Dict, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_community.chat_models.huggingface import HuggingFaceEndpoint

from .config import Settings


@dataclass
class LLMConfig:
    name: str
    constructor: Callable[[], BaseChatModel]


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
            self._registry["openai-gpt4"] = LLMConfig(
                name="openai-gpt4",
                constructor=lambda: ChatOpenAI(
                    api_key=self.settings.openai_api_key,
                    model_name="gpt-4o-mini",
                    temperature=0.2,
                ),
            )
            self._registry["openai-gpt35"] = LLMConfig(
                name="openai-gpt35",
                constructor=lambda: ChatOpenAI(
                    api_key=self.settings.openai_api_key,
                    model_name="gpt-3.5-turbo",
                    temperature=0.3,
                ),
            )
        if self.settings.ollama_model:
            self._registry["ollama"] = LLMConfig(
                name=f"ollama-{self.settings.ollama_model}",
                constructor=lambda: ChatOllama(model=self.settings.ollama_model, temperature=0.2),
            )
        if self.settings.huggingface_token:
            self._registry["huggingface"] = LLMConfig(
                name="huggingface-endpoint",
                constructor=lambda: HuggingFaceEndpoint(
                    huggingfacehub_api_token=self.settings.huggingface_token,
                    repo_id=os.getenv("HF_MODEL", "HuggingFaceH4/zephyr-7b-alpha"),
                    temperature=0.2,
                    task="text-generation",
                ),
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


def build_prompt(context: str, persona: str, task: str, question: str) -> ChatPromptTemplate:
    template = """
    You are {persona}.
    Task: {task}

    Context:
    {context}

    User question: {question}
    Answer using only the context. If insufficient, say you cannot find the answer.
    """
    return ChatPromptTemplate.from_template(template)


def run_llm(
    model: BaseChatModel,
    context: str,
    persona: str,
    task: str,
    question: str,
) -> str:
    prompt = build_prompt(context=context, persona=persona, task=task, question=question)
    chain = prompt | model
    result: AIMessage = chain.invoke(
        {"context": context, "persona": persona, "task": task, "question": question}
    )
    return result.content
