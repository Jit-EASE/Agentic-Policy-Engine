# interfaces/ai_explainer.py

from __future__ import annotations
from typing import Optional
import os

import streamlit as st
from openai import OpenAI

_client: Optional[OpenAI] = None


def _get_client() -> Optional[OpenAI]:
    global _client
    if _client is not None:
        return _client
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    _client = OpenAI(api_key=api_key)
    return _client


def explain_block(tab_name: str, context: str, max_tokens: int = 350) -> str:
    """
    Calls OpenAI to generate a smooth, non-markdown explanation.
    """
    client = _get_client()
    if client is None:
        return "Spectre Agent: OpenAI API key not configured. Please set OPENAI_API_KEY in environment."

    system_prompt = (
        "You are a calm, highly technical but clear policy modelling assistant. "
        "Explain results for an Irish agri-food policy dashboard. "
        "Do not use markdown headers or bullet markers, just smooth paragraphs."
    )

    user_prompt = (
        f"Tab: {tab_name}\n"
        f"Context: {context}\n\n"
        "Explain what these results mean for a policy-maker or researcher in practical terms, "
        "using concise, professional language."
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Spectre Agent: unable to generate explanation. Error: {e}"
