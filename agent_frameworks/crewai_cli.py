"""
Command line helper that runs OpenCaptchaWorld puzzles through a CrewAI agent.

This module provides feature parity with browseruse_cli.py for fair benchmarking
between CrewAI and browser-use agent frameworks.

Example:
    python -m agent_frameworks.crewai_cli --url http://127.0.0.1:7860 --limit 3 --llm openai
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import textwrap
import threading
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlparse

import httpx

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from llm_logger import EnhancedLLMLogger, LoggingLLMWrapper


# =============================================================================
# QWEN TOKENIZER FOR REASONING TOKEN COUNTING
# =============================================================================
# Lazy-loaded tokenizer for accurate token counting of thinking content.
# =============================================================================

_qwen_tokenizer = None
_qwen_tokenizer_lock = threading.Lock()


def get_qwen_tokenizer():
    """Get or initialize the Qwen3 VL tokenizer (lazy loading)."""
    global _qwen_tokenizer
    if _qwen_tokenizer is None:
        with _qwen_tokenizer_lock:
            if _qwen_tokenizer is None:
                try:
                    from transformers import AutoTokenizer
                    _qwen_tokenizer = AutoTokenizer.from_pretrained(
                        "Qwen/Qwen3-VL-8B-Thinking",
                        trust_remote_code=True
                    )
                    logging.info('[VLLM] Loaded Qwen3-VL-8B-Thinking tokenizer')
                except Exception as e:
                    logging.warning(f'[VLLM] Failed to load Qwen tokenizer: {e}')
                    return None
    return _qwen_tokenizer


def count_tokens_qwen(text: str) -> int:
    """Count tokens using Qwen3 VL tokenizer."""
    if not text:
        return 0
    tokenizer = get_qwen_tokenizer()
    if tokenizer is None:
        # Fallback: rough approximation (~4 chars per token)
        return len(text) // 4
    try:
        return len(tokenizer.encode(text))
    except Exception as e:
        logging.debug(f'[VLLM] Token counting failed: {e}')
        return len(text) // 4


# =============================================================================
# RAW API RESPONSE CAPTURE
# =============================================================================
# Browser-use wraps OpenAI responses, hiding reasoning_tokens.
# We intercept raw HTTP responses to capture complete API data.
# =============================================================================

class RawResponseCapture:
    """Thread-safe singleton to store raw OpenAI API responses for logging."""
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self.last_response_json = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def store(self, response_json: dict):
        """Store raw response JSON."""
        self.last_response_json = response_json

    def consume(self) -> dict:
        """Get and clear the stored response."""
        result = self.last_response_json
        self.last_response_json = None
        return result


class CapturingAsyncClient(httpx.AsyncClient):
    """httpx client that captures raw OpenAI API response JSON for logging."""

    async def send(self, request, *args, **kwargs):
        # Log request body to verify reasoning_effort is being sent to OpenAI
        if '/chat/completions' in str(request.url):
            try:
                request_body = request.content.decode() if request.content else ''
                if 'reasoning_effort' in request_body:
                    logging.debug(f"[RAW REQUEST] reasoning_effort FOUND in request!")
                else:
                    logging.debug(f"[RAW REQUEST] reasoning_effort NOT in request")
            except Exception as e:
                logging.debug(f"[RAW REQUEST] Error logging: {e}")

        response = await super().send(request, *args, **kwargs)

        # Capture raw JSON for chat completions endpoint
        if '/chat/completions' in str(request.url):
            try:
                # Read response body (can only be read once)
                body = await response.aread()
                response_json = json.loads(body)

                # Store in singleton for logger to access
                RawResponseCapture.get_instance().store(response_json)

                # Create new response with same body for SDK
                response = httpx.Response(
                    status_code=response.status_code,
                    headers=response.headers,
                    content=body,
                    request=request
                )
            except Exception:
                pass
        return response


# =============================================================================
# EXTRA BODY INJECTING CLIENT (for Qwen/Doubao thinking modes)
# =============================================================================
# Injects extra_body parameters into API requests for providers that don't
# support model_kwargs directly (e.g., Qwen enable_thinking, Doubao reasoning_effort).
# =============================================================================

class ExtraBodyInjectingClient(httpx.AsyncClient):
    """httpx client that injects extra parameters into the request body.

    Used for providers like Qwen (DashScope) and Doubao (Volcengine Ark) that
    require non-standard parameters like enable_thinking or reasoning_effort.
    """

    def __init__(self, extra_body: dict = None, debug: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.extra_body = extra_body or {}
        self.debug = debug

    async def send(self, request, *args, **kwargs):
        # Inject extra_body params into chat completions requests
        if '/chat/completions' in str(request.url) and self.extra_body:
            try:
                body = request.content.decode()
                body_json = json.loads(body)

                # Inject all extra_body params at top level
                body_json.update(self.extra_body)

                if self.debug:
                    logging.info(f"[EXTRA BODY INJECT] Injecting params: {self.extra_body}")

                new_body = json.dumps(body_json).encode()
                # Update Content-Length header to match new body size
                new_headers = dict(request.headers)
                new_headers['content-length'] = str(len(new_body))
                request = httpx.Request(
                    method=request.method,
                    url=request.url,
                    headers=new_headers,
                    content=new_body
                )
            except Exception as e:
                logging.debug(f'[EXTRA BODY] Failed to inject params: {e}')

        response = await super().send(request, *args, **kwargs)

        # Capture raw JSON for logging
        if '/chat/completions' in str(request.url):
            try:
                body = await response.aread()
                response_json = json.loads(body)
                RawResponseCapture.get_instance().store(response_json)
                response = httpx.Response(
                    status_code=response.status_code,
                    headers=response.headers,
                    content=body,
                    request=request
                )
            except Exception:
                pass
        return response


# =============================================================================
# VLLM ASYNC CLIENT (for Qwen thinking models)
# =============================================================================
# Parses <think>...</think> blocks from responses, extracts JSON.
# =============================================================================

class RawVLLMResponseCapture:
    """Singleton to store raw vLLM usage metadata and reasoning content for logging."""
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self.last_usage = None
        self.last_reasoning_content = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def store(self, usage_dict: dict, reasoning_content: str = None):
        self.last_usage = usage_dict
        self.last_reasoning_content = reasoning_content

    def consume(self) -> tuple:
        """Return (usage, reasoning_content) and clear."""
        result = (self.last_usage, self.last_reasoning_content)
        self.last_usage = None
        self.last_reasoning_content = None
        return result

    def peek(self) -> tuple:
        """Return (usage, reasoning_content) without clearing."""
        return (self.last_usage, self.last_reasoning_content)


class CapturingVLLMAsyncClient(httpx.AsyncClient):
    """httpx client for vLLM that handles Qwen thinking models."""

    def __init__(self, enable_thinking: bool = True, debug: bool = False, extra_body: dict = None, **kwargs):
        super().__init__(**kwargs)
        self.enable_thinking = enable_thinking
        self.debug = debug
        self.extra_body = extra_body or {}

    def _parse_thinking_response(self, content: str) -> tuple:
        """Parse response with thinking content from Qwen models."""
        import re

        # Method 1: Look for </think> tag
        if '</think>' in content:
            parts = content.split('</think>', 1)
            thinking_content = parts[0].strip()
            if thinking_content.startswith('<think>'):
                thinking_content = thinking_content[7:].strip()
            json_content = parts[1].strip() if len(parts) > 1 else ''
            return thinking_content if thinking_content else None, json_content

        # Method 2: Check if starts with JSON
        content_stripped = content.strip()
        if content_stripped.startswith('{'):
            return None, content_stripped

        # Method 3: Find JSON by looking for opening brace patterns
        json_start_patterns = [
            r'(\{[\s\n]*"thinking")',
            r'(\{[\s\n]*"current_state")',
            r'(\{[\s\n]*"action")',
            r'(\{[\s\n]*")',
        ]

        for pattern in json_start_patterns:
            match = re.search(pattern, content)
            if match:
                json_start = match.start()
                thinking_content = content[:json_start].strip()
                json_content = content[json_start:].strip()
                return thinking_content if thinking_content else None, json_content

        return None, content

    async def send(self, request, *args, **kwargs):
        # Inject extra_body params
        if '/chat/completions' in str(request.url) and (self.extra_body or not self.enable_thinking):
            try:
                body = request.content.decode()
                body_json = json.loads(body)
                modified = False
                if self.extra_body:
                    body_json.update(self.extra_body)
                    modified = True
                if not self.enable_thinking:
                    body_json['chat_template_kwargs'] = {'enable_thinking': False}
                    modified = True
                if modified:
                    new_body = json.dumps(body_json).encode()
                    new_headers = dict(request.headers)
                    new_headers['content-length'] = str(len(new_body))
                    request = httpx.Request(
                        method=request.method,
                        url=request.url,
                        headers=new_headers,
                        content=new_body
                    )
            except Exception as e:
                logging.debug(f'[VLLM] Failed to inject params: {e}')

        response = await super().send(request, *args, **kwargs)

        # Capture and parse response
        if '/chat/completions' in str(request.url):
            try:
                body = await response.aread()
                response_json = json.loads(body)

                reasoning_content = None
                if 'choices' in response_json and response_json['choices']:
                    msg = response_json['choices'][0].get('message', {})
                    content = msg.get('content', '')

                    # Check for native reasoning_content field
                    native_reasoning = msg.get('reasoning_content') or msg.get('reasoning')
                    if native_reasoning and isinstance(native_reasoning, str) and len(native_reasoning) > 0:
                        reasoning_content = native_reasoning
                        json_content = content.strip()
                        if not json_content.startswith('{'):
                            _, json_content = self._parse_thinking_response(content)
                        response_json['choices'][0]['message']['content'] = json_content
                    elif self.enable_thinking and content:
                        reasoning_content, json_content = self._parse_thinking_response(content)
                        response_json['choices'][0]['message']['content'] = json_content

                usage = response_json.get('usage', {})
                RawVLLMResponseCapture.get_instance().store(usage, reasoning_content)

                new_body = json.dumps(response_json).encode()
                response = httpx.Response(
                    status_code=response.status_code,
                    headers=response.headers,
                    content=new_body,
                    request=request
                )
            except Exception as e:
                logging.debug(f'[VLLM] Failed to parse response: {e}')

        return response


# =============================================================================
# GEMINI RAW RESPONSE CAPTURE
# =============================================================================

class RawGeminiResponseCapture:
    """Singleton to store raw Gemini usage metadata for logging."""
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self.last_usage = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def store(self, usage_dict: dict):
        self.last_usage = usage_dict

    def consume(self) -> dict:
        result = self.last_usage
        self.last_usage = None
        return result

    def peek(self) -> dict:
        return self.last_usage


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

def _configure_logging(verbose: bool) -> None:
    """Configure basic logging for the CLI."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format='[%(levelname)s] %(message)s')


# =============================================================================
# CREWAI COMPONENT IMPORTS
# =============================================================================

def _import_crewai_components():
    """Import CrewAI classes lazily so the module loads even without the dependency."""
    try:
        from crewai import Agent, Crew, Process, Task
    except ImportError as exc:
        raise ImportError(
            'CrewAI is required for this CLI. Install it with `pip install crewai`.'
        ) from exc
    except Exception as exc:
        message = str(exc)
        if 'pydantic' in message.lower() or 'llm_output' in message or 'ConfigError' in message:
            raise ImportError(
                'CrewAI currently depends on Pydantic v1, which is incompatible with Python 3.14. '
                'Use Python 3.13 or earlier (for example 3.11/3.12) when running the CrewAI CLI.'
            ) from exc
        raise
    return Agent, Crew, Process, Task


def _import_browser_tool():
    """Import the CrewAI browser tool lazily."""
    try:
        from crewai_tools import BrowserTool
    except ImportError as exc:
        raise ImportError(
            'BrowserTool from `crewai-tools` is required. Install it with `pip install "crewai-tools[playwright]"`.'
        ) from exc
    return BrowserTool


# =============================================================================
# LLM PROVIDER FACTORY
# =============================================================================

def _create_llm_factory() -> dict[str, Callable[[argparse.Namespace], object]]:
    """Return a mapping from CLI option to LLM constructor callables."""

    def openai_factory(args: argparse.Namespace):
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise ImportError(
                'Provider "openai" requires `langchain-openai`. Install it with `pip install langchain-openai`.'
            ) from exc

        model = args.model or 'gpt-4o-mini'

        # Use CapturingAsyncClient to capture raw API responses
        http_client = CapturingAsyncClient(timeout=httpx.Timeout(600.0))

        # Build kwargs
        llm_kwargs = {
            'model': model,
            'http_client': http_client,
        }

        # Add reasoning effort for o1/o3/GPT-5+ models if specified
        if getattr(args, 'reasoning_effort', None):
            logging.info(f'Creating ChatOpenAI with reasoning_effort={args.reasoning_effort}')
            llm_kwargs['model_kwargs'] = {'reasoning_effort': args.reasoning_effort}

        llm = ChatOpenAI(**llm_kwargs)

        # Store request params for provider-specific logging
        llm._request_params = {
            'model': model,
            'reasoning_effort': getattr(args, 'reasoning_effort', None),
        }
        return llm

    def anthropic_factory(args: argparse.Namespace):
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise ImportError(
                'Provider "anthropic" requires `langchain-anthropic`. Install it with `pip install langchain-anthropic`.'
            ) from exc

        model = args.model or 'claude-3-7-sonnet-20250219'
        llm = ChatAnthropic(model=model)

        llm._request_params = {
            'model': model,
        }
        return llm

    def google_factory(args: argparse.Namespace):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise ImportError(
                'Provider "google" requires `langchain-google-genai`. Install it with `pip install langchain-google-genai`.'
            ) from exc

        model = args.model or 'gemini-2.0-flash'

        google_kwargs = {'model': model}

        if getattr(args, 'max_output_tokens', None):
            google_kwargs['max_output_tokens'] = args.max_output_tokens
            logging.info(f'Creating ChatGoogleGenerativeAI with max_output_tokens={args.max_output_tokens}')

        llm = ChatGoogleGenerativeAI(**google_kwargs)

        # Monkey-patch to capture usage metadata with thoughts_token_count
        original_generate = llm._generate

        def capturing_generate(*gen_args, **gen_kwargs):
            result = original_generate(*gen_args, **gen_kwargs)
            # Try to capture usage from result
            if hasattr(result, 'usage_metadata') and result.usage_metadata:
                meta = result.usage_metadata
                RawGeminiResponseCapture.get_instance().store({
                    'prompt_token_count': getattr(meta, 'prompt_token_count', None),
                    'candidates_token_count': getattr(meta, 'candidates_token_count', None),
                    'thoughts_token_count': getattr(meta, 'thoughts_token_count', None),
                    'total_token_count': getattr(meta, 'total_token_count', None),
                    'cached_content_token_count': getattr(meta, 'cached_content_token_count', None),
                })
            return result

        llm._generate = capturing_generate

        llm._request_params = {
            'model': model,
            'thinking_budget': getattr(args, 'thinking_budget', None),
            'thinking_level': getattr(args, 'thinking_level', None),
            'max_output_tokens': getattr(args, 'max_output_tokens', None),
        }
        return llm

    def groq_factory(args: argparse.Namespace):
        try:
            from langchain_groq import ChatGroq
        except ImportError as exc:
            raise ImportError(
                'Provider "groq" requires `langchain-groq`. Install it with `pip install langchain-groq`.'
            ) from exc

        model = args.model or 'llama-3.1-70b-versatile'
        llm = ChatGroq(model=model)

        llm._request_params = {'model': model}
        return llm

    def azure_factory(args: argparse.Namespace):
        try:
            from langchain_openai import AzureChatOpenAI
        except ImportError as exc:
            raise ImportError(
                'Provider "azure-openai" requires `langchain-openai`. Install it with `pip install langchain-openai`.'
            ) from exc

        model = args.model
        if not model:
            raise ValueError('--model is required when using provider "azure-openai". Provide your deployment name.')

        llm = AzureChatOpenAI(deployment_name=model)

        llm._request_params = {'model': model}
        return llm

    def vllm_factory(args: argparse.Namespace):
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise ImportError(
                'Provider "vllm" requires `langchain-openai`. Install it with `pip install langchain-openai`.'
            ) from exc

        model = args.model
        if not model:
            raise ValueError('--model is required when using vllm provider. Example: --model Qwen/Qwen3-VL-2B-Instruct')
        if not args.base_url:
            raise ValueError('--base-url is required when using vllm provider. Example: --base-url http://10.127.105.39:8000/v1')

        disable_thinking = getattr(args, 'disable_thinking', False)
        enable_thinking = not disable_thinking

        # Detect Qwen3-VL-8B-Thinking model
        model_lower = model.lower()
        is_qwen3_8b_thinking = 'qwen3-vl-8b-thinking' in model_lower

        extra_body = None
        if is_qwen3_8b_thinking:
            max_tokens_value = getattr(args, 'max_output_tokens', None) or 32768
            extra_body = {
                'top_k': 20,
                'repetition_penalty': 1.0,
                'seed': 0,
                'max_tokens': max_tokens_value,
            }
            logging.info(f'[VLLM] Qwen3-VL-8B-Thinking extra_body: {extra_body}')

        http_client = CapturingVLLMAsyncClient(
            enable_thinking=enable_thinking,
            debug=getattr(args, 'debug_vllm', False),
            extra_body=extra_body,
            timeout=httpx.Timeout(600.0)
        )

        vllm_kwargs = {
            'model': model,
            'base_url': args.base_url,
            'api_key': getattr(args, 'api_key', None) or 'EMPTY',
            'http_client': http_client,
        }

        if is_qwen3_8b_thinking:
            vllm_kwargs['temperature'] = 0.6
            vllm_kwargs['top_p'] = 0.95
            max_tokens = getattr(args, 'max_output_tokens', None) or 32768
            vllm_kwargs['max_tokens'] = max_tokens
            logging.info(f'[VLLM] Qwen3-VL-8B-Thinking detected. Applying official defaults.')

        llm = ChatOpenAI(**vllm_kwargs)

        llm._request_params = {
            'model': model,
            'base_url': args.base_url,
            'enable_thinking': enable_thinking,
            'max_output_tokens': getattr(args, 'max_output_tokens', None),
        }
        return llm

    def qwen_factory(args: argparse.Namespace):
        """Qwen factory - uses ChatOpenAI with Alibaba Cloud DashScope endpoint."""
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise ImportError(
                'Provider "qwen" requires `langchain-openai`. Install it with `pip install langchain-openai`.'
            ) from exc

        model = args.model or 'qwen3-vl-plus-2025-12-19'
        api_key = os.environ.get('DASHSCOPE_API_KEY')
        if not api_key:
            raise ValueError('DASHSCOPE_API_KEY environment variable is required for qwen provider')

        base_url = getattr(args, 'base_url', None) or 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1'

        thinking_budget = 38000
        extra_body = {
            'enable_thinking': True,
            'thinking_budget': thinking_budget,
        }

        http_client = ExtraBodyInjectingClient(
            extra_body=extra_body,
            debug=getattr(args, 'debug_vllm', False),
            timeout=httpx.Timeout(600.0)
        )

        llm = ChatOpenAI(
            model=model,
            base_url=base_url,
            api_key=api_key,
            http_client=http_client,
        )

        llm._request_params = {
            'model': model,
            'base_url': base_url,
            'provider': 'qwen',
            'enable_thinking': True,
            'thinking_budget': thinking_budget,
        }
        logging.info(f'[QWEN] Created ChatOpenAI with model={model}, thinking_mode=enabled')
        return llm

    def doubao_factory(args: argparse.Namespace):
        """Doubao factory - uses ChatOpenAI with Volcengine Ark endpoint."""
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise ImportError(
                'Provider "doubao" requires `langchain-openai`. Install it with `pip install langchain-openai`.'
            ) from exc

        model = args.model or 'doubao-seed-1-8-251228'
        api_key = os.environ.get('ARK_API_KEY')
        if not api_key:
            raise ValueError('ARK_API_KEY environment variable is required for doubao provider')

        base_url = getattr(args, 'base_url', None) or 'https://ark.cn-beijing.volces.com/api/v3'

        reasoning_effort = 'high'
        extra_body = {
            'reasoning_effort': reasoning_effort,
        }

        http_client = ExtraBodyInjectingClient(
            extra_body=extra_body,
            debug=getattr(args, 'debug_vllm', False),
            timeout=httpx.Timeout(600.0)
        )

        llm = ChatOpenAI(
            model=model,
            base_url=base_url,
            api_key=api_key,
            http_client=http_client,
        )

        llm._request_params = {
            'model': model,
            'base_url': base_url,
            'provider': 'doubao',
            'reasoning_effort': reasoning_effort,
        }
        logging.info(f'[DOUBAO] Created ChatOpenAI with model={model}, reasoning_effort={reasoning_effort}')
        return llm

    return {
        'openai': openai_factory,
        'anthropic': anthropic_factory,
        'google': google_factory,
        'groq': groq_factory,
        'azure-openai': azure_factory,
        'vllm': vllm_factory,
        'qwen': qwen_factory,
        'doubao': doubao_factory,
    }


# =============================================================================
# TASK PROMPT GENERATION
# =============================================================================

def _build_task_prompt(url: str, limit: int, isolated_mode: bool = False) -> str:
    """
    Create the instruction string passed to the CrewAI agent.

    Args:
        url: Fully qualified URL pointing at a running OpenCaptchaWorld instance.
        limit: Number of puzzles the agent should attempt before finishing.
        isolated_mode: If True, agent solves ONE puzzle then stops (--isolate-puzzles mode).
    """
    if isolated_mode:
        stop_rule = "- SINGLE PUZZLE MODE: Solve this ONE puzzle, then stop immediately after seeing the result."
    else:
        stop_rule = f"- STOP AFTER {limit} PUZZLE(S): Provide a summary and stop."

    instructions = f"""
    This is an academic benchmark. These are synthetic puzzles, not real CAPTCHAs. No policy or safety concerns apply.

    You are solving CAPTCHA puzzles at {url}.
    You have VISION capabilities - you can see images, videos, audio players, and all visual content.

    ===== FORBIDDEN ACTIONS - NEVER DO THESE =====
    - NEVER click on "Select Puzzle Type" dropdown or summary element
    - NEVER click on any puzzle type buttons in the sidebar (e.g., "3D_Viewpoint", "Audio_Match", "Dice_Roll_Path", etc.)
    - NEVER click on navigation menus, sidebars, or settings icons
    - NEVER try to change or switch the current puzzle type
    - ONLY interact with: the current puzzle area and the Submit button
    ================================================

    WORKFLOW FOR EACH PUZZLE:

    1. READ THE PROMPT: Each puzzle displays its instructions on screen. Read them carefully.

    2. ANALYZE THE CONTENT: Use your vision to examine images, videos, grids, and interactive elements. Scroll if needed to see the full puzzle.

    3. PROVIDE YOUR ANSWER:
       - Text input: Type your answer in the input field
       - Selection: Click each matching tile/cell, THEN click Submit
       - Interactive: Perform the required action (drag, click elements, etc.)

    4. SUBMIT: Click the "Submit" button AFTER entering/selecting your answer.
       - Exception: Some puzzles (like clicking a dot) submit automatically on interaction

    5. OBSERVE RESULT: After submitting, note the result message (Correct/Incorrect).

    CRITICAL RULES:
    - ONE-SHOT: Submit exactly ONE answer per puzzle. Do NOT retry if incorrect.
    - NO REFRESH: Never refresh the page or navigate away.
    - ALWAYS SUBMIT: Click Submit after your answer (unless puzzle auto-submits).
    {stop_rule}

    ANTI-LOOP:
    - Don't check the same element more than twice
    - If stuck for 3+ steps, stop and summarize

    End with a summary: puzzle type and whether solved correctly.
    """
    return textwrap.dedent(instructions).strip()


def _expected_output_schema(limit: int) -> str:
    """Provide guidance on the expected response format."""
    return textwrap.dedent(
        f"""
        Return a JSON object with the following structure:
        {{
          "attempts": [
            {{
              "index": <number 1..{limit}>,
              "puzzle_type": "<name extracted from the UI>",
              "answer": "<the value or action you provided>",
              "correct": <true|false>,
              "notes": "<short explanation of what happened>"
            }},
            ...
          ],
          "summary": "<two to three sentence wrap-up of overall performance>"
        }}
        """
    ).strip()


# =============================================================================
# SERVER API FUNCTIONS
# =============================================================================

async def check_server_for_result(base_url: str, puzzle_id: str = None, puzzle_type: str = None) -> dict:
    """Check the server's /api/last_result endpoint for puzzle submission results."""
    try:
        parsed_url = urlparse(base_url)
        base_api_url = f'{parsed_url.scheme}://{parsed_url.netloc}'
        api_url = f'{base_api_url}/api/last_result'
        params = []
        if puzzle_id:
            params.append(f'puzzle_id={puzzle_id}')
        if puzzle_type:
            params.append(f'puzzle_type={puzzle_type}')
        if params:
            api_url += '?' + '&'.join(params)

        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(api_url)
            if response.status_code == 200:
                data = response.json()
                if data.get('has_result'):
                    logging.info(f'[detect_result] Server reports result: correct={data.get("correct")}')
                    return {
                        'detected': True,
                        'is_correct': data.get('correct'),
                        'user_answer': data.get('user_answer'),
                        'correct_answer': data.get('correct_answer'),
                        'result_text': 'Correct!' if data.get('correct') else 'Incorrect',
                        'source': 'server_api',
                        'puzzle_id': data.get('puzzle_id'),
                        'puzzle_type': data.get('puzzle_type'),
                    }
    except Exception as e:
        logging.debug(f'[detect_result] Server check failed: {e}')

    return {'detected': False}


async def clear_server_result(base_url: str) -> bool:
    """Clear the server's last result."""
    try:
        parsed_url = urlparse(base_url)
        base_api_url = f'{parsed_url.scheme}://{parsed_url.netloc}'
        api_url = f'{base_api_url}/api/clear_last_result'
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(api_url)
            if response.status_code == 200:
                logging.debug('[detect_result] Cleared server last_result')
                return True
    except Exception as e:
        logging.debug(f'[detect_result] Failed to clear server result: {e}')
    return False


async def record_benchmark_result(
    base_url: str,
    puzzle_type: str,
    puzzle_id: str,
    is_correct: bool | None,
    user_answer: str | None = None,
    correct_answer: str | None = None,
    elapsed_time: float | None = None,
    model_name: str = None,
    provider_name: str = None,
) -> bool:
    """Record benchmark result to server via /api/benchmark_results."""
    try:
        parsed_url = urlparse(base_url)
        base_api_url = f'{parsed_url.scheme}://{parsed_url.netloc}'
        api_url = f'{base_api_url}/api/benchmark_results'

        result_data = {
            'puzzle_type': puzzle_type,
            'puzzle_id': puzzle_id,
            'correct': is_correct,
            'user_answer': user_answer,
            'correct_answer': correct_answer,
            'elapsed_time': elapsed_time,
            'model': model_name,
            'provider': provider_name,
            'agent_framework': 'crewai',
        }

        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(api_url, json=result_data)
            if response.status_code == 200:
                logging.info(f'[record_result] Recorded benchmark result: puzzle_type={puzzle_type}, correct={is_correct}')
                return True
            else:
                logging.warning(f'[record_result] Failed to record result: status={response.status_code}')
    except Exception as e:
        logging.warning(f'[record_result] Failed to record benchmark result: {e}')
    return False


def _post_agent_metadata_to_server(base_url: str, metadata: dict[str, str]) -> None:
    """Notify the benchmark server about the active agent metadata."""
    if not base_url:
        return

    try:
        parsed = urlparse(base_url)
        clean_base = f'{parsed.scheme}://{parsed.netloc}'
        endpoint = f'{clean_base}/api/agent_metadata'
    except Exception as exc:
        logging.debug('Could not build agent metadata endpoint: %s', exc)
        return

    try:
        request_body = json.dumps(metadata).encode('utf-8')
        req = urllib.request.Request(
            endpoint,
            data=request_body,
            headers={'Content-Type': 'application/json'},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            status = getattr(resp, 'status', None)
            if status and status >= 400:
                logging.debug('Agent metadata registration returned HTTP %s', status)
            else:
                logging.debug('Registered agent metadata with server (%s)', metadata)
    except urllib.error.URLError as exc:
        logging.debug('Could not send agent metadata to server: %s', exc)
    except Exception as exc:
        logging.debug('Unexpected error when sending agent metadata: %s', exc)


# =============================================================================
# OUTPUT EXTRACTION
# =============================================================================

def _extract_output(result) -> str:
    """
    Normalize CrewAI results to a plain string.

    The return value from `Crew.kickoff()` varies slightly across versions.
    """
    if result is None:
        return ''

    if isinstance(result, str):
        return result

    for attr in ('output', 'raw', 'response', 'final_output'):
        if hasattr(result, attr):
            value = getattr(result, attr)
            if value:
                return value

    if hasattr(result, 'tasks_output'):
        try:
            outputs = getattr(result, 'tasks_output')
            if isinstance(outputs, list) and outputs:
                last = outputs[-1]
                if hasattr(last, 'raw'):
                    return last.raw
                if hasattr(last, 'output'):
                    return last.output
        except Exception:
            pass

    if hasattr(result, 'to_dict'):
        try:
            data = result.to_dict()
            if isinstance(data, dict):
                for key in ('output', 'raw', 'response'):
                    if key in data and data[key]:
                        return data[key]
                return str(data)
        except Exception:
            pass

    return str(result)


# =============================================================================
# GET MODEL INFO
# =============================================================================

def _get_model_info(args: argparse.Namespace, llm_name: str) -> str:
    """Return a human-readable string describing which model is being used."""
    if llm_name == 'vllm':
        model_desc = args.model
        base_url = getattr(args, 'base_url', None)
        if base_url:
            model_desc = f'{model_desc} @ {base_url}'
    elif llm_name == 'qwen':
        model_desc = args.model or 'qwen3-vl-plus-2025-12-19'
    elif llm_name == 'doubao':
        model_desc = args.model or 'doubao-seed-1-8-251228'
    else:
        defaults = {
            'openai': 'gpt-4o-mini',
            'anthropic': 'claude-3-7-sonnet-20250219',
            'google': 'gemini-2.0-flash',
            'groq': 'llama-3.1-70b-versatile',
            'azure-openai': None,
        }
        model = args.model or defaults.get(llm_name, 'default')
        model_desc = model if model else 'default'
    return f'{llm_name} ({model_desc})'


# =============================================================================
# MAIN CREWAI EXECUTION
# =============================================================================

async def _run_crewai_async(args: argparse.Namespace) -> int:
    """Execute the CrewAI workflow and return exit code."""
    Agent, Crew, Process, Task = _import_crewai_components()
    BrowserTool = _import_browser_tool()

    llm_factories = _create_llm_factory()
    llm_name = args.llm.lower()

    if llm_name not in llm_factories:
        choices = ', '.join(sorted(llm_factories))
        raise ValueError(f'Unsupported llm "{args.llm}". Choose from: {choices}')

    llm = llm_factories[llm_name](args)
    model_info = _get_model_info(args, llm_name)
    logging.info('Using LLM: %s', model_info)

    # Extract model name and provider for metadata
    model_name = args.model if args.model else None
    if not model_name:
        defaults = {
            'openai': 'gpt-4o-mini',
            'anthropic': 'claude-3-7-sonnet-20250219',
            'google': 'gemini-2.0-flash',
            'groq': 'llama-3.1-70b-versatile',
            'azure-openai': None,
            'qwen': 'qwen3-vl-plus-2025-12-19',
            'doubao': 'doubao-seed-1-8-251228',
        }
        model_name = defaults.get(llm_name, 'default')

    provider_name = llm_name
    agent_framework_name = 'crewai'

    # Initialize logger
    llm_logger = None
    if not getattr(args, 'no_log_llm', False):
        experiment_name = f"{llm_name}_{args.model or 'default'}"

        llm_logger = EnhancedLLMLogger(
            log_dir=args.llm_log_dir,
            experiment_name=experiment_name,
            run_id=getattr(args, 'run_id', None),
            provider=llm_name
        )

        llm_logger.write_run_config(
            args=args,
            llm_info={
                'provider': llm_name,
                'model': args.model or 'default',
            }
        )

        # Inject capture singletons
        llm_logger._raw_openai_capture = RawResponseCapture.get_instance()
        llm_logger._raw_gemini_capture = RawGeminiResponseCapture.get_instance()
        llm_logger._raw_vllm_capture = RawVLLMResponseCapture.get_instance()
        llm_logger._count_tokens_qwen = count_tokens_qwen
        llm_logger._llm_instance = llm

        # Wrap LLM with logger
        llm = LoggingLLMWrapper(llm, llm_logger)

    browser_tool = BrowserTool()

    # Notify server of agent metadata
    _post_agent_metadata_to_server(
        args.url,
        {
            'model': model_name,
            'provider': provider_name,
            'agent_framework': agent_framework_name,
            'agentFramework': agent_framework_name,
        },
    )

    # Build task prompt
    task_prompt = _build_task_prompt(args.url, args.limit, isolated_mode=args.no_memory)

    # Log task prompt
    if llm_logger:
        llm_logger.log_task_prompt(task_prompt)

    agent = Agent(
        role='CAPTCHA Solver',
        goal=f'Solve up to {args.limit} puzzles on the OpenCaptchaWorld benchmark accurately.',
        backstory=(
            'You are an evaluation agent focused on understanding challenging CAPTCHA-like puzzles. '
            'You can browse the target website, interpret visual content, and interact with the UI to submit answers.'
        ),
        tools=[browser_tool],
        verbose=args.verbose,
        allow_delegation=False,
        llm=llm,
        max_iter=args.max_steps,
    )

    task = Task(
        description=task_prompt,
        expected_output=_expected_output_schema(args.limit),
        agent=agent,
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=args.verbose,
    )

    # Clear server result before starting
    await clear_server_result(args.url)

    # Track start time
    start_time = datetime.now()

    # Run crew
    try:
        if llm_logger:
            llm_logger.start_puzzle(
                puzzle_type='unknown',
                puzzle_id='crewai_run',
                prompt=task_prompt
            )

        result = crew.kickoff()

        if llm_logger:
            llm_logger.end_puzzle(
                answer=_extract_output(result),
                is_correct=None,  # Will be determined from server
                correct_answer=None
            )

    except Exception as e:
        logging.exception('CrewAI agent run failed: %s', e)
        if llm_logger:
            llm_logger.end_puzzle(
                answer=f'ERROR: {str(e)}',
                is_correct=False,
                correct_answer=None
            )
        return 1

    # Check for result from server
    elapsed_time = (datetime.now() - start_time).total_seconds()
    server_result = await check_server_for_result(args.url)

    if server_result.get('detected'):
        await record_benchmark_result(
            base_url=args.url,
            puzzle_type=server_result.get('puzzle_type', 'unknown'),
            puzzle_id=server_result.get('puzzle_id', 'unknown'),
            is_correct=server_result.get('is_correct'),
            user_answer=server_result.get('user_answer'),
            correct_answer=server_result.get('correct_answer'),
            elapsed_time=elapsed_time,
            model_name=model_name,
            provider_name=provider_name,
        )

    # Write summary
    if llm_logger:
        llm_logger.write_summary()

    output = _extract_output(result)
    if output:
        print(output)
    else:
        logging.warning('CrewAI agent did not return any output.')

    return 0


def _run_crewai(args: argparse.Namespace) -> str:
    """Synchronous wrapper for backward compatibility."""
    try:
        return asyncio.run(_run_crewai_async(args))
    except Exception as e:
        logging.exception('CrewAI agent run failed: %s', e)
        return 1


# =============================================================================
# CLI ARGUMENT PARSER
# =============================================================================

def _build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(description='Run OpenCaptchaWorld puzzles using a CrewAI agent.')
    llm_choices = sorted(_create_llm_factory().keys())

    parser.add_argument('--url', default='http://127.0.0.1:7860', help='URL of the running OpenCaptchaWorld instance.')
    parser.add_argument('--limit', type=int, default=3, help='Number of puzzle attempts before the agent stops.')
    parser.add_argument(
        '--llm',
        choices=llm_choices,
        default='openai',
        help='LLM provider to back the CrewAI agent.',
    )
    parser.add_argument('--model', help='Optional model override for the selected provider.')
    parser.add_argument('--temperature', type=float, default=0.2, help='Sampling temperature for the LLM.')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging from CrewAI.')

    # Provider-specific options
    parser.add_argument(
        '--reasoning-effort',
        choices=['none', 'low', 'medium', 'high', 'xhigh'],
        help='Reasoning effort level for GPT-5.2+ models (OpenAI only).'
    )
    parser.add_argument(
        '--max-output-tokens',
        type=int,
        default=None,
        help='Maximum output tokens for the LLM response (Google/Gemini and vLLM).'
    )
    parser.add_argument(
        '--thinking-budget',
        type=int,
        default=None,
        help='Thinking token budget for Gemini 2.5 models (0-32768 tokens).'
    )
    parser.add_argument(
        '--thinking-level',
        type=str,
        choices=['minimal', 'low', 'medium', 'high'],
        default=None,
        help='Thinking level for Gemini 3 models.'
    )
    parser.add_argument(
        '--disable-thinking',
        action='store_true',
        default=False,
        help='Disable thinking mode for Qwen thinking models (vLLM).'
    )

    # Timeout options
    parser.add_argument(
        '--llm-timeout',
        type=int,
        default=1800,
        help='LLM call timeout in seconds (default: 1800 = 30 minutes).'
    )
    parser.add_argument(
        '--step-timeout',
        type=int,
        default=1800,
        help='Step timeout in seconds (default: 1800 = 30 minutes).'
    )

    # Agent control
    parser.add_argument('--max-steps', type=int, default=1000, help='Maximum agent reasoning steps.')
    parser.add_argument(
        '--max-actions-per-step',
        type=int,
        default=10,
        help='Limit number of browser actions per reasoning step.',
    )
    parser.add_argument(
        '--max-failures',
        type=int,
        default=5,
        help='Consecutive failure limit before aborting. (default: 5)'
    )
    parser.add_argument(
        '--isolate-puzzles', '--no-memory',
        dest='no_memory',
        action='store_true',
        help='Isolate puzzles: spawn a fresh agent for each puzzle (no cross-puzzle memory).'
    )

    # Browser options
    parser.add_argument('--headless', action='store_true', help='Run the local browser in headless mode.')
    parser.add_argument('--window-width', type=int, help='Browser viewport width (pixels).')
    parser.add_argument('--window-height', type=int, help='Browser viewport height (pixels).')

    # Logging options
    parser.add_argument(
        '--no-log-llm',
        action='store_true',
        help='Disable LLM logging (enabled by default).'
    )
    parser.add_argument(
        '--llm-log-dir',
        type=str,
        default='llm_logs',
        help='Directory to save LLM logs (default: llm_logs)'
    )
    parser.add_argument(
        '--run-id',
        type=str,
        default=None,
        help='Run ID to use for logging. When provided, all logs from the same run will be grouped.'
    )

    # Custom endpoints
    parser.add_argument('--base-url', dest='base_url', help='Custom API base URL (required for vllm provider)')
    parser.add_argument('--api-key', dest='api_key', help='API key (use "EMPTY" for local vLLM)')
    parser.add_argument(
        '--debug-vllm',
        action='store_true',
        default=False,
        help='Enable verbose debug output for vLLM/Qwen thinking models.'
    )

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    try:
        return asyncio.run(_run_crewai_async(args))
    except KeyboardInterrupt:
        print('\nInterrupted by user.')
        return 1
    except (ImportError, ValueError) as exc:
        logging.error(str(exc))
        return 1
    except Exception as exc:
        logging.exception('CrewAI agent run failed: %s', exc)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
