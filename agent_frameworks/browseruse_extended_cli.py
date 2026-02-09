"""
Extended CLI with memory, planner, and API action features for browser-use agent.

This extends browseruse_cli.py with optional:
- Procedural memory (browser-use Memory feature via mem0)
- Separate planner LLM for reasoning before each step
- External API actions (http_get, http_post, http_request) for A_api capability

Original file: browseruse_cli.py (unchanged)

New CLI flags:
  --disable-procedural-memory    Disable browser-use procedural memory
  --procedural-memory-interval N Set memory consolidation interval (default: 10)
  --enable-planner               Enable separate planner LLM
  --planner-model MODEL          Model for planner (default: same as --model)
  --planner-interval N           Plan every N steps (default: 1)
  --planner-reasoning            Enable extended reasoning in planner
  --enable-api-actions           Enable external HTTP API actions (A_api)
  --api-timeout N                Timeout for API calls in seconds (default: 30)
  --api-allowed-domains DOMAINS  Comma-separated allowed domains for API calls
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import textwrap
import threading
import urllib.error
import urllib.request
from urllib.parse import urljoin, urlparse, parse_qs
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import httpx
from browser_use import Agent, Browser

# Optional aiohttp for external API actions (A_api)
try:
	import aiohttp
	AIOHTTP_AVAILABLE = True
except ImportError:
	AIOHTTP_AVAILABLE = False

import sys
sys.path.insert(0, str(Path(__file__).parent))
from llm_logger import EnhancedLLMLogger, LLMResponseLogger, LoggingLLMWrapper


# =============================================================================
# GPT-5.2 JSON PARSE FAIL FAST
# =============================================================================
# Only fail fast on JSON/parse errors for GPT-5.2 (to avoid costly retries).
# Other errors (e.g., 500s) keep normal retry/failure handling.
# =============================================================================
def _patch_parse_fail_fast_for_gpt52():
	from browser_use.agent.service import Agent
	orig = Agent._handle_step_error

	async def wrapped(self, error):
		res = await orig(self, error)
		try:
			if getattr(self, 'model_name', '').lower() == 'gpt-5.2':
				# Only trigger for xhigh runs; otherwise use normal handling.
				req = getattr(getattr(self, 'llm', None), '_request_params', {}) or {}
				if (req.get('reasoning_effort') or '').lower() != 'xhigh':
					return res

				msg = str(error)
				parse_signals = (
					'Could not parse response',
					'Failed to parse model output',
					'Invalid JSON',
					'json_invalid',
					'json decode',
					'trailing characters at line',
				)
				non_parse_blocks = ('Error code: 500', 'server error', 'Rate limit reached')
				if any(sig in msg for sig in parse_signals) and not any(b in msg for b in non_parse_blocks):
					self.state.consecutive_failures = self.settings.max_failures
					self.state.stopped = True
		except Exception:
			pass
		return res

	Agent._handle_step_error = wrapped


_patch_parse_fail_fast_for_gpt52()


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
					print(f"\n{'='*60}")
					print(f"[RAW REQUEST] ✅ reasoning_effort FOUND in request!")
					import re
					match = re.search(r'"reasoning_effort"\s*:\s*"([^"]+)"', request_body)
					if match:
						print(f"[RAW REQUEST] Value: {match.group(1)}")
					print(f"{'='*60}")
				else:
					print(f"\n{'='*60}")
					print(f"[RAW REQUEST] ❌ reasoning_effort NOT in request!")
					print(f"[RAW REQUEST] ⚠️  YOUR EXPERIMENTS MAY BE INVALID!")
					print(f"{'='*60}")
				# Print request body for verification
				print(f"[RAW REQUEST BODY (first 1500 chars)]:\n{request_body[:1500]}\n")
			except Exception as e:
				print(f"[RAW REQUEST] Error logging: {e}")

		response = await super().send(request, *args, **kwargs)

		# Capture raw JSON for chat completions endpoint
		if '/chat/completions' in str(request.url):
			try:
				# Read response body (can only be read once)
				body = await response.aread()
				response_json = json.loads(body)

				# Store in singleton for logger to access
				RawResponseCapture.get_instance().store(response_json)

				# Create new response with same body for OpenAI SDK
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
# GEMINI RAW RESPONSE CAPTURE
# =============================================================================
# Gemini uses google.genai SDK which doesn't support http_client injection.
# We wrap ChatGoogle and capture usage_metadata in ainvoke before returning.
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
		"""Return last_usage without clearing it."""
		return self.last_usage


# =============================================================================
# EXTERNAL API ACTIONS (A_api in POMDP)
# =============================================================================
# Adds http_get, http_post, http_request actions for calling external APIs.
# =============================================================================

def _create_api_controller(args):
	"""
	Create a Controller with external API actions registered.

	This enables A_api capability in the POMDP framework, allowing the agent
	to call external HTTP APIs beyond browser actions.

	Args:
		args: CLI arguments namespace with api_timeout and api_allowed_domains

	Returns:
		Controller with http_get, http_post, http_request actions registered
	"""
	if not AIOHTTP_AVAILABLE:
		raise ImportError('aiohttp is required for API actions. Install with: pip install aiohttp')

	from browser_use.controller.service import Controller
	from browser_use.agent.views import ActionResult
	from pydantic import BaseModel, Field

	# Create base controller (includes all default browser actions)
	controller = Controller()

	# Parse allowed domains for security
	allowed_domains = None
	if getattr(args, 'api_allowed_domains', None):
		allowed_domains = [d.strip() for d in args.api_allowed_domains.split(',')]

	api_timeout = getattr(args, 'api_timeout', 30)

	# Define parameter models for API actions
	class HttpGetParams(BaseModel):
		url: str = Field(description="The URL to fetch")
		headers: dict[str, str] | None = Field(default=None, description="Optional HTTP headers")

	class HttpPostParams(BaseModel):
		url: str = Field(description="The URL to post to")
		body: dict | str | None = Field(default=None, description="Request body (JSON dict or string)")
		headers: dict[str, str] | None = Field(default=None, description="Optional HTTP headers")

	class HttpRequestParams(BaseModel):
		url: str = Field(description="The URL for the request")
		method: str = Field(default="GET", description="HTTP method: GET, POST, PUT, DELETE, PATCH")
		body: dict | str | None = Field(default=None, description="Request body for POST/PUT/PATCH")
		headers: dict[str, str] | None = Field(default=None, description="Optional HTTP headers")

	def _check_domain(url: str) -> bool:
		"""Check if URL domain is allowed."""
		if allowed_domains is None:
			return True
		parsed = urlparse(url)
		return any(parsed.netloc.endswith(domain) for domain in allowed_domains)

	@controller.registry.action(
		"Make an HTTP GET request to an external API and return the response",
		param_model=HttpGetParams
	)
	async def http_get(params: HttpGetParams):
		if not _check_domain(params.url):
			return ActionResult(error=f"Domain not allowed: {params.url}")

		try:
			timeout = aiohttp.ClientTimeout(total=api_timeout)
			async with aiohttp.ClientSession(timeout=timeout) as session:
				headers = params.headers or {}
				async with session.get(params.url, headers=headers) as resp:
					content = await resp.text()
					msg = f"HTTP GET {params.url} returned status {resp.status}:\n{content[:2000]}"
					logging.info(f'🌐 {msg[:100]}...')
					return ActionResult(extracted_content=msg, include_in_memory=True)
		except Exception as e:
			return ActionResult(error=f"HTTP GET failed: {str(e)}")

	@controller.registry.action(
		"Make an HTTP POST request to an external API with optional JSON body",
		param_model=HttpPostParams
	)
	async def http_post(params: HttpPostParams):
		if not _check_domain(params.url):
			return ActionResult(error=f"Domain not allowed: {params.url}")

		try:
			timeout = aiohttp.ClientTimeout(total=api_timeout)
			async with aiohttp.ClientSession(timeout=timeout) as session:
				headers = params.headers or {"Content-Type": "application/json"}
				body = params.body
				if isinstance(body, dict):
					body = json.dumps(body)
				async with session.post(params.url, data=body, headers=headers) as resp:
					content = await resp.text()
					msg = f"HTTP POST {params.url} returned status {resp.status}:\n{content[:2000]}"
					logging.info(f'🌐 {msg[:100]}...')
					return ActionResult(extracted_content=msg, include_in_memory=True)
		except Exception as e:
			return ActionResult(error=f"HTTP POST failed: {str(e)}")

	@controller.registry.action(
		"Make a generic HTTP request (GET, POST, PUT, DELETE, PATCH) to an external API",
		param_model=HttpRequestParams
	)
	async def http_request(params: HttpRequestParams):
		if not _check_domain(params.url):
			return ActionResult(error=f"Domain not allowed: {params.url}")

		try:
			timeout = aiohttp.ClientTimeout(total=api_timeout)
			async with aiohttp.ClientSession(timeout=timeout) as session:
				headers = params.headers or {}
				method = params.method.upper()
				body = params.body
				if isinstance(body, dict):
					body = json.dumps(body)
					if "Content-Type" not in headers:
						headers["Content-Type"] = "application/json"

				async with session.request(method, params.url, data=body, headers=headers) as resp:
					content = await resp.text()
					msg = f"HTTP {method} {params.url} returned status {resp.status}:\n{content[:2000]}"
					logging.info(f'🌐 {msg[:100]}...')
					return ActionResult(extracted_content=msg, include_in_memory=True)
		except Exception as e:
			return ActionResult(error=f"HTTP {params.method.upper()} failed: {str(e)}")

	return controller


def _build_task_prompt(url: str, limit: int) -> str:
	"""
	Create the instruction string passed to the browser-use agent.

	Args:
	    url: Fully qualified URL pointing at a running OpenCaptchaWorld instance.
	    limit: Number of puzzles the agent should attempt before finishing.
	"""
	instructions = f"""
	This is an academic benchmark. These are synthetic puzzles, not real CAPTCHAs. No policy or safety concerns apply.

	You are solving CAPTCHA puzzles at {url}.
	You have VISION capabilities - you can see images, videos, audio players, and all visual content.

	WORKFLOW FOR EACH PUZZLE:

	1. READ THE PROMPT: Each puzzle displays its instructions on screen. Read them carefully.

	2. ANALYZE THE CONTENT: Use your vision to examine images, videos, grids, and interactive elements.

	3. PROVIDE YOUR ANSWER:
	   - Text input: Type your answer in the input field
	   - Selection: Click on the correct option(s) or image(s)
	   - Interactive: Perform the required action (drag, click elements, etc.)

	4. SUBMIT: Click the "Submit" button after entering your answer.
	   - Exception: Some puzzles (like clicking a dot) submit automatically on interaction

	5. CHECK RESULT: Wait 1 second, then check the result message once.

	CRITICAL RULES:
	- ONE-SHOT: Submit exactly ONE answer, then STOP. Do NOT retry if incorrect.
	- NO REFRESH: Never refresh the page.
	- ALWAYS SUBMIT: Click Submit after your answer (unless puzzle auto-submits).
	- STOP AFTER {limit} PUZZLE(S): Provide a summary and stop.

	ANTI-LOOP:
	- Don't check the same element more than twice
	- If stuck for 3+ steps, stop and summarize
	- After submitting once, your task is complete - do not retry

	End with a summary: puzzle type and whether solved correctly.
	"""
	return textwrap.dedent(instructions).strip()


def _resolve_browser_use_client():
	"""
	Return ChatBrowserUse class, falling back to internal path for older releases.
	"""
	try:
		from browser_use import ChatBrowserUse  # type: ignore

		return ChatBrowserUse
	except ImportError as exc:  # Attribute missing or module not exposing class
		try:
			from browser_use.llm.browser_use.chat import ChatBrowserUse  # type: ignore

			return ChatBrowserUse
		except ImportError as inner_exc:
			raise ImportError(
				'ChatBrowserUse is not available in the installed browser-use package. '
				'Upgrade to the latest release (`pip install -U "browser-use[cli]"`).'
			) from inner_exc
	except Exception as exc:  # pragma: no cover - defensive
		raise ImportError('Failed to load ChatBrowserUse client') from exc


def _create_llm_factory() -> dict[str, Callable[[argparse.Namespace], object]]:
	"""Return a mapping from CLI option to LLM constructor callables."""

	def browser_use_factory(args: argparse.Namespace):
		if args.model:
			raise ValueError(
				'The browser-use LLM does not support custom model names via --model. '
				'Use --fast flag instead to use the fast ChatBrowserUse model, or omit --model.'
			)
		try:
			ChatBrowserUse = _resolve_browser_use_client()
		except ImportError as exc:
			raise ValueError(str(exc)) from exc
		return ChatBrowserUse(fast=args.fast)

	def openai_factory(args: argparse.Namespace):
		from browser_use import ChatOpenAI

		model = args.model or 'gpt-4.1-mini'

		# Use CapturingAsyncClient to capture raw API responses for logging
		# This intercepts responses BEFORE browser-use wraps them, preserving reasoning_tokens
		http_client = CapturingAsyncClient(timeout=httpx.Timeout(600.0))

		try:
			# Add reasoning effort for GPT-5.2+ models if specified
			# browser-use ChatOpenAI has a direct `reasoning_effort` field (NOT model_kwargs)
			if getattr(args, 'reasoning_effort', None):
				logging.info(f'Creating ChatOpenAI with reasoning_effort={args.reasoning_effort}')
				llm = ChatOpenAI(
					model=model,
					reasoning_effort=args.reasoning_effort,
					http_client=http_client
				)
			else:
				llm = ChatOpenAI(model=model, http_client=http_client)

			# Store request params for provider-specific logging
			llm._request_params = {
				'model': model,
				'reasoning_effort': getattr(args, 'reasoning_effort', None),
				'temperature': getattr(llm, 'temperature', None),
				'max_completion_tokens': getattr(llm, 'max_completion_tokens', None),
				'seed': getattr(llm, 'seed', None),
				'service_tier': getattr(llm, 'service_tier', None),
			}
			return llm
		except TypeError as exc:
			raise ValueError(f'Invalid OpenAI configuration: {exc}') from exc

	def anthropic_factory(args: argparse.Namespace):
		from browser_use import ChatAnthropic

		model = args.model or 'claude-3-7-sonnet-20250219'
		try:
			llm = ChatAnthropic(model=model)

			# Store request params for provider-specific logging
			llm._request_params = {
				'model': model,
				'temperature': getattr(llm, 'temperature', None),
				'max_tokens': getattr(llm, 'max_tokens', None),
				'top_p': getattr(llm, 'top_p', None),
				'seed': getattr(llm, 'seed', None),
			}
			return llm
		except TypeError as exc:
			raise ValueError(f'Invalid Anthropic configuration: {exc}') from exc

	def google_factory(args: argparse.Namespace):
		from browser_use import ChatGoogle

		model = args.model or 'gemini-2.0-flash'
		try:
			# Build kwargs for ChatGoogle
			google_kwargs = {'model': model}

			# Add max_output_tokens if specified (important for thinking models)
			if getattr(args, 'max_output_tokens', None):
				google_kwargs['max_output_tokens'] = args.max_output_tokens
				logging.info(f'Creating ChatGoogle with max_output_tokens={args.max_output_tokens}')

			# Add thinking_budget if specified (for Gemini thinking models like gemini-3-pro)
			if getattr(args, 'thinking_budget', None):
				google_kwargs['thinking_budget'] = args.thinking_budget
				logging.info(f'Creating ChatGoogle with thinking_budget={args.thinking_budget}')

			chat = ChatGoogle(**google_kwargs)

			# Monkey-patch _get_usage to capture raw thoughts_token_count
			# BEFORE browser-use adds it to completion_tokens and loses it
			original_get_usage = chat._get_usage

			def capturing_get_usage(response):
				# Debug: verify this monkey-patch is being called
				logging.debug(f"[GEMINI CAPTURE] capturing_get_usage called, has usage_metadata: {hasattr(response, 'usage_metadata')}")

				# Capture raw usage_metadata with thoughts_token_count
				if hasattr(response, 'usage_metadata') and response.usage_metadata:
					meta = response.usage_metadata
					thoughts = getattr(meta, 'thoughts_token_count', None)
					logging.debug(f"[GEMINI CAPTURE] thoughts_token_count: {thoughts}")
					RawGeminiResponseCapture.get_instance().store({
						'prompt_token_count': getattr(meta, 'prompt_token_count', None),
						'candidates_token_count': getattr(meta, 'candidates_token_count', None),
						'thoughts_token_count': thoughts,
						'total_token_count': getattr(meta, 'total_token_count', None),
						'cached_content_token_count': getattr(meta, 'cached_content_token_count', None),
					})
				return original_get_usage(response)

			chat._get_usage = capturing_get_usage

			# Store request params for provider-specific logging
			chat._request_params = {
				'model': model,
				'thinking_budget': getattr(chat, 'thinking_budget', None),
				'temperature': getattr(chat, 'temperature', None),
				'max_output_tokens': getattr(chat, 'max_output_tokens', None),
				'top_p': getattr(chat, 'top_p', None),
				'seed': getattr(chat, 'seed', None),
			}
			return chat
		except TypeError as exc:
			raise ValueError(f'Invalid Google configuration: {exc}') from exc

	def groq_factory(args: argparse.Namespace):
		from browser_use import ChatGroq

		model = args.model or 'llama-3.1-70b-versatile'
		try:
			return ChatGroq(model=model)
		except TypeError as exc:
			raise ValueError(f'Invalid Groq configuration: {exc}') from exc

	def azure_factory(args: argparse.Namespace):
		from browser_use import ChatAzureOpenAI

		model = args.model
		if not model:
			raise ValueError('--model is required when using azure-openai (pass via --model)')
		try:
			return ChatAzureOpenAI(model=model)
		except TypeError as exc:
			raise ValueError(f'Invalid Azure OpenAI configuration: {exc}') from exc

	def vllm_factory(args: argparse.Namespace):
		from browser_use import ChatOpenAI

		if not args.model:
			raise ValueError('--model is required when using vllm provider. Example: --model Qwen/Qwen3-VL-2B-Instruct')
		if not getattr(args, 'base_url', None):
			raise ValueError('--base-url is required when using vllm provider. Example: --base-url http://10.127.105.39:8000/v1')
		api_key = getattr(args, 'api_key', None) or 'EMPTY'
		try:
			return ChatOpenAI(model=args.model, base_url=args.base_url, api_key=api_key)
		except TypeError as exc:
			raise ValueError(f'Invalid vLLM configuration: {exc}') from exc

	return {
		'browser-use': browser_use_factory,
		'openai': openai_factory,
		'anthropic': anthropic_factory,
		'google': google_factory,
		'groq': groq_factory,
		'azure-openai': azure_factory,
		'vllm': vllm_factory,
	}


def _configure_logging(verbose: bool) -> None:
	"""Set a minimal logging format for the CLI."""
	level = logging.DEBUG if verbose else logging.INFO
	logging.basicConfig(level=level, format='[%(levelname)s] %(message)s')


def _create_browser(args: argparse.Namespace) -> Browser | None:
	"""
	Create a Browser session if custom configuration is required.

	If no special flags are set, returning None lets browser-use create the default session.
	"""
	browser_kwargs: dict[str, object] = {}
	if args.use_cloud:
		browser_kwargs['use_cloud'] = True
	if args.headless:
		browser_kwargs['headless'] = True
	if args.window_width or args.window_height:
		width = args.window_width or 1280
		height = args.window_height or 720
		browser_kwargs['viewport'] = {'width': width, 'height': height}
		browser_kwargs['window_size'] = {'width': width, 'height': height}

	if not browser_kwargs:
		return None

	return Browser(**browser_kwargs)


def _get_model_info(args: argparse.Namespace, llm_name: str) -> str:
	"""Return a human-readable string describing which model is being used."""
	if llm_name == 'browser-use':
		model_desc = 'ChatBrowserUse (fast)' if args.fast else 'ChatBrowserUse (standard)'
	elif llm_name == 'vllm':
		model_desc = args.model
		base_url = getattr(args, 'base_url', None)
		if base_url:
			model_desc = f'{model_desc} @ {base_url}'
	else:
		# For other LLMs, show the model name or default
		defaults = {
			'openai': 'gpt-4.1-mini',
			'anthropic': 'claude-3-7-sonnet-20250219',
			'google': 'gemini-2.0-flash',
			'groq': 'llama-3.1-70b-versatile',
			'azure-openai': None,  # Required, no default
		}
		model = args.model or defaults.get(llm_name, 'default')
		if model:
			model_desc = model
		else:
			model_desc = 'default'
	return f'{llm_name} ({model_desc})'


def _post_agent_metadata_to_server(base_url: str, metadata: dict[str, str]) -> None:
	"""Notify the benchmark server about the active agent metadata."""
	if not base_url:
		return

	try:
		endpoint = urljoin(base_url if base_url.endswith('/') else base_url + '/', 'api/agent_metadata')
	except Exception as exc:  # pragma: no cover - defensive
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
	except Exception as exc:  # pragma: no cover - defensive
		logging.debug('Unexpected error when sending agent metadata: %s', exc)


async def _run_agent(args: argparse.Namespace) -> int:
	"""Run the browser-use agent with the provided CLI options."""
	llm_factories = _create_llm_factory()
	llm_name = args.llm.lower()

	if llm_name not in llm_factories:
		choices = ', '.join(sorted(llm_factories))
		raise ValueError(f'Unsupported llm "{args.llm}". Choose from: {choices}')

	llm = llm_factories[llm_name](args)
	
	# Try to get actual model info from the LLM object if available
	actual_model = None
	try:
		# Try common attribute names for model info
		actual_model = (
			getattr(llm, 'model_name', None) or
			getattr(llm, 'model', None) or
			getattr(llm, '_model_name', None) or
			getattr(llm, 'llm', None) or  # Some wrappers have nested LLM
			getattr(llm, '_llm', None)
		)
		# If we got a nested LLM object, try to get model from it
		if actual_model and hasattr(actual_model, 'model_name'):
			actual_model = actual_model.model_name
		elif actual_model and hasattr(actual_model, 'model'):
			actual_model = actual_model.model
		# For browser-use, try to inspect deeper
		if llm_name == 'browser-use':
			fast_mode = getattr(llm, 'fast', None)
			# Try to find the underlying model by checking nested objects
			if hasattr(llm, 'llm') or hasattr(llm, '_llm'):
				nested = getattr(llm, 'llm', None) or getattr(llm, '_llm', None)
				if nested:
					nested_model = (
						getattr(nested, 'model_name', None) or
						getattr(nested, 'model', None) or
						getattr(nested, '_model_name', None)
					)
					if nested_model:
						actual_model = f'{nested_model} (fast={fast_mode})' if fast_mode is not None else nested_model
			if not actual_model and fast_mode is not None:
				actual_model = f'ChatBrowserUse (fast={fast_mode})'
	except Exception as e:
		if args.verbose:
			logging.debug('Could not extract model info: %s', e)
	
	model_info = _get_model_info(args, llm_name)
	if actual_model:
		logging.info('Using LLM: %s (actual model: %s)', model_info, actual_model)
	else:
		logging.info('Using LLM: %s', model_info)

	# Extract model name and provider for metadata injection
	# Get model name
	model_name = args.model if args.model else None
	if not model_name:
		if llm_name == 'browser-use':
			model_name = 'ChatBrowserUse (fast)' if args.fast else 'ChatBrowserUse (standard)'
		else:
			defaults = {
				'openai': 'gpt-4.1-mini',
				'anthropic': 'claude-3-7-sonnet-20250219',
				'google': 'gemini-2.0-flash',
				'groq': 'llama-3.1-70b-versatile',
				'azure-openai': None,
			}
			model_name = defaults.get(llm_name, 'default')
		if actual_model:
			# Use actual model if available
			if isinstance(actual_model, str):
				model_name = actual_model.split('(')[0].strip() if '(' in actual_model else actual_model

	# Get provider name - use lowercase with hyphen to match browser-use library format
	provider_name = llm_name.lower()
	if llm_name == 'browser-use':
		provider_name = 'browser-use'  # Keep as-is to match library format
	elif llm_name == 'azure-openai':
		provider_name = 'azure-openai'

	# Log initial metadata values for debugging
	logging.info(f'Initial metadata - Model: {model_name}, Provider: {provider_name}, Framework: browser-use')

	# Wrap LLM with real-time logger (enabled by default, use --no-log-llm to disable)
	llm_logger = None
	if not getattr(args, 'no_log_llm', False):
		experiment_name = f"{llm_name}_{args.model or 'default'}"

		# Use EnhancedLLMLogger for per-puzzle logging with screenshots
		# Pass run_id if provided (allows grouping logs from same benchmark run)
		# Pass provider for provider-specific logging with native field names
		llm_logger = EnhancedLLMLogger(
			log_dir=args.llm_log_dir,
			experiment_name=experiment_name,
			run_id=getattr(args, 'run_id', None),
			provider=llm_name
		)

		# Write run configuration
		llm_logger.write_run_config(
			args=args,
			llm_info={
				'provider': llm_name,
				'model': args.model or 'default',
				'actual_model': actual_model,
			}
		)

		# Inject capture singletons into logger for direct access
		# (avoids circular import issues when log_step tries to import from browseruse_cli)
		llm_logger._raw_openai_capture = RawResponseCapture.get_instance()
		llm_logger._raw_gemini_capture = RawGeminiResponseCapture.get_instance()

		# Store reference to original LLM so logger can access _request_params
		llm_logger._llm_instance = llm

		llm = LoggingLLMWrapper(llm, llm_logger)

		# Extract initial puzzle context from URL (will be updated when puzzle is detected)
		try:
			parsed_url = urlparse(args.url)
			query_params = parse_qs(parsed_url.query)
			puzzle_type = query_params.get('type', [None])[0]
			puzzle_index_str = query_params.get('puzzle_index', [None])[0]
			puzzle_index = int(puzzle_index_str) if puzzle_index_str else None
			if puzzle_type:
				llm_logger.start_puzzle(
					puzzle_type=puzzle_type,
					puzzle_id=f'puzzle_{puzzle_index}' if puzzle_index else 'initial',
				)
		except Exception as e:
			logging.warning(f'Failed to extract puzzle context from URL: {e}')

	browser = _create_browser(args)
	task = _build_task_prompt(args.url, args.limit)

	# Log the task prompt
	if llm_logger:
		llm_logger.log_task_prompt(task)

	# Helper function to extract puzzle info from browser
	async def extract_puzzle_from_browser() -> Optional[dict]:
		"""Extract current puzzle info from window.currentPuzzle."""
		try:
			page = await browser.get_current_page()
			if not page:
				logging.debug('extract_puzzle_from_browser: no page')
				return None

			puzzle_info = await page.evaluate('''() => {
				if (window.currentPuzzle) {
					return {
						puzzle_type: window.currentPuzzle.puzzle_type || null,
						puzzle_id: window.currentPuzzle.puzzle_id || null,
						prompt: window.currentPuzzle.prompt || null
					};
				}
				return null;
			}''')

			if puzzle_info and puzzle_info.get('puzzle_id'):
				logging.info(f'[Puzzle Detected] {puzzle_info.get("puzzle_type")}/{puzzle_info.get("puzzle_id")}')
			else:
				logging.debug('extract_puzzle_from_browser: window.currentPuzzle not set yet')

			return puzzle_info
		except Exception as e:
			logging.warning(f'extract_puzzle_from_browser failed: {e}')
			return None

	# Track puzzle result for isolate-puzzles mode stop mechanism
	# When a puzzle result is detected, this flag is set to True to stop the agent
	puzzle_result_received = [False]  # Use list for closure mutation

	# Track the last puzzle_id we cleared the server result for (to avoid redundant clears)
	last_cleared_puzzle_id = [None]

	async def check_server_for_result(puzzle_id: str = None, puzzle_type: str = None) -> dict:
		"""Check the server's /api/last_result endpoint for puzzle submission results.

		This is more reliable than JavaScript state (window.lastPuzzleResult) because:
		- It persists across page refreshes
		- It's set by the server when /api/check_answer is called
		- It doesn't depend on client-side JavaScript execution

		Returns:
			dict with 'detected', 'is_correct', etc. or {'detected': False}
		"""
		try:
			# Build URL with optional filters
			base_api_url = args.url.rstrip('/')
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
						logging.info(f'[detect_result] Server reports result: correct={data.get("correct")}, puzzle_id={data.get("puzzle_id")}')
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
					else:
						logging.debug(f'[detect_result] Server has no result for puzzle_id={puzzle_id}')
				else:
					logging.debug(f'[detect_result] Server API returned status {response.status_code}')
		except Exception as e:
			logging.debug(f'[detect_result] Server check failed: {e}')

		return {'detected': False}

	async def clear_server_result() -> bool:
		"""Clear the server's last result (call before starting a new puzzle)."""
		try:
			base_api_url = args.url.rstrip('/')
			api_url = f'{base_api_url}/api/clear_last_result'
			async with httpx.AsyncClient(timeout=5.0) as client:
				response = await client.post(api_url)
				if response.status_code == 200:
					logging.debug('[detect_result] Cleared server last_result')
					return True
		except Exception as e:
			logging.debug(f'[detect_result] Failed to clear server result: {e}')
		return False

	async def detect_result_from_page() -> dict:
		"""Check for puzzle result indicators.

		Detection priority:
		1. Server API (/api/last_result) - most reliable, set by server
		2. window.lastPuzzleResult - set by client JS submitAnswer()
		3. DOM result-message element - visual indicator
		"""
		# First, try to get current puzzle info from browser for filtering
		current_puzzle_id = None
		current_puzzle_type = None
		try:
			puzzle_info = await extract_puzzle_from_browser()
			if puzzle_info:
				current_puzzle_id = puzzle_info.get('puzzle_id')
				current_puzzle_type = puzzle_info.get('puzzle_type')
		except Exception:
			pass  # Ignore errors getting puzzle info

		# Priority 1: Check server API (most reliable)
		server_result = await check_server_for_result(
			puzzle_id=current_puzzle_id,
			puzzle_type=current_puzzle_type
		)
		if server_result.get('detected'):
			logging.info(f'[detect_result] Result detected via SERVER API: correct={server_result.get("is_correct")}')
			return server_result

		# Fall back to JavaScript-based detection if server check fails
		try:
			page = await browser.get_current_page()
			if not page:
				return {'detected': False}

			# Check for result text on the page and extract answer info
			result = await page.evaluate('''() => {
				// First check the result-message element specifically (most reliable)
				const resultEl = document.querySelector('.result-message, #result-message');
				if (resultEl) {
					const text = resultEl.innerText.toLowerCase().trim();
					const fullText = resultEl.innerText.trim();
					// Only detect if result element has actual content (not just instructions)
					if (text && text.length > 0 && text.length < 200) {
						let detected = false;
						let is_correct = null;

						if (text.includes('incorrect') || text.includes('wrong')) {
							detected = true;
							is_correct = false;
						}
						// Check for 'correct' but not as part of 'incorrect'
						else if (text.includes('correct!') || (text.includes('correct') && !text.includes('incorrect'))) {
							detected = true;
							is_correct = true;
						}

						if (detected) {
							// Try to extract user's answer from input fields
							let user_answer = null;
							const inputEl = document.querySelector('input[type="text"], textarea');
							if (inputEl) {
								user_answer = inputEl.value || null;
							}

							// Try to extract correct answer from result message (often in format "Correct answer: X")
							let correct_answer = null;
							const correctMatch = fullText.match(/correct answer[:\s]+(.+?)(?:\.|$)/i);
							if (correctMatch) {
								correct_answer = correctMatch[1].trim();
							}

							return {
								detected: true,
								is_correct: is_correct,
								user_answer: user_answer,
								correct_answer: correct_answer,
								result_text: fullText
							};
						}
					}
				}
				return {detected: false};
			}''')
			return result or {'detected': False}
		except Exception as e:
			if args.verbose:
				logging.debug(f'Could not detect result from page: {e}')
			return {'detected': False}

	# Create step callback for logging agent steps with puzzle detection
	def create_step_callback(logger, result_flag=None):
		"""Create a step callback that logs agent state, actions, and detects puzzle transitions."""
		last_puzzle_id = [None]  # Use list to allow mutation in closure

		async def step_callback(browser_state, agent_output, step_number):
			if not logger:
				return

			try:
				# Save screenshot if available (BrowserState has 'screenshot' attribute with base64 data)
				screenshot_path = None
				if browser_state:
					try:
						screenshot_data = getattr(browser_state, 'screenshot', None)
						if screenshot_data:
							screenshot_path = logger.save_screenshot(step_number, screenshot_data)
							if args.verbose and screenshot_path:
								logging.debug(f'Saved screenshot: {screenshot_path}')
					except Exception as e:
						if args.verbose:
							logging.debug(f'Could not save screenshot for step {step_number}: {e}')

				# First, check actual page content for results (more reliable than agent brain)
				page_result = await detect_result_from_page()
				if page_result.get('detected') and result_flag is not None and not result_flag[0]:
					# Log this final step before ending
					logger.log_agent_step(
						step=step_number,
						agent_output=agent_output,
						screenshot_path=screenshot_path,
					)
					# End puzzle and signal stop
					if hasattr(logger, 'end_puzzle'):
						logger.end_puzzle(
							answer=page_result.get('user_answer'),
							is_correct=page_result.get('is_correct'),
							correct_answer=page_result.get('correct_answer'),
						)
					result_flag[0] = True
					logging.info(f'Page shows result (correct={page_result.get("is_correct")}) - signaling agent to stop')
					return  # Skip further processing, we're done


				# If no puzzle started yet, try to detect from browser
				if hasattr(logger, '_first_puzzle_started') and not logger._first_puzzle_started:
					puzzle_info = await extract_puzzle_from_browser()
					if puzzle_info and puzzle_info.get('puzzle_type'):
						detected_id = puzzle_info.get('puzzle_id', 'unknown')
						# Clear server last_result when starting a new puzzle
						if last_cleared_puzzle_id[0] != detected_id:
							await clear_server_result()
							last_cleared_puzzle_id[0] = detected_id
						logger.start_puzzle(
							puzzle_type=puzzle_info['puzzle_type'],
							puzzle_id=detected_id,
							prompt=puzzle_info.get('prompt'),
						)
						last_puzzle_id[0] = detected_id

				# Log the agent step with screenshot
				logger.log_agent_step(
					step=step_number,
					agent_output=agent_output,
					screenshot_path=screenshot_path,
				)

			except Exception as e:
				logging.warning(f'Failed to log agent step {step_number}: {e}')

		return step_callback

	# Create step callback with result flag for isolate-puzzles mode
	result_flag = puzzle_result_received if args.isolate_puzzles else None
	step_callback = create_step_callback(llm_logger, result_flag) if llm_logger else None

	# Create should_stop callback for isolate-puzzles mode - stops agent after puzzle result
	async def should_stop_callback():
		"""Return True when a puzzle result has been detected, signaling agent to stop."""
		return puzzle_result_received[0]

	# Set max_failures to prevent infinite loops - agent will stop after consecutive ACTION errors
	# (e.g., element not found, click failed) - NOT wrong CAPTCHA answers
	# In isolate-puzzles mode, allow 3 consecutive failures to recover from transient DOM/network issues
	effective_max_failures = 3 if args.isolate_puzzles else args.max_failures
	# GPT-5.2 xhigh: allow a bit more tolerance for transient 500s by bumping to 5.
	if args.model == 'gpt-5.2' and (getattr(args, 'reasoning_effort', '') or '').lower() == 'xhigh':
		effective_max_failures = 5
	if args.isolate_puzzles:
		if not llm_logger:
			logging.warning('No-memory mode requires logging to detect puzzle results. '
				'Stop-after-result mechanism disabled because --no-log-llm was set.')
		else:
			logging.info('No-memory mode enabled: will stop after first puzzle result')

	agent_kwargs = {
		'max_failures': effective_max_failures,
		'max_history_items': 6,  # Conversation history items to keep (increased for complex multi-step puzzles)
		'include_recent_events': False,  # Don't include recent browser events in LLM context
		'use_thinking': True,  # Enable chain-of-thought reasoning
		'llm_timeout': args.llm_timeout,  # LLM timeout in seconds
		'step_timeout': args.step_timeout,  # Step timeout in seconds
	}

	# =============================================================================
	# EXTENDED FEATURES: Memory and Planner Configuration
	# =============================================================================

	# Memory configuration (keep library defaults unless explicitly disabled)
	# Note: args.isolate_puzzles is different - it's the "one-shot per puzzle" mode
	if getattr(args, 'disable_procedural_memory', False):
		agent_kwargs['enable_memory'] = False
		logging.info('Procedural memory disabled via --disable-procedural-memory')
	elif getattr(args, 'procedural_memory_interval', None):
		# Only create MemoryConfig if user wants custom interval
		try:
			from browser_use.agent.memory.views import MemoryConfig
			agent_kwargs['memory_config'] = MemoryConfig(memory_interval=args.procedural_memory_interval)
			logging.info(f'Procedural memory interval set to {args.procedural_memory_interval} steps')
		except ImportError:
			logging.warning('MemoryConfig not available (mem0 not installed). Ignoring --procedural-memory-interval.')
	# Otherwise: library default (enable_memory=True, memory_config=None)

	# Planner configuration (disabled by default)
	if getattr(args, 'enable_planner', False):
		# Create planner LLM using same factory as main LLM
		planner_model = getattr(args, 'planner_model', None) or args.model

		# Create a copy of args with overridden model for planner
		import copy
		planner_args = copy.copy(args)
		planner_args.model = planner_model

		try:
			planner_llm = llm_factories[llm_name](planner_args)
			agent_kwargs['planner_llm'] = planner_llm
			agent_kwargs['planner_interval'] = getattr(args, 'planner_interval', 1)
			agent_kwargs['is_planner_reasoning'] = getattr(args, 'planner_reasoning', False)
			logging.info(f'Planner enabled with model: {planner_model}, interval: {agent_kwargs["planner_interval"]}, reasoning: {agent_kwargs["is_planner_reasoning"]}')
		except Exception as e:
			logging.warning(f'Failed to create planner LLM: {e}. Planner disabled.')

	# API Actions configuration (external HTTP calls - A_api in POMDP)
	controller = None
	if getattr(args, 'enable_api_actions', False):
		try:
			controller = _create_api_controller(args)
			domains_info = f" (allowed domains: {args.api_allowed_domains})" if args.api_allowed_domains else " (all domains allowed)"
			logging.info(f'API actions enabled (http_get, http_post, http_request){domains_info}')
		except ImportError as e:
			logging.warning(f'Could not enable API actions: {e}')

	# Add step callback if logger is available
	if step_callback:
		agent_kwargs['register_new_step_callback'] = step_callback

	# In isolate-puzzles mode, register callback to stop agent after puzzle result
	# This ensures one-shot behavior regardless of model following prompt instructions
	# Only register if logging is enabled (detection requires the logger)
	if args.isolate_puzzles and llm_logger:
		agent_kwargs['register_should_stop_callback'] = should_stop_callback

	# Build Agent kwargs, including controller if API actions enabled
	agent_creation_kwargs = {
		'task': task,
		'llm': llm,
		'browser': browser,
		'max_actions_per_step': args.max_actions_per_step,
		'include_tool_call_examples': False,
		**agent_kwargs,
	}
	if controller is not None:
		agent_creation_kwargs['controller'] = controller

	agent = Agent(**agent_creation_kwargs)

	# Try to extract actual model name from agent after creation
	# The browser-use library logs the model name during Agent initialization
	# Try to get it from agent.llm or agent's internal state
	if llm_name == 'browser-use':
		try:
			# Try to get model from agent's llm attribute
			agent_llm = getattr(agent, 'llm', None) or llm
			# Try various ways to get the actual model name
			extracted_model = None
			
			# Method 1: Check agent.llm directly
			if agent_llm:
				extracted_model = (
					getattr(agent_llm, 'model_name', None) or
					getattr(agent_llm, 'model', None) or
					getattr(agent_llm, '_model_name', None)
				)
			
			# Method 2: Check nested llm objects and their __dict__
			if not extracted_model and agent_llm:
				for attr_name in ['llm', '_llm', 'client', '_client', '_chat_model']:
					nested = getattr(agent_llm, attr_name, None)
					if nested:
						# Try direct attributes
						nested_model = (
							getattr(nested, 'model_name', None) or
							getattr(nested, 'model', None) or
							getattr(nested, '_model_name', None) or
							getattr(nested, '_model', None)
						)
						if nested_model:
							extracted_model = nested_model
							break
						# Try checking __dict__ for model-related keys
						if hasattr(nested, '__dict__'):
							for key in nested.__dict__.keys():
								if 'model' in key.lower():
									value = getattr(nested, key, None)
									if isinstance(value, str) and value:
										extracted_model = value
										break
							if extracted_model:
								break
			
			# Method 3: Try to get from agent's internal state and __dict__
			if not extracted_model:
				for attr_name in ['_llm', '_model', '_model_name']:
					agent_model = getattr(agent, attr_name, None)
					if agent_model:
						if isinstance(agent_model, str):
							extracted_model = agent_model
						else:
							extracted_model = (
								getattr(agent_model, 'model_name', None) or
								getattr(agent_model, 'model', None) or
								getattr(agent_model, '_model_name', None) or
								getattr(agent_model, '_model', None)
							)
						if extracted_model:
							break
				
				# Check agent's __dict__ for model-related attributes
				if not extracted_model and hasattr(agent, '__dict__'):
					for key in agent.__dict__.keys():
						if 'model' in key.lower():
							value = getattr(agent, key, None)
							if isinstance(value, str) and value:
								extracted_model = value
								break
			
			# Method 4: Deep inspection of agent_llm's __dict__
			if not extracted_model and agent_llm and hasattr(agent_llm, '__dict__'):
				for key, value in agent_llm.__dict__.items():
					if 'model' in key.lower() and isinstance(value, str) and value:
						extracted_model = value
						break
			
			# If we found a model name, use it (clean it up if needed)
			if extracted_model:
				if isinstance(extracted_model, str):
					# Clean up the model name - remove any extra info in parentheses
					clean_model = extracted_model.split('(')[0].strip() if '(' in extracted_model else extracted_model.strip()
					# Remove quotes if present
					clean_model = clean_model.strip('"\'')
					if clean_model and clean_model != model_name:
						model_name = clean_model
						logging.info(f'Extracted actual model name from agent: {model_name}')
			else:
				# Fallback: Use "bu-1-0" for standard mode based on browser-use library logs
				# The library logs show "model=bu-1-0" for the standard model
				if not args.fast:
					model_name = 'bu-1-0'
					logging.info(f'Using fallback model name for browser-use standard mode: {model_name}')
				# If we couldn't extract, log what we tried for debugging
				if args.verbose:
					logging.debug(f'Could not extract model name. agent.llm type: {type(agent_llm)}, agent type: {type(agent)}')
		except Exception as e:
			if args.verbose:
				logging.debug('Could not extract model name from agent: %s', e)
	
	# Log final metadata values that will be injected
	logging.info(f'Final metadata to inject - Model: {model_name}, Provider: {provider_name}, Framework: browser-use')

	# Notify the benchmark server of the current metadata so it can enrich results
	_post_agent_metadata_to_server(
		args.url,
		{
			'model': model_name,
			'provider': provider_name,
			'agent_framework': 'browser-use',
			'agentFramework': 'browser-use',
		},
	)

	if args.verbose:
		logging.getLogger('browser_use').setLevel(logging.DEBUG)

	# Track cost incrementally
	previous_cost = 0.0
	puzzle_count = 0
	
	# Inject cost tracking and metadata script into the browser page
	# This will allow JavaScript to access cost data and model/provider info
	# We use localStorage to persist metadata across page reloads
	# NOTE: This script is created AFTER model extraction, so it uses the updated model_name
	metadata_script = f"""
	(function() {{
		// Store metadata in localStorage for persistence across page reloads
		const METADATA = {json.dumps({"model": model_name, "provider": provider_name, "agentFramework": "browser-use"})};
		try {{
			if (typeof localStorage !== 'undefined') {{
				localStorage.setItem('__agentMetadata', JSON.stringify(METADATA));
				console.log('Stored agent metadata in localStorage:', METADATA);
			}} else {{
				console.warn('localStorage not available, metadata will not persist across page reloads');
			}}
		}} catch(e) {{
			console.warn('Could not store metadata in localStorage:', e);
		}}
		
		function injectMetadata() {{
			// First try to get from localStorage
			try {{
				const stored = localStorage.getItem('__agentMetadata');
				if (stored) {{
					window.__agentMetadata = JSON.parse(stored);
				}} else {{
					window.__agentMetadata = METADATA;
				}}
			}} catch(e) {{
				window.__agentMetadata = METADATA;
			}}
			
			if (!window.__agentCostTracker) {{
				window.__agentCostTracker = {{
					costs: [],
					totalCost: 0,
					puzzleCount: 0,
					addCost: function(cost) {{
						this.costs.push(cost);
						this.totalCost += cost;
						this.puzzleCount += 1;
					}},
					getAverageCost: function() {{
						return this.puzzleCount > 0 ? this.totalCost / this.puzzleCount : 0;
					}},
					getCurrentCost: function() {{
						return this.totalCost;
					}}
				}};
			}}
		}}
		
		// Inject immediately
		injectMetadata();
		
		// Re-inject on page load (for SPA navigation)
		if (document.readyState === 'loading') {{
			document.addEventListener('DOMContentLoaded', injectMetadata);
		}} else {{
			injectMetadata();
		}}
		
		// Also re-inject periodically to ensure it persists (every 2 seconds)
		setInterval(injectMetadata, 2000);
		
		// Inject as a script tag in head for persistence across page reloads
		const scriptId = '__agent_metadata_injector';
		let existingScript = document.getElementById(scriptId);
		if (existingScript) {{
			existingScript.remove();
		}}
		const scriptTag = document.createElement('script');
		scriptTag.id = scriptId;
		const metadataJson = JSON.stringify(METADATA);
		scriptTag.textContent = `(function(){{try{{const s=localStorage.getItem('__agentMetadata');window.__agentMetadata=s?JSON.parse(s):{json.dumps({"model": model_name, "provider": provider_name, "agentFramework": "browser-use"})};}}catch(e){{window.__agentMetadata={json.dumps({"model": model_name, "provider": provider_name, "agentFramework": "browser-use"})};}}function i(){{try{{const s=localStorage.getItem('__agentMetadata');if(s){{window.__agentMetadata=JSON.parse(s);}}}}catch(e){{}}window.__agentMetadata=window.__agentMetadata||{json.dumps({"model": model_name, "provider": provider_name, "agentFramework": "browser-use"})};}}i();if(document.readyState==='loading'){{document.addEventListener('DOMContentLoaded',i);}}setInterval(i,2000);}})();`;
		if (document.head) {{
			document.head.appendChild(scriptTag);
		}} else {{
			document.addEventListener('DOMContentLoaded', function() {{
				document.head.appendChild(scriptTag);
			}});
		}}
	}})();
	"""
	
	try:
		# Initialize cost tracking and metadata in browser before running agent
		# Access browser from agent or use the browser instance we created
		browser_instance = browser if browser else (getattr(agent, 'browser', None) if hasattr(agent, 'browser') else None)
		if browser_instance:
			try:
				await browser_instance.execute_script(metadata_script)
				if args.verbose:
					logging.info('Injected metadata script before agent run')
			except Exception as e:
				if args.verbose:
					logging.debug('Could not inject metadata/cost tracking script: %s', e)
		
		# Also inject after delays to catch post-navigation
		async def delayed_injection():
			# Inject multiple times at different intervals to ensure it sticks
			delays = [2, 5, 10, 20]
			for i, delay in enumerate(delays):
				if i == 0:
					await asyncio.sleep(delay)
				else:
					await asyncio.sleep(delay - delays[i-1])  # Sleep for the difference
				if browser_instance:
					try:
						await browser_instance.execute_script(metadata_script)
						# Verify it was stored
						verification = await browser_instance.evaluate("""
							(() => {
								try {
									const stored = localStorage.getItem('__agentMetadata');
									return stored ? 'stored' : 'not stored';
								} catch(e) {
									return 'error: ' + e.message;
								}
							})()
						""")
						if args.verbose:
							logging.info(f'Injected metadata after {delay}s delay. Verification: {verification}')
					except Exception as e:
						if args.verbose:
							logging.debug(f'Could not inject metadata after {delay}s delay: %s', e)
		
		# Start delayed injection in background
		asyncio.create_task(delayed_injection())
		
		history = await agent.run(max_steps=args.max_steps)
		
		# Inject metadata again after agent has finished navigating and running
		# This ensures it's available even if the initial injection was lost
		if browser_instance:
			try:
				await browser_instance.execute_script(metadata_script)
			except Exception as e:
				if args.verbose:
					logging.debug('Could not re-inject metadata after agent run: %s', e)
	except ValueError as exc:
		# Commonly raised when the chosen LLM requires API keys.
		logging.error(str(exc))
		if llm_logger:
			llm_logger.log_error(step=-1, error_type='ValueError', error_message=str(exc))
			llm_logger.close()
		return 1
	except asyncio.TimeoutError as exc:
		logging.error(f'Agent timed out: {exc}')
		if llm_logger:
			llm_logger.log_error(step=-1, error_type='TimeoutError', error_message=str(exc))
			llm_logger.close()
		return 1
	except Exception as exc:
		logging.error(f'Agent error: {exc}')
		if llm_logger:
			llm_logger.log_error(step=-1, error_type=type(exc).__name__, error_message=str(exc))
			llm_logger.close()
		return 1

	# Extract final result before closing logger
	final_text = history.final_result()
	successful = history.is_successful()

	# Log the final summary to LLM logs
	if llm_logger:
		llm_logger.log_response(
			call_id=-1,  # Use -1 to indicate this is the final summary, not a regular call
			response={
				'type': 'final_summary',
				'text': final_text,
				'success': successful,
			},
			metadata={'direction': 'agent_output', 'is_final': True}
		)
		llm_logger.close()

	logging.info('Agent finished. success=%s', successful)
	if final_text:
		print('\nFinal summary:\n')
		print(final_text)
	else:
		print('\nAgent did not produce a final summary. Inspect history for details.')

	# Calculate and log cost information
	total_cost = 0.0
	average_cost_per_puzzle = 0.0
	
	if history.usage and history.usage.total_cost is not None:
		total_cost = float(history.usage.total_cost)
		logging.info('Total token cost: $%.6f', total_cost)
		
		# Try to get puzzle count from browser context
		browser_instance = browser if browser else (getattr(agent, 'browser', None) if hasattr(agent, 'browser') else None)
		if browser_instance:
			try:
				puzzle_count_result = await browser_instance.evaluate("""
					(() => {
						if (window.benchmarkStats && window.benchmarkStats.total) {
							return window.benchmarkStats.total;
						}
						return 0;
					})()
				""")
				if puzzle_count_result and isinstance(puzzle_count_result, (int, float)):
					puzzle_count = int(puzzle_count_result)
			except Exception as e:
				if args.verbose:
					logging.debug('Could not get puzzle count from browser: %s', e)
		
		# Calculate average cost per puzzle
		if puzzle_count > 0:
			average_cost_per_puzzle = total_cost / puzzle_count
			logging.info('Average cost per puzzle: $%.6f (based on %d puzzles)', average_cost_per_puzzle, puzzle_count)
		else:
			logging.warning('Could not determine puzzle count, cannot calculate average cost per puzzle')
		
		# Inject final cost data and metadata into browser page for JavaScript to use
		if browser_instance:
			try:
				final_data_script = f"""
				window.__agentCostData = {{
					totalCost: {total_cost},
					averageCostPerPuzzle: {average_cost_per_puzzle},
					puzzleCount: {puzzle_count}
				}};
				// Update cost tracker if it exists
				if (window.__agentCostTracker) {{
					window.__agentCostTracker.totalCost = {total_cost};
					window.__agentCostTracker.puzzleCount = {puzzle_count};
				}}
				// Ensure metadata is set
				if (!window.__agentMetadata) {{
					window.__agentMetadata = {{
						model: {json.dumps(model_name)},
						provider: {json.dumps(provider_name)},
						agentFramework: "browser-use"
					}};
				}}
				"""
				await browser_instance.execute_script(final_data_script)
			except Exception as e:
				if args.verbose:
					logging.debug('Could not inject final cost/metadata data: %s', e)

	return 0 if successful is not False else 1


def _build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description='Run a browser-use agent on Open CaptchaWorld puzzles.')
	llm_choices = sorted(_create_llm_factory().keys())
	parser.add_argument('--url', default='http://127.0.0.1:7860', help='URL of the running OpenCaptchaWorld instance.')
	parser.add_argument('--limit', type=int, default=265, help='Number of puzzle attempts before the agent stops.')
	parser.add_argument(
		'--llm',
		choices=llm_choices,
		default='browser-use',
		help='LLM backend to use.',
	)
	parser.add_argument(
		'--model',
		help='Override the model name for the selected LLM (if supported). '
		'Note: browser-use backend uses fixed internal models and does not support --model.'
	)
	parser.add_argument(
		'--fast',
		action='store_true',
		help='Use the fast ChatBrowserUse model when --llm browser-use. '
		'Standard mode uses a more powerful model; fast mode uses a faster/cheaper model. '
		'Both are fixed models managed by the browser-use library.'
	)
	parser.add_argument(
		'--reasoning-effort',
		choices=['none', 'low', 'medium', 'high', 'xhigh'],
		help='Reasoning effort level for GPT-5.2+ models (OpenAI only). '
		'Options: none, low, medium, high, xhigh. Default is model-specific.'
	)
	parser.add_argument(
		'--max-output-tokens',
		type=int,
		default=None,
		help='Maximum output tokens for the LLM response (Google/Gemini only). '
		'For thinking models like gemini-3-pro, set higher (e.g., 65536) to allow room for both thinking and response.'
	)
	parser.add_argument(
		'--thinking-budget',
		type=int,
		default=None,
		help='Thinking token budget for Gemini thinking models (Google only). '
		'Controls how many tokens the model can use for internal reasoning. '
		'For maximum thinking, set to 24576 or higher. Example: --thinking-budget 24576'
	)
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
		help='Step timeout in seconds (default: 1800 = 30 minutes). Each agent step must complete within this time.'
	)
	parser.add_argument('--max-steps', type=int, default=1000, help='Maximum agent reasoning steps.')
	parser.add_argument(
		'--max-actions-per-step',
		type=int,
		default=10,
		help='Limit number of browser actions the agent may take per reasoning step.',
	)
	parser.add_argument(
		'--max-failures',
		type=int,
		default=5,
		help='Consecutive failure limit before aborting. Lower values prevent infinite loops. (default: 5)'
	)
	parser.add_argument(
		'--isolate-puzzles',
		dest='isolate_puzzles',
		action='store_true',
		help='Isolate puzzles: spawn a fresh agent for each puzzle (no cross-puzzle memory). '
		'Agent still has full memory within each puzzle attempt. Stops after solving/failing one puzzle.'
	)
	parser.add_argument('--use-cloud', action='store_true', help='Launch the browser in the Browser Use Cloud.')
	parser.add_argument('--headless', action='store_true', help='Run the local browser in headless mode.')
	parser.add_argument('--window-width', type=int, help='Browser viewport width (pixels).')
	parser.add_argument('--window-height', type=int, help='Browser viewport height (pixels).')
	parser.add_argument('--verbose', action='store_true', help='Enable verbose logging.')
	parser.add_argument(
		'--no-log-llm',
		action='store_true',
		help='Disable LLM logging (enabled by default). When logging is on, saves per-puzzle LLM logs to llm_logs/.'
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
		help='Run ID to use for logging. When provided, all logs from the same run will be grouped '
		'in the same directory. Useful for benchmarking where multiple puzzles should share the same run.'
	)

	# =============================================================================
	# EXTENDED FEATURES: Memory and Planner
	# =============================================================================
	# Note: --isolate-puzzles already exists (means "one-shot mode per puzzle")
	# Use different names to avoid confusion

	# Memory options (browser-use procedural memory via mem0)
	parser.add_argument(
		'--disable-procedural-memory',
		action='store_true',
		help='Disable browser-use procedural memory feature (enabled by default if mem0 installed).'
	)
	parser.add_argument(
		'--procedural-memory-interval',
		type=int,
		default=None,
		help='Create procedural memory every N steps (2-99). Default: 10'
	)

	# Planner options (separate LLM for planning)
	parser.add_argument(
		'--enable-planner',
		action='store_true',
		help='Enable separate planner LLM for reasoning before each step.'
	)
	parser.add_argument(
		'--planner-model',
		type=str,
		default=None,
		help='Model name for planner LLM. Uses same provider as --llm. Default: same as --model'
	)
	parser.add_argument(
		'--planner-interval',
		type=int,
		default=1,
		help='Run planner every N steps. Default: 1 (every step)'
	)
	parser.add_argument(
		'--planner-reasoning',
		action='store_true',
		help='Enable extended reasoning format in planner prompts.'
	)

	# API Actions options (external HTTP calls - A_api in POMDP)
	parser.add_argument(
		'--enable-api-actions',
		action='store_true',
		help='Enable external API calling actions (http_get, http_post, http_request). Adds A_api capability.'
	)
	parser.add_argument(
		'--api-timeout',
		type=int,
		default=30,
		help='Timeout in seconds for external API calls. Default: 30'
	)
	parser.add_argument(
		'--api-allowed-domains',
		type=str,
		default=None,
		help='Comma-separated list of allowed domains for API calls. Default: all domains allowed.'
	)
	parser.add_argument('--base-url', dest='base_url', help='Custom API base URL (required for vllm provider)')
	parser.add_argument('--api-key', dest='api_key', help='API key (use "EMPTY" for local vLLM)')

	return parser


def main(argv: Optional[list[str]] = None) -> int:
	parser = _build_parser()
	args = parser.parse_args(argv)
	_configure_logging(args.verbose)
	try:
		return asyncio.run(_run_agent(args))
	except KeyboardInterrupt:
		print('\nInterrupted by user.')
		return 1
	except ValueError as exc:
		# User input errors - show clean message without traceback
		logging.error('%s', exc)
		return 1
	except Exception as exc:  # pylint: disable=broad-except
		# Unexpected errors - show full traceback
		logging.exception('Agent run failed: %s', exc)
		return 1


if __name__ == '__main__':
	raise SystemExit(main())
