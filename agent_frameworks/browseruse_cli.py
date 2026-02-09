"""
Simple CLI to drive the OpenCaptchaWorld benchmark with a browser-use agent.

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

import sys
sys.path.insert(0, str(Path(__file__).parent))
from llm_logger import EnhancedLLMLogger, LLMResponseLogger, LoggingLLMWrapper


# =============================================================================
# GPT-5.2 JSON PARSE FAIL FAST
# =============================================================================
# We only want to fail fast (no retries) on JSON/parse errors for GPT-5.2.
# Implemented by monkeypatching Agent._handle_step_error to treat parse errors
# as max failures when model_name == 'gpt-5.2'. Other error types (e.g., 500s)
# continue to use normal retry/failure handling.
# Context: OpenAI gpt-5.2 server with reasoning_effort=xhigh has been unstable
# (~75% HTTP 500s, ~10% invalid JSON). We stop immediately on bad JSON to avoid
# repeated billable retries from that flaky server.
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
				# Only punish JSON/parse failures (not 5xx or other errors).
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
					# Force immediate stop to avoid repeated billable retries on bad JSON.
					self.state.consecutive_failures = self.settings.max_failures
					self.state.stopped = True
		except Exception:
			pass
		return res

	Agent._handle_step_error = wrapped


_patch_parse_fail_fast_for_gpt52()


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
					print(f"\n{'='*60}")
					print(f"[RAW REQUEST] ✅ reasoning_effort FOUND in request!")
					import re
					match = re.search(r'"reasoning_effort"\s*:\s*"([^"]+)"', request_body)
					if match:
						print(f"[RAW REQUEST] Value: {match.group(1)}")
					print(f"{'='*60}")
					# Debug: print stack trace to identify why this is called multiple times
					import traceback
					print(f"[RAW REQUEST] Call stack:")
					traceback.print_stack(limit=15)
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

				# Debug: print raw response content to see what model actually returned
				content = response_json.get('choices', [{}])[0].get('message', {}).get('content', '')
				if content:
					print(f"\n{'='*60}")
					print(f"[RAW RESPONSE CONTENT] (first 3000 chars):")
					print(f"{'='*60}")
					print(content[:3000])
					if len(content) > 3000:
						print(f"\n... (truncated, total length: {len(content)} chars)")
					print(f"{'='*60}\n")

				# Create new response with body for OpenAI SDK
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
					print(f"\n{'='*60}")
					print(f"[EXTRA BODY INJECT] Injecting params: {self.extra_body}")
					print(f"{'='*60}")

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
# ANTHROPIC RAW RESPONSE CAPTURE
# =============================================================================

class RawAnthropicResponseCapture:
	"""Thread-safe singleton to store raw Anthropic API responses for logging.

	Separate from RawResponseCapture (OpenAI) to avoid conflicts when both
	providers are used in the same session.
	"""
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


# =============================================================================
# ANTHROPIC EXTENDED THINKING CLIENT
# =============================================================================
# Injects thinking parameter into Anthropic API requests for extended thinking.
# =============================================================================

class ThinkingAnthropicHttpClient(httpx.AsyncClient):
	"""httpx client that injects extended thinking and effort parameters into Anthropic API requests.

	Anthropic extended thinking requires:
	- thinking: {"type": "enabled", "budget_tokens": N}
	- Minimum budget is 1024 tokens
	- Supported on: claude-haiku-4-5, claude-sonnet-4-5, claude-opus-4-5

	Anthropic effort parameter (Opus 4.5 only):
	- output_config: {"effort": "low"|"medium"|"high"}
	- Requires beta header: anthropic-beta: effort-2025-11-24
	"""

	def __init__(self, thinking_budget: int = None, effort: str = None, debug: bool = False, **kwargs):
		super().__init__(**kwargs)
		self.thinking_budget = thinking_budget
		self.effort = effort
		self.debug = debug
		# Store last response for logging
		self._last_thinking_tokens = None

	async def send(self, request, *args, **kwargs):
		# Inject thinking and/or effort params into Anthropic messages requests
		if '/v1/messages' in str(request.url) and (self.thinking_budget or self.effort):
			try:
				body = request.content.decode()
				body_json = json.loads(body)
				new_headers = dict(request.headers)
				modified = False

				# Inject thinking configuration
				if self.thinking_budget:
					body_json['thinking'] = {
						'type': 'enabled',
						'budget_tokens': self.thinking_budget
					}
					modified = True

					# Check if tool_choice forces a specific tool (incompatible with thinking)
					tool_choice = body_json.get('tool_choice', None)
					tool_choice_forces_tool = (
						isinstance(tool_choice, dict) and
						tool_choice.get('type') in ('tool', 'any')
					)

					# CRITICAL: Change forced tool_choice to "auto" when thinking is enabled
					# Anthropic API: "Thinking may not be enabled when tool_choice forces tool use"
					# Solution: Use tool_choice="auto" and trust the model to use the tool
					if tool_choice_forces_tool:
						original_tool_choice = tool_choice.copy()
						body_json['tool_choice'] = {'type': 'auto'}
						logging.info(
							f'[ANTHROPIC] Changed tool_choice from {original_tool_choice} to "auto" '
							f'(required for extended thinking compatibility)'
						)
						if self.debug:
							print(f"\n{'='*60}")
							print(f"[ANTHROPIC THINKING] Enabling with tool_choice override:")
							print(f"  budget_tokens: {self.thinking_budget}")
							print(f"  original tool_choice: {original_tool_choice}")
							print(f"  new tool_choice: auto (required for thinking)")
							print(f"  model: {body_json.get('model', 'unknown')}")
							print(f"{'='*60}")
					elif self.debug:
						print(f"\n{'='*60}")
						print(f"[ANTHROPIC THINKING] Injecting thinking config:")
						print(f"  budget_tokens: {self.thinking_budget}")
						print(f"  tool_choice: {tool_choice or 'auto (default)'}")
						print(f"  model: {body_json.get('model', 'unknown')}")
						print(f"{'='*60}")

				# Inject effort configuration (Opus 4.5 only)
				if self.effort:
					body_json['output_config'] = {'effort': self.effort}
					# Add required beta header for effort (combine with existing beta headers)
					existing_beta = new_headers.get('anthropic-beta', '')
					if 'effort-2025-11-24' not in existing_beta:
						beta_headers = [h for h in existing_beta.split(',') if h.strip()]
						beta_headers.append('effort-2025-11-24')
						new_headers['anthropic-beta'] = ','.join(beta_headers)
					modified = True
					if self.debug:
						print(f"\n{'='*60}")
						print(f"[ANTHROPIC EFFORT] Injecting effort config:")
						print(f"  effort: {self.effort}")
						print(f"  beta header: {new_headers.get('anthropic-beta')}")
						print(f"  model: {body_json.get('model', 'unknown')}")
						print(f"{'='*60}")

				if modified:
					new_body = json.dumps(body_json).encode()
					new_headers['content-length'] = str(len(new_body))
					request = httpx.Request(
						method=request.method,
						url=request.url,
						headers=new_headers,
						content=new_body
					)
			except Exception as e:
				logging.warning(f'[ANTHROPIC] Failed to inject params: {e}')

		response = await super().send(request, *args, **kwargs)

		# Capture response for logging (extract thinking token usage)
		if '/v1/messages' in str(request.url):
			try:
				# Read the full response body
				body = await response.aread()

				# Try to parse JSON - handle potential compression issues
				try:
					response_json = json.loads(body)

					# Store raw response for logging (use Anthropic-specific capture)
					RawAnthropicResponseCapture.get_instance().store(response_json)

					# Extract thinking token usage if available
					usage = response_json.get('usage', {})
					if usage and self.debug:
						print(f"\n[ANTHROPIC THINKING] Response usage:")
						print(f"  input_tokens: {usage.get('input_tokens', 0)}")
						print(f"  output_tokens: {usage.get('output_tokens', 0)}")
				except json.JSONDecodeError:
					# If body isn't valid JSON, just continue without parsing
					if self.debug:
						print(f"[ANTHROPIC] Response body not JSON, skipping capture")

				# Rebuild response with the captured body
				response = httpx.Response(
					status_code=response.status_code,
					headers=response.headers,
					content=body,
					request=request
				)
			except Exception as e:
				if self.debug:
					print(f"[ANTHROPIC] Error capturing response: {e}")

		return response


# =============================================================================
# VLLM ASYNC CLIENT (for Qwen thinking models)
# =============================================================================
# Parses <think>...</think> blocks from responses, extracts JSON for browser-use.
# =============================================================================

class CapturingVLLMAsyncClient(httpx.AsyncClient):
	"""httpx client for vLLM that handles Qwen thinking models.

	When thinking is enabled (default):
	- Parses <think>...</think> blocks from response content
	- Extracts JSON after </think> for browser-use
	- Stores reasoning content for logging

	When thinking is disabled:
	- Injects chat_template_kwargs: {enable_thinking: false} into requests

	For Qwen3-VL-8B-Thinking:
	- Injects top_k and repetition_penalty via extra_body
	"""

	def __init__(self, enable_thinking: bool = True, debug: bool = False, extra_body: dict = None, **kwargs):
		super().__init__(**kwargs)
		self.enable_thinking = enable_thinking
		self.debug = debug
		self.extra_body = extra_body or {}

	def _parse_thinking_response(self, content: str) -> tuple:
		"""
		Parse response with thinking content from Qwen models.
		Returns (thinking_content, json_content).

		Qwen3-VL-8B-Thinking outputs in format:
		  reasoning text here...
		  </think>

		  {"json": "here"}

		Note: There's often NO opening <think> tag, only a closing </think>.
		"""
		import re

		# Method 1: Look for </think> tag (Qwen format - may not have opening <think>)
		if '</think>' in content:
			parts = content.split('</think>', 1)
			thinking_content = parts[0].strip()
			# Remove opening <think> if present
			if thinking_content.startswith('<think>'):
				thinking_content = thinking_content[7:].strip()
			json_content = parts[1].strip() if len(parts) > 1 else ''
			if self.debug:
				print(f"[VLLM PARSE] Method: </think> tag split, reasoning: {len(thinking_content)} chars")
			logging.debug(f'[VLLM] Found </think> tag, reasoning: {len(thinking_content)} chars')
			return thinking_content if thinking_content else None, json_content

		# Method 2: Check if starts with JSON (pure JSON, no thinking)
		content_stripped = content.strip()
		if content_stripped.startswith('{'):
			if self.debug:
				print(f"[VLLM PARSE] Method: Pure JSON (no thinking content)")
			return None, content_stripped

		# Method 3: Find JSON by looking for opening brace patterns
		json_start_patterns = [
			r'(\{[\s\n]*"thinking")',
			r'(\{[\s\n]*"current_state")',
			r'(\{[\s\n]*"action")',
			r'(\{[\s\n]*")',  # Generic JSON start
		]

		for pattern in json_start_patterns:
			match = re.search(pattern, content)
			if match:
				json_start = match.start()
				thinking_content = content[:json_start].strip()
				json_content = content[json_start:].strip()
				if thinking_content:
					logging.debug(f'[VLLM] Found thinking without tags: {len(thinking_content)} chars')
				return thinking_content if thinking_content else None, json_content

		# Fallback: try to find any { followed by "
		brace_match = re.search(r'\{[\s\n]*"', content)
		if brace_match:
			json_start = brace_match.start()
			thinking_content = content[:json_start].strip()
			json_content = content[json_start:].strip()
			return thinking_content if thinking_content else None, json_content

		# Method 5: Search backwards from end for JSON object
		# Sometimes models put JSON at the end without proper tags
		last_brace = content.rfind('{')
		if last_brace != -1:
			# Check if there's a valid JSON-like structure starting here
			potential_json = content[last_brace:].strip()
			if re.match(r'\{[\s\n]*"', potential_json):
				thinking_content = content[:last_brace].strip()
				if self.debug:
					print(f"[VLLM PARSE] Method: Found JSON at end (pos {last_brace}), reasoning: {len(thinking_content)} chars")
				return thinking_content if thinking_content else None, potential_json

		# No JSON found - log warning and return entire content as-is
		# This will cause browser-use to fail parsing, triggering a retry
		if self.debug:
			print(f"[VLLM PARSE] WARNING: No JSON found in {len(content)} chars of content")
			print(f"[VLLM PARSE] Content starts with: {content[:100]}...")
			print(f"[VLLM PARSE] Content ends with: ...{content[-100:]}")
		return None, content

	async def send(self, request, *args, **kwargs):
		# Debug: Log request details for vLLM/Qwen thinking models
		if self.debug and '/chat/completions' in str(request.url):
			try:
				request_body = request.content.decode() if request.content else ''
				print(f"\n{'='*60}")
				print(f"[VLLM REQUEST] Qwen Thinking Model Request")
				print(f"[VLLM REQUEST] enable_thinking: {self.enable_thinking}")
				print(f"[VLLM REQUEST] URL: {request.url}")
				print(f"{'='*60}")
				print(f"[VLLM REQUEST BODY (first 1500 chars)]:\n{request_body[:1500]}\n")
			except Exception as e:
				print(f"[VLLM REQUEST] Error logging request: {e}")

		# Inject extra_body params for Qwen3-VL-8B-Thinking and/or chat_template_kwargs
		if '/chat/completions' in str(request.url) and (self.extra_body or not self.enable_thinking):
			try:
				body = request.content.decode()
				body_json = json.loads(body)
				modified = False
				# Inject extra_body params (top_k, repetition_penalty, max_tokens, seed) for Qwen3-VL-8B-Thinking only
				if self.extra_body:
					body_json.update(self.extra_body)
					modified = True
				# vLLM accepts chat_template_kwargs at top level
				if not self.enable_thinking:
					body_json['chat_template_kwargs'] = {'enable_thinking': False}
					modified = True
				if modified:
					new_body = json.dumps(body_json).encode()
					# Update Content-Length header to match new body size (required for Qwen3-VL-8B-Thinking)
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

		# Capture and modify response for chat completions
		if '/chat/completions' in str(request.url):
			try:
				body = await response.aread()
				response_json = json.loads(body)


				# Extract content from response
				reasoning_content = None
				if 'choices' in response_json and response_json['choices']:
					msg = response_json['choices'][0].get('message', {})
					content = msg.get('content', '')

					# Method 1: Use reasoning_content field if vLLM provides it
					native_reasoning = msg.get('reasoning_content') or msg.get('reasoning')
					if native_reasoning and isinstance(native_reasoning, str) and len(native_reasoning) > 0:
						reasoning_content = native_reasoning
						# Content should already be just JSON, but verify
						json_content = content.strip()
						if not json_content.startswith('{'):
							# Content still has reasoning, parse it
							_, json_content = self._parse_thinking_response(content)
						logging.info(f'[VLLM] Using native reasoning_content: {len(reasoning_content)} chars')
						response_json['choices'][0]['message']['content'] = json_content
					elif self.enable_thinking and content:
						# Method 2: Parse <think>...</think> blocks or plain text reasoning
						reasoning_content, json_content = self._parse_thinking_response(content)
						if reasoning_content:
							logging.info(f'[VLLM] Extracted thinking from content: {len(reasoning_content)} chars')
						# ALWAYS modify response to contain only the JSON part
						# This ensures browser-use gets the cleanest possible content
						# If no JSON was found, json_content is the original content and will fail gracefully
						response_json['choices'][0]['message']['content'] = json_content

				# Store usage and reasoning for logging
				usage = response_json.get('usage', {})
				RawVLLMResponseCapture.get_instance().store(usage, reasoning_content)

				# Debug: Log response details
				if self.debug:
					print(f"\n{'='*60}")
					print(f"[VLLM RESPONSE] Qwen Thinking Model Response")
					# Log finish_reason to detect truncation
					finish_reason = response_json['choices'][0].get('finish_reason', 'unknown')
					print(f"[VLLM RESPONSE] finish_reason: {finish_reason}")
					# Check for </think> tag in original content
					has_think_tag = '</think>' in content
					print(f"[VLLM RESPONSE] Has </think> tag: {has_think_tag}")
					if reasoning_content:
						reasoning_tokens = count_tokens_qwen(reasoning_content)
						print(f"[VLLM RESPONSE] Reasoning extracted: {len(reasoning_content)} chars ({reasoning_tokens} tokens)")
						print(f"[VLLM RESPONSE] Reasoning preview (first 500 chars):")
						print(f"{reasoning_content[:500]}...")
					else:
						print(f"[VLLM RESPONSE] No reasoning content found")
						# Show why parsing failed
						if not has_think_tag:
							print(f"[VLLM RESPONSE] WARNING: No </think> tag - model may have been truncated or skipped thinking format")
					# Get final JSON content
					final_content = response_json['choices'][0].get('message', {}).get('content', '')
					print(f"[VLLM RESPONSE] JSON content size: {len(final_content)} chars")
					if final_content:
						print(f"[VLLM RESPONSE] JSON preview: {final_content[:300]}...")
					print(f"{'='*60}\n")

				# Recreate response with modified content
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
# VLLM RAW RESPONSE CAPTURE (for Qwen thinking models)
# =============================================================================
# vLLM/Qwen thinking models output <think>...</think> blocks before JSON.
# We parse these client-side and store reasoning content for logging.
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


def _build_task_prompt(url: str, limit: int, isolated_mode: bool = False) -> str:
	"""
	Create the instruction string passed to the browser-use agent.

	Args:
	    url: Fully qualified URL pointing at a running OpenCaptchaWorld instance.
	    limit: Number of puzzles the agent should attempt before finishing.
	    isolated_mode: If True, agent solves ONE puzzle then stops (--isolate-puzzles mode).
	                   If False, agent solves multiple puzzles up to limit.
	"""
	# Mode-specific instructions
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
	- Output exactly ONE JSON response per step (never multiple)

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
		# CRITICAL: Use args.llm_timeout for GPT-5.2 xhigh which can take 30+ minutes!
		timeout_seconds = getattr(args, 'llm_timeout', 1800) or 1800
		logging.info(f'Creating OpenAI client with HTTP timeout={timeout_seconds}s (model={model})')
		http_client = CapturingAsyncClient(timeout=httpx.Timeout(float(timeout_seconds)))

		try:
			# Build kwargs for ChatOpenAI
			openai_kwargs = {
				'model': model,
				'http_client': http_client,
				'max_retries': 0,  # Disable SDK auto-retries to avoid paying for failed requests
			}

			# Add reasoning effort for GPT-5.2+ models if specified
			if getattr(args, 'reasoning_effort', None):
				openai_kwargs['reasoning_effort'] = args.reasoning_effort
				logging.info(f'Creating ChatOpenAI with reasoning_effort={args.reasoning_effort}')

			# Add max_completion_tokens directly (browser-use 0.11.1 ChatOpenAI parameter)
			# GPT-5.2 supports up to 128K output tokens
			if getattr(args, 'max_output_tokens', None):
				openai_kwargs['max_completion_tokens'] = args.max_output_tokens
				logging.info(f'Creating ChatOpenAI with max_completion_tokens={args.max_output_tokens}')

			llm = ChatOpenAI(**openai_kwargs)

			# Store request params for provider-specific logging
			llm._request_params = {
				'model': model,
				'reasoning_effort': getattr(args, 'reasoning_effort', None),
				'temperature': getattr(llm, 'temperature', None),
				'max_completion_tokens': getattr(args, 'max_output_tokens', None),
				'seed': getattr(llm, 'seed', None),
				'service_tier': getattr(llm, 'service_tier', None),
			}
			return llm
		except TypeError as exc:
			raise ValueError(f'Invalid OpenAI configuration: {exc}') from exc

	def anthropic_factory(args: argparse.Namespace):
		from browser_use import ChatAnthropic

		model = args.model or 'claude-sonnet-4-20250514'
		thinking_budget = getattr(args, 'anthropic_thinking_budget', None)
		effort = getattr(args, 'anthropic_effort', None)

		try:
			# Build kwargs for ChatAnthropic
			anthropic_kwargs = {'model': model}

			# Add max_tokens if specified (important for thinking models)
			if getattr(args, 'max_output_tokens', None):
				anthropic_kwargs['max_tokens'] = args.max_output_tokens
				logging.info(f'Creating ChatAnthropic with max_tokens={args.max_output_tokens}')

			# If extended thinking or effort is enabled, use custom HTTP client
			if thinking_budget or effort:
				if thinking_budget and thinking_budget < 1024:
					raise ValueError(f'Anthropic thinking budget must be at least 1024 tokens (got {thinking_budget})')
				if effort:
					logging.info(f'Creating ChatAnthropic with effort={effort} (Opus 4.5 only)')
				if thinking_budget:
					logging.info(f'Creating ChatAnthropic with extended thinking (budget_tokens={thinking_budget})')

				http_client = ThinkingAnthropicHttpClient(
					thinking_budget=thinking_budget,
					effort=effort,
					debug=getattr(args, 'debug_vllm', False),
					timeout=httpx.Timeout(600.0)
				)
				anthropic_kwargs['http_client'] = http_client

			# Add API key if provided (strip whitespace to avoid header injection errors)
			api_key = getattr(args, 'api_key', None)
			if api_key:
				anthropic_kwargs['api_key'] = api_key.strip()
			else:
				# Also check and strip environment variable if present
				import os
				env_key = os.environ.get('ANTHROPIC_API_KEY', '')
				if env_key and env_key != env_key.strip():
					logging.warning('[ANTHROPIC] Stripping whitespace from ANTHROPIC_API_KEY env var')
					os.environ['ANTHROPIC_API_KEY'] = env_key.strip()

			llm = ChatAnthropic(**anthropic_kwargs)

			# Store request params for provider-specific logging
			llm._request_params = {
				'model': model,
				'thinking_budget': thinking_budget,
				'effort': effort,
				'max_tokens': getattr(llm, 'max_tokens', None),
				'temperature': getattr(llm, 'temperature', None),
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

			# Add thinking_budget if specified (Gemini 2.5)
			if getattr(args, 'thinking_budget', None) is not None:
				google_kwargs['thinking_budget'] = args.thinking_budget
				logging.info(f'Creating ChatGoogle with thinking_budget={args.thinking_budget}')

			# Add thinking_level if specified (Gemini 3)
			if getattr(args, 'thinking_level', None):
				google_kwargs['thinking_level'] = args.thinking_level
				logging.info(f'Creating ChatGoogle with thinking_level={args.thinking_level}')

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
				'thinking_budget': getattr(args, 'thinking_budget', None),
				'thinking_level': getattr(args, 'thinking_level', None),
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
		"""vLLM factory - uses ChatOpenAI with OpenAI-compatible vLLM endpoint.

		For Qwen thinking models:
		- Default: thinking enabled, parses <think>...</think> blocks
		- Use --disable-thinking to suppress reasoning
		- Applies official Qwen3-VL-8B-Thinking defaults automatically
		"""
		from browser_use import ChatOpenAI

		model = args.model
		disable_thinking = getattr(args, 'disable_thinking', False)
		enable_thinking = not disable_thinking  # Default: thinking enabled

		# Detect Qwen3-VL-8B-Thinking model specifically (case-insensitive, flexible matching)
		model_lower = model.lower()
		is_qwen3_8b_thinking = 'qwen3-vl-8b-thinking' in model_lower or model_lower == 'qwen/qwen3-vl-8b-thinking'
		logging.info(f'[VLLM] Model: {model}, is_qwen3_8b_thinking: {is_qwen3_8b_thinking}')

		# Build extra_body for Qwen3-VL-8B-Thinking (injected via HTTP client)
		# All vLLM-specific params go here since ChatOpenAI doesn't accept them
		extra_body = None
		if is_qwen3_8b_thinking:
			# Default to 32768 for thinking models if not specified
			max_tokens_value = getattr(args, 'max_output_tokens', None) or 32768
			extra_body = {
				'top_k': 20,  # Official default
				'repetition_penalty': 1.0,  # Official default
				'seed': 0,  # For reproducibility
				'max_tokens': max_tokens_value,  # Critical: override browser-use's 4096 default
			}
			logging.info(f'[VLLM] Qwen3-VL-8B-Thinking extra_body: {extra_body}')

		# Use custom client for vLLM that handles thinking models
		http_client = CapturingVLLMAsyncClient(
			enable_thinking=enable_thinking,
			debug=getattr(args, 'debug_vllm', False),
			extra_body=extra_body,  # Only for Qwen3-VL-8B-Thinking
			timeout=httpx.Timeout(600.0)
		)

		# Build kwargs with model-specific defaults
		vllm_kwargs = {
			'model': model,
			'base_url': args.base_url,
			'api_key': getattr(args, 'api_key', None) or 'EMPTY',
			'http_client': http_client,
		}

		# Apply Qwen3-VL-8B-Thinking official defaults for THINKING mode
		# Reference: https://qwen.readthedocs.io/en/latest/getting_started/quickstart.html
		# For Qwen3-Thinking models: temperature=0.6, top_p=0.95, top_k=20
		if is_qwen3_8b_thinking:
			vllm_kwargs['temperature'] = 0.6  # Official default for Qwen3 THINKING models
			vllm_kwargs['top_p'] = 0.95  # Official default
			# IMPORTANT: Set max_completion_tokens to override browser-use's default of 4096
			# Without this, browser-use sends max_completion_tokens=4096 which truncates responses
			# Default to 32768 for Qwen3-VL-8B-Thinking if not specified (thinking needs more tokens)
			max_tokens = getattr(args, 'max_output_tokens', None) or 32768
			vllm_kwargs['max_completion_tokens'] = max_tokens
			logging.info(f'[VLLM] Qwen3-VL-8B-Thinking: Setting max_completion_tokens={max_tokens}')
			# Note: seed, top_k, repetition_penalty are in extra_body (injected via HTTP client)
			logging.info(f'[VLLM] Qwen3-VL-8B-Thinking detected. Applying official defaults: '
						 f'temperature=0.6, top_p=0.95, top_k=20, repetition_penalty=1.0, seed=0')
		else:
			# For other models, use model defaults
			vllm_kwargs['temperature'] = None
			vllm_kwargs['frequency_penalty'] = None
			# Add max_tokens for non-Qwen models via ChatOpenAI
			if getattr(args, 'max_output_tokens', None):
				vllm_kwargs['max_completion_tokens'] = args.max_output_tokens
				logging.info(f'Creating vLLM ChatOpenAI with max_completion_tokens={args.max_output_tokens}')

		if enable_thinking:
			logging.info(f'Creating vLLM ChatOpenAI with thinking enabled (will parse <think> blocks)')
		else:
			logging.info(f'Creating vLLM ChatOpenAI with thinking disabled')

		try:
			llm = ChatOpenAI(**vllm_kwargs)

			# Store request params for provider-specific logging
			request_params = {
				'model': model,
				'base_url': args.base_url,
				'enable_thinking': enable_thinking,
			}
			# Add Qwen3-VL-8B-Thinking specific params for logging
			if is_qwen3_8b_thinking:
				max_tokens_for_log = getattr(args, 'max_output_tokens', None) or 32768
				request_params['max_output_tokens'] = max_tokens_for_log  # For logger compatibility
				request_params['inference_params'] = {
					'temperature': 0.6,  # Official default for Qwen3 THINKING models
					'top_p': 0.95,
					'top_k': 20,
					'repetition_penalty': 1.0,
					'seed': 0,
					'max_tokens': max_tokens_for_log,
					'max_completion_tokens': max_tokens_for_log,
				}
			else:
				request_params['max_output_tokens'] = getattr(args, 'max_output_tokens', None)
			llm._request_params = request_params

			return llm
		except TypeError as exc:
			raise ValueError(f'Invalid vLLM configuration: {exc}') from exc

	def qwen_factory(args: argparse.Namespace):
		"""Qwen factory - uses ChatOpenAI with Alibaba Cloud DashScope OpenAI-compatible endpoint.

		Supports Qwen3-VL models via DashScope API with thinking mode enabled by default.
		Requires DASHSCOPE_API_KEY environment variable.
		"""
		import os
		from browser_use import ChatOpenAI

		model = args.model or 'qwen3-vl-plus-2025-12-19'
		api_key = os.environ.get('DASHSCOPE_API_KEY')
		if not api_key:
			raise ValueError('DASHSCOPE_API_KEY environment variable is required for qwen provider')

		# DashScope OpenAI-compatible endpoint (Singapore region by default)
		base_url = getattr(args, 'base_url', None) or 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1'

		# Enable thinking mode with high thinking effort (38000 tokens is near max for most Qwen3-VL models)
		thinking_budget = 38000
		extra_body = {
			'enable_thinking': True,
			'thinking_budget': thinking_budget,
		}

		# Use ExtraBodyInjectingClient to inject thinking params into requests
		http_client = ExtraBodyInjectingClient(
			extra_body=extra_body,
			debug=getattr(args, 'debug_vllm', False),
			timeout=httpx.Timeout(600.0)
		)

		try:
			llm = ChatOpenAI(
				model=model,
				base_url=base_url,
				api_key=api_key,
				http_client=http_client,
			)

			# Store request params for provider-specific logging
			llm._request_params = {
				'model': model,
				'base_url': base_url,
				'provider': 'qwen',
				'enable_thinking': True,
				'thinking_budget': thinking_budget,
			}
			logging.info(f'[QWEN] Created ChatOpenAI with model={model}, base_url={base_url}, thinking_mode=enabled, thinking_budget={thinking_budget}')
			return llm
		except TypeError as exc:
			raise ValueError(f'Invalid Qwen configuration: {exc}') from exc

	def doubao_factory(args: argparse.Namespace):
		"""Doubao factory - uses ChatOpenAI with Volcengine Ark OpenAI-compatible endpoint.

		Supports Doubao-Seed models via Ark API with thinking mode enabled by default.
		Requires ARK_API_KEY environment variable.
		"""
		import os
		from browser_use import ChatOpenAI

		model = args.model or 'doubao-seed-1-8-251228'
		api_key = os.environ.get('ARK_API_KEY')
		if not api_key:
			raise ValueError('ARK_API_KEY environment variable is required for doubao provider')

		# Volcengine Ark OpenAI-compatible endpoint
		base_url = getattr(args, 'base_url', None) or 'https://ark.cn-beijing.volces.com/api/v3'

		# Enable thinking mode with high reasoning effort
		# Volcengine Doubao uses reasoning_effort parameter
		# reasoning_effort options: low, medium, high, minimal
		reasoning_effort = 'high'
		extra_body = {
			'reasoning_effort': reasoning_effort,
		}

		# Use ExtraBodyInjectingClient to inject reasoning params into requests
		http_client = ExtraBodyInjectingClient(
			extra_body=extra_body,
			debug=getattr(args, 'debug_vllm', False),
			timeout=httpx.Timeout(600.0)
		)

		try:
			llm = ChatOpenAI(
				model=model,
				base_url=base_url,
				api_key=api_key,
				http_client=http_client,
			)

			# Store request params for provider-specific logging
			llm._request_params = {
				'model': model,
				'base_url': base_url,
				'provider': 'doubao',
				'reasoning_effort': reasoning_effort,
			}
			logging.info(f'[DOUBAO] Created ChatOpenAI with model={model}, base_url={base_url}, reasoning_effort={reasoning_effort}')
			return llm
		except TypeError as exc:
			raise ValueError(f'Invalid Doubao configuration: {exc}') from exc

	return {
		'browser-use': browser_use_factory,
		'openai': openai_factory,
		'anthropic': anthropic_factory,
		'google': google_factory,
		'groq': groq_factory,
		'azure-openai': azure_factory,
		'vllm': vllm_factory,
		'qwen': qwen_factory,
		'doubao': doubao_factory,
	}


def _configure_logging(verbose: bool) -> None:
	"""Set a minimal logging format for the CLI."""
	level = logging.DEBUG if verbose else logging.INFO
	logging.basicConfig(level=level, format='[%(levelname)s] %(message)s')


def _create_browser(args: argparse.Namespace) -> Browser:
	"""
	Create a Browser session.

	Always creates a Browser so our callbacks can access it for puzzle detection.
	The same Browser is passed to the Agent, so there's no duplication.
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
	elif llm_name == 'qwen':
		model_desc = args.model or 'qwen3-vl-plus-2025-12-19'
	elif llm_name == 'doubao':
		model_desc = args.model or 'doubao-seed-1-8-251228'
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
		# Extract base URL (scheme + netloc) in case base_url contains query params
		parsed = urlparse(base_url)
		clean_base = f'{parsed.scheme}://{parsed.netloc}'
		endpoint = f'{clean_base}/api/agent_metadata'
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
		llm_logger._raw_vllm_capture = RawVLLMResponseCapture.get_instance()
		llm_logger._raw_anthropic_capture = RawAnthropicResponseCapture.get_instance()

		# Inject Qwen tokenizer function for vLLM reasoning token counting
		llm_logger._count_tokens_qwen = count_tokens_qwen

		# Store reference to original LLM so logger can access _request_params
		llm_logger._llm_instance = llm

		llm = LoggingLLMWrapper(llm, llm_logger)

		# Puzzle logging will start when real puzzle_id is detected from browser
		# (in step_callback via extract_puzzle_from_browser)

	browser = _create_browser(args)
	task = _build_task_prompt(args.url, args.limit, isolated_mode=args.isolate_puzzles)

	# Log the task prompt
	if llm_logger:
		llm_logger.log_task_prompt(task)

	# Holder for agent reference - allows callbacks to access agent's browser after creation
	agent_holder = [None]

	# Helper function to extract puzzle info from browser
	# Uses browser.get_current_page() to access the page and evaluate JavaScript
	async def extract_puzzle_from_browser() -> Optional[dict]:
		"""Extract current puzzle info from window.currentPuzzle."""
		try:
			# Get browser - use agent's browser if local browser is None
			browser_instance = browser if browser else (agent_holder[0].browser if agent_holder[0] else None)
			if not browser_instance:
				logging.debug('extract_puzzle_from_browser: no browser available')
				return None
			page = await browser_instance.get_current_page()
			if not page:
				logging.debug('extract_puzzle_from_browser: no page')
				return None

			# Note: page.evaluate requires arrow function format and auto-calls it
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

			# page.evaluate may return JSON string - parse it
			if isinstance(puzzle_info, str):
				try:
					puzzle_info = json.loads(puzzle_info)
				except json.JSONDecodeError:
					puzzle_info = None

			# Ensure we have a dict
			if puzzle_info and not isinstance(puzzle_info, dict):
				puzzle_info = None

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
			# Extract base URL (scheme + netloc) from args.url which may contain query params
			# e.g., http://127.0.0.1:7869/?type=X&puzzle_index=0 -> http://127.0.0.1:7869
			parsed_url = urlparse(args.url)
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
			# Extract base URL (scheme + netloc) from args.url which may contain query params
			parsed_url = urlparse(args.url)
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

	def classify_agent_errors(history) -> str:
		"""
		Analyze agent history errors and return appropriate user_answer value.

		Checks both history.errors() AND history.extracted_content() because
		browser-use v0.11.x puts some error-like messages in extracted_content
		instead of error (e.g., "Element index X not available").

		Error patterns:
		- "Element with index X does not exist" (error field, v0.1.47)
		- "Element index X not available" (extracted_content field, v0.11.x)
		- "Element not clickable with index X"
		- "Could not parse response"

		Returns one of:
		- NO_SUBMIT_INVALID_INDEX: Element index errors
		- NO_SUBMIT_NOT_CLICKABLE: Element not clickable errors
		- NO_SUBMIT_FORMAT_ERROR: LLM parsing errors
		- NO_SUBMIT_OTHER_ERROR: Other errors
		- NO_SUBMISSION_MAX_STEPS: No errors detected
		"""
		if not history:
			return 'NO_SUBMISSION_MAX_STEPS'

		# Collect all error-like messages from both sources
		all_messages = []

		# Source 1: history.errors() - actual error field
		if hasattr(history, 'errors'):
			errors = history.errors()
			if errors:
				all_messages.extend([e for e in errors if e is not None])

		# Source 2: history.extracted_content() - browser-use v0.11.x puts errors here
		if hasattr(history, 'extracted_content'):
			contents = history.extracted_content()
			if contents:
				# Only include messages that look like errors
				error_patterns = ['not available', 'does not exist', 'not clickable', 'failed', 'error', 'could not']
				for content in contents:
					if content and any(p in content.lower() for p in error_patterns):
						all_messages.append(content)

		if not all_messages:
			return 'NO_SUBMISSION_MAX_STEPS'

		# Count error types based on verified error messages
		invalid_index = 0
		not_clickable = 0
		format_error = 0
		other_error = 0

		for msg in all_messages:
			msg_lower = str(msg).lower()

			# Element index errors (most common)
			# Matches: "Element with index X does not exist" (v0.1.47)
			# Matches: "Element index X not available" (v0.11.x)
			if 'index' in msg_lower and ('does not exist' in msg_lower or 'not available' in msg_lower):
				invalid_index += 1
			# Element not clickable
			# Matches: "Element not clickable with index X"
			elif 'not clickable' in msg_lower:
				not_clickable += 1
			# LLM parsing errors
			# Matches: "Could not parse response"
			elif 'could not parse' in msg_lower or 'parse' in msg_lower:
				format_error += 1
			# Everything else that looks like an error
			elif any(p in msg_lower for p in ['failed', 'error']):
				other_error += 1

		# Return the most frequent error type
		counts = [
			(invalid_index, 'NO_SUBMIT_INVALID_INDEX'),
			(not_clickable, 'NO_SUBMIT_NOT_CLICKABLE'),
			(format_error, 'NO_SUBMIT_FORMAT_ERROR'),
			(other_error, 'NO_SUBMIT_OTHER_ERROR'),
		]

		max_count, error_type = max(counts, key=lambda x: x[0])

		if max_count > 0:
			logging.info(f'[classify_errors] Errors: index={invalid_index}, not_clickable={not_clickable}, format={format_error}, other={other_error} -> {error_type}')
			return error_type

		return 'NO_SUBMISSION_MAX_STEPS'

	async def record_benchmark_result(
		puzzle_type: str,
		puzzle_id: str,
		is_correct: bool | None,
		user_answer: str | None = None,
		correct_answer: str | None = None,
		elapsed_time: float | None = None,
	) -> bool:
		"""
		Record benchmark result to server via /api/benchmark_results.

		This ensures every puzzle attempt has a recorded result, even if the agent
		fails to submit (e.g., max-steps reached, timeout, navigation error).
		"""
		try:
			parsed_url = urlparse(args.url)
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
				'agent_framework': 'browser-use',
			}

			async with httpx.AsyncClient(timeout=5.0) as client:
				response = await client.post(api_url, json=result_data)
				if response.status_code == 200:
					logging.info(f'[record_result] Recorded benchmark result: puzzle_type={puzzle_type}, puzzle_id={puzzle_id}, correct={is_correct}')
					return True
				else:
					logging.warning(f'[record_result] Failed to record result: status={response.status_code}')
		except Exception as e:
			logging.warning(f'[record_result] Failed to record benchmark result: {e}')
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

		# If browser extraction failed, use expected_puzzle_type from URL as fallback
		# This ensures we don't accept stale results from previous/different puzzle types
		effective_puzzle_type = current_puzzle_type or expected_puzzle_type[0]

		# Priority 1: Check server API (most reliable)
		server_result = await check_server_for_result(
			puzzle_id=current_puzzle_id,
			puzzle_type=effective_puzzle_type
		)
		if server_result.get('detected'):
			# Extra validation: ensure the result's puzzle_type matches what we expect
			result_puzzle_type = server_result.get('puzzle_type')
			if effective_puzzle_type and result_puzzle_type and result_puzzle_type != effective_puzzle_type:
				logging.warning(f'[detect_result] Ignoring stale result: server has {result_puzzle_type} but expected {effective_puzzle_type}')
			else:
				logging.info(f'[detect_result] Result detected via SERVER API: correct={server_result.get("is_correct")}')
				return server_result

		# Fall back to JavaScript-based detection if server check fails
		try:
			# Get browser - use agent's browser if local browser is None
			browser_instance = browser if browser else (agent_holder[0].browser if agent_holder[0] else None)
			if not browser_instance:
				logging.debug('[detect_result] No browser available')
				return {'detected': False}
			page = await browser_instance.get_current_page()
			if not page:
				logging.debug('[detect_result] No page available')
				return {'detected': False}

			# Check for result text on the page and extract answer info
			result = await page.evaluate(r'''() => {
				// First check the result-message element specifically (most reliable)
				const resultEl = document.querySelector('.result-message, #result-message');

				// DEBUG: Log what we find
				const debugInfo = {
					elementFound: !!resultEl,
					elementText: resultEl ? resultEl.innerText : null,
					elementClass: resultEl ? resultEl.className : null,
					puzzleId: window.currentPuzzle ? window.currentPuzzle.puzzle_id : null,
					lastPuzzleResult: window.lastPuzzleResult || null
				};
				console.log('[detect_result DEBUG]', JSON.stringify(debugInfo));

				// First, check window.lastPuzzleResult (set by submitAnswer, survives puzzle transitions)
				if (window.lastPuzzleResult && window.lastPuzzleResult.correct !== undefined) {
					const lpr = window.lastPuzzleResult;
					// Only use if it's recent (within last 120 seconds) to avoid stale results
					const age = Date.now() - (lpr.timestamp || 0);
					debugInfo.lastPuzzleResultAge = age;
					console.log('[detect_result] lastPuzzleResult age:', age, 'ms, correct:', lpr.correct);
					if (age < 120000) {
						return {
							detected: true,
							is_correct: lpr.correct,
							user_answer: null,
							correct_answer: lpr.correct_answer || null,
							result_text: lpr.correct ? 'Correct!' : 'Incorrect',
							source: 'window.lastPuzzleResult',
							age_ms: age,
							debug: debugInfo
						};
					} else {
						console.log('[detect_result] lastPuzzleResult too old, skipping');
					}
				}

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
								result_text: fullText,
								source: 'resultEl',
								debug: debugInfo
							};
						}
					}
				}
				return {detected: false, debug: debugInfo};
			}''')
			# Log the raw result for debugging
			logging.info(f'[detect_result] Raw result: {result}')
			# page.evaluate may return JSON string - parse it
			if isinstance(result, str):
				try:
					result = json.loads(result)
				except json.JSONDecodeError:
					return {'detected': False}
			# Ensure we have a dict
			if not isinstance(result, dict):
				return {'detected': False}
			return result
		except Exception as e:
			if args.verbose:
				logging.debug(f'Could not detect result from page: {e}')
			return {'detected': False}

	# Parse expected puzzle type from URL for navigation detection
	# URL format: http://127.0.0.1:7869/?type=3D_Viewpoint&puzzle_index=0&...
	expected_puzzle_type = [None]  # Use list for closure mutation
	try:
		parsed_url = urlparse(args.url)
		query_params = parse_qs(parsed_url.query)
		if 'type' in query_params:
			expected_puzzle_type[0] = query_params['type'][0]
			logging.info(f'[navigation_guard] Expected puzzle type from URL: {expected_puzzle_type[0]}')
	except Exception as e:
		logging.debug(f'[navigation_guard] Could not parse expected puzzle type from URL: {e}')

	# Track consecutive action failures to stop agent early
	# Browser-use v0.11.x puts "not available" in extracted_content instead of error,
	# so max_failures doesn't trigger. We track this ourselves.
	consecutive_action_failures = [0]
	MAX_CONSECUTIVE_ACTION_FAILURES = 3
	last_failure_type = ['']
	last_checked_history_len = [0]  # Track which history length we already checked to avoid double-counting

	# Create step callback for logging agent steps with puzzle detection
	def create_step_callback(logger, result_flag=None, agent_ref=None):
		"""Create a step callback that logs agent state, actions, and detects puzzle transitions."""
		last_puzzle_id = [None]  # Use list to allow mutation in closure
		first_puzzle_type = [None]  # Track the first puzzle type loaded (should match expected)

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
				logging.info(f'[step_callback] Step {step_number}: Checking for result on page...')
				page_result = await detect_result_from_page()
				logging.info(f'[step_callback] Step {step_number}: detect_result returned detected={page_result.get("detected")}, is_correct={page_result.get("is_correct")}')

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


				# Try to detect real puzzle_id from browser
				puzzle_info = await extract_puzzle_from_browser()
				logging.info(f'[step_callback] Step {step_number}: Current puzzle_id={puzzle_info.get("puzzle_id") if puzzle_info else None}, last_puzzle_id={last_puzzle_id[0]}')

				if puzzle_info and puzzle_info.get('puzzle_id'):
					detected_id = puzzle_info.get('puzzle_id')
					detected_type = puzzle_info.get('puzzle_type')

					# NAVIGATION GUARD: Check if agent navigated to a different puzzle TYPE
					# This happens when agent clicks on forbidden navigation elements
					if expected_puzzle_type[0] and detected_type and detected_type != expected_puzzle_type[0]:
						logging.error(f'[navigation_guard] NAVIGATION FAILURE: Agent navigated from expected {expected_puzzle_type[0]} to {detected_type}')
						# End current puzzle as failed due to navigation
						if hasattr(logger, 'end_puzzle'):
							logger.end_puzzle(
								answer='NAVIGATION_FAILURE',
								is_correct=False,
								correct_answer=f'Agent navigated away to {detected_type}',
							)
						# Signal agent to stop (in isolate-puzzles mode)
						if result_flag is not None:
							result_flag[0] = True
							logging.info('[navigation_guard] Signaling agent to stop due to navigation failure')
						return  # Stop processing this step

					# Check if this is a different puzzle than before
					if detected_id != last_puzzle_id[0]:
						logging.info(f'[step_callback] Puzzle transition detected: {last_puzzle_id[0]} -> {detected_id}')
						# End previous puzzle if exists
						if last_puzzle_id[0] is not None and hasattr(logger, 'end_puzzle'):
							logging.warning(f'[step_callback] Ending previous puzzle WITHOUT result (correct=None)')
							logger.end_puzzle()

						# Clear server last_result when starting a new puzzle
						# This ensures we don't detect stale results from previous puzzles
						if last_cleared_puzzle_id[0] != detected_id:
							await clear_server_result()
							last_cleared_puzzle_id[0] = detected_id

						# Start new puzzle with REAL ID
						if hasattr(logger, 'start_puzzle'):
							logger.start_puzzle(
								puzzle_type=detected_type,
								puzzle_id=detected_id,
								prompt=puzzle_info.get('prompt')
							)

						last_puzzle_id[0] = detected_id

				# Only log if puzzle has been detected
				if last_puzzle_id[0] is not None:
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

	# Create stop callback for isolate-puzzles mode - stops agent after puzzle result
	# NOTE: This callback is checked AFTER LLM response but BEFORE actions execute
	async def check_puzzle_result_and_stop():
		"""
		Return True to stop agent when:
		1. A puzzle result has been detected, or
		2. MAX_CONSECUTIVE_ACTION_FAILURES steps had all actions fail
		"""
		if puzzle_result_received[0]:
			logging.info('[check_puzzle_result_and_stop] Result flag already set, stopping agent')
			agent_holder[0].state.stopped = True
			return True

		# Check for consecutive action failures in previous step
		# Browser-use v0.11.x puts "not available" in extracted_content, not error
		# NOTE: This callback is invoked multiple times per step by browser-use (lines 1024, 1085, 1091 in service.py)
		# but history is only updated at the END of each step in _finalize(). We must track history length
		# to avoid counting the same failed step multiple times.
		agent = agent_holder[0]
		if agent and len(agent.history.history) > 0:
			current_history_len = len(agent.history.history)

			# Only check if history has grown (new step added since last check)
			if current_history_len > last_checked_history_len[0]:
				last_checked_history_len[0] = current_history_len
				last_step = agent.history.history[-1]

				# Check if ALL actions in this step failed with "not available" errors
				all_failed = True
				for r in last_step.result:
					# ActionResult has: error (str|None), extracted_content (str|None)
					msg = ((r.error or '') + ' ' + (r.extracted_content or '')).lower()
					if 'not available' not in msg and 'does not exist' not in msg:
						all_failed = False
						break

				if all_failed and len(last_step.result) > 0:
					consecutive_action_failures[0] += 1
					logging.warning(f'[action_failure] Step had all actions fail ({consecutive_action_failures[0]}/{MAX_CONSECUTIVE_ACTION_FAILURES})')
					if consecutive_action_failures[0] >= MAX_CONSECUTIVE_ACTION_FAILURES:
						logging.error(f'[action_failure] Stopping agent after {MAX_CONSECUTIVE_ACTION_FAILURES} consecutive failed steps')
						last_failure_type[0] = 'NO_SUBMIT_INVALID_INDEX'
						agent.state.stopped = True
						return True
				else:
					consecutive_action_failures[0] = 0

		# Check the page directly in case submit just happened
		page_result = await detect_result_from_page()
		if page_result.get('detected'):
			logging.info(f'[check_puzzle_result_and_stop] Page shows result (correct={page_result.get("is_correct")}) - stopping agent')
			puzzle_result_received[0] = True
			if llm_logger:
				llm_logger.end_puzzle(
					answer=page_result.get('user_answer'),
					is_correct=page_result.get('is_correct'),
					correct_answer=page_result.get('correct_answer'),
				)
			agent_holder[0].state.stopped = True
			return True

		return False

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

	# Add step callback if logger is available
	if step_callback:
		agent_kwargs['register_new_step_callback'] = step_callback

	# In isolate-puzzles mode, register callback to stop agent after puzzle result
	# This ensures one-shot behavior regardless of model following prompt instructions
	# Only register if logging is enabled (detection requires the logger)
	# Uses browser-use's register_external_agent_status_raise_error_callback which
	# raises InterruptedError when the callback returns True
	if args.isolate_puzzles and llm_logger:
		agent_kwargs['register_external_agent_status_raise_error_callback'] = check_puzzle_result_and_stop

	agent = Agent(
		task=task,
		llm=llm,
		browser=browser,
		max_actions_per_step=args.max_actions_per_step,
		include_tool_call_examples=False,
		**agent_kwargs,
	)

	# Store agent reference for callback to use
	agent_holder[0] = agent

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

		# Clear any stale server result BEFORE starting the agent
		# This prevents detection of results from previous puzzle runs
		await clear_server_result()
		logging.info('[agent_start] Cleared server last_result before starting agent')

		history = await agent.run(max_steps=args.max_steps)
		
		# Inject metadata again after agent has finished navigating and running
		# This ensures it's available even if the initial injection was lost
		if browser_instance:
			try:
				await browser_instance.execute_script(metadata_script)
			except Exception as e:
				if args.verbose:
					logging.debug('Could not re-inject metadata after agent run: %s', e)

		# Check if a result was recorded during the agent run
		# If not (agent reached max-steps without submitting), record it now
		if not puzzle_result_received[0]:
			logging.info('[agent_end] No submission detected during agent run, analyzing errors')

			# Use last_failure_type if agent was stopped due to consecutive failures
			# Otherwise classify errors from history
			error_type = last_failure_type[0] if last_failure_type[0] else classify_agent_errors(history)

			# Get puzzle info from URL or browser
			puzzle_type = expected_puzzle_type[0]
			puzzle_id = None
			try:
				puzzle_info = await extract_puzzle_from_browser()
				if puzzle_info:
					puzzle_type = puzzle_info.get('puzzle_type') or puzzle_type
					puzzle_id = puzzle_info.get('puzzle_id')
			except Exception:
				pass

			if puzzle_type:
				await record_benchmark_result(
					puzzle_type=puzzle_type,
					puzzle_id=puzzle_id or 'unknown',
					is_correct=False,
					user_answer=error_type,
				)
			else:
				logging.warning('[agent_end] Could not determine puzzle_type, skipping result recording')

	except ValueError as exc:
		# Commonly raised when the chosen LLM requires API keys.
		logging.error(str(exc))
		if llm_logger:
			llm_logger.log_error(step=-1, error_type='ValueError', error_message=str(exc))
			llm_logger.close()
		# Record result for error case
		if expected_puzzle_type[0] and not puzzle_result_received[0]:
			await record_benchmark_result(
				puzzle_type=expected_puzzle_type[0],
				puzzle_id='unknown',
				is_correct=False,
				user_answer=f'ERROR_VALUE_ERROR',
			)
		return 1
	except asyncio.TimeoutError as exc:
		logging.error(f'Agent timed out: {exc}')
		if llm_logger:
			llm_logger.log_error(step=-1, error_type='TimeoutError', error_message=str(exc))
			llm_logger.close()
		# Record result for timeout case
		if expected_puzzle_type[0] and not puzzle_result_received[0]:
			await record_benchmark_result(
				puzzle_type=expected_puzzle_type[0],
				puzzle_id='unknown',
				is_correct=False,
				user_answer='ERROR_TIMEOUT',
			)
		return 1
	except Exception as exc:
		logging.error(f'Agent error: {exc}')
		if llm_logger:
			llm_logger.log_error(step=-1, error_type=type(exc).__name__, error_message=str(exc))
			llm_logger.close()
		# Record result for error case
		if expected_puzzle_type[0] and not puzzle_result_received[0]:
			await record_benchmark_result(
				puzzle_type=expected_puzzle_type[0],
				puzzle_id='unknown',
				is_correct=False,
				user_answer=f'ERROR_{type(exc).__name__.upper()}',
			)
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
		help='Maximum output tokens for the LLM response (OpenAI, Google/Gemini, vLLM). '
		'For GPT-5.2 xhigh: set 65536-128000. For gemini-3-pro: set 40960-65536. '
		'For Qwen3-VL-8B-Thinking: set 32768-40960. GPT-5.2 supports up to 128K output tokens.'
	)
	parser.add_argument(
		'--thinking-budget',
		type=int,
		default=None,
		help='Thinking token budget for Gemini 2.5 models (0-32768 tokens). '
		'Use -1 for dynamic allocation. Example: --thinking-budget 24576'
	)
	parser.add_argument(
		'--thinking-level',
		type=str,
		choices=['minimal', 'low', 'medium', 'high'],
		default=None,
		help='Thinking level for Gemini 3 models. '
		'Pro: low/high. Flash: minimal/low/medium/high. Example: --thinking-level high'
	)
	parser.add_argument(
		'--anthropic-thinking-budget',
		type=int,
		default=None,
		help='Token budget for Claude extended thinking (min 1024, recommended 10000-32768). '
		'Supported on: claude-haiku-4-5, claude-sonnet-4-5, claude-opus-4-5. '
		'Example: --anthropic-thinking-budget 10000'
	)
	parser.add_argument(
		'--anthropic-effort',
		type=str,
		choices=['low', 'medium', 'high'],
		default=None,
		help='Effort level for Claude Opus 4.5 ONLY (controls token spending). '
		'Requires beta header effort-2025-11-24. Options: low (efficient), medium (balanced), high (max). '
		'Example: --anthropic-effort high'
	)
	parser.add_argument(
		'--disable-thinking',
		action='store_true',
		default=False,
		help='Disable thinking mode for Qwen thinking models (vLLM). '
		'When disabled, injects enable_thinking=False to suppress reasoning. '
		'Default: thinking enabled (client parses <think> blocks).'
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
		help='Limit number of browser actions the agent may take per reasoning step. Lower values reduce DOM staleness issues.',
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
	parser.add_argument('--base-url', dest='base_url', help='Custom API base URL (required for vllm provider)')
	parser.add_argument('--api-key', dest='api_key', help='API key (use "EMPTY" for local vLLM)')
	parser.add_argument(
		'--debug-vllm',
		action='store_true',
		default=False,
		help='Enable verbose debug output for vLLM/Qwen thinking models. '
		'Shows request body, response reasoning preview, and parsing details.'
	)
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
