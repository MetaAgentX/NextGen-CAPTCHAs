"""
LLM Response Logger for capturing detailed LLM outputs during agent runs.

Provides two logger implementations:
- EnhancedLLMLogger: Per-puzzle logging with screenshots (recommended)
- LegacyLLMLogger: Single-file logging (backward compatible)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import base64
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union


# =============================================================================
# PROVIDER-SPECIFIC USAGE EXTRACTION FUNCTIONS
# =============================================================================
# Each provider returns token usage in a different format.
# These functions normalize them to a common schema for easy comparison.
#
# Common output schema:
# {
#     'input_tokens': int,      # Tokens in the prompt
#     'output_tokens': int,     # Tokens in the response
#     'total_tokens': int,      # Total tokens used
#     'reasoning_tokens': int,  # Thinking/reasoning tokens (if available)
#     'cached_tokens': int,     # Cached input tokens (if available)
# }
# =============================================================================


# Cached tiktoken encoder (lazy-loaded)
_tiktoken_encoder = None


def _count_tokens_tiktoken(text: str) -> Optional[int]:
	"""
	Count tokens using tiktoken (cl100k_base encoding).

	This is the same tokenizer used by GPT-4, GPT-3.5-turbo, and text-embedding-ada-002.
	While not exact for Claude models, it provides a reasonable approximation
	that is far more accurate than character-based estimation.

	Args:
		text: The text to tokenize

	Returns:
		Token count, or None if tiktoken is not installed
	"""
	global _tiktoken_encoder

	if not text:
		return None

	try:
		if _tiktoken_encoder is None:
			import tiktoken
			_tiktoken_encoder = tiktoken.get_encoding("cl100k_base")
		return len(_tiktoken_encoder.encode(text))
	except ImportError:
		logging.warning(
			"tiktoken not installed. Cannot count tokens accurately. "
			"Install with: pip install tiktoken"
		)
		return None
	except Exception as e:
		logging.warning(f"Failed to count tokens with tiktoken: {e}")
		return None


def extract_openai_usage(response: Any) -> dict:
	"""
	Extract usage from OpenAI API response (GPT-5, o1, o3 series).

	OpenAI Response Structure:
	{
	    "usage": {
	        "prompt_tokens": 100,
	        "completion_tokens": 500,
	        "total_tokens": 600,
	        "prompt_tokens_details": {
	            "cached_tokens": 0
	        },
	        "completion_tokens_details": {
	            "reasoning_tokens": 200  # <-- Thinking tokens for o1/o3/GPT-5
	        }
	    }
	}
	"""
	usage = {
		'input_tokens': None,
		'output_tokens': None,
		'total_tokens': None,
		'reasoning_tokens': None,
		'cached_tokens': None,
	}

	usage_obj = getattr(response, 'usage', None)
	if not usage_obj:
		return usage

	# Core token counts
	usage['input_tokens'] = getattr(usage_obj, 'prompt_tokens', None)
	usage['output_tokens'] = getattr(usage_obj, 'completion_tokens', None)
	usage['total_tokens'] = getattr(usage_obj, 'total_tokens', None)

	# Reasoning tokens (nested in completion_tokens_details)
	details = getattr(usage_obj, 'completion_tokens_details', None)
	if details:
		usage['reasoning_tokens'] = getattr(details, 'reasoning_tokens', None)

	# Cached tokens (nested in prompt_tokens_details)
	prompt_details = getattr(usage_obj, 'prompt_tokens_details', None)
	if prompt_details:
		usage['cached_tokens'] = getattr(prompt_details, 'cached_tokens', None)

	return usage


def extract_anthropic_usage(response: Any) -> dict:
	"""
	Extract usage from Anthropic API response (Claude 4.5 Opus, Claude 4 Sonnet).

	Anthropic Response Structure:
	{
	    "usage": {
	        "input_tokens": 100,
	        "output_tokens": 500,
	        "cache_creation_input_tokens": 0,
	        "cache_read_input_tokens": 0
	    },
	    "content": [
	        {"type": "thinking", "thinking": "Let me think..."},  # <-- Thinking here!
	        {"type": "text", "text": "The answer is..."}
	    ]
	}

	NOTE: Claude does NOT have a separate 'reasoning_tokens' field!
	Thinking tokens are INCLUDED in output_tokens. To count them separately,
	you must parse the content blocks where type == 'thinking'.
	"""
	usage = {
		'input_tokens': None,
		'output_tokens': None,
		'total_tokens': None,
		'reasoning_tokens': None,
		'cached_tokens': None,
	}

	usage_obj = getattr(response, 'usage', None)
	if not usage_obj:
		return usage

	# Core token counts
	usage['input_tokens'] = getattr(usage_obj, 'input_tokens', None)
	usage['output_tokens'] = getattr(usage_obj, 'output_tokens', None)

	# Anthropic doesn't provide total_tokens, compute it
	if usage['input_tokens'] and usage['output_tokens']:
		usage['total_tokens'] = usage['input_tokens'] + usage['output_tokens']

	# Cached tokens
	usage['cached_tokens'] = getattr(usage_obj, 'cache_read_input_tokens', None)

	# Extract thinking tokens from content blocks (Claude-specific)
	# Anthropic does NOT provide reasoning token counts in the API response.
	# We use tiktoken (cl100k_base encoding) to count tokens accurately.
	content = getattr(response, 'content', None)
	if content and isinstance(content, list):
		thinking_text = ""
		for block in content:
			if getattr(block, 'type', None) == 'thinking':
				thinking_text += getattr(block, 'thinking', '') or ''

		if thinking_text:
			usage['reasoning_tokens'] = _count_tokens_tiktoken(thinking_text)

	return usage


def extract_gemini_usage(response: Any) -> dict:
	"""
	Extract usage from Google Gemini API response (Gemini 3 Pro, Gemini 2.5).

	Gemini Response Structure:
	{
	    "usage_metadata": {
	        "prompt_token_count": 100,
	        "candidates_token_count": 500,
	        "total_token_count": 600,
	        "thoughts_token_count": 200,  # <-- Thinking tokens (direct field!)
	        "cached_content_token_count": 0
	    }
	}

	NOTE: Gemini uses 'usage_metadata' not 'usage'!
	"""
	usage = {
		'input_tokens': None,
		'output_tokens': None,
		'total_tokens': None,
		'reasoning_tokens': None,
		'cached_tokens': None,
	}

	# Gemini uses 'usage_metadata' instead of 'usage'
	usage_obj = getattr(response, 'usage_metadata', None)
	if not usage_obj:
		return usage

	# Core token counts (Gemini naming convention)
	usage['input_tokens'] = getattr(usage_obj, 'prompt_token_count', None)
	usage['output_tokens'] = getattr(usage_obj, 'candidates_token_count', None)
	usage['total_tokens'] = getattr(usage_obj, 'total_token_count', None)

	# Thinking tokens (direct field in Gemini!)
	usage['reasoning_tokens'] = getattr(usage_obj, 'thoughts_token_count', None)

	# Cached tokens
	usage['cached_tokens'] = getattr(usage_obj, 'cached_content_token_count', None)

	return usage


def extract_usage(response: Any, provider: str = None) -> dict:
	"""
	Extract usage from any LLM provider response.

	Args:
		response: The LLM response object
		provider: Provider name ('openai', 'anthropic', 'google') - REQUIRED

	Returns:
		Unified usage dict with normalized field names

	Raises:
		ValueError: If provider is not specified or unknown
	"""
	if provider is None:
		raise ValueError(
			"Provider must be explicitly specified. "
			"Use provider='openai', 'anthropic', or 'google'."
		)

	# Route to provider-specific extractor
	if provider == 'google':
		return extract_gemini_usage(response)
	elif provider == 'anthropic':
		return extract_anthropic_usage(response)
	elif provider == 'openai':
		return extract_openai_usage(response)
	else:
		raise ValueError(f"Unknown provider: {provider}. Use 'openai', 'anthropic', or 'google'.")


class EnhancedLLMLogger:
	"""
	Enhanced logger with per-puzzle file structure and screenshot support.

	Terminology (aligned with browser-use):
	- "step" in browser-use = one agent reasoning cycle (think → decide → act)
	- "call" here = one LLM invocation (a step may involve multiple LLM calls)
	- "puzzle_step" = LLM call counter within the current puzzle

	Directory structure:
	llm_logs/{run_id}/
	├── run_config.json
	├── summary.json
	└── puzzles/
	    ├── dice_count_0000/       # Uses puzzle_id directly as directory name
	    │   ├── log.jsonl          # Detailed events (for debugging)
	    │   ├── steps.jsonl        # Clean input/output pairs (for fine-tuning)
	    │   └── screenshots/
	    │       ├── step_001.jpg
	    │       └── ...
	    └── ...

	steps.jsonl format (each line, matching browser-use structure):
	{
	    "call_id": 1,           # LLM invocation number (global)
	    "puzzle_step": 1,       # LLM call counter within this puzzle
	    "timing": {"start_time": "...", "end_time": "...", "duration_ms": 1234.5},
	    "input": {"messages": [{"role": "...", "content": "..."}]},
	    "output": {
	        "content": "...",
	        "current_state": {
	            "evaluation_previous_goal": "...",
	            "memory": "...",
	            "next_goal": "..."
	        },
	        "actions": [...]
	    },
	    "metadata": {"usage": {"input_tokens": ..., "output_tokens": ..., "reasoning_tokens": ...}}
	}
	"""

	def __init__(self, log_dir: str = 'llm_logs', experiment_name: str = None, run_id: str = None, provider: str = None):
		"""Initialize the enhanced logger with structured directory.

		Args:
			log_dir: Directory for storing logs (default: 'llm_logs')
			experiment_name: Optional experiment name suffix
			run_id: Optional run ID to use instead of generating a new timestamp.
			        This allows multiple logger instances to share the same run directory.
			provider: LLM provider name ('openai', 'google', 'anthropic', etc.)
			          Used for provider-specific logging with native field names.
		"""
		self.provider = provider

		# Use provided run_id or generate unique one
		if run_id:
			exp_suffix = f'_{experiment_name}' if experiment_name else ''
			self.run_id = f'{run_id}{exp_suffix}'
		else:
			timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
			exp_suffix = f'_{experiment_name}' if experiment_name else ''
			self.run_id = f'{timestamp}{exp_suffix}'

		# Create directory structure
		self.log_dir = Path(log_dir)
		self.run_dir = self.log_dir / self.run_id
		self.puzzles_dir = self.run_dir / 'puzzles'
		self.puzzles_dir.mkdir(parents=True, exist_ok=True)

		# File paths
		self.run_config_file = self.run_dir / 'run_config.json'
		self.summary_file = self.run_dir / 'summary.json'

		# State tracking
		self.step_counter = 0
		self.puzzle_counter = 0
		self.current_puzzle_dir: Optional[Path] = None
		self.current_puzzle_log: Optional[Path] = None
		self.current_puzzle_steps_log: Optional[Path] = None  # For fine-tuning pairs
		self.current_puzzle_type: Optional[str] = None
		self.current_puzzle_id: Optional[str] = None
		self.current_puzzle_start: Optional[datetime] = None
		self.current_puzzle_steps = 0
		self.current_puzzle_input_tokens = 0
		self.current_puzzle_output_tokens = 0
		self.current_puzzle_reasoning_tokens = 0
		self.puzzle_stats: list[dict] = []
		self.puzzle_step_counter = 0  # Track steps within current puzzle (for steps.jsonl)

		# Load existing summary if continuing a run (e.g., isolate-puzzles mode with shared run_id)
		if self.summary_file.exists():
			try:
				with open(self.summary_file, 'r', encoding='utf-8') as f:
					existing_summary = json.load(f)
				self.puzzle_stats = existing_summary.get('puzzles', [])
				self.puzzle_counter = len(self.puzzle_stats)
				self.step_counter = existing_summary.get('total_llm_calls', 0)
				logging.info(f'Loaded existing summary: {self.puzzle_counter} puzzles, {self.step_counter} LLM calls')
			except (json.JSONDecodeError, KeyError) as e:
				logging.warning(f'Could not load existing summary: {e}')

		# For backward compatibility with set_puzzle_context
		self.puzzle_type = None
		self.puzzle_index = None

		# Track if first puzzle has been started
		self._first_puzzle_started = False
		self._early_call_buffer = []  # Buffer LLM calls before puzzle detection (for log.jsonl)
		self._early_step_buffer = []  # Buffer LLM step entries before puzzle detection (for steps.jsonl)
		self._early_screenshot_buffer = []  # Buffer screenshots before puzzle detection

		logging.info(f'Enhanced LLM logger initialized. Run ID: {self.run_id}')
		logging.info(f'Logs will be saved to: {self.run_dir}')

	def write_run_config(self, args: Any = None, llm_info: dict = None):
		"""Write run configuration to run_config.json."""
		config = {
			'run_id': self.run_id,
			'start_time': datetime.now().isoformat(),
			'cli_args': {},
			'llm_info': llm_info or {},
			'environment': {
				'python_version': sys.version,
				'platform': sys.platform,
				'cwd': os.getcwd(),
			}
		}

		# Extract CLI args if provided
		if args:
			config['cli_args'] = {
				'url': getattr(args, 'url', None),
				'limit': getattr(args, 'limit', None),
				'llm': getattr(args, 'llm', None),
				'model': getattr(args, 'model', None),
				'fast': getattr(args, 'fast', False),
				'reasoning_effort': getattr(args, 'reasoning_effort', None),
				'max_steps': getattr(args, 'max_steps', None),
				'max_actions_per_step': getattr(args, 'max_actions_per_step', None),
				'max_failures': getattr(args, 'max_failures', None),
				'isolate_puzzles': getattr(args, 'isolate_puzzles', False),
				'use_cloud': getattr(args, 'use_cloud', False),
				'headless': getattr(args, 'headless', False),
				'window_width': getattr(args, 'window_width', None),
				'window_height': getattr(args, 'window_height', None),
				'verbose': getattr(args, 'verbose', False),
				'llm_log_dir': getattr(args, 'llm_log_dir', None),
			}

		# Store task_prompt placeholder - will be updated when log_task_prompt is called
		config['task_prompt'] = None

		with open(self.run_config_file, 'w', encoding='utf-8') as f:
			json.dump(config, f, indent=2, ensure_ascii=False)

		logging.info(f'Run config written to: {self.run_config_file}')

	def start_puzzle(self, puzzle_type: str, puzzle_id: str, prompt: str = None):
		"""Start logging for a new puzzle - creates new directory and log file."""
		# Close previous puzzle if any
		if self.current_puzzle_log:
			self.end_puzzle()

		self.puzzle_counter += 1
		self.current_puzzle_type = puzzle_type
		self.current_puzzle_id = puzzle_id
		self.current_puzzle_start = datetime.now()
		self.current_puzzle_steps = 0
		self._first_puzzle_started = True

		# Also set for backward compatibility
		self.puzzle_type = puzzle_type
		self.puzzle_index = self.puzzle_counter

		# Create safe directory name using puzzle_id (e.g., dice_count_0000)
		safe_id = puzzle_id.replace('/', '_').replace('\\', '_').replace('.', '_')
		dir_name = safe_id

		self.current_puzzle_dir = self.puzzles_dir / dir_name
		self.current_puzzle_dir.mkdir(parents=True, exist_ok=True)
		self.current_puzzle_log = self.current_puzzle_dir / 'log.jsonl'
		self.current_puzzle_steps_log = self.current_puzzle_dir / 'steps.jsonl'  # Clean input/output pairs
		self.puzzle_step_counter = 0  # Reset step counter for new puzzle

		# Write puzzle_start event
		start_event = {
			'event': 'puzzle_start',
			'timestamp': self.current_puzzle_start.isoformat(),
			'puzzle_number': self.puzzle_counter,
			'puzzle_type': puzzle_type,
			'puzzle_id': puzzle_id,
			'prompt': prompt,
		}
		self._write_to_puzzle_log(start_event)

		logging.info(f'Started puzzle {self.puzzle_counter}: {puzzle_type} ({puzzle_id})')

		# Flush buffered early calls now that puzzle is started
		if self._early_call_buffer:
			buffered_count = len(self._early_call_buffer)
			for buffered_entry in self._early_call_buffer:
				# Update with real puzzle info
				buffered_entry['puzzle_type'] = puzzle_type
				buffered_entry['puzzle_id'] = puzzle_id
				# Accumulate tokens
				usage = buffered_entry.get('usage', {})
				if usage.get('input_tokens'):
					self.current_puzzle_input_tokens += usage['input_tokens']
				if usage.get('output_tokens'):
					self.current_puzzle_output_tokens += usage['output_tokens']
				if usage.get('reasoning_tokens'):
					self.current_puzzle_reasoning_tokens += usage['reasoning_tokens']
				# Count as puzzle step if it was a response
				if buffered_entry.get('metadata', {}).get('direction') == 'response':
					self.current_puzzle_steps += 1
				# Write to log
				self._write_to_puzzle_log(buffered_entry)
			self._early_call_buffer = []
			logging.info(f'Flushed {buffered_count} buffered early LLM calls')

		# Flush buffered early step entries (for steps.jsonl) now that puzzle is started
		if self._early_step_buffer:
			step_buffered_count = len(self._early_step_buffer)
			for step_entry in self._early_step_buffer:
				# Update with real puzzle info
				step_entry['puzzle_type'] = puzzle_type
				step_entry['puzzle_id'] = puzzle_id
				# Increment puzzle step counter
				self.puzzle_step_counter += 1
				step_entry['puzzle_step'] = self.puzzle_step_counter
				# Accumulate tokens from usage (handle Gemini, OpenAI, and Anthropic formats)
				usage = step_entry.get('usage', {})
				if usage:
					# Gemini format
					if usage.get('prompt_token_count'):
						self.current_puzzle_input_tokens += usage['prompt_token_count']
					if usage.get('candidates_token_count'):
						self.current_puzzle_output_tokens += usage['candidates_token_count']
					if usage.get('thoughts_token_count'):
						self.current_puzzle_reasoning_tokens += usage['thoughts_token_count']
					# OpenAI format
					if usage.get('prompt_tokens'):
						self.current_puzzle_input_tokens += usage['prompt_tokens']
					if usage.get('completion_tokens'):
						self.current_puzzle_output_tokens += usage['completion_tokens']
					details = usage.get('completion_tokens_details') or {}
					if details.get('reasoning_tokens'):
						self.current_puzzle_reasoning_tokens += details['reasoning_tokens']
					# Anthropic format
					if usage.get('input_tokens'):
						self.current_puzzle_input_tokens += usage['input_tokens']
					if usage.get('output_tokens'):
						self.current_puzzle_output_tokens += usage['output_tokens']
				# Anthropic thinking tokens (stored separately in thinking_content)
				thinking_content = step_entry.get('thinking_content')
				if thinking_content:
					thinking_tokens = _count_tokens_tiktoken(thinking_content)
					if thinking_tokens is not None:
						self.current_puzzle_reasoning_tokens += thinking_tokens
				# vLLM (and other providers) may store reasoning tokens at the top level
				# (e.g., _log_step_vllm sets 'reasoning_tokens' directly).
				reason_tokens = step_entry.get('reasoning_tokens')
				if reason_tokens:
					self.current_puzzle_reasoning_tokens += reason_tokens
				# Write to steps.jsonl
				with open(self.current_puzzle_steps_log, 'a', encoding='utf-8') as f:
					f.write(json.dumps(step_entry, ensure_ascii=False) + '\n')
			self._early_step_buffer = []
			logging.info(f'Flushed {step_buffered_count} buffered early step entries')

		# Flush buffered early screenshots now that puzzle is started
		if self._early_screenshot_buffer:
			screenshot_buffered_count = len(self._early_screenshot_buffer)
			screenshots_dir = self.current_puzzle_dir / 'screenshots'
			screenshots_dir.mkdir(exist_ok=True)
			for step, screenshot_data in self._early_screenshot_buffer:
				try:
					filename = f'step_{step:03d}.jpg'
					filepath = screenshots_dir / filename
					# Handle both bytes and base64 string
					if isinstance(screenshot_data, str):
						if screenshot_data.startswith('data:'):
							screenshot_data = screenshot_data.split(',', 1)[1]
						screenshot_bytes = base64.b64decode(screenshot_data)
					else:
						screenshot_bytes = screenshot_data
					with open(filepath, 'wb') as f:
						f.write(screenshot_bytes)
				except Exception as e:
					logging.warning(f'Failed to save buffered screenshot for step {step}: {e}')
			self._early_screenshot_buffer = []
			logging.info(f'Flushed {screenshot_buffered_count} buffered early screenshots')

	def end_puzzle(self, answer: Any = None, is_correct: bool = None,
				   correct_answer: Any = None):
		"""End current puzzle logging with result."""
		if not self.current_puzzle_log:
			return

		end_time = datetime.now()
		duration = (end_time - self.current_puzzle_start).total_seconds() if self.current_puzzle_start else 0

		# Compute total tokens from components.
		# For Anthropic, reasoning tokens are INCLUDED in output_tokens (per API docs).
		# Only add reasoning_tokens separately for providers that track them independently.
		if self.provider == 'anthropic':
			total_tokens = (
				self.current_puzzle_input_tokens +
				self.current_puzzle_output_tokens
			)
		else:
			total_tokens = (
				self.current_puzzle_input_tokens +
				self.current_puzzle_output_tokens +
				self.current_puzzle_reasoning_tokens
			)

		end_event = {
			'event': 'puzzle_end',
			'timestamp': end_time.isoformat(),
			'puzzle_number': self.puzzle_counter,
			'puzzle_type': self.current_puzzle_type,
			'puzzle_id': self.current_puzzle_id,
			'duration_seconds': duration,
			'total_steps': self.current_puzzle_steps,
			'total_tokens': total_tokens,
			'input_tokens': self.current_puzzle_input_tokens,
			'output_tokens': self.current_puzzle_output_tokens,
			'reasoning_tokens': self.current_puzzle_reasoning_tokens,
			'user_answer': answer,
			'is_correct': is_correct,
			'correct_answer': correct_answer,
		}
		self._write_to_puzzle_log(end_event)

		# Add to stats for summary
		self.puzzle_stats.append({
			'puzzle_number': self.puzzle_counter,
			'puzzle_type': self.current_puzzle_type,
			'puzzle_id': self.current_puzzle_id,
			'is_correct': is_correct,
			'duration_seconds': duration,
			'steps': self.current_puzzle_steps,
			'tokens': total_tokens,
			'input_tokens': self.current_puzzle_input_tokens,
			'output_tokens': self.current_puzzle_output_tokens,
			'reasoning_tokens': self.current_puzzle_reasoning_tokens,
			'directory': self.current_puzzle_dir.name if self.current_puzzle_dir else None,
		})

		logging.info(f'Ended puzzle {self.puzzle_counter}: correct={is_correct}, duration={duration:.1f}s')

		# Reset current puzzle state
		self.current_puzzle_dir = None
		self.current_puzzle_log = None
		self.current_puzzle_type = None
		self.current_puzzle_id = None
		self.current_puzzle_start = None
		self.current_puzzle_input_tokens = 0
		self.current_puzzle_output_tokens = 0
		self.current_puzzle_reasoning_tokens = 0

	def _write_to_puzzle_log(self, entry: dict):
		"""Write entry to current puzzle's JSONL log file."""
		if self.current_puzzle_log:
			with open(self.current_puzzle_log, 'a', encoding='utf-8') as f:
				f.write(json.dumps(entry, ensure_ascii=False) + '\n')

	def _is_puzzle_started(self) -> bool:
		"""Check if a puzzle log has been started.

		Returns:
			True if a puzzle is active and logging should proceed,
			False if no puzzle detected yet and logging should be skipped.

		Note: Does NOT create fallback - puzzle must be started via start_puzzle()
		when real puzzle_id is detected from browser.
		"""
		return self._first_puzzle_started

	def save_screenshot(self, step: int, screenshot_data: Union[bytes, str]) -> Optional[str]:
		"""
		Save screenshot to current puzzle's screenshots folder.

		Args:
			step: Current step number
			screenshot_data: Screenshot as bytes or base64 string

		Returns:
			Relative path to saved screenshot, or None if failed
		"""
		# If puzzle not started yet, buffer the screenshot for later
		if not self._first_puzzle_started:
			self._early_screenshot_buffer.append((step, screenshot_data))
			logging.debug(f'[save_screenshot] Buffered early screenshot for step {step} (total: {len(self._early_screenshot_buffer)})')
			return None  # Will be saved when puzzle starts

		if not self.current_puzzle_dir:
			return None

		try:
			screenshots_dir = self.current_puzzle_dir / 'screenshots'
			screenshots_dir.mkdir(exist_ok=True)

			filename = f'step_{step:03d}.jpg'
			filepath = screenshots_dir / filename

			# Handle both bytes and base64 string
			if isinstance(screenshot_data, str):
				# Remove data URL prefix if present
				if screenshot_data.startswith('data:'):
					screenshot_data = screenshot_data.split(',', 1)[1]
				screenshot_bytes = base64.b64decode(screenshot_data)
			else:
				screenshot_bytes = screenshot_data

			with open(filepath, 'wb') as f:
				f.write(screenshot_bytes)

			return str(filepath.relative_to(self.run_dir))
		except Exception as e:
			# Include data preview for debugging (first 100 chars if string)
			data_preview = ''
			if isinstance(screenshot_data, str):
				data_preview = f' (data: {screenshot_data[:100]}...)'
			elif isinstance(screenshot_data, bytes):
				data_preview = f' (bytes len: {len(screenshot_data)})'
			logging.warning(f'Failed to save screenshot: {e}{data_preview}')
			return None

	def detect_puzzle_transition(self, agent_output: Any) -> dict:
		"""
		Detect if the agent has completed a puzzle based on memory/goal text.

		Args:
			agent_output: The agent's output from the current step

		Returns:
			dict with:
			- transition_detected: bool
			- is_correct: Optional[bool]
			- answer_text: Optional[str] - extracted answer if found
		"""
		result = {'transition_detected': False, 'is_correct': None, 'answer_text': None}

		if not agent_output:
			return result

		# Try to get brain state
		brain = getattr(agent_output, 'current_state', None)
		if not brain:
			return result

		memory = str(getattr(brain, 'memory', '') or '')
		evaluation = str(getattr(brain, 'evaluation_previous_goal', '') or '')
		next_goal = str(getattr(brain, 'next_goal', '') or '')

		combined_text = f'{memory} {evaluation} {next_goal}'.lower()

		# Keywords indicating puzzle completion
		# NOTE: Avoid generic words like "incorrect" alone - for Mirror puzzles,
		# the agent identifies "incorrect mirrors" as part of the task itself.
		# Page-level detection handles most cases; this is a fallback.
		completion_keywords = [
			'correct!', 'wrong answer', 'right answer',
			'puzzle solved', 'puzzle complete',
			'answered correctly', 'answered incorrectly',
		]

		for keyword in completion_keywords:
			if keyword in combined_text:
				result['transition_detected'] = True

				# Try to determine correctness
				if any(w in combined_text for w in ['incorrect', 'wrong']):
					result['is_correct'] = False
				elif any(w in combined_text for w in ['correct!', 'right answer', 'answered correctly', 'puzzle solved']):
					result['is_correct'] = True

				break

		return result

	def set_puzzle_context(self, puzzle_type: str = None, puzzle_id: int = None):
		"""Set puzzle context (backward compatibility with LegacyLLMLogger)."""
		self.puzzle_type = puzzle_type
		self.puzzle_index = puzzle_id  # Keep internal attribute name for compatibility

		# Only start a puzzle if we have BOTH puzzle_type AND puzzle_id.
		# Do NOT use fallback values like 'unknown' - that masks debugging issues.
		if puzzle_type and puzzle_id is not None and not self._first_puzzle_started:
			self.start_puzzle(
				puzzle_type=puzzle_type,
				puzzle_id=f'puzzle_{puzzle_id}',
			)

	def log_response(self, call_id: int, response: Any, metadata: dict = None):
		"""Log an LLM response (request or response)."""
		# Only count actual LLM responses, not request logging
		# LoggingLLMWrapper calls this twice: once for request (direction='request')
		# and once for response (direction='response'). Only count responses.
		is_response = metadata and metadata.get('direction') == 'response'
		if is_response:
			self.step_counter += 1
			if self._first_puzzle_started:
				self.current_puzzle_steps += 1

		log_entry = {
			'event': 'llm_call',
			'call_id': call_id,                   # LLM call ID from LoggingLLMWrapper
			'call_sequence': self.step_counter,   # LLM call sequence number
			'timestamp': datetime.now().isoformat(),
			'puzzle_type': self.current_puzzle_type,  # Will be None if buffered
			'puzzle_id': self.current_puzzle_id,      # Will be None if buffered
			'metadata': metadata or {},
		}

		# Extract content from response
		if hasattr(response, 'content'):
			log_entry['content'] = str(response.content)
		if hasattr(response, 'text'):
			log_entry['text'] = str(response.text)
		if hasattr(response, 'message'):
			log_entry['message'] = str(response.message)
		if isinstance(response, str):
			log_entry['text'] = response
		if isinstance(response, dict):
			log_entry['response_data'] = response

		# Extract usage info using provider-specific extractors
		try:
			usage = extract_usage(response, provider=self.provider)

			# For Google provider, try to get usage from raw capture (has thoughts_token_count)
			if self.provider == 'google' and hasattr(self, '_raw_gemini_capture') and self._raw_gemini_capture:
				raw_gemini = self._raw_gemini_capture.peek()
				if raw_gemini:
					usage = {
						'input_tokens': raw_gemini.get('prompt_token_count'),
						'output_tokens': raw_gemini.get('candidates_token_count'),
						'total_tokens': raw_gemini.get('total_token_count'),
						'reasoning_tokens': raw_gemini.get('thoughts_token_count'),
						'cached_tokens': raw_gemini.get('cached_content_token_count'),
					}

			logging.debug(f'[log_response] extracted usage: {usage}')
			if any(v is not None for v in usage.values()):
				log_entry['usage'] = usage
		except Exception as e:
			logging.debug(f'[log_response] extract_usage failed: {e}')

		# Extract model info
		if hasattr(response, 'model'):
			log_entry['model'] = str(response.model)

		# Capture raw response if nothing else extracted
		if not any(k in log_entry for k in ['content', 'text', 'message', 'response_data']):
			log_entry['raw_response'] = str(response)
			if hasattr(response, '__dict__'):
				try:
					log_entry['response_dict'] = {
						k: str(v) for k, v in response.__dict__.items()
					}
				except Exception:
					pass

		# If puzzle not started yet, buffer the entry for later
		if not self._first_puzzle_started:
			self._early_call_buffer.append(log_entry)
			logging.debug(f'[log_response] Buffered early call (total: {len(self._early_call_buffer)})')
			return log_entry

		# Puzzle started - write to log
		# NOTE: Token accumulation is now ONLY handled by _log_step_*() methods
		# which use raw API responses for accurate counts. Do NOT accumulate here
		# to avoid double-counting.

		self._write_to_puzzle_log(log_entry)
		return log_entry

	def log_step(self, call_id: int, input_messages: list, output_response: Any,
				 start_time: str = None, end_time: str = None, duration_ms: float = None,
				 usage: dict = None, error: str = None):
		"""
		Log a complete LLM call (input + output) for fine-tuning.

		Routes to provider-specific logging methods based on self.provider.
		Each provider uses its native field names for maximum clarity.

		Note: `call_id` is the LLM invocation counter, NOT the browser-use agent step.
		Browser-use "step" = agent reasoning cycle. A single step may have multiple LLM calls.
		"""
		# Use current time if end_time not provided
		if not end_time:
			end_time = datetime.now().isoformat()

		# Capture raw API responses from injected capture references
		raw_openai = None
		raw_gemini = None
		if hasattr(self, '_raw_openai_capture') and self._raw_openai_capture:
			try:
				raw_openai = self._raw_openai_capture.consume()
			except Exception as e:
				logging.debug(f'Failed to capture raw OpenAI response: {e}')

		if hasattr(self, '_raw_gemini_capture') and self._raw_gemini_capture:
			try:
				raw_gemini = self._raw_gemini_capture.consume()
			except Exception as e:
				logging.debug(f'Failed to capture raw Gemini response: {e}')

		# Capture raw vLLM response (usage + reasoning_content)
		raw_vllm = None
		reasoning_content = None
		if hasattr(self, '_raw_vllm_capture') and self._raw_vllm_capture:
			try:
				raw_vllm, reasoning_content = self._raw_vllm_capture.consume()
			except Exception as e:
				logging.debug(f'Failed to capture raw vLLM response: {e}')

		# Capture raw Anthropic response (for extended thinking blocks)
		raw_anthropic = None
		if hasattr(self, '_raw_anthropic_capture') and self._raw_anthropic_capture:
			try:
				raw_anthropic = self._raw_anthropic_capture.consume()
			except Exception as e:
				logging.debug(f'Failed to capture raw Anthropic response: {e}')

		# Route to provider-specific logging method
		if self.provider == 'google':
			return self._log_step_gemini(
				call_id, input_messages, output_response,
				start_time, end_time, duration_ms,
				raw_gemini, error
			)
		elif self.provider == 'openai':
			return self._log_step_openai(
				call_id, input_messages, output_response,
				start_time, end_time, duration_ms,
				raw_openai, error
			)
		elif self.provider in ('qwen', 'doubao'):
			# Qwen and Doubao use OpenAI-compatible APIs via ExtraBodyInjectingClient
			# which stores raw responses in RawResponseCapture (same as OpenAI)
			return self._log_step_openai(
				call_id, input_messages, output_response,
				start_time, end_time, duration_ms,
				raw_openai, error
			)
		elif self.provider == 'anthropic':
			return self._log_step_anthropic(
				call_id, input_messages, output_response,
				start_time, end_time, duration_ms,
				raw_anthropic, error
			)
		elif self.provider == 'vllm':
			return self._log_step_vllm(
				call_id, input_messages, output_response,
				start_time, end_time, duration_ms,
				raw_vllm, reasoning_content, error
			)

		# Fallback: Generic logging for unknown providers (original behavior)
		# Build step entry (puzzle_step will be set later if buffered)
		step_entry = {
			'call_id': call_id,
			'puzzle_step': None,  # Will be set when written or flushed
			'puzzle_type': self.current_puzzle_type,
			'puzzle_id': self.current_puzzle_id,
			'timing': {
				'start_time': start_time,
				'end_time': end_time,
				'duration_ms': duration_ms,
			},
			'input': {
				'messages': input_messages,
			},
			'output': {},
			'metadata': {
				'usage': usage,
			}
		}

		# Store raw API responses (already consumed at lines 694-707)
		if raw_openai:
			step_entry['raw_openai_response'] = raw_openai
		if raw_gemini:
			step_entry['raw_gemini_response'] = raw_gemini

		# Extract output content
		if error:
			step_entry['output']['error'] = error
		elif output_response:
			# Extract the main content
			if hasattr(output_response, 'content'):
				step_entry['output']['content'] = str(output_response.content)
			elif hasattr(output_response, 'text'):
				step_entry['output']['content'] = str(output_response.text)
			elif isinstance(output_response, str):
				step_entry['output']['content'] = output_response
			elif isinstance(output_response, dict):
				step_entry['output'] = output_response

			# Extract structured fields if present (browser-use agent format)
			if hasattr(output_response, 'current_state'):
				brain = output_response.current_state
				step_entry['output']['current_state'] = {
					'evaluation_previous_goal': getattr(brain, 'evaluation_previous_goal', None),
					'memory': getattr(brain, 'memory', None),
					'next_goal': getattr(brain, 'next_goal', None),
				}

			# Extract actions if present
			if hasattr(output_response, 'action'):
				try:
					actions = []
					for action in output_response.action:
						action_dict = action.model_dump() if hasattr(action, 'model_dump') else {}
						action_info = {k: v for k, v in action_dict.items() if v is not None}
						actions.append(action_info)
					step_entry['output']['actions'] = actions
				except Exception:
					pass

			# Extract usage from response if not provided
			# Note: This fallback path is for unknown providers. Only attempt extraction
			# if we have a provider set (extract_usage requires explicit provider).
			if not usage and self.provider:
				try:
					extracted_usage = extract_usage(output_response, provider=self.provider)
					if any(v is not None for v in extracted_usage.values()):
						step_entry['metadata']['usage'] = extracted_usage
				except Exception:
					pass

		# Now merge reasoning_tokens from raw captures into final usage
		# This must happen AFTER extract_usage to avoid being overwritten
		try:
			current_usage = step_entry['metadata']['usage']
			reasoning_tokens = None

			# Extract from OpenAI raw response
			if raw_openai:
				raw_usage = raw_openai.get('usage', {})
				details = raw_usage.get('completion_tokens_details', {})
				if details and details.get('reasoning_tokens'):
					reasoning_tokens = details['reasoning_tokens']

			# Extract from Gemini raw response (thoughts_token_count)
			if raw_gemini and raw_gemini.get('thoughts_token_count'):
				reasoning_tokens = raw_gemini['thoughts_token_count']

			# Merge reasoning_tokens into usage
			if reasoning_tokens:
				if current_usage is None:
					step_entry['metadata']['usage'] = {'reasoning_tokens': reasoning_tokens}
				elif isinstance(current_usage, dict):
					current_usage['reasoning_tokens'] = reasoning_tokens
				else:
					# Convert Pydantic model or other object to dict
					try:
						usage_dict = current_usage.model_dump() if hasattr(current_usage, 'model_dump') else dict(current_usage)
						usage_dict['reasoning_tokens'] = reasoning_tokens
						step_entry['metadata']['usage'] = usage_dict
					except Exception:
						pass
		except Exception as e:
			logging.debug(f'Failed to merge reasoning_tokens: {e}')

		# If puzzle not started yet, buffer the entry for later
		if not self._first_puzzle_started:
			self._early_step_buffer.append(step_entry)
			logging.debug(f'[log_step] Buffered early step entry (total: {len(self._early_step_buffer)})')
			return step_entry

		# Puzzle started - set puzzle_step and write
		self.puzzle_step_counter += 1
		step_entry['puzzle_step'] = self.puzzle_step_counter

		# Write to steps.jsonl
		if self.current_puzzle_steps_log:
			with open(self.current_puzzle_steps_log, 'a', encoding='utf-8') as f:
				f.write(json.dumps(step_entry, ensure_ascii=False) + '\n')

		return step_entry

	def _log_step_gemini(self, call_id: int, input_messages: list, output_response: Any,
						  start_time: str, end_time: str, duration_ms: float,
						  raw_gemini: dict, error: str = None) -> dict:
		"""
		Provider-specific logging for Google Gemini with native field names.

		Uses Gemini's native naming: prompt_token_count, candidates_token_count,
		thoughts_token_count, total_token_count, cached_content_token_count
		"""
		# Get request params from LLM instance
		request_params = {}
		if hasattr(self, '_llm_instance') and hasattr(self._llm_instance, '_request_params'):
			request_params = self._llm_instance._request_params

		# Build step entry (puzzle_step will be set later if buffered)
		step_entry = {
			'call_id': call_id,
			'puzzle_step': None,  # Will be set when written or flushed
			'provider': 'google',
			'model': request_params.get('model'),
			'puzzle_type': self.current_puzzle_type,
			'puzzle_id': self.current_puzzle_id,
			'timing': {
				'start_time': start_time,
				'end_time': end_time,
				'duration_ms': duration_ms,
			},
			'request_params': {
				'thinking_budget': request_params.get('thinking_budget'),
				'temperature': request_params.get('temperature'),
				'max_output_tokens': request_params.get('max_output_tokens'),
				'top_p': request_params.get('top_p'),
				'seed': request_params.get('seed'),
			},
			'input': {
				'messages': input_messages,
			},
			'output': {},
			# Gemini native usage field names
			'usage': raw_gemini if raw_gemini else {
				'prompt_token_count': None,
				'candidates_token_count': None,
				'thoughts_token_count': None,
				'total_token_count': None,
				'cached_content_token_count': None,
			},
		}

		# Extract output content
		self._extract_output_content(step_entry, output_response, error)

		# If puzzle not started yet, buffer the entry for later
		if not self._first_puzzle_started:
			self._early_step_buffer.append(step_entry)
			logging.debug(f'[_log_step_gemini] Buffered early step entry (total: {len(self._early_step_buffer)})')
			return step_entry

		# Puzzle started - set puzzle_step, accumulate tokens, and write
		self.puzzle_step_counter += 1
		step_entry['puzzle_step'] = self.puzzle_step_counter

		# Accumulate token counts for summary (using Gemini native field names)
		if raw_gemini:
			if raw_gemini.get('prompt_token_count'):
				self.current_puzzle_input_tokens += raw_gemini['prompt_token_count']
			if raw_gemini.get('candidates_token_count'):
				self.current_puzzle_output_tokens += raw_gemini['candidates_token_count']
			if raw_gemini.get('thoughts_token_count'):
				self.current_puzzle_reasoning_tokens += raw_gemini['thoughts_token_count']

		# Write to steps.jsonl
		if self.current_puzzle_steps_log:
			with open(self.current_puzzle_steps_log, 'a', encoding='utf-8') as f:
				f.write(json.dumps(step_entry, ensure_ascii=False) + '\n')

		return step_entry

	def _log_step_openai(self, call_id: int, input_messages: list, output_response: Any,
						  start_time: str, end_time: str, duration_ms: float,
						  raw_openai: dict, error: str = None) -> dict:
		"""
		Provider-specific logging for OpenAI with native field names.

		Uses OpenAI's native naming: prompt_tokens, completion_tokens,
		reasoning_tokens (in completion_tokens_details), cached_tokens (in prompt_tokens_details)
		"""
		# Get request params from LLM instance
		request_params = {}
		if hasattr(self, '_llm_instance') and hasattr(self._llm_instance, '_request_params'):
			request_params = self._llm_instance._request_params

		# Extract OpenAI native usage from raw response
		usage = {}
		if raw_openai and 'usage' in raw_openai:
			raw_usage = raw_openai['usage']
			usage = {
				'prompt_tokens': raw_usage.get('prompt_tokens'),
				'completion_tokens': raw_usage.get('completion_tokens'),
				'total_tokens': raw_usage.get('total_tokens'),
				'prompt_tokens_details': raw_usage.get('prompt_tokens_details'),
				'completion_tokens_details': raw_usage.get('completion_tokens_details'),
			}

		# Build step entry (puzzle_step will be set later if buffered)
		step_entry = {
			'call_id': call_id,
			'puzzle_step': None,  # Will be set when written or flushed
			'provider': 'openai',
			'model': request_params.get('model'),
			'puzzle_type': self.current_puzzle_type,
			'puzzle_id': self.current_puzzle_id,
			'timing': {
				'start_time': start_time,
				'end_time': end_time,
				'duration_ms': duration_ms,
			},
			'request_params': {
				'reasoning_effort': request_params.get('reasoning_effort'),
				'temperature': request_params.get('temperature'),
				'max_completion_tokens': request_params.get('max_completion_tokens'),
				'seed': request_params.get('seed'),
				'service_tier': request_params.get('service_tier'),
			},
			'input': {
				'messages': input_messages,
			},
			'output': {},
			# OpenAI native usage field names
			'usage': usage,
		}

		# Extract output content
		self._extract_output_content(step_entry, output_response, error)

		# If puzzle not started yet, buffer the entry for later
		if not self._first_puzzle_started:
			self._early_step_buffer.append(step_entry)
			logging.debug(f'[_log_step_openai] Buffered early step entry (total: {len(self._early_step_buffer)})')
			return step_entry

		# Puzzle started - set puzzle_step, accumulate tokens, and write
		self.puzzle_step_counter += 1
		step_entry['puzzle_step'] = self.puzzle_step_counter

		# Accumulate token counts for summary (using OpenAI native field names)
		if usage:
			if usage.get('prompt_tokens'):
				self.current_puzzle_input_tokens += usage['prompt_tokens']
			if usage.get('completion_tokens'):
				self.current_puzzle_output_tokens += usage['completion_tokens']
			# OpenAI reasoning tokens are nested in completion_tokens_details
			details = usage.get('completion_tokens_details') or {}
			if details.get('reasoning_tokens'):
				self.current_puzzle_reasoning_tokens += details['reasoning_tokens']

		# Write to steps.jsonl
		if self.current_puzzle_steps_log:
			with open(self.current_puzzle_steps_log, 'a', encoding='utf-8') as f:
				f.write(json.dumps(step_entry, ensure_ascii=False) + '\n')

		return step_entry

	def _log_step_vllm(self, call_id: int, input_messages: list, output_response: Any,
					   start_time: str, end_time: str, duration_ms: float,
					   raw_vllm: dict, reasoning_content: str = None, error: str = None) -> dict:
		"""
		Provider-specific logging for vLLM/Qwen with native field names.

		Captures: prompt_tokens, completion_tokens, reasoning_content (for thinking models)
		Uses Qwen tokenizer for accurate reasoning token counting.
		"""
		# Get request params from LLM instance
		request_params = {}
		if hasattr(self, '_llm_instance') and hasattr(self._llm_instance, '_request_params'):
			request_params = self._llm_instance._request_params

		# Extract usage from raw vLLM response
		usage = {
			'prompt_tokens': raw_vllm.get('prompt_tokens') if raw_vllm else None,
			'completion_tokens': raw_vllm.get('completion_tokens') if raw_vllm else None,
			'total_tokens': raw_vllm.get('total_tokens') if raw_vllm else None,
		}

		# Count reasoning tokens using Qwen tokenizer
		reasoning_tokens = None
		if reasoning_content:
			if hasattr(self, '_count_tokens_qwen') and self._count_tokens_qwen:
				reasoning_tokens = self._count_tokens_qwen(reasoning_content)
			else:
				# Fallback: rough approximation
				reasoning_tokens = len(reasoning_content) // 4

		# Build step entry
		step_entry = {
			'call_id': call_id,
			'puzzle_step': None,  # Will be set when written or flushed
			'provider': 'vllm',
			'model': request_params.get('model'),
			'puzzle_type': self.current_puzzle_type,
			'puzzle_id': self.current_puzzle_id,
			'timing': {
				'start_time': start_time,
				'end_time': end_time,
				'duration_ms': duration_ms,
			},
			'request_params': {
				'enable_thinking': request_params.get('enable_thinking'),
				'max_output_tokens': request_params.get('max_output_tokens'),
			},
			'input': {
				'messages': input_messages,
			},
			'output': {},
			# vLLM native usage field names
			'usage': usage,
			# Reasoning content and tokens (for thinking models)
			'reasoning_content': reasoning_content,
			'reasoning_tokens': reasoning_tokens,
		}

		# Extract output content
		self._extract_output_content(step_entry, output_response, error)

		# If puzzle not started yet, buffer the entry for later
		if not self._first_puzzle_started:
			self._early_step_buffer.append(step_entry)
			logging.debug(f'[_log_step_vllm] Buffered early step entry (total: {len(self._early_step_buffer)})')
			return step_entry

		# Puzzle started - set puzzle_step, accumulate tokens, and write
		self.puzzle_step_counter += 1
		step_entry['puzzle_step'] = self.puzzle_step_counter

		# Accumulate token counts for summary
		if usage.get('prompt_tokens'):
			self.current_puzzle_input_tokens += usage['prompt_tokens']
		if usage.get('completion_tokens'):
			self.current_puzzle_output_tokens += usage['completion_tokens']
		if reasoning_tokens:
			self.current_puzzle_reasoning_tokens += reasoning_tokens

		# Write to steps.jsonl
		if self.current_puzzle_steps_log:
			with open(self.current_puzzle_steps_log, 'a', encoding='utf-8') as f:
				f.write(json.dumps(step_entry, ensure_ascii=False) + '\n')

		return step_entry

	def _log_step_anthropic(self, call_id: int, input_messages: list, output_response: Any,
							 start_time: str, end_time: str, duration_ms: float,
							 raw_anthropic: dict = None, error: str = None) -> dict:
		"""
		Provider-specific logging for Anthropic with native field names.

		Uses Anthropic's native naming: input_tokens, output_tokens,
		cache_creation_input_tokens, cache_read_input_tokens

		Args:
			raw_anthropic: Raw API response dict from RawResponseCapture (contains thinking blocks)
		"""
		# Get request params from LLM instance
		request_params = {}
		if hasattr(self, '_llm_instance') and hasattr(self._llm_instance, '_request_params'):
			request_params = self._llm_instance._request_params

		# Extract Anthropic native usage - prefer raw response over wrapper
		usage = {
			'input_tokens': None,
			'output_tokens': None,
			'cache_creation_input_tokens': None,
			'cache_read_input_tokens': None,
		}
		# First try to get usage from raw API response (most accurate)
		if raw_anthropic and 'usage' in raw_anthropic:
			raw_usage = raw_anthropic['usage']
			usage = {
				'input_tokens': raw_usage.get('input_tokens'),
				'output_tokens': raw_usage.get('output_tokens'),
				'cache_creation_input_tokens': raw_usage.get('cache_creation_input_tokens'),
				'cache_read_input_tokens': raw_usage.get('cache_read_input_tokens'),
			}
		# Fallback to wrapper response
		elif output_response and hasattr(output_response, 'usage'):
			usage_obj = output_response.usage
			usage = {
				'input_tokens': getattr(usage_obj, 'input_tokens', None),
				'output_tokens': getattr(usage_obj, 'output_tokens', None),
				'cache_creation_input_tokens': getattr(usage_obj, 'cache_creation_input_tokens', None),
				'cache_read_input_tokens': getattr(usage_obj, 'cache_read_input_tokens', None),
			}

		# Extract thinking content from RAW API response (not wrapper)
		# The browser_use ChatAnthropic wrapper discards thinking blocks,
		# so we must extract from raw_anthropic['content'] array
		thinking_content = None
		if raw_anthropic and 'content' in raw_anthropic:
			# Raw API response has content as array of dicts
			for block in raw_anthropic['content']:
				if isinstance(block, dict) and block.get('type') == 'thinking':
					thinking_content = block.get('thinking')
					break
		# Fallback: try output_response.content (for non-thinking responses)
		elif output_response and hasattr(output_response, 'content'):
			content = output_response.content
			if isinstance(content, list):
				for block in content:
					if getattr(block, 'type', None) == 'thinking':
						thinking_content = getattr(block, 'thinking', None)
						break

		# Build step entry (puzzle_step will be set later if buffered)
		step_entry = {
			'call_id': call_id,
			'puzzle_step': None,  # Will be set when written or flushed
			'provider': 'anthropic',
			'model': request_params.get('model'),
			'puzzle_type': self.current_puzzle_type,
			'puzzle_id': self.current_puzzle_id,
			'timing': {
				'start_time': start_time,
				'end_time': end_time,
				'duration_ms': duration_ms,
			},
			'request_params': {
				'temperature': request_params.get('temperature'),
				'max_tokens': request_params.get('max_tokens'),
				'top_p': request_params.get('top_p'),
				'seed': request_params.get('seed'),
				'thinking_budget': request_params.get('thinking_budget'),
				'effort': request_params.get('effort'),
			},
			'input': {
				'messages': input_messages,
			},
			'output': {},
			# Anthropic native usage field names
			'usage': usage,
			# Claude thinking content (if present)
			'thinking_content': thinking_content,
		}

		# Extract output content
		self._extract_output_content(step_entry, output_response, error)

		# If puzzle not started yet, buffer the entry for later
		if not self._first_puzzle_started:
			self._early_step_buffer.append(step_entry)
			logging.debug(f'[_log_step_anthropic] Buffered early step entry (total: {len(self._early_step_buffer)})')
			return step_entry

		# Puzzle started - set puzzle_step, accumulate tokens, and write
		self.puzzle_step_counter += 1
		step_entry['puzzle_step'] = self.puzzle_step_counter

		# Accumulate token counts for summary (using Anthropic native field names)
		if usage:
			if usage.get('input_tokens'):
				self.current_puzzle_input_tokens += usage['input_tokens']
			if usage.get('output_tokens'):
				self.current_puzzle_output_tokens += usage['output_tokens']
		# Count Anthropic thinking tokens using tiktoken (accurate tokenization)
		# Anthropic doesn't provide native thinking token counts in the API response.
		logging.debug(f'[ANTHROPIC THINKING] thinking_content={bool(thinking_content)}, raw_anthropic={bool(raw_anthropic)}')
		if thinking_content:
			thinking_tokens = _count_tokens_tiktoken(thinking_content)
			logging.info(f'[ANTHROPIC THINKING] Counted {thinking_tokens} thinking tokens from content ({len(thinking_content)} chars)')
			if thinking_tokens is not None:
				self.current_puzzle_reasoning_tokens += thinking_tokens
				logging.info(f'[ANTHROPIC THINKING] Updated current_puzzle_reasoning_tokens to {self.current_puzzle_reasoning_tokens}')
		else:
			logging.debug(f'[ANTHROPIC THINKING] No thinking_content found in step {call_id}')

		# Write to steps.jsonl
		if self.current_puzzle_steps_log:
			with open(self.current_puzzle_steps_log, 'a', encoding='utf-8') as f:
				f.write(json.dumps(step_entry, ensure_ascii=False) + '\n')

		return step_entry

	def _extract_output_content(self, step_entry: dict, output_response: Any, error: str = None):
		"""
		Common helper to extract output content from LLM response.
		Modifies step_entry in place.

		Handles multiple response formats:
		- Standard LLM responses with .content or .text
		- Browser-use AgentOutput with .current_state and .action
		- Pydantic models with .model_dump()
		"""
		if error:
			step_entry['output']['error'] = error
		elif output_response:
			# Try to serialize Pydantic model first (most complete)
			if hasattr(output_response, 'model_dump'):
				try:
					dumped = output_response.model_dump()
					# Filter out None values and very large fields
					step_entry['output'] = {
						k: v for k, v in dumped.items()
						if v is not None and k not in ('model_output_raw',)
					}
				except Exception:
					pass

			# Extract thinking/reasoning if present (Gemini, Claude extended thinking)
			if hasattr(output_response, 'thinking') and output_response.thinking:
				step_entry['output']['thinking'] = str(output_response.thinking)

			# Extract the main content
			if hasattr(output_response, 'content') and output_response.content:
				step_entry['output']['content'] = str(output_response.content)
			elif hasattr(output_response, 'text') and output_response.text:
				step_entry['output']['content'] = str(output_response.text)
			elif isinstance(output_response, str):
				step_entry['output']['content'] = output_response
			elif isinstance(output_response, dict) and not step_entry['output']:
				step_entry['output'] = output_response

			# Extract structured fields if present (browser-use agent format)
			if hasattr(output_response, 'current_state') and output_response.current_state:
				brain = output_response.current_state
				step_entry['output']['current_state'] = {
					'evaluation_previous_goal': getattr(brain, 'evaluation_previous_goal', None),
					'memory': getattr(brain, 'memory', None),
					'next_goal': getattr(brain, 'next_goal', None),
				}

			# Extract actions if present
			if hasattr(output_response, 'action') and output_response.action:
				try:
					actions = []
					for action in output_response.action:
						action_dict = action.model_dump() if hasattr(action, 'model_dump') else {}
						action_info = {k: v for k, v in action_dict.items() if v is not None}
						actions.append(action_info)
					step_entry['output']['actions'] = actions
				except Exception:
					pass

	def log_task_prompt(self, task: str):
		"""Log the task prompt given to the agent."""
		if self._is_puzzle_started():
			log_entry = {
				'event': 'task_prompt',
				'timestamp': datetime.now().isoformat(),
				'puzzle_type': self.current_puzzle_type,
				'puzzle_id': self.current_puzzle_id,
				'task': task,
			}
			self._write_to_puzzle_log(log_entry)
		else:
			logging.debug('[log_task_prompt] No puzzle started yet, skipping puzzle log')

		# Also update run_config.json with the task prompt
		if self.run_config_file.exists():
			try:
				with open(self.run_config_file, 'r', encoding='utf-8') as f:
					config = json.load(f)
				config['task_prompt'] = task
				with open(self.run_config_file, 'w', encoding='utf-8') as f:
					json.dump(config, f, indent=2, ensure_ascii=False)
			except Exception as e:
				logging.warning(f'Failed to update run_config with task_prompt: {e}')

		logging.info('[Task Prompt] Logged to run_config')

	def log_agent_step(self, step: int, agent_output: Any,
					   actions_taken: list = None, action_results: list = None,
					   screenshot_path: str = None):
		"""Log agent step information including eval, memory, next_goal, and actions."""
		if not self._is_puzzle_started():
			logging.debug(f'[log_agent_step] No puzzle started yet, skipping step {step}')
			return

		log_entry = {
			'event': 'agent_step',
			'step': step,
			'timestamp': datetime.now().isoformat(),
			'puzzle_type': self.current_puzzle_type,
			'puzzle_id': self.current_puzzle_id,
		}

		# Extract agent brain state (using browser-use field names)
		if agent_output and hasattr(agent_output, 'current_state'):
			brain = agent_output.current_state
			log_entry['current_state'] = {
				'evaluation_previous_goal': getattr(brain, 'evaluation_previous_goal', None),
				'memory': getattr(brain, 'memory', None),
				'next_goal': getattr(brain, 'next_goal', None),
			}

		# Extract actions
		if agent_output and hasattr(agent_output, 'action'):
			try:
				actions = []
				for action in agent_output.action:
					action_dict = action.model_dump() if hasattr(action, 'model_dump') else {}
					action_info = {k: v for k, v in action_dict.items() if v is not None}
					actions.append(action_info)
				log_entry['actions'] = actions
			except Exception as e:
				log_entry['actions_error'] = str(e)

		# Add action results if provided
		if action_results:
			try:
				results = []
				for result in action_results:
					result_info = {
						'is_done': getattr(result, 'is_done', None),
						'success': getattr(result, 'success', None),
						'extracted_content': getattr(result, 'extracted_content', None),
						'error': getattr(result, 'error', None),
					}
					results.append(result_info)
				log_entry['action_results'] = results
			except Exception as e:
				log_entry['results_error'] = str(e)

		# Add screenshot path if provided
		if screenshot_path:
			log_entry['screenshot'] = screenshot_path

		self._write_to_puzzle_log(log_entry)

	def log_error(self, step: int, error_type: str, error_message: str, details: dict = None):
		"""Log errors and timeouts."""
		if not self._is_puzzle_started():
			logging.warning(f'[log_error] No puzzle started, cannot log error: {error_type}: {error_message}')
			return
		log_entry = {
			'event': 'error',
			'step': step,
			'timestamp': datetime.now().isoformat(),
			'puzzle_type': self.current_puzzle_type,
			'puzzle_id': self.current_puzzle_id,
			'error_type': error_type,
			'error_message': error_message,
			'details': details or {},
		}
		self._write_to_puzzle_log(log_entry)
		logging.info(f'[Error @ Step {step}] {error_type}: {error_message}')

	def log_history_item(self, item: Any, index: int = None):
		"""Log a history item (backward compatibility)."""
		if not self._is_puzzle_started():
			logging.debug('[log_history_item] No puzzle started yet, skipping')
			return
		log_entry = {
			'event': 'history_item',
			'index': index,
			'timestamp': datetime.now().isoformat(),
		}

		if hasattr(item, 'role'):
			log_entry['role'] = str(item.role)
		if hasattr(item, 'content'):
			log_entry['content'] = str(item.content)
		if hasattr(item, 'text'):
			log_entry['text'] = str(item.text)

		self._write_to_puzzle_log(log_entry)

	def close(self):
		"""Close the logger and write summary."""
		# End any open puzzle
		if self.current_puzzle_log:
			self.end_puzzle()

		# Calculate aggregate stats
		total_puzzles = len(self.puzzle_stats)
		correct_puzzles = sum(1 for p in self.puzzle_stats if p.get('is_correct') is True)
		incorrect_puzzles = sum(1 for p in self.puzzle_stats if p.get('is_correct') is False)
		unknown_puzzles = total_puzzles - correct_puzzles - incorrect_puzzles
		total_duration = sum(p.get('duration_seconds', 0) for p in self.puzzle_stats)
		total_tokens = sum(p.get('tokens', 0) for p in self.puzzle_stats)
		total_input_tokens = sum(p.get('input_tokens', 0) for p in self.puzzle_stats)
		total_output_tokens = sum(p.get('output_tokens', 0) for p in self.puzzle_stats)
		total_reasoning_tokens = sum(p.get('reasoning_tokens', 0) for p in self.puzzle_stats)

		# Write summary.json
		summary = {
			'run_id': self.run_id,
			'end_time': datetime.now().isoformat(),
			'total_puzzles': total_puzzles,
			'correct_puzzles': correct_puzzles,
			'incorrect_puzzles': incorrect_puzzles,
			'unknown_puzzles': unknown_puzzles,
			'accuracy': correct_puzzles / total_puzzles if total_puzzles > 0 else 0,
			'total_duration_seconds': total_duration,
			'total_llm_calls': self.step_counter,
			'total_tokens': total_tokens,
			'total_input_tokens': total_input_tokens,
			'total_output_tokens': total_output_tokens,
			'total_reasoning_tokens': total_reasoning_tokens,
			'puzzles': self.puzzle_stats,
		}

		with open(self.summary_file, 'w', encoding='utf-8') as f:
			json.dump(summary, f, indent=2, ensure_ascii=False)

		logging.info(f'Enhanced LLM logger closed. Run ID: {self.run_id}')
		logging.info(f'Total puzzles: {total_puzzles}, Correct: {correct_puzzles}, Accuracy: {summary["accuracy"]:.1%}')
		logging.info(f'Logs saved to: {self.run_dir}')


# Backward compatibility alias
LegacyLLMLogger = None  # Will be set below after LLMResponseLogger is defined


class LLMResponseLogger:
	"""Logger to capture and save detailed LLM responses."""

	def __init__(self, log_dir: str = 'llm_logs', experiment_name: str = None, provider: str = None):
		self.log_dir = Path(log_dir)
		self.log_dir.mkdir(exist_ok=True)
		self.provider = provider  # Provider for usage extraction ('openai', 'anthropic', 'google')

		timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
		exp_name = f'_{experiment_name}' if experiment_name else ''
		self.log_file = self.log_dir / f'llm_responses_{timestamp}{exp_name}.jsonl'
		self.step_counter = 0

		# Puzzle context - will be set via set_puzzle_context()
		self.puzzle_type = None
		self.puzzle_id = None

		logging.info(f'LLM responses will be logged to: {self.log_file}')

	def set_puzzle_context(self, puzzle_type: str = None, puzzle_id: int = None):
		"""Set the current puzzle context to be included in all log entries.

		Args:
			puzzle_type: Type of puzzle (e.g., 'Mirror', 'Dice')
			puzzle_id: Puzzle number/ID
		"""
		self.puzzle_type = puzzle_type
		self.puzzle_id = puzzle_id
		logging.info(f'Puzzle context set: type={puzzle_type}, id={puzzle_id}')

	def log_response(self, call_id: int, response: Any, metadata: dict = None):
		"""Log a single LLM response with full details."""
		self.step_counter += 1
		log_entry = {
			'call_id': call_id,                   # LLM call ID from LoggingLLMWrapper
			'call_sequence': self.step_counter,   # LLM call sequence number
			'timestamp': datetime.now().isoformat(),
			'puzzle_type': self.puzzle_type,
			'puzzle_id': self.puzzle_id,
			'metadata': metadata or {},
		}

		# Try to extract detailed information from the response
		# Handle different response types
		if hasattr(response, 'content'):
			log_entry['content'] = str(response.content)
		if hasattr(response, 'text'):
			log_entry['text'] = str(response.text)
		if hasattr(response, 'message'):
			log_entry['message'] = str(response.message)
		if isinstance(response, str):
			log_entry['text'] = response

		# Extract usage info using provider-specific extractors
		try:
			if self.provider:
				usage = extract_usage(response, provider=self.provider)
				if any(v is not None for v in usage.values()):
					log_entry['usage'] = usage
			# Skip usage extraction if no provider - extract_usage now requires explicit provider
		except Exception:
			pass

		# Extract model info
		if hasattr(response, 'model'):
			log_entry['model'] = str(response.model)

		# Extract tool calls
		if hasattr(response, 'tool_calls'):
			try:
				log_entry['tool_calls'] = [
					{
						'id': getattr(tc, 'id', None),
						'type': getattr(tc, 'type', None),
						'function': {
							'name': getattr(tc.function, 'name', None) if hasattr(tc, 'function') else None,
							'arguments': getattr(tc.function, 'arguments', None) if hasattr(tc, 'function') else None,
						}
					}
					for tc in response.tool_calls
				] if response.tool_calls else []
			except Exception:
				log_entry['tool_calls'] = str(response.tool_calls)

		# Capture raw response for debugging
		if not any(k in log_entry for k in ['content', 'text', 'message']):
			log_entry['raw_response'] = str(response)
			# Try to get dict representation
			if hasattr(response, '__dict__'):
				try:
					log_entry['response_dict'] = {
						k: str(v) for k, v in response.__dict__.items()
					}
				except Exception:
					pass

		# Write to file
		with open(self.log_file, 'a', encoding='utf-8') as f:
			f.write(json.dumps(log_entry, ensure_ascii=False, indent=None) + '\n')

		# Also log summary to console
		logging.info(f'[LLM Response #{self.step_counter} @ Step {step}] Logged to file')

		return log_entry

	def log_history_item(self, item: Any, index: int = None):
		"""Log a history item from the agent's conversation history."""
		log_entry = {
			'event': 'history_item',
			'index': index,
			'call_sequence': self.step_counter + 1,  # Next LLM call sequence number
			'timestamp': datetime.now().isoformat(),
		}

		# Extract role and content
		if hasattr(item, 'role'):
			log_entry['role'] = str(item.role)
		if hasattr(item, 'content'):
			log_entry['content'] = str(item.content)
		if hasattr(item, 'text'):
			log_entry['text'] = str(item.text)

		# Try to get full dict
		if hasattr(item, '__dict__'):
			try:
				log_entry['item_dict'] = {
					k: str(v) for k, v in item.__dict__.items()
				}
			except Exception:
				log_entry['raw_item'] = str(item)
		else:
			log_entry['raw_item'] = str(item)

		# Write to file
		with open(self.log_file, 'a', encoding='utf-8') as f:
			f.write(json.dumps(log_entry, ensure_ascii=False, indent=None) + '\n')

	def log_task_prompt(self, task: str):
		"""Log the task prompt given to the agent."""
		log_entry = {
			'event': 'task_prompt',
			'timestamp': datetime.now().isoformat(),
			'puzzle_type': self.puzzle_type,
			'puzzle_id': self.puzzle_id,
			'task': task,
		}
		with open(self.log_file, 'a', encoding='utf-8') as f:
			f.write(json.dumps(log_entry, ensure_ascii=False, indent=None) + '\n')
		logging.info('[Task Prompt] Logged to file')

	def log_agent_step(self, step: int, agent_output: Any, actions_taken: list = None, action_results: list = None):
		"""Log agent step information including eval, memory, next_goal, and actions."""
		log_entry = {
			'event': 'agent_step',
			'step': step,
			'timestamp': datetime.now().isoformat(),
			'puzzle_type': self.puzzle_type,
			'puzzle_id': self.puzzle_id,
		}

		# Extract agent brain state (using browser-use field names)
		if agent_output and hasattr(agent_output, 'current_state'):
			brain = agent_output.current_state
			log_entry['current_state'] = {
				'evaluation_previous_goal': getattr(brain, 'evaluation_previous_goal', None),
				'memory': getattr(brain, 'memory', None),
				'next_goal': getattr(brain, 'next_goal', None),
			}

		# Extract actions from agent output
		if agent_output and hasattr(agent_output, 'action'):
			try:
				actions = []
				for action in agent_output.action:
					# Get the action name and parameters
					action_dict = action.model_dump() if hasattr(action, 'model_dump') else {}
					# Filter out None values to get the actual action
					action_info = {k: v for k, v in action_dict.items() if v is not None}
					actions.append(action_info)
				log_entry['actions'] = actions
			except Exception as e:
				log_entry['actions_error'] = str(e)

		# Add action results if provided
		if action_results:
			try:
				results = []
				for result in action_results:
					result_info = {
						'is_done': getattr(result, 'is_done', None),
						'success': getattr(result, 'success', None),
						'extracted_content': getattr(result, 'extracted_content', None),
						'error': getattr(result, 'error', None),
					}
					results.append(result_info)
				log_entry['action_results'] = results
			except Exception as e:
				log_entry['results_error'] = str(e)

		with open(self.log_file, 'a', encoding='utf-8') as f:
			f.write(json.dumps(log_entry, ensure_ascii=False, indent=None) + '\n')

	def log_error(self, step: int, error_type: str, error_message: str, details: dict = None):
		"""Log errors and timeouts."""
		log_entry = {
			'event': 'error',
			'step': step,
			'timestamp': datetime.now().isoformat(),
			'puzzle_type': self.puzzle_type,
			'puzzle_id': self.puzzle_id,
			'error_type': error_type,
			'error_message': error_message,
			'details': details or {},
		}
		with open(self.log_file, 'a', encoding='utf-8') as f:
			f.write(json.dumps(log_entry, ensure_ascii=False, indent=None) + '\n')
		logging.info(f'[Error @ Step {step}] {error_type}: {error_message}')

	def close(self):
		"""Close the logger and write summary."""
		summary = {
			'event': 'summary',
			'total_responses': self.step_counter,
			'timestamp': datetime.now().isoformat(),
			'log_file': str(self.log_file),
		}

		with open(self.log_file, 'a', encoding='utf-8') as f:
			f.write(json.dumps(summary, ensure_ascii=False) + '\n')

		logging.info(f'LLM logger closed. Total responses logged: {self.step_counter}')


class LoggingLLMWrapper:
	"""Wrapper that logs LLM calls in real-time while forwarding to the actual LLM."""

	def __init__(self, llm: Any, logger: Union[EnhancedLLMLogger, 'LLMResponseLogger']):
		self.llm = llm
		self.logger = logger
		self.call_counter = 0

		# Copy over all attributes from the wrapped LLM
		for attr in dir(llm):
			if not attr.startswith('_') and not hasattr(self, attr):
				try:
					setattr(self, attr, getattr(llm, attr))
				except AttributeError:
					pass

	def __getattr__(self, name: str) -> Any:
		"""Forward attribute access to the wrapped LLM."""
		return getattr(self.llm, name)

	def with_structured_output(self, *args, **kwargs) -> 'LoggingStructuredLLMWrapper':
		"""Override with_structured_output to return a wrapped structured LLM.

		Browser-use calls llm.with_structured_output().ainvoke() which would
		bypass this wrapper. By overriding this method, we ensure the returned
		structured LLM is also wrapped and its ainvoke() calls are logged.
		"""
		structured_llm = self.llm.with_structured_output(*args, **kwargs)
		return LoggingStructuredLLMWrapper(structured_llm, self)

	def _extract_messages_data(self, messages: Any) -> list:
		"""Extract messages into a serializable format.

		Handles multiple content formats:
		- Dict content (LangChain native): {'type': 'text', 'text': '...'}
		- Pydantic/object content: ContentPartTextParam(text='...')
		- Simple string content (system messages)

		Text content is fully preserved. Images are replaced with placeholders.
		"""
		messages_data = []
		if isinstance(messages, list):
			for msg in messages:
				if hasattr(msg, 'content'):
					content = msg.content
					# Handle multimodal content (list of content parts)
					if isinstance(content, list):
						extracted_parts = []
						for part in content:
							# Case 1: Dict content parts (LangChain native format)
							if isinstance(part, dict):
								if part.get('type') == 'text':
									extracted_parts.append({'type': 'text', 'text': part.get('text', '')})
								elif part.get('type') == 'image_url':
									extracted_parts.append({'type': 'image', 'image': '<image_placeholder>'})
								else:
									extracted_parts.append(part)
							# Case 2: Pydantic/object with .text attribute (text content)
							elif hasattr(part, 'text') and isinstance(getattr(part, 'text', None), str):
								extracted_parts.append({'type': 'text', 'text': part.text})
							# Case 3: Pydantic/object with .image_url attribute (image content)
							elif hasattr(part, 'image_url'):
								extracted_parts.append({'type': 'image', 'image': '<image_placeholder>'})
							else:
								# Fallback: try to serialize with model_dump or str
								if hasattr(part, 'model_dump'):
									extracted_parts.append(part.model_dump())
								else:
									extracted_parts.append({'type': 'unknown', 'value': str(part)[:1000]})
						messages_data.append({
							'role': getattr(msg, 'role', 'unknown'),
							'content': extracted_parts
						})
					else:
						# Simple string content - preserve fully
						messages_data.append({
							'role': getattr(msg, 'role', 'unknown'),
							'content': str(content)
						})
				else:
					messages_data.append(str(msg))
		else:
			messages_data = [str(messages)]
		return messages_data

	async def ainvoke(self, messages: Any, output_format: Any = None, **kwargs: Any) -> Any:
		"""Intercept async LLM calls and log them in real-time."""
		import time
		self.call_counter += 1
		call_id = self.call_counter

		# Extract messages for logging
		messages_data = self._extract_messages_data(messages)

		# Capture start timestamp
		start_timestamp = datetime.now().isoformat()
		start_time = time.time()

		# Log request to detailed log
		try:
			self.logger.log_response(
				call_id=call_id,
				response={'type': 'request', 'messages': messages_data, 'kwargs': str(kwargs)},
				metadata={'call_id': call_id, 'direction': 'request', 'timestamp': start_timestamp}
			)
		except Exception as e:
			logging.warning(f'Failed to log LLM request: {e}')

		# Call the actual LLM with timing
		try:
			response = await self.llm.ainvoke(messages, output_format, **kwargs)
			end_time = time.time()
			end_timestamp = datetime.now().isoformat()
			duration_ms = (end_time - start_time) * 1000

			# Log response to detailed log
			try:
				self.logger.log_response(
					call_id=call_id,
					response=response,
					metadata={'call_id': call_id, 'direction': 'response', 'timestamp': end_timestamp, 'duration_ms': duration_ms}
				)
			except Exception as e:
				logging.warning(f'Failed to log LLM response: {e}')

			# Log combined LLM call for fine-tuning (if logger supports it)
			if hasattr(self.logger, 'log_step'):
				try:
					self.logger.log_step(
						call_id=call_id,
						input_messages=messages_data,
						output_response=response,
						start_time=start_timestamp,
						end_time=end_timestamp,
						duration_ms=duration_ms
					)
				except Exception as e:
					logging.warning(f'Failed to log LLM call: {e}')

			return response

		except Exception as e:
			end_time = time.time()
			end_timestamp = datetime.now().isoformat()
			duration_ms = (end_time - start_time) * 1000

			# Log error to detailed log
			try:
				self.logger.log_response(
					call_id=call_id,
					response={'type': 'error', 'error': str(e)},
					metadata={'call_id': call_id, 'direction': 'error', 'timestamp': end_timestamp, 'duration_ms': duration_ms}
				)
			except Exception as log_err:
				logging.warning(f'Failed to log LLM error: {log_err}')

			# Log error LLM call for fine-tuning
			if hasattr(self.logger, 'log_step'):
				try:
					self.logger.log_step(
						call_id=call_id,
						input_messages=messages_data,
						output_response=None,
						start_time=start_timestamp,
						end_time=end_timestamp,
						duration_ms=duration_ms,
						error=str(e)
					)
				except Exception:
					pass

			# Re-raise the original error
			raise

	def invoke(self, messages: Any, output_format: Any = None, **kwargs: Any) -> Any:
		"""Intercept sync LLM calls and log them in real-time."""
		import time
		self.call_counter += 1
		call_id = self.call_counter

		# Extract messages for logging
		messages_data = self._extract_messages_data(messages)

		# Capture start timestamp
		start_timestamp = datetime.now().isoformat()
		start_time = time.time()

		# Log request to detailed log
		try:
			self.logger.log_response(
				call_id=call_id,
				response={'type': 'request', 'messages': messages_data, 'kwargs': str(kwargs)},
				metadata={'call_id': call_id, 'direction': 'request', 'timestamp': start_timestamp}
			)
		except Exception as e:
			logging.warning(f'Failed to log LLM request: {e}')

		# Call the actual LLM with timing
		try:
			response = self.llm.invoke(messages, output_format, **kwargs)
			end_time = time.time()
			end_timestamp = datetime.now().isoformat()
			duration_ms = (end_time - start_time) * 1000

			# Log response to detailed log
			try:
				self.logger.log_response(
					call_id=call_id,
					response=response,
					metadata={'call_id': call_id, 'direction': 'response', 'timestamp': end_timestamp, 'duration_ms': duration_ms}
				)
			except Exception as e:
				logging.warning(f'Failed to log LLM response: {e}')

			# Log combined LLM call for fine-tuning (if logger supports it)
			if hasattr(self.logger, 'log_step'):
				try:
					self.logger.log_step(
						call_id=call_id,
						input_messages=messages_data,
						output_response=response,
						start_time=start_timestamp,
						end_time=end_timestamp,
						duration_ms=duration_ms
					)
				except Exception as e:
					logging.warning(f'Failed to log LLM call: {e}')

			return response

		except Exception as e:
			end_time = time.time()
			end_timestamp = datetime.now().isoformat()
			duration_ms = (end_time - start_time) * 1000

			# Log error to detailed log
			try:
				self.logger.log_response(
					call_id=call_id,
					response={'type': 'error', 'error': str(e)},
					metadata={'call_id': call_id, 'direction': 'error', 'timestamp': end_timestamp, 'duration_ms': duration_ms}
				)
			except Exception as log_err:
				logging.warning(f'Failed to log LLM error: {log_err}')

			# Log error LLM call for fine-tuning
			if hasattr(self.logger, 'log_step'):
				try:
					self.logger.log_step(
						call_id=call_id,
						input_messages=messages_data,
						output_response=None,
						start_time=start_timestamp,
						end_time=end_timestamp,
						duration_ms=duration_ms,
						error=str(e)
					)
				except Exception:
					pass

			# Re-raise the original error
			raise


class LoggingStructuredLLMWrapper:
	"""Wrapper for structured LLM output that logs ainvoke() calls.

	This wrapper is returned by LoggingLLMWrapper.with_structured_output() to ensure
	that browser-use's calls to structured_llm.ainvoke() are properly logged.

	Without this, ~93% of API calls would bypass logging because browser-use calls:
		structured_llm = llm.with_structured_output(...)
		response = await structured_llm.ainvoke(messages)
	"""

	def __init__(self, structured_llm: Any, parent_wrapper: 'LoggingLLMWrapper'):
		self.structured_llm = structured_llm
		self.parent = parent_wrapper

		# Copy over all attributes from the wrapped structured LLM
		for attr in dir(structured_llm):
			if not attr.startswith('_') and not hasattr(self, attr):
				try:
					setattr(self, attr, getattr(structured_llm, attr))
				except AttributeError:
					pass

	def __getattr__(self, name: str) -> Any:
		"""Forward attribute access to the wrapped structured LLM."""
		return getattr(self.structured_llm, name)

	async def ainvoke(self, messages: Any, **kwargs: Any) -> Any:
		"""Intercept async structured LLM calls and log them."""
		import time

		# Use parent's call counter for consistent call_id tracking
		self.parent.call_counter += 1
		call_id = self.parent.call_counter

		# Extract messages for logging
		messages_data = self.parent._extract_messages_data(messages)

		# Capture start timestamp
		start_timestamp = datetime.now().isoformat()
		start_time = time.time()

		# Log request to detailed log
		try:
			self.parent.logger.log_response(
				call_id=call_id,
				response={'type': 'request', 'messages': messages_data, 'kwargs': str(kwargs), 'structured': True},
				metadata={'call_id': call_id, 'direction': 'request', 'timestamp': start_timestamp}
			)
		except Exception as e:
			logging.warning(f'Failed to log structured LLM request: {e}')

		# Call the actual structured LLM with timing
		try:
			response = await self.structured_llm.ainvoke(messages, **kwargs)
			end_time = time.time()
			end_timestamp = datetime.now().isoformat()
			duration_ms = (end_time - start_time) * 1000

			# Log response to detailed log
			try:
				self.parent.logger.log_response(
					call_id=call_id,
					response=response,
					metadata={'call_id': call_id, 'direction': 'response', 'timestamp': end_timestamp, 'duration_ms': duration_ms, 'structured': True}
				)
			except Exception as e:
				logging.warning(f'Failed to log structured LLM response: {e}')

			# Log combined LLM call for fine-tuning (if logger supports it)
			if hasattr(self.parent.logger, 'log_step'):
				try:
					# For structured output, the response is a dict with 'parsed' and 'raw'
					# We need to extract the raw message for proper token counting
					raw_response = response.get('raw') if isinstance(response, dict) else response
					self.parent.logger.log_step(
						call_id=call_id,
						input_messages=messages_data,
						output_response=raw_response,
						start_time=start_timestamp,
						end_time=end_timestamp,
						duration_ms=duration_ms
					)
				except Exception as e:
					logging.warning(f'Failed to log structured LLM call: {e}')

			return response

		except Exception as e:
			end_time = time.time()
			end_timestamp = datetime.now().isoformat()
			duration_ms = (end_time - start_time) * 1000

			# Log error to detailed log
			try:
				self.parent.logger.log_response(
					call_id=call_id,
					response={'type': 'error', 'error': str(e)},
					metadata={'call_id': call_id, 'direction': 'error', 'timestamp': end_timestamp, 'duration_ms': duration_ms, 'structured': True}
				)
			except Exception as log_err:
				logging.warning(f'Failed to log structured LLM error: {log_err}')

			# Log error LLM call for fine-tuning
			if hasattr(self.parent.logger, 'log_step'):
				try:
					self.parent.logger.log_step(
						call_id=call_id,
						input_messages=messages_data,
						output_response=None,
						start_time=start_timestamp,
						end_time=end_timestamp,
						duration_ms=duration_ms,
						error=str(e)
					)
				except Exception:
					pass

			# Re-raise the original error
			raise

	def invoke(self, messages: Any, **kwargs: Any) -> Any:
		"""Intercept sync structured LLM calls and log them."""
		import time

		# Use parent's call counter for consistent call_id tracking
		self.parent.call_counter += 1
		call_id = self.parent.call_counter

		# Extract messages for logging
		messages_data = self.parent._extract_messages_data(messages)

		# Capture start timestamp
		start_timestamp = datetime.now().isoformat()
		start_time = time.time()

		# Log request to detailed log
		try:
			self.parent.logger.log_response(
				call_id=call_id,
				response={'type': 'request', 'messages': messages_data, 'kwargs': str(kwargs), 'structured': True},
				metadata={'call_id': call_id, 'direction': 'request', 'timestamp': start_timestamp}
			)
		except Exception as e:
			logging.warning(f'Failed to log structured LLM request: {e}')

		# Call the actual structured LLM with timing
		try:
			response = self.structured_llm.invoke(messages, **kwargs)
			end_time = time.time()
			end_timestamp = datetime.now().isoformat()
			duration_ms = (end_time - start_time) * 1000

			# Log response to detailed log
			try:
				self.parent.logger.log_response(
					call_id=call_id,
					response=response,
					metadata={'call_id': call_id, 'direction': 'response', 'timestamp': end_timestamp, 'duration_ms': duration_ms, 'structured': True}
				)
			except Exception as e:
				logging.warning(f'Failed to log structured LLM response: {e}')

			# Log combined LLM call for fine-tuning (if logger supports it)
			if hasattr(self.parent.logger, 'log_step'):
				try:
					raw_response = response.get('raw') if isinstance(response, dict) else response
					self.parent.logger.log_step(
						call_id=call_id,
						input_messages=messages_data,
						output_response=raw_response,
						start_time=start_timestamp,
						end_time=end_timestamp,
						duration_ms=duration_ms
					)
				except Exception as e:
					logging.warning(f'Failed to log structured LLM call: {e}')

			return response

		except Exception as e:
			end_time = time.time()
			end_timestamp = datetime.now().isoformat()
			duration_ms = (end_time - start_time) * 1000

			# Log error to detailed log
			try:
				self.parent.logger.log_response(
					call_id=call_id,
					response={'type': 'error', 'error': str(e)},
					metadata={'call_id': call_id, 'direction': 'error', 'timestamp': end_timestamp, 'duration_ms': duration_ms, 'structured': True}
				)
			except Exception as log_err:
				logging.warning(f'Failed to log structured LLM error: {log_err}')

			# Log error LLM call for fine-tuning
			if hasattr(self.parent.logger, 'log_step'):
				try:
					self.parent.logger.log_step(
						call_id=call_id,
						input_messages=messages_data,
						output_response=None,
						start_time=start_timestamp,
						end_time=end_timestamp,
						duration_ms=duration_ms,
						error=str(e)
					)
				except Exception:
					pass

			# Re-raise the original error
			raise


# Backward compatibility: LegacyLLMLogger is the same as LLMResponseLogger
LegacyLLMLogger = LLMResponseLogger
