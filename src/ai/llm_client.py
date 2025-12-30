"""
Unified LLM Client - All LLM calls route through Komilion/OpenRouter.

This is the ONLY place in Raiden that makes LLM API calls.
Benefits:
- Single API key (KOMILION_API_KEY)
- Unified billing through OpenRouter
- Easy model switching via modes (frugal/balanced/premium)
- One place to update if API changes
- Automatic retry with exponential backoff
- Mode escalation on rate limits (balanced → premium)
- Connection pooling for efficiency
- Cost tracking across all calls

NOTE: Google Street View API (GOOGLE_API_KEY) is separate - that's for
fetching images, not LLM calls.

Usage:
    from src.ai.llm_client import get_llm_client, LLMClient

    # Get singleton client
    client = get_llm_client()

    # Text chat
    response = client.chat("What is the capital of Sweden?")

    # Vision analysis
    response = client.analyze_image(
        image_path="facade.jpg",
        prompt="Describe this building facade",
    )

    # Multi-image analysis
    response = client.analyze_images(
        image_paths=["north.jpg", "south.jpg"],
        prompt="Compare these two facades",
    )

    # Check total cost
    print(f"Total spent: ${client.total_cost:.4f}")

    # With retry and escalation disabled
    response = client.chat("...", max_retries=1, escalate_on_failure=False)
"""

import base64
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure module logger
logger = logging.getLogger(__name__)

# Mode escalation order (free → paid)
MODE_ESCALATION = ["frugal", "balanced", "premium"]

# Retry configuration
DEFAULT_MAX_RETRIES = 3
RETRY_BACKOFF_FACTOR = 1.5  # 1.5s, 2.25s, 3.375s...
RETRY_STATUS_CODES = [429, 500, 502, 503, 504]


@dataclass
class LLMResponse:
    """Response from LLM call."""
    content: str
    model: str
    cost: float
    raw: Dict[str, Any]
    retries_used: int = 0
    mode_escalated: bool = False

    def as_json(self) -> Optional[Dict]:
        """Parse content as JSON if possible."""
        try:
            text = self.content.strip()

            # Handle markdown code blocks
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

            # Extract JSON object
            first_brace = text.find('{')
            last_brace = text.rfind('}')
            if first_brace != -1 and last_brace != -1:
                text = text[first_brace:last_brace + 1]

            return json.loads(text)
        except (json.JSONDecodeError, Exception):
            return None


@dataclass
class CostTracker:
    """Thread-safe cost accumulator."""
    _total: float = 0.0
    _call_count: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, cost: float) -> None:
        """Add cost from a call."""
        with self._lock:
            self._total += cost
            self._call_count += 1

    @property
    def total(self) -> float:
        """Total cost across all calls."""
        with self._lock:
            return self._total

    @property
    def call_count(self) -> int:
        """Number of API calls made."""
        with self._lock:
            return self._call_count

    def reset(self) -> None:
        """Reset tracking (for testing)."""
        with self._lock:
            self._total = 0.0
            self._call_count = 0


class LLMClient:
    """
    Production-grade LLM client - ALL calls route through Komilion/OpenRouter.

    Features:
    - Connection pooling via requests.Session
    - Automatic retries with exponential backoff
    - Rate limit handling (429) with mode escalation
    - Thread-safe cost tracking
    - Proper logging

    Modes:
    - frugal: Cheapest models (text-only, NO vision)
    - balanced: Vision-capable free models (default)
    - premium: Best quality (expensive)

    For vision tasks, balanced is recommended (free + capable).
    """

    KOMILION_URL = "https://www.komilion.com/api/chat"
    TIMEOUT_SECONDS = 90
    CONNECT_TIMEOUT = 10

    def __init__(
        self,
        mode: Literal["frugal", "balanced", "premium"] = "balanced",
        api_key: Optional[str] = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        escalate_on_rate_limit: bool = True,
    ):
        """
        Initialize the LLM client.

        Args:
            mode: Komilion routing mode (frugal/balanced/premium)
            api_key: Komilion API key (defaults to KOMILION_API_KEY env var)
            max_retries: Maximum retry attempts (default: 3)
            escalate_on_rate_limit: Upgrade mode on 429 errors (default: True)
        """
        self.mode = mode
        self.api_key = api_key or os.environ.get("KOMILION_API_KEY")
        self.max_retries = max_retries
        self.escalate_on_rate_limit = escalate_on_rate_limit
        self._initialized = False
        self._cost_tracker = CostTracker()

        # Connection pooling with retry adapter
        self._session = requests.Session()
        retry_strategy = Retry(
            total=0,  # We handle retries manually for mode escalation
            backoff_factor=RETRY_BACKOFF_FACTOR,
            status_forcelist=[],  # We handle status codes manually
        )
        adapter = HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=retry_strategy,
        )
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)

        if not self.api_key:
            logger.warning("KOMILION_API_KEY not set - LLM calls will fail")

    def _ensure_initialized(self) -> bool:
        """Check if client is ready to use."""
        if not self.api_key:
            logger.error("KOMILION_API_KEY required for LLM calls")
            return False
        self._initialized = True
        return True

    def _encode_image(self, image_path: Union[str, Path]) -> tuple[str, str]:
        """Encode image to base64 with mime type."""
        path = Path(image_path)
        suffix = path.suffix.lower()

        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        mime_type = mime_types.get(suffix, "image/jpeg")

        with open(path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        return image_data, mime_type

    def _get_next_mode(self, current_mode: str) -> Optional[str]:
        """Get next escalation mode, or None if at premium."""
        try:
            idx = MODE_ESCALATION.index(current_mode)
            if idx < len(MODE_ESCALATION) - 1:
                return MODE_ESCALATION[idx + 1]
        except ValueError:
            pass
        return None

    def _make_request(
        self,
        messages: List[Dict],
        mode: Optional[str] = None,
        max_retries: Optional[int] = None,
        escalate_on_failure: bool = True,
    ) -> Optional[LLMResponse]:
        """
        Make request to Komilion API with retry and escalation logic.

        Args:
            messages: Chat messages to send
            mode: Override default mode
            max_retries: Override default retry count
            escalate_on_failure: Escalate mode on rate limit (default: True)

        Returns:
            LLMResponse or None on error
        """
        if not self._ensure_initialized():
            return None

        current_mode = mode or self.mode
        retries = max_retries if max_retries is not None else self.max_retries
        attempt = 0
        mode_escalated = False

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        while attempt <= retries:
            payload = {
                "messages": messages,
                "mode": current_mode,
            }

            try:
                logger.debug(f"LLM request attempt {attempt + 1}/{retries + 1}, mode={current_mode}")

                response = self._session.post(
                    self.KOMILION_URL,
                    headers=headers,
                    json=payload,
                    timeout=(self.CONNECT_TIMEOUT, self.TIMEOUT_SECONDS),
                )

                # Success
                if response.ok:
                    result = response.json()

                    # Extract response content
                    content = ""
                    if isinstance(result, dict):
                        if "content" in result:
                            content = result["content"]
                        elif "message" in result:
                            content = result["message"]
                        elif "choices" in result:
                            content = result["choices"][0]["message"]["content"]
                        elif "response" in result:
                            content = result["response"]
                        else:
                            content = str(result)

                        model = result.get("model", "unknown")
                        cost = result.get("cost", 0)

                        # Track cost
                        self._cost_tracker.add(cost)

                        logger.info(f"Komilion response: model={model}, cost=${cost:.4f}, total=${self.total_cost:.4f}")

                        return LLMResponse(
                            content=content,
                            model=model,
                            cost=cost,
                            raw=result,
                            retries_used=attempt,
                            mode_escalated=mode_escalated,
                        )
                    else:
                        return LLMResponse(
                            content=str(result),
                            model="unknown",
                            cost=0,
                            raw={"raw": result},
                            retries_used=attempt,
                            mode_escalated=mode_escalated,
                        )

                # Rate limited - try escalating mode
                elif response.status_code == 429:
                    logger.warning(f"Rate limited (429) on mode={current_mode}")

                    if escalate_on_failure and self.escalate_on_rate_limit:
                        next_mode = self._get_next_mode(current_mode)
                        if next_mode:
                            logger.info(f"Escalating mode: {current_mode} → {next_mode}")
                            current_mode = next_mode
                            mode_escalated = True
                            # Don't count as retry, just mode switch
                            continue

                    # Can't escalate, wait and retry
                    retry_after = int(response.headers.get("Retry-After", 5))
                    logger.info(f"Waiting {retry_after}s before retry")
                    time.sleep(retry_after)

                # Server error - retry with backoff
                elif response.status_code in RETRY_STATUS_CODES:
                    logger.warning(f"Server error {response.status_code}, retrying...")

                # Client error - don't retry
                else:
                    logger.error(f"Komilion API error: {response.status_code} - {response.text[:200]}")
                    return None

            except requests.Timeout:
                logger.warning(f"Request timeout (attempt {attempt + 1})")
            except requests.ConnectionError as e:
                logger.warning(f"Connection error: {e}")
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                return None

            # Exponential backoff
            attempt += 1
            if attempt <= retries:
                sleep_time = RETRY_BACKOFF_FACTOR ** attempt
                logger.debug(f"Retrying in {sleep_time:.1f}s...")
                time.sleep(sleep_time)

        logger.error(f"All {retries + 1} attempts failed")
        return None

    @property
    def total_cost(self) -> float:
        """Total cost across all API calls."""
        return self._cost_tracker.total

    @property
    def call_count(self) -> int:
        """Number of API calls made."""
        return self._cost_tracker.call_count

    def reset_cost_tracking(self) -> None:
        """Reset cost tracking (for testing or new analysis run)."""
        self._cost_tracker.reset()

    def chat(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        mode: Optional[str] = None,
        max_retries: Optional[int] = None,
        escalate_on_failure: bool = True,
    ) -> Optional[LLMResponse]:
        """
        Text chat (no images).

        Args:
            prompt: User message
            system_prompt: Optional system instructions
            mode: Override default mode
            max_retries: Override default retry count
            escalate_on_failure: Escalate mode on rate limit

        Returns:
            LLMResponse or None on error
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        return self._make_request(
            messages,
            mode=mode or self.mode,
            max_retries=max_retries,
            escalate_on_failure=escalate_on_failure,
        )

    def analyze_image(
        self,
        image_path: Union[str, Path],
        prompt: str,
        system_prompt: Optional[str] = None,
        mode: Optional[str] = None,
        max_retries: Optional[int] = None,
        escalate_on_failure: bool = True,
    ) -> Optional[LLMResponse]:
        """
        Analyze a single image.

        Args:
            image_path: Path to image file
            prompt: Analysis instructions
            system_prompt: Optional system instructions
            mode: Override mode (vision requires balanced or premium)
            max_retries: Override default retry count
            escalate_on_failure: Escalate mode on rate limit

        Returns:
            LLMResponse or None on error
        """
        # Vision requires balanced or premium
        effective_mode = mode or self.mode
        if effective_mode == "frugal":
            logger.debug("Vision requires balanced mode, upgrading from frugal")
            effective_mode = "balanced"

        image_data, mime_type = self._encode_image(image_path)

        content = []
        if system_prompt:
            content.append({"type": "text", "text": system_prompt})

        content.append({"type": "text", "text": prompt})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{image_data}"},
        })

        messages = [{"role": "user", "content": content}]

        return self._make_request(
            messages,
            mode=effective_mode,
            max_retries=max_retries,
            escalate_on_failure=escalate_on_failure,
        )

    def analyze_images(
        self,
        image_paths: List[Union[str, Path]],
        prompt: str,
        system_prompt: Optional[str] = None,
        mode: Optional[str] = None,
        max_images: int = 4,
        max_retries: Optional[int] = None,
        escalate_on_failure: bool = True,
    ) -> Optional[LLMResponse]:
        """
        Analyze multiple images in one call.

        Args:
            image_paths: List of image file paths
            prompt: Analysis instructions
            system_prompt: Optional system instructions
            mode: Override mode (vision requires balanced or premium)
            max_images: Maximum images to include
            max_retries: Override default retry count
            escalate_on_failure: Escalate mode on rate limit

        Returns:
            LLMResponse or None on error
        """
        # Vision requires balanced or premium
        effective_mode = mode or self.mode
        if effective_mode == "frugal":
            logger.debug("Vision requires balanced mode, upgrading from frugal")
            effective_mode = "balanced"

        content = []
        if system_prompt:
            content.append({"type": "text", "text": system_prompt})

        content.append({"type": "text", "text": prompt})

        # Add images (up to max)
        for i, path in enumerate(image_paths[:max_images]):
            image_data, mime_type = self._encode_image(path)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{image_data}"},
            })

        messages = [{"role": "user", "content": content}]

        return self._make_request(
            messages,
            mode=effective_mode,
            max_retries=max_retries,
            escalate_on_failure=escalate_on_failure,
        )

    def close(self) -> None:
        """Close the session (cleanup)."""
        self._session.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup session."""
        self.close()


# Singleton instance
_client: Optional[LLMClient] = None
_client_lock = threading.Lock()


def get_llm_client(
    mode: Literal["frugal", "balanced", "premium"] = "balanced",
) -> LLMClient:
    """
    Get the global LLM client singleton.

    Thread-safe singleton pattern.

    Args:
        mode: Komilion routing mode

    Returns:
        LLMClient instance
    """
    global _client
    with _client_lock:
        if _client is None:
            _client = LLMClient(mode=mode)
    return _client


def reset_llm_client() -> None:
    """Reset the singleton (for testing)."""
    global _client
    with _client_lock:
        if _client is not None:
            _client.close()
        _client = None
