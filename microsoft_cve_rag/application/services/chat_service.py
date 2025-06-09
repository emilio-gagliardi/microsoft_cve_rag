import asyncio
import base64
import logging
import mimetypes
import os
import time
from pathlib import Path
from typing import (  # Added AsyncGenerator
    Any,
    AsyncGenerator,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import aiohttp
import litellm
import tiktoken
from application.app_utils import get_openai_api_key
from litellm.utils import ModelResponse  # Added ModelResponse import

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

_llm_call_lock = asyncio.Lock()
_last_api_call_time = 0.0
MIN_LLM_CALL_INTERVAL_SECONDS = float(
    os.getenv("MIN_LLM_CALL_INTERVAL", "10.0")
)
# litellm._turn_on_debug()
litellm.drop_params = True


class LLMClient:
    """
    A simple client to interact with LLM providers via LiteLLM.
    Configuration (API keys, base URLs) is primarily handled by LiteLLM
    reading environment variables (e.g., OPENAI_API_KEY, OPENROUTER_API_KEY).
    """

    def __init__(self, mock_mode: bool = False):
        # Basic check if LiteLLM is available
        if not hasattr(litellm, "completion"):
            raise ImportError(
                "LiteLLM library not found or installed incorrectly."
            )
        logger.info(
            "LLMClient initialized. LiteLLM will use environment variables for"
            " API keys."
        )
        # You could add checks here for specific keys if needed, e.g.:
        # if not os.getenv("OPENAI_API_KEY") and not os.getenv("OPENROUTER_API_KEY"):
        #     logging.warning("Neither OPENAI_API_KEY nor OPENROUTER_API_KEY found in environment.")
        self.mock_mode = mock_mode
        self.supported_image_mime_types = {
            '.png': 'image/png',
            '.jpeg': 'image/jpeg',
            '.jpg': 'image/jpeg',
            '.webp': 'image/webp',
        }

    @staticmethod
    def _count_tokens(text: str, model_name: str) -> int:
        """
        Counts tokens in a text string for a given model.

        Attempts to map various model names (especially those in TOKEN_PRICING)
        to appropriate tiktoken model keys. Falls back to 'cl100k_base'
        if a specific encoding is not found.

        Args:
            text: The text string to count tokens for.
            model_name: The name of the model, which could be a key from
                        TOKEN_PRICING or another model identifier.

        Returns:
            The number of tokens in the text.
        """
        if not text:  # Handle empty or None strings
            return 0

        # Normalize model_name: extract base name (e.g., from OpenRouter) and lowercase
        normalized_model_name = model_name.split('/')[-1].lower()

        # Default tiktoken model key, used if no specific mapping is found
        tiktoken_model_key = "cl100k_base"

        # Mappings based on TOKEN_PRICING keys and common patterns.
        # Order is important: most specific matches should come first.
        if (
            "gpt-4o-mini" == normalized_model_name
            or "gpt-o4-mini-high" == normalized_model_name
        ):
            tiktoken_model_key = "gpt-4o-mini"
        elif "gpt-4o" == normalized_model_name:
            # As of tiktoken 0.7.0, "gpt-4o" is a valid model key
            tiktoken_model_key = "gpt-4o"
        elif (
            "gpt-4" in normalized_model_name
        ):  # Catches general "gpt-4" if not a more specific variant
            tiktoken_model_key = "gpt-4"
        elif "gpt-o3" == normalized_model_name:
            tiktoken_model_key = "gpt-3.5-turbo"
        elif "gpt-3.5" in normalized_model_name:  # Catches general "gpt-3.5"
            tiktoken_model_key = "gpt-3.5-turbo"
        elif "gemini" in normalized_model_name:
            # Tiktoken does not have specific encodings for Gemini models.
            # Using cl100k_base is a common practice.
            # You can uncomment the logger line below if you want to be notified.
            # logger.info(
            #     f"No specific tiktoken encoding for Gemini model '{model_name}'. "
            #     f"Using '{tiktoken_model_key}' (cl100k_base)."
            # )
            pass  # tiktoken_model_key is already "cl100k_base" as per default
        # For unmapped keys like "default_llm_pricing", it will correctly use "cl100k_base".

        try:
            encoding = tiktoken.encoding_for_model(tiktoken_model_key)
        except KeyError:
            # This warning is active to alert when a derived key isn't found by tiktoken.
            logger.warning(
                "Tiktoken encoding not found for derived key "
                f"'{tiktoken_model_key}' (from original model_name "
                f"'{model_name}'). Using cl100k_base as a fallback."
            )
            encoding = tiktoken.get_encoding("cl100k_base")

        return len(encoding.encode(text))

    def _encode_image_to_base64(
        self, image_path: Path
    ) -> Optional[Tuple[str, str]]:
        """
        Reads an image file, encodes it to base64, and determines its MIME type.

        Args:
            image_path: Path object pointing to the image file.

        Returns:
            A tuple containing (mime_type, base64_encoded_string) or None if encoding fails
            or the image type is unsupported.
        """
        if not image_path.is_file():
            logger.warning(f"Image file not found: {image_path}")
            return None

        suffix = image_path.suffix.lower()
        mime_type = self.supported_image_mime_types.get(suffix)

        if not mime_type:
            logger.warning(
                f"Unsupported image file type: {suffix} for file {image_path}."
                " Skipping."
            )
            return None

        try:
            with open(image_path, "rb") as image_file:
                binary_data = image_file.read()
            base64_encoded_data = base64.b64encode(binary_data)
            base64_string = base64_encoded_data.decode('utf-8')
            logger.info(f"Successfully encoded image: {image_path}")
            return mime_type, base64_string
        except Exception as e:
            logger.error(
                f"Error encoding image file {image_path}: {e}", exc_info=True
            )
            return None

    def get_completion(
        self,
        model: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        image_paths: Optional[List[Path]] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 0.95,
        stream: bool = False,
        stop: Optional[Union[str, List[str]]] = None,
        presence_penalty: float = 0.4,
        frequency_penalty: float = 0.2,
        response_format: Optional[dict] = None,
        seed: Optional[int] = None,
        tools: Optional[List] = None,
        tool_choice: Optional[str] = None,
        logprobs: bool = False,
        top_logprobs: int = 0,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        deployment_id: Optional[str] = None,
        timeout: int = 120,
        metadata: Optional[dict] = None,
        **kwargs: Any,
    ) -> str:
        """
        Gets a completion from the specified model using LiteLLM's synchronous call.
        Supports sending images along with the text prompt for multimodal models.
        Accepts all major LiteLLM parameters as arguments.
        """
        if self.mock_mode:
            mock_image_info = (
                f" (with {len(image_paths)} images)" if image_paths else ""
            )
            # Extract task_type from kwargs, defaulting to 'default'
            task_type = kwargs.get('task_type', 'default')
            # logger.info(f"Using task_type '{task_type}' for mock completion.")
            return self._get_mock_completion(
                prompt + mock_image_info,
                system_prompt=system_prompt,
                model_name=model,
                task_type=task_type,
            )[0]

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Process image paths
        image_path_obj_list: List[Path] = []
        if image_paths:
            if isinstance(image_paths, str):
                image_path_obj_list.append(Path(image_paths))
            elif isinstance(image_paths, Path):
                image_path_obj_list.append(image_paths)
            elif isinstance(image_paths, list):
                for p in image_paths:
                    if isinstance(p, str):
                        image_path_obj_list.append(Path(p))
                    elif isinstance(p, Path):
                        image_path_obj_list.append(p)
                    else:
                        logger.warning(
                            f"Unsupported image path type in list: {type(p)}."
                            " Skipping."
                        )
            else:
                logger.warning(
                    f"Unsupported image_paths type: {type(image_paths)}."
                    " Skipping image processing."
                )

        # Construct user message content
        # Default image_detail, can be overridden by kwargs if needed
        image_detail = kwargs.pop('image_detail', 'auto')

        if image_path_obj_list:  # Multi-modal message
            user_content_parts: List[Dict[str, Any]] = []
            if prompt:  # Text part is always first if present
                user_content_parts.append({"type": "text", "text": prompt})

            for img_path_obj in image_path_obj_list:
                encoded_image_data = self._encode_image_to_base64(img_path_obj)
                if encoded_image_data:
                    mime_type, base64_string = encoded_image_data
                    image_url_content = {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{base64_string}",
                            "detail": image_detail,
                        },
                    }
                    user_content_parts.append(image_url_content)
                else:
                    logger.warning(
                        f"Could not encode image at {img_path_obj}. Skipping."
                    )

            if (
                not user_content_parts
            ):  # Should only happen if prompt was empty and all images failed
                logger.error(
                    "No content (prompt or image) to send to LLM for"
                    " multi-modal call."
                )
                # _get_mock_completion returns (text, usage_dict)
                return self._get_mock_completion(
                    prompt if prompt else "",
                    system_prompt=system_prompt,
                    model_name=model,
                    task_type="error",
                )[0]
            messages.append({"role": "user", "content": user_content_parts})
        elif prompt:  # Text-only message
            messages.append({"role": "user", "content": prompt})
        else:  # No prompt and no images
            logger.error("No prompt provided and no images to send.")
            return self._get_mock_completion(
                "",
                system_prompt=system_prompt,
                model_name=model,
                task_type="error",
            )[0]

        # Dynamically build the arguments dictionary for litellm.completion
        completion_args = {
            "model": model,
            "messages": messages,
            "timeout": timeout,
        }
        if temperature is not None:
            completion_args["temperature"] = temperature
        if max_tokens is not None:
            completion_args["max_tokens"] = max_tokens
        if top_p is not None:
            completion_args["top_p"] = top_p
        if stream is not None:
            completion_args["stream"] = stream
        if stop is not None:
            completion_args["stop"] = stop
        if presence_penalty is not None:
            completion_args["presence_penalty"] = presence_penalty
        if frequency_penalty is not None:
            completion_args["frequency_penalty"] = frequency_penalty
        if response_format is not None:
            completion_args["response_format"] = response_format
        if seed is not None:
            completion_args["seed"] = seed
        if tools is not None:
            completion_args["tools"] = tools
        if tool_choice is not None:
            completion_args["tool_choice"] = tool_choice

        # Handle logprobs and top_logprobs carefully based on method signature parameters
        if logprobs:  # logprobs parameter from method signature is True
            completion_args['logprobs'] = True
            if (
                top_logprobs is not None
            ):  # top_logprobs from method signature (can be 0)
                completion_args['top_logprobs'] = top_logprobs
        else:  # logprobs parameter from method signature is False
            completion_args['logprobs'] = False
            # Do NOT add top_logprobs to completion_args if logprobs is False

        if api_key is not None:
            completion_args["api_key"] = api_key
        if api_base is not None:
            completion_args["api_base"] = api_base
        if api_version is not None:
            completion_args["api_version"] = api_version
        if deployment_id is not None:
            completion_args["deployment_id"] = deployment_id
        if metadata is not None:
            completion_args["metadata"] = metadata

        # Remove logprobs and top_logprobs from kwargs before update, as they are handled explicitly from method params
        kwargs.pop('logprobs', None)
        kwargs.pop('top_logprobs', None)
        kwargs.pop(
            'task_type', None
        )  # Ensure task_type is not passed to LiteLLM

        # Add any other kwargs passed directly
        completion_args.update(kwargs)

        logger.debug(
            f"Calling litellm.completion with args: {completion_args}"
        )
        try:
            response_obj = litellm.completion(**completion_args)
            completion_text = ""
            if (
                response_obj
                and response_obj.choices
                and response_obj.choices[0].message
            ):
                completion_text = response_obj.choices[0].message.content or ""

            usage = response_obj.usage
            if usage:
                logger.info(
                    f"LiteLLM call to {model} successful. Usage: Prompt"
                    f" tokens: {usage.prompt_tokens}, Completion tokens:"
                    f" {usage.completion_tokens}, Total tokens:"
                    f" {usage.total_tokens}"
                )
            else:
                logger.info(
                    f"Received completion from {model}. Length:"
                    f" {len(completion_text)}. Usage data not available."
                )
            return completion_text
        except Exception as e:
            logger.error(
                f"Error getting completion from {model} via LiteLLM: {e}",
                exc_info=True,
            )
            raise

    async def aget_completion(
        self,
        model: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        image_paths: Optional[List[Path]] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 0.95,
        stream: bool = False,
        stop: Optional[Union[str, List[str]]] = None,
        presence_penalty: float = 0.4,
        frequency_penalty: float = 0.2,
        response_format: Optional[dict] = None,
        seed: Optional[int] = None,
        tools: Optional[List] = None,
        tool_choice: Optional[str] = None,
        logprobs: bool = False,
        top_logprobs: int = 0,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        deployment_id: Optional[str] = None,
        timeout: int = 120,
        metadata: Optional[dict] = None,
        **kwargs: Any,
    ) -> Tuple[str, Optional[Dict[str, int]]]:
        """
        Async version of get_completion for use in async FastAPI routes.
        Accepts all major LiteLLM parameters as arguments.
        """
        global _last_api_call_time  # Declare intent to modify module-level variable

        if self.mock_mode:
            mock_image_info = (
                f" (with {len(image_paths)} images)" if image_paths else ""
            )
            task_type = kwargs.get('task_type', 'default')
            logger.info(
                "\n\nLLM Client (MOCK) received prompts for task"
                f" '{task_type}':\nsystem prompt:\n{system_prompt}\nuser"
                f" prompt:\n{prompt}{mock_image_info}\n\n"
            )
            # _get_mock_completion now returns (text, usage_dict)
            return self._get_mock_completion(
                prompt + mock_image_info,
                system_prompt=system_prompt,
                model_name=model,
                task_type=task_type,
            )
        logger.info(
            f"Type of 'prompt' variable: {type(prompt)}, Value: {prompt}"
        )
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Process image paths
        image_path_obj_list: List[Path] = []
        if image_paths:
            if isinstance(image_paths, str):
                image_path_obj_list.append(Path(image_paths))
            elif isinstance(image_paths, Path):
                image_path_obj_list.append(image_paths)
            elif isinstance(image_paths, list):
                for p in image_paths:
                    if isinstance(p, str):
                        image_path_obj_list.append(Path(p))
                    elif isinstance(p, Path):
                        image_path_obj_list.append(p)
                    else:
                        logger.warning(
                            f"Unsupported image path type in list: {type(p)}."
                            " Skipping."
                        )
            else:
                logger.warning(
                    f"Unsupported image_paths type: {type(image_paths)}."
                    " Skipping image processing."
                )

        # Construct user message content
        # Default image_detail, can be overridden by kwargs if needed
        image_detail = kwargs.pop('image_detail', 'auto')

        if image_path_obj_list:  # Multi-modal message
            user_content_parts: List[Dict[str, Any]] = []
            if prompt:  # Text part is always first if present
                user_content_parts.append({"type": "text", "text": prompt})

            for img_path_obj in image_path_obj_list:
                encoded_image_data = self._encode_image_to_base64(img_path_obj)
                if encoded_image_data:
                    mime_type, base64_string = encoded_image_data
                    image_url_content = {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{base64_string}",
                            "detail": image_detail,
                        },
                    }
                    user_content_parts.append(image_url_content)
                else:
                    logger.warning(
                        f"Could not encode image at {img_path_obj}. Skipping."
                    )

            if (
                not user_content_parts
            ):  # Should only happen if prompt was empty and all images failed
                logger.error(
                    "No content (prompt or image) to send to LLM for"
                    " multi-modal call."
                )
                return self._get_mock_completion(
                    prompt if prompt else "",
                    system_prompt=system_prompt,
                    model_name=model,
                    task_type="error",
                )
            messages.append({"role": "user", "content": user_content_parts})
        elif prompt:  # Text-only message
            messages.append({"role": "user", "content": prompt})
        else:  # No prompt and no images
            logger.error("No prompt provided and no images to send.")
            return self._get_mock_completion(
                "",
                system_prompt=system_prompt,
                model_name=model,
                task_type="error",
            )

        # Dynamically build the arguments dictionary for litellm.acompletion
        kwargs_for_acompletion = {}

        if temperature is not None:
            kwargs_for_acompletion["temperature"] = temperature
        if max_tokens is not None:
            kwargs_for_acompletion["max_tokens"] = max_tokens
        if top_p is not None:
            kwargs_for_acompletion["top_p"] = top_p
        if stream is not None:
            kwargs_for_acompletion["stream"] = stream  # stream can be False
        if stop is not None:
            kwargs_for_acompletion["stop"] = stop
        if presence_penalty is not None:
            kwargs_for_acompletion["presence_penalty"] = presence_penalty
        if frequency_penalty is not None:
            kwargs_for_acompletion["frequency_penalty"] = frequency_penalty
        if response_format is not None:
            kwargs_for_acompletion["response_format"] = response_format
        if seed is not None:
            kwargs_for_acompletion["seed"] = seed
        if tools is not None:
            kwargs_for_acompletion["tools"] = tools
        if tool_choice is not None:
            kwargs_for_acompletion["tool_choice"] = tool_choice

        # Handle logprobs and top_logprobs carefully based on method signature parameters
        if logprobs:  # logprobs parameter from method signature is True
            kwargs_for_acompletion['logprobs'] = True
            if (
                top_logprobs is not None
            ):  # top_logprobs from method signature (can be 0)
                kwargs_for_acompletion['top_logprobs'] = top_logprobs
        else:  # logprobs parameter from method signature is False
            kwargs_for_acompletion['logprobs'] = False
            # Do NOT add top_logprobs to kwargs_for_acompletion if logprobs is False

        if api_key is not None:
            kwargs_for_acompletion["api_key"] = api_key
        if api_base is not None:
            kwargs_for_acompletion["api_base"] = api_base
        if api_version is not None:
            kwargs_for_acompletion["api_version"] = api_version
        if deployment_id is not None:
            kwargs_for_acompletion["deployment_id"] = deployment_id
        if timeout is not None:
            kwargs_for_acompletion["timeout"] = timeout
        if metadata is not None:
            kwargs_for_acompletion["metadata"] = metadata

        # Remove logprobs and top_logprobs from kwargs before update, as they are handled explicitly from method params
        kwargs.pop('logprobs', None)
        kwargs.pop('top_logprobs', None)
        kwargs.pop(
            'task_type', None
        )  # Ensure task_type is not passed to LiteLLM

        # Add any other kwargs passed directly
        kwargs_for_acompletion.update(kwargs)

        async with _llm_call_lock:
            current_time = time.monotonic()
            time_since_last_call = current_time - _last_api_call_time

            if time_since_last_call < MIN_LLM_CALL_INTERVAL_SECONDS:
                delay_needed = (
                    MIN_LLM_CALL_INTERVAL_SECONDS - time_since_last_call
                )
                logger.info(
                    "Throttling LLM call. Waiting for"
                    f" {delay_needed:.2f} seconds."
                )
                await asyncio.sleep(delay_needed)

            logger.debug(
                f"Calling litellm.acompletion with model '{model}' and"
                f" messages: {messages}. Full args: {kwargs_for_acompletion}"
            )
            try:
                response_obj: Union[
                    ModelResponse, AsyncGenerator[ModelResponse, None]
                ] = await litellm.acompletion(
                    model=model, messages=messages, **kwargs_for_acompletion
                )
                completion_text = ""
                usage_dict: Optional[Dict[str, int]] = None

                if isinstance(response_obj, ModelResponse):  # Non-streaming
                    if (
                        response_obj.choices
                        and response_obj.choices[0].message
                    ):
                        completion_text = (
                            response_obj.choices[0].message.content or ""
                        )
                    if response_obj.usage:
                        usage_dict = {
                            "prompt_tokens": response_obj.usage.prompt_tokens,
                            "completion_tokens": (
                                response_obj.usage.completion_tokens
                            ),
                            "total_tokens": response_obj.usage.total_tokens,
                        }
                        logger.info(
                            f"LiteLLM async call to {model} successful. Usage:"
                            f" Prompt tokens: {usage_dict['prompt_tokens']},"
                            " Completion tokens:"
                            f" {usage_dict['completion_tokens']}, Total"
                            f" tokens: {usage_dict['total_tokens']}"
                        )
                    else:
                        logger.info(
                            f"Received async completion from {model}. Length:"
                            f" {len(completion_text)}. Usage data not"
                            " available."
                        )
                elif isinstance(response_obj, AsyncGenerator):  # Streaming
                    logger.info(
                        f"Receiving streamed async completion from {model}..."
                    )
                    full_response_content = []
                    final_response_obj = None  # To store the chunk that might contain usage data
                    async for chunk in response_obj:
                        if (
                            chunk.choices
                            and chunk.choices[0].delta
                            and chunk.choices[0].delta.content
                        ):
                            full_response_content.append(
                                chunk.choices[0].delta.content
                            )
                        # Capture the last chunk or a chunk that explicitly contains usage data
                        if (
                            hasattr(chunk, 'usage') and chunk.usage
                        ):  # Check if usage is present and not None
                            final_response_obj = chunk
                    completion_text = "".join(full_response_content)
                    if final_response_obj and final_response_obj.usage:
                        usage_dict = {
                            "prompt_tokens": (
                                final_response_obj.usage.prompt_tokens
                            ),
                            "completion_tokens": (
                                final_response_obj.usage.completion_tokens
                            ),
                            "total_tokens": (
                                final_response_obj.usage.total_tokens
                            ),
                        }
                    logger.info(
                        "Finished receiving streamed async completion from"
                        f" {model}. Total length: {len(completion_text)}."
                    )
                else:
                    logger.warning(
                        "Unexpected response_obj type from"
                        f" litellm.acompletion: {type(response_obj)}"
                    )
                    completion_text = "Error: Unexpected response type"

                _last_api_call_time = time.monotonic()
                return completion_text, usage_dict
            except Exception as e:
                logger.error(
                    f"Error (async) getting completion from {model} via"
                    f" LiteLLM: {e}",
                    exc_info=True,
                )
                _last_api_call_time = time.monotonic()
                return f"LiteLLM API Error: {e}", None

    def _get_mock_completion(
        self,
        prompt: str,
        system_prompt: Optional[str],
        model_name: str,
        task_type: str = "default",
    ) -> Tuple[str, Dict[str, int]]:
        """Return a mock LLM response based on task_type."""
        logger.debug(
            f"Generating mock completion for task_type: {task_type} with"
            f" model: {model_name}"
        )
        # Use task_type directly instead of searching prompt
        if task_type == "executive_summary":
            mock_text = (
                "This quarterly deep-dive report synthesizes findings across"
                " ten key sections, analyzing Microsoft CVE trends from [Start"
                " Date] to [End Date]. Overall vulnerability volume remained"
                " consistent with previous quarters, however, analysis of"
                " severity distribution revealed a concerning 4% increase in"
                " 'Critical' rated vulnerabilities, primarily driven by flaws"
                " in Windows OS and Azure services. Exploitability assessments"
                " indicated that approximately 15% of disclosed CVEs had"
                " publicly available exploit code or were observed being"
                " actively exploited, highlighting the urgency for rapid"
                " patching. Remediation metrics showed an average patch"
                " deployment time of 18 days for critical vulnerabilities, a"
                " slight improvement but still lagging behind industry best"
                " practices.\n\nKey section findings include: Section 3"
                " (Severity Trends) confirmed the dominance of Critical/High"
                " vulnerabilities, with detailed charts illustrating peaks"
                " associated with specific product releases. Section 5"
                " (Product Impact) identified Windows Server and Microsoft"
                " Exchange as the most impacted products based on CVE count"
                " and severity weighting. Section 7 (Exploit Analysis)"
                " provided statistical breakdowns of exploit types (RCE, EoP,"
                " DoS) and correlated them with CVSSv3 scores, noting a rise"
                " in privilege escalation exploits. Section 9 (Remediation"
                " Effectiveness) analyzed patch success rates and identified"
                " common obstacles reported by enterprise security teams,"
                " offering data-driven insights into patching"
                " challenges.\n\nBased on the comprehensive data analysis"
                " presented across all sections, strategic recommendations"
                " include: prioritizing patch deployment for critical RCE and"
                " EoP vulnerabilities within 7-14 days, enhancing monitoring"
                " for Azure service configurations, and reviewing internal"
                " security awareness training related to phishing vectors"
                " often associated with initial access exploits. Continuous"
                " vigilance and adaptation of security strategies based on"
                " these quarterly trends are crucial for mitigating risk"
                " within the Microsoft ecosystem. Detailed statistics, charts,"
                " and section-specific insights supporting these conclusions"
                " can be found within the full report body."
            )
        elif task_type == "report_conclusion":
            mock_text = (
                "In conclusion, this quarterly analysis highlights the"
                " persistent and evolving nature of security vulnerabilities"
                " within the Microsoft ecosystem. While overall CVE volume"
                " remained relatively stable, the data presented across ten"
                " sections reveals critical trends, particularly the continued"
                " prevalence of high-severity flaws and the increasing"
                " sophistication of exploit techniques observed in the wild."
                " Key takeaways include the concentration of risk in specific"
                " product areas like Windows Server and Azure, and the ongoing"
                " challenge of achieving timely patch deployment across"
                " diverse enterprise environments. The statistical evidence"
                " underscores the necessity for a proactive, data-driven"
                " approach to vulnerability management.\n\nMoving forward,"
                " organizations must leverage these insights to refine their"
                " security strategies, focusing on risk-based prioritization,"
                " enhanced detection capabilities, and streamlined remediation"
                " workflows. The findings reinforce that security is not a"
                " static goal but a continuous process requiring adaptation"
                " and investment. By addressing the specific vulnerability"
                " patterns and challenges identified in this report,"
                " businesses can significantly improve their resilience"
                " against emerging threats in the subsequent quarter and"
                " beyond."
            )
        elif task_type == "callout":
            mock_text = (
                "Key takeaway: Organizations should prioritize remediation of"
                " high-severity vulnerabilities identified this quarter, with"
                " special attention to RCE and privilege escalation issues."
            )
        elif task_type == "chart_insight":
            mock_text = (
                "This chart highlights a notable trend: the majority of"
                " vulnerabilities are clustered in the high-severity range,"
                " with a spike observed in March due to a major Windows"
                " update."
            )
        elif task_type == "narrative":
            mock_text = (
                "This section delves into the narrative derived from the"
                " quarterly vulnerability data, highlighting key trends and"
                " contextual factors. The analysis reveals a consistent"
                " pattern in the types of vulnerabilities disclosed,"
                " predominantly affecting core Windows components and related"
                " services. We observed a slight increase in remote code"
                " execution flaws compared to the previous quarter, demanding"
                " continued vigilance from security teams. Furthermore, the"
                " data reflects the impact of Microsoft's ongoing efforts to"
                " improve security posture through proactive fuzzing and"
                " variant analysis programs.\n\nExamining the timeline,"
                " vulnerability disclosures aligned closely with Microsoft's"
                " established Patch Tuesday cycle, indicating a predictable"
                " rhythm for remediation planning. However, several"
                " out-of-band patches were also issued, primarily addressing"
                " zero-day exploits actively used in the wild, underscoring"
                " the dynamic threat landscape. The distribution of severity"
                " scores remained heavily weighted towards 'Critical' and"
                " 'Important', reinforcing the need for rapid patch"
                " deployment. Geopolitical factors and major software releases"
                " appeared to correlate with specific spikes in vulnerability"
                " reporting during the period.\n\nOverall, the narrative"
                " underscores a complex interplay between vendor patching"
                " cadences, attacker focus, and the inherent complexity of the"
                " Microsoft ecosystem. While proactive measures show promise,"
                " the persistent volume of high-impact vulnerabilities"
                " necessitates robust, risk-based patch management strategies."
                " Organizations should integrate these findings into their"
                " threat modeling and prioritize resources towards mitigating"
                " the most significant risks identified. Continuous monitoring"
                " and adaptation remain crucial for navigating the evolving"
                " security challenges presented each quarter."
            )
        elif task_type == "section_summary":
            mock_text = (
                "In summary, this section analyzed the trends in vulnerability"
                " severity scores throughout the quarter. The data clearly"
                " indicates a sustained high volume of 'Critical' rated CVEs,"
                " particularly impacting server-side components. While"
                " mitigation efforts are ongoing, the concentration of risk in"
                " these critical areas warrants continued focus and potential"
                " architectural review. Key performance indicators suggest"
                " that patch latency for critical flaws improved slightly but"
                " remains a crucial area for operational enhancement."
            )
        else:  # Default case
            logger.warning(
                f"Unknown task_type '{task_type}' for mock completion,"
                " returning default response."
            )
            mock_text = (
                "This analysis reveals several important security trends. Key"
                " metrics indicate a shift in vulnerability patterns."
                " Statistical evidence supports ongoing security improvements."
                " Specific components require prioritized attention."
                " Recommendations focus on proactive security measures."
            )

        mock_input_tokens = 100 + (
            50 if system_prompt else 0
        )  # Rough estimate for prompt + system
        mock_completion_tokens = (
            len(mock_text.split()) // 2
        )  # Very rough estimate for completion

        mock_usage = {
            "prompt_tokens": mock_input_tokens,
            "completion_tokens": mock_completion_tokens,
            "total_tokens": mock_input_tokens + mock_completion_tokens,
        }

        return mock_text, mock_usage


# litellm.set_verbose = True # Already set at module level if you keep the previous change
# For more detailed logs as per LiteLLM's new recommendation:
# os.environ['LITELLM_LOG'] = 'DEBUG' # You might need to set this before importing litellm for it to take full effect


async def test_litellm_direct_call():
    try:
        # === TEST CASE: Multi-modal with actual image and specific prompt ===
        image_path = r"C:\Users\emili\PycharmProjects\microsoft_cve_rag\microsoft_cve_rag\application\data\reports\quarterly_deep_dive\quarterly_deep_dive_jan_2024_mar_2024\images\chart-volume-by-severity-monthly.png"
        prompt_text = "which category had the highest count in Febrary?"

        try:
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode(
                    'utf-8'
                )

            mime_type, _ = mimetypes.guess_type(image_path)
            if mime_type is None:
                mime_type = (  # Default if guess fails, e.g. for .webp or other types not in standard mimetypes
                    "image/png"
                )

            image_data_url = f"data:{mime_type};base64,{base64_image}"

            messages_payload = [
                {
                    'role': 'system',
                    'content': 'You are a data-focused security analyst.',
                },
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': prompt_text},
                        {
                            'type': 'image_url',
                            'image_url': {
                                'url': image_data_url,
                                'detail': 'auto',
                            },
                        },
                    ],
                },
            ]
            model_to_test = "gpt-4o-mini"
        except FileNotFoundError:
            print(f"Error: Image file not found at {image_path}")
            return
        except Exception as e:
            print(f"Error processing image: {e}")
            return

        print(
            "\n--- Attempting direct call to litellm.acompletion with model:"
            f" {model_to_test} ---"
        )
        print(f"Messages payload: {messages_payload}")

        # Ensure API key is loaded (consider moving os.environ.get outside the function if key is constant)
        api_key = get_openai_api_key()
        if not api_key:
            print("Error: OPENAI_API_KEY not found in environment variables.")
            return

        response = await litellm.acompletion(
            model=model_to_test,
            messages=messages_payload,
            api_key=api_key,
            temperature=0.7,
            max_tokens=150,
        )
        print("LiteLLM Direct Call Response:")
        # print(response) # Full response object can be verbose
        if (
            response.choices
            and response.choices[0].message
            and response.choices[0].message.content
        ):
            print("\nContent:")
            print(response.choices[0].message.content)
        else:
            print("\nNo content in response or unexpected response structure.")
            print(f"Full response object: {response}")

        if response.usage:
            print("\nUsage:")
            print(response.usage)

    except litellm.exceptions.BadRequestError as e:
        print(
            f"LiteLLM BadRequestError (Direct Call for {model_to_test}): {e}"
        )
        # Try to get more details from the error response if available
        error_response_text = "No detailed error response text available."
        if (
            hasattr(e, 'response')
            and e.response is not None
            and hasattr(e.response, 'text')
        ):
            error_response_text = e.response.text
        elif hasattr(
            e, 'message'
        ):  # Fallback to the direct message if response object is not as expected
            error_response_text = str(e.message)
        print(f"Error details: {error_response_text}")

    except Exception as e:
        print(
            "An unexpected error occurred (Direct Call for"
            f" {model_to_test}): {e}"
        )
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_litellm_direct_call())  # If running as a script
    # Or call from an existing async context:
    # await test_litellm_direct_call()
    pass
