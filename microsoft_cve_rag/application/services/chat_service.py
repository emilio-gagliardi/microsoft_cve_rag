import litellm
import base64
from pathlib import Path
from typing import Optional, Tuple, Any, List, Dict, Union
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LLMClient:
    """
    A simple client to interact with LLM providers via LiteLLM.
    Configuration (API keys, base URLs) is primarily handled by LiteLLM
    reading environment variables (e.g., OPENAI_API_KEY, OPENROUTER_API_KEY).
    """
    def __init__(self, mock_mode: bool = False):
        # Basic check if LiteLLM is available
        if not hasattr(litellm, "completion"):
            raise ImportError("LiteLLM library not found or installed incorrectly.")
        logger.info("LLMClient initialized. LiteLLM will use environment variables for API keys.")
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

    def _encode_image_to_base64(self, image_path: Path) -> Optional[Tuple[str, str]]:
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
            logger.warning(f"Unsupported image file type: {suffix} for file {image_path}. Skipping.")
            return None

        try:
            with open(image_path, "rb") as image_file:
                binary_data = image_file.read()
            base64_encoded_data = base64.b64encode(binary_data)
            base64_string = base64_encoded_data.decode('utf-8')
            logger.info(f"Successfully encoded image: {image_path}")
            return mime_type, base64_string
        except Exception as e:
            logger.error(f"Error encoding image file {image_path}: {e}", exc_info=True)
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
        **kwargs: Any
    ) -> str:
        """
        Gets a completion from the specified model using LiteLLM's synchronous call.
        Supports sending images along with the text prompt for multimodal models.
        Accepts all major LiteLLM parameters as arguments.
        """
        if self.mock_mode:
            mock_image_info = f" (with {len(image_paths)} images)" if image_paths else ""
            # Extract task_type from kwargs, defaulting to 'default'
            task_type = kwargs.get('task_type', 'default')
            # logger.info(f"Using task_type '{task_type}' for mock completion.")
            return self._get_mock_completion(prompt + mock_image_info, task_type=task_type)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        user_content: List[Dict[str, Any]] = []
        user_content.append({"type": "text", "content": prompt})
        if image_paths:
            logger.info(f"Processing {len(image_paths)} image(s) for the prompt.")
            for img_path in image_paths:
                encoded_image_data = self._encode_image_to_base64(img_path)
                if encoded_image_data:
                    mime_type, base64_image = encoded_image_data
                    image_message_part = {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{base64_image}",
                            "detail": "auto"
                        }
                    }
                    user_content.append(image_message_part)
        messages.append({"role": "user", "content": user_content})
        try:
            import litellm
            response = litellm.completion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stream=stream,
                stop=stop,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                response_format=response_format,
                seed=seed,
                tools=tools,
                tool_choice=tool_choice,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
                api_key=api_key,
                api_base=api_base,
                api_version=api_version,
                deployment_id=deployment_id,
                timeout=timeout,
                metadata=metadata,
                **kwargs
            )
            completion_text = response['choices'][0]['message']['content']
            logger.info(f"Received completion from {model}. Length: {len(completion_text)}")
            return completion_text
        except Exception as e:
            logger.error(f"Error getting completion from {model} via LiteLLM: {e}", exc_info=True)
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
        **kwargs: Any
    ) -> str:
        """
        Async version of get_completion for use in async FastAPI routes.
        Accepts all major LiteLLM parameters as arguments.
        """
        if self.mock_mode:
            mock_image_info = f" (with {len(image_paths)} images)" if image_paths else ""
            # Extract task_type from kwargs, defaulting to 'default'
            task_type = kwargs.get('task_type', 'default')
            return self._get_mock_completion(prompt + mock_image_info, task_type=task_type)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        user_content: List[Dict[str, Any]] = []
        user_content.append({"type": "text", "content": prompt})
        if image_paths:
            logger.info(f"Processing {len(image_paths)} image(s) for the prompt.")
            for img_path in image_paths:
                encoded_image_data = self._encode_image_to_base64(img_path)
                if encoded_image_data:
                    mime_type, base64_image = encoded_image_data
                    image_message_part = {
                        "type": "image_data",
                        "image_data": {
                            "data": f"data:{mime_type};base64,{base64_image}"
                        }
                    }
                    user_content.append(image_message_part)
        messages.append({"role": "user", "content": user_content})
        try:
            import litellm
            response = await litellm.acompletion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stream=stream,
                stop=stop,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                response_format=response_format,
                seed=seed,
                tools=tools,
                tool_choice=tool_choice,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
                api_key=api_key,
                api_base=api_base,
                api_version=api_version,
                deployment_id=deployment_id,
                timeout=timeout,
                metadata=metadata,
                **kwargs
            )
            completion_text = response['choices'][0]['message']['content']
            logger.info(f"Received async completion from {model}. Length: {len(completion_text)}")
            return completion_text
        except Exception as e:
            logger.error(f"Error (async) getting completion from {model} via LiteLLM: {e}", exc_info=True)
            raise

    def _get_mock_completion(self, prompt: str, task_type: str = "default") -> str:
        """Return a mock LLM response based on task_type."""
        logger.debug(f"Generating mock completion for task_type: {task_type}")
        # Use task_type directly instead of searching prompt
        if task_type == "executive_summary":
            return (
                "This quarterly deep-dive report synthesizes findings across ten key sections, analyzing Microsoft CVE trends from [Start Date] to [End Date]. "
                "Overall vulnerability volume remained consistent with previous quarters, however, analysis of severity distribution revealed a concerning 4% increase in 'Critical' rated vulnerabilities, primarily driven by flaws in Windows OS and Azure services. "
                "Exploitability assessments indicated that approximately 15% of disclosed CVEs had publicly available exploit code or were observed being actively exploited, highlighting the urgency for rapid patching. "
                "Remediation metrics showed an average patch deployment time of 18 days for critical vulnerabilities, a slight improvement but still lagging behind industry best practices.\n\n"

                "Key section findings include: Section 3 (Severity Trends) confirmed the dominance of Critical/High vulnerabilities, with detailed charts illustrating peaks associated with specific product releases. "
                "Section 5 (Product Impact) identified Windows Server and Microsoft Exchange as the most impacted products based on CVE count and severity weighting. "
                "Section 7 (Exploit Analysis) provided statistical breakdowns of exploit types (RCE, EoP, DoS) and correlated them with CVSSv3 scores, noting a rise in privilege escalation exploits. "
                "Section 9 (Remediation Effectiveness) analyzed patch success rates and identified common obstacles reported by enterprise security teams, offering data-driven insights into patching challenges.\n\n"

                "Based on the comprehensive data analysis presented across all sections, strategic recommendations include: prioritizing patch deployment for critical RCE and EoP vulnerabilities within 7-14 days, enhancing monitoring for Azure service configurations, and reviewing internal security awareness training related to phishing vectors often associated with initial access exploits. "
                "Continuous vigilance and adaptation of security strategies based on these quarterly trends are crucial for mitigating risk within the Microsoft ecosystem. "
                "Detailed statistics, charts, and section-specific insights supporting these conclusions can be found within the full report body."
            )
        elif task_type == "report_conclusion":
            return (
                "In conclusion, this quarterly analysis highlights the persistent and evolving nature of security vulnerabilities within the Microsoft ecosystem. "
                "While overall CVE volume remained relatively stable, the data presented across ten sections reveals critical trends, particularly the continued prevalence of high-severity flaws and the increasing sophistication of exploit techniques observed in the wild. "
                "Key takeaways include the concentration of risk in specific product areas like Windows Server and Azure, and the ongoing challenge of achieving timely patch deployment across diverse enterprise environments. "
                "The statistical evidence underscores the necessity for a proactive, data-driven approach to vulnerability management.\n\n"

                "Moving forward, organizations must leverage these insights to refine their security strategies, focusing on risk-based prioritization, enhanced detection capabilities, and streamlined remediation workflows. "
                "The findings reinforce that security is not a static goal but a continuous process requiring adaptation and investment. "
                "By addressing the specific vulnerability patterns and challenges identified in this report, businesses can significantly improve their resilience against emerging threats in the subsequent quarter and beyond."
            )
        elif task_type == "callout":
            return (
                "Key takeaway: Organizations should prioritize remediation of high-severity vulnerabilities identified this quarter, with special attention to RCE and privilege escalation issues."
            )
        elif task_type == "chart_insight":
            return (
                "This chart highlights a notable trend: the majority of vulnerabilities are clustered in the high-severity range, with a spike observed in March due to a major Windows update."
            )
        elif task_type == "narrative":
            return (
                "This section delves into the narrative derived from the quarterly vulnerability data, highlighting key trends and contextual factors. "
                "The analysis reveals a consistent pattern in the types of vulnerabilities disclosed, predominantly affecting core Windows components and related services. "
                "We observed a slight increase in remote code execution flaws compared to the previous quarter, demanding continued vigilance from security teams. "
                "Furthermore, the data reflects the impact of Microsoft's ongoing efforts to improve security posture through proactive fuzzing and variant analysis programs.\n\n"

                "Examining the timeline, vulnerability disclosures aligned closely with Microsoft's established Patch Tuesday cycle, indicating a predictable rhythm for remediation planning. "
                "However, several out-of-band patches were also issued, primarily addressing zero-day exploits actively used in the wild, underscoring the dynamic threat landscape. "
                "The distribution of severity scores remained heavily weighted towards 'Critical' and 'Important', reinforcing the need for rapid patch deployment. "
                "Geopolitical factors and major software releases appeared to correlate with specific spikes in vulnerability reporting during the period.\n\n"

                "Overall, the narrative underscores a complex interplay between vendor patching cadences, attacker focus, and the inherent complexity of the Microsoft ecosystem. "
                "While proactive measures show promise, the persistent volume of high-impact vulnerabilities necessitates robust, risk-based patch management strategies. "
                "Organizations should integrate these findings into their threat modeling and prioritize resources towards mitigating the most significant risks identified. "
                "Continuous monitoring and adaptation remain crucial for navigating the evolving security challenges presented each quarter."
            )
        elif task_type == "section_summary":
            return (
                "In summary, this section analyzed the trends in vulnerability severity scores throughout the quarter. "
                "The data clearly indicates a sustained high volume of 'Critical' rated CVEs, particularly impacting server-side components. "
                "While mitigation efforts are ongoing, the concentration of risk in these critical areas warrants continued focus and potential architectural review. "
                "Key performance indicators suggest that patch latency for critical flaws improved slightly but remains a crucial area for operational enhancement."
            )
        else: # Default case
            logger.warning(f"Unknown task_type '{task_type}' for mock completion, returning default response.")
            return (
                "This analysis reveals several important security trends. Key metrics indicate a shift in vulnerability patterns. "
                "Statistical evidence supports ongoing security improvements. Specific components require prioritized attention. "
                "Recommendations focus on proactive security measures."
            )
