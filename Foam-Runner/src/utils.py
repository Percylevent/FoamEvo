# utils.py
import re
import math
from experience_conditions import applicability, task_context
import subprocess
import os
import signal
from typing import Optional, Any, Type, TypedDict, List, Dict
from pydantic import BaseModel, Field, ValidationError
from langchain.chat_models import init_chat_model
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
import tiktoken
from langchain_anthropic import ChatAnthropic
from pathlib import Path
import requests
import time
import random
import shutil
from config import Config
from openfoam_runtime import execute_script
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    HuggingFaceEmbeddings = None


# Global dictionary to store loaded FAISS databases. Loading is deferred until
# retrieval so importing a service module does not download embedding models.
FAISS_DB_CACHE: Dict[str, Any] = {}
_FAISS_DB_CACHE_LOADED = False

def get_embedding_model(config: Optional[Config] = None):
    """Return an embedding model based on the provided config.

    Note: historically this module accessed Config.* class attributes at import time.
    That works for defaults but breaks when callers pass a customized Config instance.
    """
    cfg = config or Config()

    provider = (cfg.embedding_provider or "openai").lower()
    model = cfg.embedding_model

    if provider == "openai":
        return OpenAIEmbeddings(model=model)
    if provider == "huggingface":
        if HuggingFaceEmbeddings is None:
            raise ImportError(
                "langchain-huggingface is not installed. Please install it to use HuggingFace embeddings."
            )
        return HuggingFaceEmbeddings(model_name=model)
    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model=model)

    raise ValueError(f"Unsupported embedding provider: {provider}")


def load_faiss_dbs(config: Optional[Config] = None):
    cfg = config or Config()
    embedding_model = get_embedding_model(cfg)

    base_dir = Path(__file__).resolve().parent.parent / "database" / "faiss"

    # Sanitize model name for directory usage
    model_dir_name = (cfg.embedding_model or "").replace("/", "_").replace(":", "_")
    db_path = base_dir / model_dir_name

    print(f"Loading FAISS indices from: {db_path} with model: {cfg.embedding_model}")

    dbs = {}
    indices = [
        "openfoam_allrun_scripts",
        "openfoam_tutorials_structure",
        "openfoam_tutorials_details",
        "openfoam_command_help",
    ]

    for index in indices:
        index_path = db_path / index
        if index_path.exists():
            try:
                dbs[index] = FAISS.load_local(
                    str(index_path), embedding_model, allow_dangerous_deserialization=True
                )
            except Exception as e:
                print(f"Failed to load index {index}: {e}")
        else:
            print(f"Warning: Index path does not exist: {index_path}")

    return dbs


class FoamfilePydantic(BaseModel):
    file_name: str = Field(description="Name of the OpenFOAM input file")
    folder_name: str = Field(description="Folder where the foamfile should be stored")
    content: str = Field(description="Content of the OpenFOAM file, written in OpenFOAM dictionary format")

class FoamPydantic(BaseModel):
    list_foamfile: List[FoamfilePydantic] = Field(description="List of OpenFOAM configuration files")

class _CodexResponsesWrapper:
    """Wrapper for an OpenAI Responses-compatible endpoint.

    This mimics the minimal interface LLMService expects from LangChain chat models:
    - invoke(messages) -> object with .content
    - get_num_tokens(text) -> int

    We support two wire endpoints:
    - OpenAI Platform: https://api.openai.com/v1/responses (API key / some OAuth tokens)
    - ChatGPT/Codex subscription backend: https://chatgpt.com/backend-api/codex/responses

    The ChatGPT backend requires a non-empty `instructions` field that matches the Codex harness
    expectations. We ship a default copy in `src/codex_instructions_default.txt`.
    """

    class _Resp:
        def __init__(self, content: str):
            self.content = content

    def __init__(
        self,
        token: str,
        model: str,
        temperature: float = 0.0,
        *,
        base_url: str = "https://api.openai.com/v1",
        account_id: Optional[str] = None,
        instructions: Optional[str] = None,
        stream: bool = False,
        http_timeout: int = 300,
    ):
        self._token = token
        self._model = model
        self._temperature = temperature
        self._base_url = base_url.rstrip("/")
        self._account_id = account_id
        self._instructions = instructions
        self._stream = stream
        self._http_timeout = http_timeout
        # Token counting (best-effort). Exact tokenization may differ by model.
        # We default to a modern tokenizer; adjust if you need model-specific counting.
        try:
            self._enc = tiktoken.get_encoding("o200k_base")
        except Exception:
            self._enc = tiktoken.get_encoding("cl100k_base")

    def get_num_tokens(self, text: str) -> int:
        return len(self._enc.encode(text or ""))

    @staticmethod
    def _extract_json_object(text: str) -> str:
        """Best-effort extraction of a JSON object from a model response."""
        if not text:
            raise ValueError("Empty response; expected JSON")

        s = text.strip()
        # Strip fenced code blocks.
        if s.startswith("```"):
            s = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", s)
            s = re.sub(r"\n```\s*$", "", s).strip()

        # If it's already a JSON object, return it.
        if s.startswith("{") and s.endswith("}"):
            return s

        # Otherwise, find the first {...} block.
        m = re.search(r"\{[\s\S]*\}", s)
        if not m:
            raise ValueError(f"Could not find a JSON object in response: {s[:200]}")
        return m.group(0)

    def with_structured_output(self, pydantic_obj: Type[BaseModel]):
        """Return a wrapper that parses the model output into a Pydantic object.

        This is a minimal compatibility shim for LangChain's `.with_structured_output()`.
        """

        parent = self

        class _StructuredWrapper:
            def get_num_tokens(self, text: str) -> int:
                return parent.get_num_tokens(text)

            def invoke(self, messages):
                schema = pydantic_obj.model_json_schema()
                schema_hint = (
                    "Return ONLY valid JSON (no markdown) that matches this JSON Schema:\n"
                    + str(schema)
                )

                patched = list(messages)
                # Prepend a system constraint for JSON output.
                patched.insert(0, {"role": "system", "content": schema_hint})

                resp = parent.invoke(patched)
                raw = getattr(resp, "content", "")
                json_text = parent._extract_json_object(raw)
                return pydantic_obj.model_validate_json(json_text)

        return _StructuredWrapper()

    @staticmethod
    def _to_responses_input(messages):
        out = []
        for m in messages:
            role = m.get("role")
            content = m.get("content", "")
            # Responses API supports rich content; we use simple input_text.
            out.append({"role": role, "content": [{"type": "input_text", "text": content}]})
        return out

    @staticmethod
    def _extract_output_text(resp_json: dict) -> str:
        # Newer APIs often include output_text; fall back to traversing.
        if isinstance(resp_json, dict) and isinstance(resp_json.get("output_text"), str):
            return resp_json["output_text"]

        texts = []
        for item in resp_json.get("output", []) if isinstance(resp_json, dict) else []:
            for c in item.get("content", []) if isinstance(item, dict) else []:
                if isinstance(c, dict):
                    if c.get("type") in {"output_text", "text"} and isinstance(c.get("text"), str):
                        texts.append(c["text"])
        return "\n".join(texts).strip()

    def _build_payload(self, messages):
        payload = {
            "model": self._model,
            "input": self._to_responses_input(messages),
        }

        # OpenAI Platform supports temperature.
        if "chatgpt.com" not in self._base_url:
            payload["temperature"] = self._temperature

        # ChatGPT/Codex subscription backend expects these extra keys.
        if "chatgpt.com" in self._base_url:
            payload.update(
                {
                    "instructions": self._instructions or "",
                    "tools": [],
                    "tool_choice": "auto",
                    "parallel_tool_calls": False,
                    "reasoning": {"summary": "auto"},
                    "store": False,
                    "stream": bool(self._stream),
                    "include": ["reasoning.encrypted_content"],
                }
            )
        return payload

    @staticmethod
    def _iter_sse_text(resp: requests.Response):
        """Yield decoded SSE 'data:' payloads as strings."""
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw:
                continue
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="ignore")
            line = str(raw).strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                break
            yield data

    def invoke(self, messages):
        url = f"{self._base_url}/responses"
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
            "Accept": "application/json" if not self._stream else "text/event-stream",
            "User-Agent": "Foam-Runner",
        }
        if self._account_id:
            headers["ChatGPT-Account-Id"] = self._account_id

        payload = self._build_payload(messages)

        # ChatGPT Codex backend can take 60-180s on complex prompts when
        # reasoning.summary=auto is enabled — the model spends time on the
        # reasoning trace before emitting tokens. The previous hardcoded 60s
        # was too tight: prompts with non-trivial BC topology (e.g. a 2-inlet
        # elbow channel) deterministically exceeded it on gpt-5.5, raising
        # `HTTPSConnectionPool ... Read timed out. (read timeout=60)` and
        # failing the workflow. Keep the timeout in the resolved YAML config.
        timeout = self._http_timeout
        r = requests.post(url, headers=headers, json=payload, timeout=timeout, stream=bool(self._stream))

        # If we get an error, surface the response body to aid debugging.
        if not r.ok:
            try:
                detail = r.text[:2000]
            except Exception:
                detail = ""
            raise requests.HTTPError(
                f"HTTP {r.status_code} for {url}. Body: {detail}", response=r
            )

        if not self._stream:
            data = r.json()
            return self._Resp(self._extract_output_text(data))

        # Streaming: accumulate text deltas.
        import json

        chunks: list[str] = []
        for s in self._iter_sse_text(r):
            try:
                j = json.loads(s)
            except Exception:
                continue

            # Codex backend streams OpenAI Responses-style events.
            if isinstance(j, dict):
                t = j.get("type")
                if t == "response.output_text.delta" and isinstance(j.get("delta"), str):
                    chunks.append(j["delta"])
                    continue
                if t == "response.output_text.done" and isinstance(j.get("text"), str):
                    # Some clients rely on done; we already collected deltas but keep as fallback.
                    if not chunks:
                        chunks.append(j["text"])
                    continue

            # Fallback: try generic extraction.
            t2 = self._extract_output_text(j)
            if t2:
                chunks.append(t2)

        return self._Resp("".join(chunks).strip())


class LLMService:
    @staticmethod
    def _load_codex_access_token_from_auth_json(auth_json_path: Path) -> str:
        import json

        data = json.loads(auth_json_path.read_text(encoding="utf-8"))

        # Be permissive: different Codex versions may store different shapes.
        # Common patterns we try:
        #   {"access_token": "..."}
        #   {"token": "..."}
        #   {"auth": {"access_token": "..."}}
        #   {"credentials": {"access_token": "..."}}
        candidates = []

        def maybe_add(v):
            if isinstance(v, str) and v.strip():
                candidates.append(v.strip())

        if isinstance(data, dict):
            maybe_add(data.get("access_token"))
            maybe_add(data.get("token"))

            for k in ("auth", "credentials", "session"):
                v = data.get(k)
                if isinstance(v, dict):
                    maybe_add(v.get("access_token"))
                    maybe_add(v.get("token"))

        if not candidates:
            raise ValueError(
                f"Could not find an access token in {auth_json_path}. "
                "Expected keys like access_token/token/id_token."
            )

        # Prefer access_token-like strings first (we already appended in that order)
        return candidates[0]

    @staticmethod
    def _load_codex_oauth_from_clawdbot_auth_profiles(auth_profiles_path: Path) -> tuple[str, Optional[str]]:
        """Load (access token, account id) from Clawdbot's OpenAI-Codex OAuth cache.

        Expected shape (v1):
          {"profiles": {"openai-codex:default": {"access": "...", "accountId": "...", ...}}}

        We also fall back to "openai-codex" or any first profile that looks usable.
        """
        import json

        data = json.loads(auth_profiles_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Unexpected JSON in {auth_profiles_path}")

        profiles = data.get("profiles")
        if not isinstance(profiles, dict):
            raise ValueError(f"Missing 'profiles' in {auth_profiles_path}")

        preferred_keys = ["openai-codex:default", "openai-codex"]
        for k in preferred_keys:
            v = profiles.get(k)
            if isinstance(v, dict):
                token = v.get("access")
                account_id = v.get("accountId")
                if isinstance(token, str) and token.strip():
                    return token.strip(), account_id if isinstance(account_id, str) else None

        # Fallback: scan any profile entry that has an 'access' string
        for _, v in profiles.items():
            if isinstance(v, dict):
                token = v.get("access")
                account_id = v.get("accountId")
                if isinstance(token, str) and token.strip():
                    return token.strip(), account_id if isinstance(account_id, str) else None

        raise ValueError(
            f"Could not find an 'access' token in {auth_profiles_path}. "
            "Expected profiles[*].access"
        )

    def _load_codex_oauth(self) -> tuple[str, Optional[str]]:
        """Load the Codex/ChatGPT OAuth token from a local auth cache.

        Supported locations (first match wins):
        1) $CODEX_HOME/auth.json (Codex CLI file-based cache)
        2) ~/.codex/auth.json (Codex CLI default)
        3) ~/.clawdbot/agents/main/agent/auth-profiles.json (Clawdbot OpenAI-Codex OAuth cache)

        Note: These files contain access/refresh tokens. Treat them like passwords.
        """
        candidates: list[Path] = []

        codex_home = os.getenv("CODEX_HOME")
        if codex_home:
            candidates.append(Path(codex_home) / "auth.json")

        candidates.append(Path.home() / ".codex" / "auth.json")

        # Clawdbot stores the OpenAI-Codex OAuth profile here.
        candidates.append(
            Path.home()
            / ".clawdbot"
            / "agents"
            / "main"
            / "agent"
            / "auth-profiles.json"
        )

        for p in candidates:
            if not p.exists():
                continue

            # Codex CLI cache
            if p.name == "auth.json":
                return self._load_codex_access_token_from_auth_json(p), None

            # Clawdbot cache
            if p.name == "auth-profiles.json":
                return self._load_codex_oauth_from_clawdbot_auth_profiles(p)

        raise FileNotFoundError(
            "Could not find a Codex/ChatGPT OAuth cache. Looked for: "
            + ", ".join(str(x) for x in candidates)
            + ". "
            "If you used the Codex CLI, run `codex login` and ensure file-based credential storage. "
            "If you used Clawdbot, make sure you completed OpenAI Codex OAuth in onboarding."
        )

    def __init__(self, config: object):
        self.model_version = getattr(config, "model_version", "gpt-4o")
        self.temperature = getattr(config, "temperature", 0)
        self.model_provider = getattr(config, "model_provider", "openai")
        self._config = config

        # Initialize statistics
        self.total_calls = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.failed_calls = 0
        self.retry_count = 0
        
        # Initialize the LLM
        if self.model_provider.lower() == "bedrock":
            try:
                from langchain_aws import ChatBedrockConverse
                import tracking_aws
            except ImportError as exc:
                raise ImportError(
                    "Bedrock dependencies are not installed. "
                    "Install them with `pip install -e '.[bedrock]'`."
                ) from exc
            bedrock_runtime = tracking_aws.new_default_client()
            self.llm = ChatBedrockConverse(
                client=bedrock_runtime, 
                model_id=self.model_version, 
                temperature=self.temperature, 
                max_tokens=8192
            )
        elif self.model_provider.lower() == "anthropic":
            api_key = getattr(self._config, "anthropic_api_key", "")
            base_url = (
                getattr(self._config, "anthropic_base_url", "")
                or "https://api.anthropic.com"
            )
            self.llm = ChatAnthropic(
                model=self.model_version,
                temperature=self.temperature,
                api_key=api_key,
                base_url=base_url,
                timeout=float(getattr(self._config, "http_timeout", 300)),
                max_retries=0,
            )

        elif self.model_provider.lower() == "openai":
            # Usage-based API access. Credentials come from FOAM_AGENT_* env.
            api_key = getattr(self._config, "openai_api_key", "")
            base_url = getattr(self._config, "openai_base_url", "") or "https://api.openai.com/v1"
            timeout = int(getattr(self._config, "http_timeout", 300))
            api_style = getattr(self._config, "openai_api_style", "chat_completions")
            if api_style == "responses":
                # Some relays serve only the Responses API and reject
                # /chat/completions outright, so the wire protocol has to be
                # selectable independently of the credential.
                #
                # `instructions` is deliberately not sent: a relay may inject
                # its own and refuse any request that also supplies one. The
                # agent's own role prompts travel as ordinary system messages,
                # which is enough -- they are honoured even when a harness
                # preamble was prepended upstream.
                self.llm = _CodexResponsesWrapper(
                    token=api_key,
                    model=self.model_version,
                    temperature=self.temperature,
                    base_url=base_url,
                    stream=False,
                    http_timeout=timeout,
                )
            else:
                self.llm = init_chat_model(
                    self.model_version,
                    model_provider=self.model_provider,
                    temperature=self.temperature,
                    api_key=api_key,
                    base_url=base_url,
                    timeout=timeout,
                )
        elif self.model_provider.lower() in {"openai-codex", "codex", "chatgpt-oauth"}:
            # Subscription-based access via "Sign in with ChatGPT" (Codex auth cache).
            # We use the OpenAI Responses API, which is the typical surface for Codex subscription access.
            token, account_id = self._load_codex_oauth()

            # ChatGPT/Codex subscription route: use the same endpoint as Codex CLI.
            # This avoids requiring Platform API scopes like api.responses.write.
            instructions_path = Path(__file__).resolve().parent / "codex_instructions_default.txt"
            try:
                instructions = instructions_path.read_text(encoding="utf-8")
            except Exception:
                instructions = "You are Codex, based on GPT-5. You are running as a coding agent in the Codex CLI on a user's computer."

            self.llm = _CodexResponsesWrapper(
                token=token,
                account_id=account_id,
                model=self.model_version,
                temperature=self.temperature,
                base_url="https://chatgpt.com/backend-api/codex",
                instructions=instructions,
                stream=True,
                http_timeout=int(getattr(self._config, "http_timeout", 300)),
            )
        elif self.model_provider.lower() == "ollama":
            try:
                from langchain_ollama import ChatOllama
            except ImportError as exc:
                raise ImportError(
                    "Ollama dependencies are not installed. "
                    "Install them with `pip install -e '.[ollama]'`."
                ) from exc
            try:
                response = requests.get("http://localhost:11434/api/version", timeout=2)
                # If request successful, service is running
            except requests.exceptions.RequestException:
                print("Ollama is not running, starting it...")
                subprocess.Popen(["ollama", "serve"], 
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
                # Wait for service to start
                time.sleep(5)  # Give it 3 seconds to initialize

            self.llm = ChatOllama(
                model=self.model_version, 
                temperature=self.temperature,
                num_predict=-1,
                num_ctx=131072,
                base_url="http://localhost:11434"
            )
        elif self.model_provider.lower() == "deepseek":
            from langchain_openai import ChatOpenAI
            reasoning = getattr(self._config, "reasoning_effort", "max")
            if reasoning not in ("low", "medium", "high", "max"):
                reasoning = "max"
            # Note: temperature is ignored by DeepSeek in thinking mode.
            api_key = getattr(self._config, "deepseek_api_key", "")
            base_url = getattr(self._config, "deepseek_base_url", "") or "https://api.deepseek.com/v1"
            self.llm = ChatOpenAI(
                model=self.model_version,
                temperature=self.temperature,
                base_url=base_url,
                api_key=api_key,
                reasoning_effort=reasoning,
                extra_body={"thinking": {"type": "enabled"}},
            )
        else:
            raise ValueError(f"{self.model_provider} is not a supported model_provider")
    
    # Transient HTTP statuses worth retrying instead of aborting the workflow.
    # 429 rate limit; 500/502/503 upstream faults; 504/520/522/524 gateway or
    # origin read timeouts. 524 matters in particular: a Cloudflare-fronted relay
    # returns it when the model takes longer than the proxy read window, which is
    # common for long reasoning responses and is retryable.
    _RETRYABLE_STATUS_CODES = (429, 500, 502, 503, 504, 520, 522, 524)
    # Status code alone is not sufficient. Relays answer with 400 for
    # transient upstream problems, which is semantically closer to 503 than
    # to a genuine client error. Treating the whole 400 class as fast-fail
    # cost one verified fix its promotion: four replay trials died on the
    # first LLM call with "the requested model is not available for the
    # configured upstream account" and the batch stalled at
    # infrastructure_blocked.
    _RETRYABLE_MESSAGE_MARKERS = (
        'upstream request failed',
        'not available for the configured upstream account',
        'upstream account',
    )

    def _is_throttling_error(self, error: Exception) -> bool:
        """
        Check whether an exception is transient, so the call should be retried.

        Covers provider throttling plus gateway/upstream failures. Anything not
        matched here propagates and aborts the entire workflow, so a missing
        case turns a recoverable blip into a lost run.

        Args:
            error: The exception to check

        Returns:
            True if the call should be retried, False otherwise
        """
        # Check ClientError with specific error codes
        try:
            from botocore.exceptions import ClientError
        except ImportError:
            ClientError = ()
        if isinstance(error, ClientError):
            error_code = error.response.get('Error', {}).get('Code', '')
            return error_code in ('Throttling', 'TooManyRequestsException', 'ThrottlingException')

        # Check for ThrottlingException and throttling-related error messages
        error_type = type(error).__name__
        error_str = str(error)

        # OpenAI/Anthropic SDK exceptions carry the HTTP status; prefer reading it
        # over matching strings.
        status_code = getattr(error, 'status_code', None)
        if status_code is None:
            response = getattr(error, 'response', None)
            status_code = getattr(response, 'status_code', None)
        if isinstance(status_code, int) and status_code in self._RETRYABLE_STATUS_CODES:
            return True

        throttling_indicators = (
            error_type == 'ThrottlingException',
            error_type in ('RateLimitError', 'APIConnectionError',
                           'APITimeoutError', 'InternalServerError',
                           'ServiceUnavailableError'),
            'ThrottlingException' in error_str,
            'Too many tokens' in error_str,
            'reached max retries' in error_str,
            'overloaded_error' in error_str,
            'Too Many Requests' in error_str,
            # Relays report the upstream status inside the message body, e.g.
            # "Error code: 524 - {...Proxy Read Timeout...}".
            any(f'Error code: {code}' in error_str
                for code in self._RETRYABLE_STATUS_CODES),
            # Message-body markers, independent of the status code.
            any(m in error_str.lower()
                for m in self._RETRYABLE_MESSAGE_MARKERS),
        )

        return any(throttling_indicators)
    
    def _handle_throttling_retry(self, error: Exception, retry_count: int, max_retries: int) -> Optional[int]:
        """
        Handle throttling error by implementing exponential backoff retry logic.
        
        Args:
            error: The throttling exception
            retry_count: Current retry attempt number
            max_retries: Maximum number of retries allowed
            
        Returns:
            The updated retry count if retry should continue, None if max retries exceeded
        """
        retry_count += 1
        self.retry_count += 1
        
        if retry_count > max_retries:

            print(f"Maximum retries ({max_retries}) exceeded: {str(error)}")
            return None
        
        # Exponential backoff with jitter
        base_delay = 1.0
        max_delay = 60.0
        delay = min(max_delay, base_delay * (2 ** (retry_count - 1)))
        jitter = random.uniform(0, 0.1 * delay)
        sleep_time = delay + jitter
        
        print(f"Transient LLM error: {str(error)}. Retrying in {sleep_time:.2f} seconds (attempt {retry_count}/{max_retries})")
        time.sleep(sleep_time)
        
        return retry_count

    def invoke(self,
              user_prompt: str, 
              system_prompt: Optional[str] = None, 
              pydantic_obj: Optional[Type[BaseModel]] = None,
              max_retries: int = 10) -> Any:
        """
        Invoke the LLM with the given prompts and return the response.
        
        Args:
            user_prompt: The user's prompt
            system_prompt: Optional system prompt
            pydantic_obj: Optional Pydantic model for structured output
            max_retries: Maximum number of retries for throttling errors
            
        Returns:
            The LLM response with token usage statistics
        """
        self.total_calls += 1
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        
        # Calculate prompt tokens
        prompt_tokens = 0
        for message in messages:
            prompt_tokens += self._get_num_tokens(message["content"])

        retry_count = 0
        structured_retry_count = 0
        structured_retry_limit = int(
            getattr(getattr(self, "_config", None), "structured_output_retries", 2)
        )
        while True:
            try:
                if pydantic_obj:
                    if self.model_provider.lower() == "deepseek":
                        # DeepSeek thinking mode does not support response_format,
                        # so with_structured_output fails. Use JSON prompt fallback.
                        schema = pydantic_obj.model_json_schema()
                        json_instruction = (
                            "Return ONLY valid JSON (no markdown, no extra text) matching this schema:\n"
                            + str(schema)
                        )
                        json_messages = list(messages)
                        json_messages.append({"role": "user", "content": json_instruction})
                        raw_response = self.llm.invoke(json_messages)
                        raw_text = raw_response.content
                        # Strip markdown fences if present
                        t = raw_text.strip()
                        if t.startswith("```"):
                            t = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", t)
                            t = re.sub(r"\n?```\s*$", "", t).strip()
                        response = pydantic_obj.model_validate_json(t)
                    else:
                        structured_llm = self.llm.with_structured_output(pydantic_obj)
                        response = structured_llm.invoke(messages)
                    if response is None:
                        raise ValueError(
                            f"Structured output for {pydantic_obj.__name__} was empty"
                        )
                else:
                    response = self.llm.invoke(messages)
                    response = response.content

                # Calculate completion tokens
                response_content = str(response)
                completion_tokens = self._get_num_tokens(response_content)
                total_tokens = prompt_tokens + completion_tokens
                
                # Update statistics
                self.total_prompt_tokens += prompt_tokens
                self.total_completion_tokens += completion_tokens
                self.total_tokens += total_tokens

                return response
                
            except Exception as e:
                if pydantic_obj and self._is_structured_output_error(e):
                    if structured_retry_count < structured_retry_limit:
                        structured_retry_count += 1
                        self.retry_count += 1
                        print(
                            "Structured output validation failed: "
                            f"{str(e)}. Retrying: {structured_retry_count}/"
                            f"{structured_retry_limit}"
                        )
                        continue
                if self._is_throttling_error(e):
                    print(f"Transient LLM error occurred: {str(e)}.")
                    print(f"Retrying: {retry_count + 1}/{max_retries}")
                    retry_count = self._handle_throttling_retry(e, retry_count, max_retries)
                    if retry_count is None:
                        # Max retries exceeded
                        self.failed_calls += 1
                        terminal_error = Exception(
                            f"Maximum retries ({max_retries}) exceeded for "
                            f"throttling error: {str(e)}"
                        )
                        raise terminal_error
                    continue  # Retry the request
                else:
                    print(f"Non-throttling error occurred: {str(e)}.")

                    # Non-throttling error: log and raise
                    print(f"Error occurred in LLM service: {str(e)}")
                    try:
                        from botocore.exceptions import ClientError
                    except ImportError:
                        ClientError = ()
                    if isinstance(e, ClientError):
                        print(e.response)
                    self.failed_calls += 1
                    raise e

    @staticmethod
    def _is_structured_output_error(error: Exception) -> bool:
        if isinstance(error, ValidationError):
            return True
        error_type = type(error).__name__
        error_text = str(error).lower()
        if error_type in {"OutputParserException", "JsonOutputParserException"}:
            return True
        return any(
            marker in error_text
            for marker in (
                "structured output",
                "valid dictionary or instance",
                "input should be a valid list",
                "tool call",
            )
        )

    def _get_num_tokens(self, text: str) -> int:
        """Count tokens without requiring a provider-specific tokenizer download."""
        if self.model_provider.lower() == "anthropic":
            try:
                encoding = tiktoken.get_encoding("cl100k_base")
                return len(encoding.encode(text or ""))
            except Exception:
                return max(1, len(text or "") // 4)
        try:
            return self.llm.get_num_tokens(text)
        except Exception as error:
            print(f"Provider token counter unavailable: {error}. Using tiktoken fallback.")
            try:
                encoding = tiktoken.get_encoding("cl100k_base")
                return len(encoding.encode(text or ""))
            except Exception:
                return max(1, len(text or "") // 4)
    
    def get_statistics(self) -> dict:
        """
        Get the current statistics of the LLM service.
        
        Returns:
            Dictionary containing various statistics
        """
        return {
            "total_calls": self.total_calls,
            "failed_calls": self.failed_calls,
            "retry_count": self.retry_count,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "average_prompt_tokens": self.total_prompt_tokens / self.total_calls if self.total_calls > 0 else 0,
            "average_completion_tokens": self.total_completion_tokens / self.total_calls if self.total_calls > 0 else 0,
            "average_tokens": self.total_tokens / self.total_calls if self.total_calls > 0 else 0
        }
    
    def print_statistics(self) -> None:
        """
        Print the current statistics of the LLM service.
        """
        stats = self.get_statistics()
        print("\n<LLM Service Statistics>")
        print(f"Total calls: {stats['total_calls']}")
        print(f"Failed calls: {stats['failed_calls']}")
        print(f"Total retries: {stats['retry_count']}")
        print(f"Total prompt tokens: {stats['total_prompt_tokens']}")
        print(f"Total completion tokens: {stats['total_completion_tokens']}")
        print(f"Total tokens: {stats['total_tokens']}")
        print(f"Average prompt tokens per call: {stats['average_prompt_tokens']:.2f}")
        print(f"Average completion tokens per call: {stats['average_completion_tokens']:.2f}")
        print(f"Average tokens per call: {stats['average_tokens']:.2f}\n")
        print("</LLM Service Statistics>")

class GraphState(TypedDict):
    user_requirement: str
    config: Config
    case_dir: str
    tutorial: str
    case_name: str
    subtasks: List[dict]
    current_subtask_index: int
    error_command: Optional[str]
    error_content: Optional[str]
    loop_count: int
    # Additional state fields that will be added during execution
    llm_service: Optional['LLMService']
    case_stats: Optional[dict]
    tutorial_reference: Optional[str]
    case_path_reference: Optional[str]
    dir_structure_reference: Optional[str]
    case_info: Optional[str]
    allrun_reference: Optional[str]
    dir_structure: Optional[dict]
    commands: Optional[List[str]]
    foamfiles: Optional[dict]
    error_logs: Optional[List[str]]
    history_text: Optional[List[str]]
    case_domain: Optional[str]
    case_category: Optional[str]
    case_solver: Optional[str]
    # Mesh-related state fields
    mesh_info: Optional[dict]
    mesh_commands: Optional[List[str]]
    custom_mesh_used: Optional[bool]
    mesh_type: Optional[str]
    custom_mesh_path: Optional[str]
    # Review and rewrite related fields
    review_analysis: Optional[str]
    rewrite_plan: Optional[dict]
    last_rewrite_changed: Optional[bool]
    changed_files: Optional[List[str]]
    error_signature: Optional[str]
    error_signature_history: Optional[List[str]]
    repeated_error_count: int
    # Consecutive runner passes blocked by an infrastructure precondition.
    # LangGraph only carries declared state fields across transitions.
    consecutive_infrastructure_failures: int
    input_writer_mode: Optional[str]
    similar_case_advice: Optional[dict]
    experience_guidance: Optional[str]
    # Routing decision cache
    requires_hpc: Optional[bool]
    requires_visualization: Optional[bool]
    # HPC-related fields
    job_id: Optional[str]
    cluster_info: Optional[dict]
    slurm_script_path: Optional[str]
    termination_reason: Optional[str]

def tokenize(text: str) -> str:
    # Replace underscores with spaces
    text = text.replace('_', ' ')
    # Insert a space between a lowercase letter and an uppercase letter (global match)
    text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)
    return text.lower()

def save_file(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(content)
    print(f"Saved file at {path}")

def read_file(path: str) -> str:
    if os.path.exists(path):
        with open(path, 'r') as f:
            return f.read()
    return ""

def remove_files(directory: str, prefix: str) -> None:
    for file in os.listdir(directory):
        if file.startswith(prefix):
            os.remove(os.path.join(directory, file))
    print(f"Removed files with prefix '{prefix}' in {directory}")

def remove_file(path: str) -> None:
    if os.path.exists(path):
        os.remove(path)
        print(f"Removed file {path}")

def remove_numeric_folders(case_dir: str) -> None:
    """
    Remove all folders in case_dir that represent numeric values, including those with decimal points,
    except for the "0" folder.
    
    Args:
        case_dir (str): The directory path to process
    """
    for item in os.listdir(case_dir):
        item_path = os.path.join(case_dir, item)
        if os.path.isdir(item_path) and item != "0":
            try:
                # Try to convert to float to check if it's a numeric value
                float(item)
                # If conversion succeeds, it's a numeric folder
                try:
                    shutil.rmtree(item_path)
                    print(f"Removed numeric folder: {item_path}")
                except Exception as e:
                    print(f"Error removing folder {item_path}: {str(e)}")
            except ValueError:
                # Not a numeric value, so we keep this folder
                pass


def scan_case_directory(case_dir: str) -> Dict[str, List[str]]:
    """
    Scan an OpenFOAM case directory and return the directory structure.
    
    This function traverses the case directory one level deep and collects
    the files in each subdirectory (typically 'system', 'constant', '0', etc.).
    
    Args:
        case_dir (str): Path to the OpenFOAM case directory
    
    Returns:
        Dict[str, List[str]]: Dictionary mapping folder names to lists of file names
            Example: {"system": ["controlDict", "fvSchemes"], "constant": ["transportProperties"]}
    
    Raises:
        FileNotFoundError: If case_dir does not exist
        PermissionError: If directory cannot be accessed
    
    Example:
        >>> structure = scan_case_directory("/path/to/case")
        >>> print(structure["system"])  # ["controlDict", "fvSchemes", "fvSolution"]
    """
    if not os.path.exists(case_dir):
        raise FileNotFoundError(f"Case directory does not exist: {case_dir}")
    
    dir_structure = {}
    base_depth = case_dir.rstrip(os.sep).count(os.sep)
    
    # Walk through the directory tree
    for root, dirs, files in os.walk(case_dir):
        # Only process directories one level below case_dir
        current_depth = root.rstrip(os.sep).count(os.sep)
        if current_depth == base_depth + 1:
            folder_name = os.path.relpath(root, case_dir)
            # Skip solver-generated time-step output directories (numeric names
            # like "0.05", "0.1", "1", "1.5", etc.) but keep the "0" directory
            # which holds initial/boundary conditions.
            if folder_name != "0":
                try:
                    float(folder_name)
                    continue  # numeric time-step directory — skip it
                except ValueError:
                    pass
            # Filter out hidden files and only include regular files
            regular_files = [f for f in files if not f.startswith('.') and os.path.isfile(os.path.join(root, f))]
            if regular_files:
                dir_structure[folder_name] = regular_files
    
    return dir_structure


def read_case_foamfiles(case_dir: str, dir_structure: Optional[Dict[str, List[str]]] = None) -> 'FoamPydantic':
    """
    Read OpenFOAM files from a case directory and convert to FoamPydantic format.
    
    This function reads all OpenFOAM configuration files from the case directory
    (typically from 'system', 'constant', '0' folders) and creates a FoamPydantic
    object containing the file contents.
    
    Args:
        case_dir (str): Path to the OpenFOAM case directory
        dir_structure (Optional[Dict[str, List[str]]]): Pre-scanned directory structure.
            If None, will scan the directory automatically.
    
    Returns:
        FoamPydantic: Object containing list of FoamfilePydantic objects with file metadata
    
    Raises:
        FileNotFoundError: If case_dir does not exist
        UnicodeDecodeError: If files contain invalid encoding (will skip those files)
    
    Example:
        >>> foamfiles = read_case_foamfiles("/path/to/case")
        >>> print(len(foamfiles.list_foamfile))  # Number of files read
        >>> print(foamfiles.list_foamfile[0].file_name)  # "controlDict"
    """
    if not os.path.exists(case_dir):
        raise FileNotFoundError(f"Case directory does not exist: {case_dir}")
    
    # Scan directory structure if not provided
    if dir_structure is None:
        dir_structure = scan_case_directory(case_dir)
    
    foamfile_list = []
    
    # Read files from each folder
    for folder_name, file_names in dir_structure.items():
        for file_name in file_names:
            file_path = os.path.join(case_dir, folder_name, file_name)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                foamfile_list.append(FoamfilePydantic(
                    file_name=file_name,
                    folder_name=folder_name,
                    content=content
                ))
            except UnicodeDecodeError:
                print(f"Warning: Skipping file due to encoding error: {file_path}")
            except Exception as e:
                print(f"Warning: Error reading file {file_path}: {e}")
    
    return FoamPydantic(list_foamfile=foamfile_list)

def run_command(
    script_path: str,
    out_file: str,
    err_file: str,
    working_dir: str,
    max_time_limit: int,
):
    print(f"Executing script {script_path} in {working_dir}")
    result = execute_script(
        script_path,
        stdout_path=out_file,
        stderr_path=err_file,
        working_dir=working_dir,
        timeout_s=max_time_limit,
    )
    print(f"Executed script {script_path}")
    return result

def parse_context(text: str) -> str:
    match = re.search(r'FoamFile\s*\{.*?(?=```|$)', text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(0).strip()
    
    print("Warning: Could not parse context; returning original text.")
    return text


def retrieve_faiss(database_name: str, query: str, topk: int = 1) -> dict:
    """
    Retrieve a similar case from a FAISS database.
    """

    global FAISS_DB_CACHE, _FAISS_DB_CACHE_LOADED
    if not _FAISS_DB_CACHE_LOADED:
        FAISS_DB_CACHE = load_faiss_dbs()
        _FAISS_DB_CACHE_LOADED = True

    if database_name not in FAISS_DB_CACHE:
        raise ValueError(f"Database '{database_name}' is not loaded.")

    # Tokenize the query
    query = tokenize(query)

    vectordb = FAISS_DB_CACHE[database_name]
    try:
        docs_and_scores = vectordb.similarity_search_with_score(query, k=topk)
        docs = [d for d, _ in docs_and_scores]
        scores = [s for _, s in docs_and_scores]
    except Exception:
        docs = vectordb.similarity_search(query, k=topk)
        scores = [None] * len(docs)

    if not docs:
        raise ValueError(f"No documents found for query: {query}")

    formatted_results = []
    for doc, score in zip(docs, scores):
        metadata = doc.metadata or {}

        if database_name == "openfoam_allrun_scripts":
            formatted_results.append({
                "index": doc.page_content,
                "full_content": metadata.get("full_content", "unknown"),
                "case_name": metadata.get("case_name", "unknown"),
                "case_domain": metadata.get("case_domain", "unknown"),
                "case_category": metadata.get("case_category", "unknown"),
                "case_solver": metadata.get("case_solver", "unknown"),
                "dir_structure": metadata.get("dir_structure", "unknown"),
                "allrun_script": metadata.get("allrun_script", "N/A"),
                "score": score,
            })
        elif database_name == "openfoam_command_help":
            formatted_results.append({
                "index": doc.page_content,
                "full_content": metadata.get("full_content", "unknown"),
                "command": metadata.get("command", "unknown"),
                "help_text": metadata.get("help_text", "unknown"),
                "score": score,
            })
        elif database_name == "openfoam_tutorials_structure":
            formatted_results.append({
                "index": doc.page_content,
                "full_content": metadata.get("full_content", "unknown"),
                "case_name": metadata.get("case_name", "unknown"),
                "case_domain": metadata.get("case_domain", "unknown"),
                "case_category": metadata.get("case_category", "unknown"),
                "case_solver": metadata.get("case_solver", "unknown"),
                "dir_structure": metadata.get("dir_structure", "unknown"),
                "score": score,
            })
        elif database_name == "openfoam_tutorials_details":
            formatted_results.append({
                "index": doc.page_content,
                "full_content": metadata.get("full_content", "unknown"),
                "case_name": metadata.get("case_name", "unknown"),
                "case_domain": metadata.get("case_domain", "unknown"),
                "case_category": metadata.get("case_category", "unknown"),
                "case_solver": metadata.get("case_solver", "unknown"),
                "dir_structure": metadata.get("dir_structure", "unknown"),
                "tutorials": metadata.get("tutorials", "N/A"),
                "score": score,
            })
        else:
            raise ValueError(f"Unknown database name: {database_name}")

    return formatted_results
        

def parse_directory_structure(data: str) -> dict:
    """
    Parses the directory structure string and returns a dictionary where:
      - Keys: directory names
      - Values: count of files in that directory.
    """
    directory_file_counts = {}

    # Find all <dir>...</dir> blocks in the input string.
    dir_blocks = re.findall(r'<dir>(.*?)</dir>', data, re.DOTALL)

    for block in dir_blocks:
        # Extract the directory name (everything after "directory name:" until the first period)
        dir_name_match = re.search(r'directory name:\s*(.*?)\.', block)
        # Extract the list of file names within square brackets
        files_match = re.search(r'File names in this directory:\s*\[(.*?)\]', block)
        
        if dir_name_match and files_match:
            dir_name = dir_name_match.group(1).strip()
            files_str = files_match.group(1)
            # Split the file names by comma, removing any surrounding whitespace
            file_list = [filename.strip() for filename in files_str.split(',')]
            directory_file_counts[dir_name] = len(file_list)

    return directory_file_counts


# --- Experience RAG ---------------------------------------------------------
# A second, independent index holding guidance distilled from past failures.
# It cannot ride on FAISS_DB_CACHE: retrieve_faiss() raises on unknown database
# names (see above) and its _FAISS_DB_CACHE_LOADED latch loads a fixed index
# list exactly once, so a late-arriving index would never be picked up.
EXPERIENCE_DB_CACHE: Optional[Any] = None
_EXPERIENCE_DB_LOADED = False


def load_experience_db(config: Optional[Config] = None):
    """Load the experience index, or return None when it is absent.

    Absence is the normal state before the first entry is promoted, so it is
    reported once and never raised.
    """
    cfg = config or Config()
    embedding_model = getattr(cfg, "embedding_model", None)
    if not embedding_model:
        # A config-like object without embedding settings cannot tell us which
        # index to load, and guessing would load one built by another model.
        print(
            "<experience_db>no embedding_model on the supplied config; "
            "retrieval disabled</experience_db>"
        )
        return None

    model_dir_name = embedding_model.replace("/", "_").replace(":", "_")
    db_path = (
        Path(__file__).resolve().parent.parent
        / "database" / "experience" / "faiss" / model_dir_name / "experience_guidance"
    )

    if not db_path.exists():
        print(f"<experience_db>absent at {db_path}; retrieval disabled</experience_db>")
        return None

    try:
        db = FAISS.load_local(
            str(db_path), get_embedding_model(cfg), allow_dangerous_deserialization=True
        )
    except Exception as e:
        print(f"<experience_db>failed to load {db_path}: {e}; retrieval disabled</experience_db>")
        return None

    count = getattr(getattr(db, "index", None), "ntotal", "?")
    print(f"<experience_db>loaded {count} entries from {db_path}</experience_db>")
    return db


# Maximum L2 distance for a guidance entry to be injected. FAISS is loaded with
# langchain's default EUCLIDEAN_DISTANCE strategy, so this is a distance and
# lower means closer -- 0.0 is an exact match.
#
# Without a ceiling the library behaves as "always hit": FAISS returns the k
# nearest neighbours unconditionally, so once any entry exists, every query gets
# guidance, however unrelated. The solver and fork filters already remove the
# dangerous cross-domain hits; this removes the merely irrelevant ones, which
# are prompt noise rather than a correctness problem.
#
# 0.9 is deliberately permissive. The one real retrieval measured end to end
# scored 0.354, and absurd queries land above 1.0, so the cutoff sits well clear
# of known-good matches. Guidance that survives this and is still wrong is
# caught downstream: replay regresses, promotion is blocked, and the next batch
# retires the entry.
EXPERIENCE_MAX_DISTANCE = 0.9


def retrieve_experience(
    query: str,
    topk: int = 3,
    config: Optional[Config] = None,
    solver: str = "",
    fork: str = "",
    max_distance: float = EXPERIENCE_MAX_DISTANCE,
    version: str = "",
    context: Optional[dict] = None,
) -> List[dict]:
    """Retrieve guidance entries for a query; returns [] when the library is absent.

    Never raises: a missing or broken experience index must not change how a
    case runs, only whether it gets extra advice.

    ``solver`` and ``fork`` apply hard filters after recall, mirroring the domain
    constraint the tutorial retrieval already enforces (services/plan.py). Pure
    similarity is not enough here: an entry about one solver can outrank a
    relevant one on shared keywords, and guidance written for the other OpenFOAM
    fork is not merely less relevant, it is wrong. Entries declaring ``any`` opt
    out of the corresponding filter.

    ``max_distance`` drops matches that are merely the nearest available rather
    than actually relevant; see EXPERIENCE_MAX_DISTANCE.
    """
    global EXPERIENCE_DB_CACHE, _EXPERIENCE_DB_LOADED
    if topk <= 0 or not math.isfinite(max_distance) or max_distance < 0:
        return []
    version = version or (getattr(config, "openfoam_version", "") if config else os.getenv("WM_PROJECT_VERSION", ""))
    context = task_context(config) if context is None else context
    if not _EXPERIENCE_DB_LOADED:
        try:
            EXPERIENCE_DB_CACHE = load_experience_db(config)
        except Exception as e:
            print(f"<experience_db>load failed: {e}; retrieval disabled</experience_db>")
            EXPERIENCE_DB_CACHE = None
        _EXPERIENCE_DB_LOADED = True

    if EXPERIENCE_DB_CACHE is None:
        return []

    # Recall wider than requested: filtering happens below, so asking for only
    # topk would let filtered-out entries consume the budget.
    recall_k = max(10, topk * 4)
    try:
        docs_and_scores = EXPERIENCE_DB_CACHE.similarity_search_with_score(
            tokenize(query), k=recall_k
        )
    except Exception as e:
        print(f"<experience_db>search failed: {e}</experience_db>")
        return []

    results = []
    for doc, score in docs_and_scores:
        metadata = doc.metadata or {}
        entry_id = metadata.get("entry_id", "unknown")

        try:
            distance = float(score)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(distance) or distance < 0 or distance > max_distance:
            continue
        accepted, reason = applicability(metadata, solver=solver, fork=fork,
                                          version=version, context=context)
        if not accepted:
            print(f"<experience_excluded>{entry_id} reason={reason}</experience_excluded>")
            continue
        entry_solver = metadata.get("solver") or ""
        entry_fork = metadata.get("fork") or ""
        # entry_id, not trigger_type: several entries may share a trigger_type,
        # so only the id identifies which guidance a run actually consumed.
        print(
            f"<experience_hit>{entry_id} trigger={metadata.get('trigger_type', 'unknown')} "
            f"score={score}</experience_hit>"
        )
        results.append({
            "full_content": metadata.get("full_content", ""),
            "entry_id": entry_id,
            "trigger_type": metadata.get("trigger_type", "unknown"),
            "solver": entry_solver or "unknown",
            "fork": entry_fork or "unknown",
            "score": score,
        })
        if len(results) >= topk:
            break
    return results


def format_experience_guidance(entries: List[dict]) -> str:
    """Render entries for a prompt, or "" when there are none.

    An empty string keeps prompts byte-identical to their pre-feature form,
    which is what makes the empty-library case verifiable.
    """
    if not entries:
        return ""
    bodies = "\n\n".join(e.get("full_content", "") for e in entries if e.get("full_content"))
    if not bodies:
        return ""
    return (
        "<experience_guidance>\n"
        "Guidance filtered by declared applicability metadata. Check its remaining "
        "physical and numerical assumptions against the current case before applying it.\n"
        f"{bodies}\n"
        "</experience_guidance>\n"
    )
