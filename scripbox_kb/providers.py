from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass


class ProviderError(Exception):
    """Base exception for LLM provider failures."""

    def __init__(self, message: str, provider: str, original: Exception | None = None):
        super().__init__(message)
        self.provider = provider
        self.original = original


class ProviderRateLimitError(ProviderError): ...


class ProviderAuthError(ProviderError): ...


class ProviderModelNotFoundError(ProviderError): ...


class ProviderUnavailableError(ProviderError): ...


def classify_error(exc: Exception, provider: str) -> ProviderError:
    err_text = str(exc).lower()
    if "429" in err_text or "rate_limit" in err_text:
        return ProviderRateLimitError(str(exc), provider, exc)
    if "401" in err_text or "invalid_api_key" in err_text:
        return ProviderAuthError(str(exc), provider, exc)
    if "404" in err_text or "model_not_found" in err_text or "decommissioned" in err_text:
        return ProviderModelNotFoundError(str(exc), provider, exc)

    return ProviderUnavailableError(str(exc), provider, exc)


def groq_stream(
    prompt: str,
    *,
    api_key: str,
    model: str,
    system_prompt: str,
    max_tokens: int = 1024,
    temperature: float = 0.2,
) -> Iterator[str]:
    from groq import Groq

    try:
        client = Groq(api_key=api_key)
        stream = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
    except Exception as e:
        raise classify_error(e, "groq") from e


def gemini_stream(prompt: str, *, api_key: str, model: str, system_prompt: str) -> Iterator[str]:
    import google.generativeai as genai

    try:
        genai.configure(api_key=api_key)
        genai_model = genai.GenerativeModel(
            model,
            system_instruction=system_prompt,
        )
        stream = genai_model.generate_content(prompt, stream=True)
        for chunk in stream:
            if chunk.text:
                yield chunk.text
    except Exception as e:
        raise classify_error(e, "gemini") from e


@dataclass
class ProviderResult:
    provider: str  # "groq" or "gemini"
    stream: Iterator[str]
    fallback_reason: str | None = None  # why we fell back, if applicable


def stream_with_fallback(
    prompt: str,
    *,
    groq_api_key: str,
    gemini_api_key: str,
    groq_model: str,
    gemini_model: str,
    system_prompt: str,
    max_tokens: int = 1024,
) -> ProviderResult:
    if not groq_api_key and not gemini_api_key:
        raise ProviderUnavailableError("No provider API keys configured.", "none")

    fallback_reason = None
    if groq_api_key:
        try:
            stream_gen = groq_stream(
                prompt,
                api_key=groq_api_key,
                model=groq_model,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
            )
            # Try getting first element to trigger any initial errors.
            first_chunk = next(stream_gen)

            def chain_generator(first: str, gen: Iterator[str]) -> Iterator[str]:
                yield first
                yield from gen

            return ProviderResult(provider="groq", stream=chain_generator(first_chunk, stream_gen))

        except ProviderAuthError:
            raise  # Don't silently fall back on bad auth
        except (ProviderRateLimitError, ProviderModelNotFoundError) as e:
            fallback_reason = f"Groq fallback: {type(e).__name__}"
        except StopIteration:
            return ProviderResult(provider="groq", stream=iter([]))
        except Exception as e:
            fallback_reason = f"Groq fallback: {type(e).__name__}"

    if gemini_api_key:
        try:
            stream_gen = gemini_stream(
                prompt, api_key=gemini_api_key, model=gemini_model, system_prompt=system_prompt
            )
            # Ensure it works immediately
            first_chunk = next(stream_gen)

            def chain_generator(first: str, gen: Iterator[str]) -> Iterator[str]:
                yield first
                yield from gen

            return ProviderResult(
                provider="gemini",
                stream=chain_generator(first_chunk, stream_gen),
                fallback_reason=fallback_reason,
            )
        except StopIteration:
            return ProviderResult(
                provider="gemini", stream=iter([]), fallback_reason=fallback_reason
            )
        except Exception as e:
            raise classify_error(e, "gemini") from e

    raise ProviderUnavailableError("Providers failed or unavailable.", "none")
