import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.llm_providers import (
    PROVIDERS,
    _friendly_error,
    call_llm,
    call_llm_text,
    call_llm_with_usage,
    model_options,
    secret_name,
    test_connection,
)


class LLMProviderSettingsTests(unittest.TestCase):
    def test_all_requested_providers_have_models_and_key_names(self):
        self.assertEqual(list(PROVIDERS), ["Claude", "Cohere.ai", "Gemini", "ChatGPT", "OpenRouter"])
        expected_secrets = {
            "Claude": "ANTHROPIC_API_KEY",
            "Cohere.ai": "COHERE_API_KEY",
            "Gemini": "GEMINI_API_KEY",
            "ChatGPT": "OPENAI_API_KEY",
            "OpenRouter": "OPENROUTER_API_KEY",
        }
        for provider, env_name in expected_secrets.items():
            self.assertTrue(model_options(provider))
            self.assertEqual(secret_name(provider), env_name)
        self.assertEqual(model_options("OpenRouter")[0], "google/gemma-4-31b-it:free")
        self.assertIn("nvidia/nemotron-3.5-super-vl:free", model_options("OpenRouter"))

    def test_rejects_unknown_provider(self):
        with self.assertRaises(ValueError):
            model_options("Unknown")
        with self.assertRaises(ValueError):
            secret_name("Unknown")

    def test_provider_errors_are_friendly(self):
        unauthorized = type("ProviderError", (Exception,), {"status_code": 401})("hidden details")
        self.assertEqual(
            _friendly_error("Gemini", unauthorized),
            "Gemini rejected the API key. Check the key in LLM Settings.",
        )
        rate_limited = type("ProviderError", (Exception,), {"status_code": 429})("hidden details")
        self.assertIn("quota reached", _friendly_error("ChatGPT", rate_limited))
        payment_required = type("ProviderError", (Exception,), {"status_code": 402})("hidden details")
        self.assertIn("available credits or billing", _friendly_error("OpenRouter", payment_required))

    @patch("requests.post")
    def test_gemini_api_key_is_sent_in_header_not_url(self, post):
        response = Mock()
        response.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": '{"ok": true}'}]}}],
            "usageMetadata": {
                "promptTokenCount": 21,
                "candidatesTokenCount": 8,
            },
        }
        post.return_value = response
        self.assertEqual(call_llm("Gemini", "secret-value", "gemini-model", "prompt"), '{"ok": true}')
        generation = call_llm_with_usage(
            "Gemini", "secret-value", "gemini-model", "prompt"
        )
        self.assertEqual((generation.input_tokens, generation.output_tokens), (21, 8))
        kwargs = post.call_args.kwargs
        self.assertEqual(kwargs["headers"], {"x-goog-api-key": "secret-value"})
        self.assertNotIn("secret-value", post.call_args.args[0])
        self.assertNotIn("params", kwargs)

    @patch("requests.post")
    def test_connection_test_calls_gemini_without_supplier_data(self, post):
        post.return_value = Mock()
        test_connection("Gemini", "secret-value", "gemini-model")
        self.assertEqual(
            post.call_args.kwargs["json"]["contents"][0]["parts"][0]["text"],
            "Reply with the word connected.",
        )

    @patch("cohere.ClientV2")
    def test_cohere_adapter_returns_generated_text(self, client_factory):
        response = Mock()
        response.message.content = [SimpleNamespace(text='{"provider":"cohere"}')]
        response.usage.tokens = SimpleNamespace(input_tokens=12, output_tokens=7)
        client_factory.return_value.chat.return_value = response
        generation = call_llm_with_usage(
            "Cohere.ai", "key", "command-a-plus-05-2026", "prompt"
        )
        self.assertEqual(generation.text, '{"provider":"cohere"}')
        self.assertEqual(generation.input_tokens, 12)
        self.assertEqual(generation.output_tokens, 7)
        client_factory.assert_called_once_with(api_key="key", timeout=180)
        self.assertEqual(
            client_factory.return_value.chat.call_args.kwargs["max_tokens"],
            16384,
        )
        self.assertEqual(
            client_factory.return_value.chat.call_args.kwargs["thinking"],
            {"type": "disabled"},
        )

    @patch("cohere.ClientV2")
    def test_cohere_adapter_rejects_token_limited_response(self, client_factory):
        response = Mock()
        response.finish_reason = "MAX_TOKENS"
        response.message.content = [SimpleNamespace(text='{"incomplete":')]
        client_factory.return_value.chat.return_value = response

        with self.assertRaisesRegex(RuntimeError, "output token limit"):
            call_llm("Cohere.ai", "key", "command-a-plus-05-2026", "prompt")

    @patch("cohere.ClientV2")
    def test_cohere_empty_answer_has_actionable_error(self, client_factory):
        response = Mock()
        response.finish_reason = "COMPLETE"
        response.message.content = []
        client_factory.return_value.chat.return_value = response

        with self.assertRaisesRegex(RuntimeError, "returned no text content"):
            call_llm("Cohere.ai", "key", "command-a-plus-05-2026", "prompt")

    @patch("cohere.ClientV2")
    def test_cohere_legacy_model_omits_reasoning_option(self, client_factory):
        response = Mock()
        response.message.content = [SimpleNamespace(text='{"provider":"cohere"}')]
        client_factory.return_value.chat.return_value = response

        call_llm("Cohere.ai", "key", "command-a-03-2025", "prompt")

        self.assertNotIn(
            "thinking",
            client_factory.return_value.chat.call_args.kwargs,
        )
        self.assertEqual(
            client_factory.return_value.chat.call_args.kwargs["max_tokens"],
            4096,
        )

    @patch("cohere.ClientV2")
    def test_cohere_adapter_skips_thinking_content_blocks(self, client_factory):
        response = Mock()
        response.message.content = [
            SimpleNamespace(type="thinking", thinking="Working through the response."),
            SimpleNamespace(type="text", text='{"provider":"cohere"}'),
        ]
        client_factory.return_value.chat.return_value = response
        text = call_llm("Cohere.ai", "key", "command-a-plus-05-2026", "prompt")
        self.assertEqual(text, '{"provider":"cohere"}')

    @patch("cohere.ClientV2")
    def test_cohere_text_adapter_returns_plain_answer(self, client_factory):
        response = Mock()
        response.message.content = [SimpleNamespace(text="Supported answer [S1].")]
        client_factory.return_value.chat.return_value = response
        text = call_llm_text("Cohere.ai", "key", "command-a-03-2025", "question")
        self.assertEqual(text, "Supported answer [S1].")
        self.assertNotIn("response_format", client_factory.return_value.chat.call_args.kwargs)

    def test_read_timeout_error_is_reported_as_timeout(self):
        read_timeout = type("ReadTimeout", (Exception,), {})()
        self.assertEqual(
            _friendly_error("Cohere.ai", read_timeout),
            "Cohere.ai request timed out. Retry when the provider is reachable.",
        )

    @patch("anthropic.Anthropic")
    def test_claude_adapter_returns_generated_text(self, client_factory):
        client_factory.return_value.messages.create.return_value.content = [
            SimpleNamespace(type="text", text='{"provider":"claude"}')
        ]
        client_factory.return_value.messages.create.return_value.usage = SimpleNamespace(
            input_tokens=31, output_tokens=9
        )
        text = call_llm("Claude", "key", "claude-sonnet-5-5", "prompt")
        self.assertEqual(text, '{"provider":"claude"}')
        generation = call_llm_with_usage(
            "Claude", "key", "claude-sonnet-5-5", "prompt"
        )
        self.assertEqual((generation.input_tokens, generation.output_tokens), (31, 9))

    @patch("openai.OpenAI")
    def test_chatgpt_adapter_returns_generated_text(self, client_factory):
        client_factory.return_value.chat.completions.create.return_value.choices = [
            SimpleNamespace(message=SimpleNamespace(content='{"provider":"chatgpt"}'))
        ]
        client_factory.return_value.chat.completions.create.return_value.usage = (
            SimpleNamespace(prompt_tokens=42, completion_tokens=13)
        )
        text = call_llm("ChatGPT", "key", "gpt-4.1", "prompt")
        self.assertEqual(text, '{"provider":"chatgpt"}')
        generation = call_llm_with_usage("ChatGPT", "key", "gpt-4.1", "prompt")
        self.assertEqual((generation.input_tokens, generation.output_tokens), (42, 13))

    @patch("openai.OpenAI")
    def test_openrouter_adapter_uses_openrouter_endpoint_and_returns_json(self, client_factory):
        client_factory.return_value.chat.completions.create.return_value.choices = [
            SimpleNamespace(message=SimpleNamespace(content='{"provider":"openrouter"}'))
        ]
        client_factory.return_value.chat.completions.create.return_value.usage = (
            SimpleNamespace(prompt_tokens=52, completion_tokens=17)
        )
        generation = call_llm_with_usage(
            "OpenRouter", "key", "openai/gpt-4.1-mini", "prompt"
        )
        self.assertEqual(generation.text, '{"provider":"openrouter"}')
        self.assertEqual((generation.input_tokens, generation.output_tokens), (52, 17))
        client_factory.assert_called_once_with(
            api_key="key",
            base_url="https://openrouter.ai/api/v1",
            timeout=90,
            default_headers={"X-Title": "Tender Forge Steel AI"},
        )
        self.assertEqual(
            client_factory.return_value.chat.completions.create.call_args.kwargs[
                "response_format"
            ],
            {"type": "json_object"},
        )

    @patch("anthropic.Anthropic")
    def test_usage_is_unavailable_when_provider_omits_metadata(self, client_factory):
        response = client_factory.return_value.messages.create.return_value
        response.content = [SimpleNamespace(type="text", text='{"ok": true}')]
        response.usage = None

        generation = call_llm_with_usage(
            "Claude", "key", "claude-sonnet-5-5", "prompt"
        )

        self.assertIsNone(generation.input_tokens)
        self.assertIsNone(generation.output_tokens)

    @patch("openai.OpenAI")
    def test_openrouter_connection_test_uses_selected_model(self, client_factory):
        test_connection("OpenRouter", "key", "openai/gpt-4.1-mini")
        client_factory.return_value.chat.completions.create.assert_called_once()
        request = client_factory.return_value.chat.completions.create.call_args.kwargs
        self.assertEqual(request["model"], "openai/gpt-4.1-mini")
        self.assertEqual(request["response_format"], {"type": "json_object"})


    @patch("openai.OpenAI")
    def test_openrouter_text_adapter_returns_plain_answer(self, client_factory):
        client_factory.return_value.chat.completions.create.return_value.choices = [
            SimpleNamespace(message=SimpleNamespace(content="Supported answer [S1]."))
        ]
        text = call_llm_text("OpenRouter", "key", "openai/gpt-4.1-mini", "question")
        self.assertEqual(text, "Supported answer [S1].")
        self.assertEqual(
            client_factory.return_value.chat.completions.create.call_args.kwargs["max_tokens"],
            2048,
        )


if __name__ == "__main__":
    unittest.main()
