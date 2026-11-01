import unittest

from src.credentials import persist_api_key


class ApiKeySessionStateTests(unittest.TestCase):
    def test_copies_widget_value_to_non_widget_session_state(self):
        state = {"api_key_input_Cohere.ai": "secret-value"}

        persist_api_key("Cohere.ai", state)

        self.assertEqual(state["api_key_session_Cohere.ai"], "secret-value")

    def test_clearing_widget_removes_stored_key(self):
        state = {
            "api_key_input_Cohere.ai": "",
            "api_key_session_Cohere.ai": "old-secret",
        }

        persist_api_key("Cohere.ai", state)

        self.assertNotIn("api_key_session_Cohere.ai", state)

    def test_provider_keys_are_kept_separate(self):
        state = {
            "api_key_input_Cohere.ai": "cohere-secret",
            "api_key_input_Claude": "claude-secret",
        }

        persist_api_key("Cohere.ai", state)
        persist_api_key("Claude", state)

        self.assertEqual(state["api_key_session_Cohere.ai"], "cohere-secret")
        self.assertEqual(state["api_key_session_Claude"], "claude-secret")


if __name__ == "__main__":
    unittest.main()
