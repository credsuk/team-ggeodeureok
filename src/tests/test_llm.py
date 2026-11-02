import json
import unittest
from unittest.mock import patch

from src.llm import OllamaClient


class FakeResponse:
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False
    def read(self):
        return b'{"message":{"content":"{\\"ok\\":true}"}}'


class OllamaClientTests(unittest.TestCase):
    def test_sends_schema_and_keeps_local_model_warm_for_followup_stage(self):
        schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
        with patch("urllib.request.urlopen", return_value=FakeResponse()) as open_url:
            client = OllamaClient("http://localhost:11434", "local-model", keep_alive="5m")
            self.assertEqual(client.chat_json("system", "user", schema), {"ok": True})
        request = open_url.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(payload["format"], schema)
        self.assertEqual(payload["keep_alive"], "5m")
        self.assertEqual(request.full_url, "http://localhost:11434/api/chat")


if __name__ == "__main__":
    unittest.main()
