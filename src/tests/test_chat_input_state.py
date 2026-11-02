import unittest

from src.app import chat_input_disabled, finish_chat_prompt, queue_chat_prompt


class ChatInputStateTests(unittest.TestCase):
    def test_queuing_prompt_saves_prior_history_and_disables_input(self):
        prior = [{"role": "user", "content": f"q{i}"} for i in range(10)]
        state = {"messages": list(prior)}

        self.assertTrue(queue_chat_prompt(state, "next question"))

        self.assertEqual(state["pending_prompt"], "next question")
        self.assertEqual(state["pending_history"], prior[-8:])
        self.assertEqual(state["messages"][-1], {"role": "user", "content": "next question"})
        self.assertTrue(chat_input_disabled(state))

    def test_second_prompt_is_rejected_while_one_is_pending(self):
        state = {"messages": []}
        self.assertTrue(queue_chat_prompt(state, "first"))

        self.assertFalse(queue_chat_prompt(state, "second"))
        self.assertEqual(state["pending_prompt"], "first")
        self.assertEqual(len(state["messages"]), 1)

    def test_finishing_request_reenables_input_and_clears_pending_state(self):
        state = {"messages": []}
        queue_chat_prompt(state, "question")

        finish_chat_prompt(state)

        self.assertFalse(chat_input_disabled(state))
        self.assertIsNone(state["pending_prompt"])
        self.assertEqual(state["pending_history"], [])


if __name__ == "__main__":
    unittest.main()
