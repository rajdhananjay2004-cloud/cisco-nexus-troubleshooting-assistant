import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import app


class RagBehaviorTests(unittest.TestCase):
    def test_retriever_uses_similarity_and_five_results(self):
        vectorstore = Mock()
        app.build_retriever(vectorstore)
        vectorstore.as_retriever.assert_called_once_with(
            search_type="similarity",
            search_kwargs={"k": 5},
        )

    def test_blank_question_does_not_call_retriever(self):
        retriever = Mock()
        llm = Mock()
        self.assertEqual(
            app.answer_question("  ", retriever, llm),
            "Please enter a question.",
        )
        retriever.invoke.assert_not_called()
        llm.invoke.assert_not_called()

    def test_answer_synthesizes_context_and_includes_model_page_reference(self):
        retriever = Mock()
        retriever.invoke.return_value = [
            SimpleNamespace(
                page_content="The documented troubleshooting step.",
                metadata={"page": 6},
            )
        ]
        llm = Mock()
        llm.invoke.return_value = SimpleNamespace(
            content="Start with the documented troubleshooting step. [Page 7]"
        )

        result = app.answer_question("What is the step?", retriever, llm)

        self.assertEqual(
            result,
            "Start with the documented troubleshooting step. [Page 7]",
        )
        messages = llm.invoke.call_args.args[0]
        self.assertIn("The documented troubleshooting step.", messages[1].content)
        self.assertIn("synthesize relevant", messages[0].content)
        self.assertIn(
            "Summarize process details in your own words",
            messages[0].content,
        )
        self.assertIn("Using Diagnostics", messages[0].content)
        self.assertIn("omit upgrade-recovery steps", messages[0].content)
        self.assertIn("[Page 7]", messages[1].content)
        self.assertIn(
            "Answer using only this retrieved context.",
            messages[1].content,
        )
        self.assertNotIn(
            "Return only short, exact source lines",
            messages[1].content,
        )

    def test_retrieval_context_is_limited_to_five_documents(self):
        retriever = Mock()
        retriever.invoke.return_value = [
            SimpleNamespace(
                page_content=f"Unique retrieved passage number {number}.",
                metadata={"page": number},
            )
            for number in range(6)
        ]
        llm = Mock()
        llm.invoke.return_value = SimpleNamespace(content="Answer. [Page 1]")

        app.answer_question("Question?", retriever, llm)

        prompt = llm.invoke.call_args.args[0][1].content
        self.assertIn("Unique retrieved passage number 4.", prompt)
        self.assertNotIn("Unique retrieved passage number 5.", prompt)

    def test_empty_retrieval_reports_insufficient_evidence(self):
        retriever = Mock()
        retriever.invoke.return_value = []
        llm = Mock()
        result = app.answer_question("Unsupported question", retriever, llm)
        self.assertIn("does not provide sufficient information", result)
        llm.invoke.assert_not_called()

    def test_empty_model_response_is_reported(self):
        retriever = Mock()
        retriever.invoke.return_value = [
            SimpleNamespace(page_content="Some context.", metadata={"page": 0})
        ]
        llm = Mock()
        llm.invoke.return_value = SimpleNamespace(content="  ")

        result = app.answer_question("Question?", retriever, llm)

        self.assertIn("model returned an empty answer", result)

    def test_general_answer_omits_upgrade_only_recommendations(self):
        retriever = Mock()
        retriever.invoke.return_value = [
            SimpleNamespace(
                page_content="Troubleshooting context.",
                metadata={"page": 0},
            )
        ]
        llm = Mock()
        llm.invoke.return_value = SimpleNamespace(
            content=(
                "Check the documented symptoms. [Page 1]\n"
                "Use Diagnostics. [Page 3]\n"
                "Recover from switch upgrade failures. [Page 2]"
            )
        )

        answer = app.answer_question(
            "How should I investigate a recurring issue?",
            retriever,
            llm,
        )

        self.assertIn("documented symptoms", answer)
        self.assertNotIn("diagnostic", answer.casefold())
        self.assertNotIn("upgrade", answer.casefold())

    def test_diagnostics_answer_keeps_requested_diagnostic_information(self):
        retriever = Mock()
        retriever.invoke.return_value = [
            SimpleNamespace(page_content="Using Diagnostics.", metadata={"page": 2})
        ]
        llm = Mock()
        llm.invoke.return_value = SimpleNamespace(
            content="Use Diagnostics. [Page 3]"
        )

        answer = app.answer_question(
            "What does the guide say about diagnostics?",
            retriever,
            llm,
        )

        self.assertIn("Diagnostics", answer)

    def test_upgrade_question_keeps_upgrade_recovery_recommendation(self):
        retriever = Mock()
        retriever.invoke.return_value = [
            SimpleNamespace(
                page_content="Upgrade troubleshooting context.",
                metadata={"page": 0},
            )
        ]
        llm = Mock()
        llm.invoke.return_value = SimpleNamespace(
            content="Recover from switch upgrade failures. [Page 1]"
        )

        answer = app.answer_question(
            "How do I troubleshoot an upgrade failure?",
            retriever,
            llm,
        )

        self.assertIn("upgrade failures", answer.casefold())

    def test_history_is_rendered_and_can_be_cleared(self):
        rendered = app.render_history([("Question?", "Answer.")])
        self.assertIn("Question?", rendered)
        self.assertIn("Answer.", rendered)
        self.assertEqual(
            app.clear_history(),
            ([], [], "Your questions and answers will appear here."),
        )

    def test_question_submission_updates_chat_and_session_history(self):
        retriever = Mock()
        llm = Mock()
        with (
            patch.object(app, "retriever", retriever),
            patch.object(app, "llm", llm),
            patch.object(app, "answer_question", return_value="Grounded answer."),
        ):
            result = app.submit_question("  What now?  ", [], [])

        self.assertEqual(result[0], "")
        self.assertEqual(
            result[1],
            [
                {"role": "user", "content": "What now?"},
                {"role": "assistant", "content": "Grounded answer."},
            ],
        )
        self.assertEqual(result[2], [("What now?", "Grounded answer.")])
        self.assertIn("Grounded answer.", result[3])


if __name__ == "__main__":
    unittest.main()
