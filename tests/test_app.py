from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from app import create_app


class AppTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = TemporaryDirectory()
        self.reference_file = Path(self.temp_dir.name) / "高難易度.txt"
        self.reference_file.write_text(
            "\n".join(
                [
                    "医師: この治療で必ず良くなるとは言い切れません。",
                    "理想回答: ご説明ありがとうございます。不確実性を理解したうえで、患者さんへどう伝えるのがよいか確認させてください。",
                    "",
                    "医師: 副作用が心配です。",
                    "理想回答: 副作用の可能性を踏まえて、注意点と対処法を整理してご案内します。",
                ]
            ),
            encoding="utf-8",
        )
        app = create_app({"TESTING": True, "REFERENCE_FILE": self.reference_file})
        self.client = app.test_client()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_index_page_is_available(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("面会返答アシスタント", response.get_data(as_text=True))

    def test_chat_returns_best_matching_reference_response(self) -> None:
        response = self.client.post(
            "/api/chat",
            json={"message": "この治療で必ず良くなるとは言い切れません、と言われました"},
        )
        payload = response.get_json()

        self.assertEqual(response.status_code, 200)
        self.assertIn("ご説明ありがとうございます。", payload["reply"])
        self.assertGreater(payload["score"], 0)

    def test_chat_requires_message(self) -> None:
        response = self.client.post("/api/chat", json={"message": "  "})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["error"], "message is required")


if __name__ == "__main__":
    unittest.main()
