"""不連線 LINE 的發送流程測試。"""
import argparse
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import send_weather_report as sender


class SendingTests(unittest.TestCase):
    def test_send_skip_and_confirm_resend(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            day = datetime.now(sender.TAIWAN).strftime("%Y%m%d")
            image = root / f"{day}_080000_v002.png"
            image.write_bytes(b"image")
            (root / f"{day}_090000_v001.png").write_bytes(b"older-version")
            (root / "19990101_080000_v999.png").write_bytes(b"old-date")
            subscribers = root / "subscribers.json"
            subscribers.write_text(json.dumps({"subscribers": [{"oa_basic_id": "@test", "recipient_id": "Utest"}]}))
            env = root / ".env"
            env.write_text("LINE_CHANNEL_ACCESS_TOKEN_TEST=fake-token\n")
            args = argparse.Namespace(media=root, subscribers=subscribers, env=env,
                                      base_url="https://example.com/media", history=root / "history.json",
                                      message=None, dry_run=False)
            self.assertEqual(sender.latest_image(root, day), image)
            response = Mock(status_code=200, headers={}, content=b"image")
            with patch.object(sender.requests, "get", return_value=response), patch.object(sender.requests, "post", return_value=response) as post:
                self.assertEqual(sender.run(args), 0)
                self.assertEqual(post.call_count, 1)
                state = json.loads(args.history.read_text())
                self.assertEqual(state["reports"][image.name]["attempts"][0]["status"], "accepted")
                with patch.object(sender.sys.stdin, "isatty", return_value=True), patch("builtins.input", return_value="n"):
                    self.assertEqual(sender.run(args), 0)
                self.assertEqual(post.call_count, 1)
                with patch.object(sender.sys.stdin, "isatty", return_value=True), patch("builtins.input", return_value="y"):
                    self.assertEqual(sender.run(args), 0)
                self.assertEqual(post.call_count, 2)
                with patch.object(sender.sys.stdin, "isatty", return_value=True), patch("builtins.input", return_value="y"), patch.object(sender.requests, "post", side_effect=sender.requests.Timeout):
                    self.assertEqual(sender.run(args), 1)
                state = json.loads(args.history.read_text())
                self.assertEqual(state["reports"][image.name]["attempts"][-1]["status"], "unknown")
                self.assertFalse(args.history.with_suffix(".json.lock").exists())

    def test_no_today_image(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                sender.latest_image(Path(temporary), "20261005")


if __name__ == "__main__":
    unittest.main()
