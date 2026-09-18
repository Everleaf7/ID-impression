from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.web_demo.app import (
    AvatarGenerator,
    GeneratorConfig,
    PublicError,
    RateLimiter,
    validate_exposure,
    validate_id,
    validate_seed,
    valid_host_header,
)


class WebDemoSecurityTest(unittest.TestCase):
    def test_id_validation_accepts_mixed_visible_characters(self):
        self.assertEqual(validate_id(" 示例黑猫_928 "), "示例黑猫_928")

    def test_id_validation_rejects_empty_long_and_control_characters(self):
        for value in ("", "a" * 41, "line\nbreak", "hidden\u200bvalue"):
            with self.subTest(value=repr(value)), self.assertRaises(PublicError):
                validate_id(value)

    def test_seed_validation_is_bounded(self):
        self.assertEqual(validate_seed("42"), 42)
        for value in (-1, 2**32, True, "not-a-number"):
            with self.subTest(value=value), self.assertRaises(PublicError):
                validate_seed(value)

    def test_remote_binding_requires_explicit_flag_and_strong_token(self):
        validate_exposure("127.0.0.1", "", False)
        with self.assertRaises(PublicError):
            validate_exposure("0.0.0.0", "", False)
        with self.assertRaises(PublicError):
            validate_exposure("0.0.0.0", "short", True)
        validate_exposure("0.0.0.0", "a" * 24, True)

    def test_loopback_host_header_blocks_dns_rebinding(self):
        allowed = frozenset({"127.0.0.1:7860", "localhost:7860"})
        self.assertTrue(valid_host_header("127.0.0.1:7860", allowed))
        self.assertTrue(valid_host_header("localhost:7860", allowed))
        self.assertFalse(valid_host_header("attacker.example", allowed))
        self.assertFalse(valid_host_header("localhost:7860\r\nX-Test: value", allowed))

    def test_rate_limiter_expires_old_entries(self):
        limiter = RateLimiter(limit=2, window_seconds=10)
        self.assertTrue(limiter.allow("client", now=0))
        self.assertTrue(limiter.allow("client", now=1))
        self.assertFalse(limiter.allow("client", now=2))
        self.assertTrue(limiter.allow("client", now=11))

    def test_command_keeps_untrusted_id_in_one_argument(self):
        with tempfile.TemporaryDirectory() as temp:
            generator = AvatarGenerator(GeneratorConfig(
                backend="placeholder", gpu="0", output_root=Path(temp)
            ))
            hostile = "name; touch /tmp/not-executed"
            command = generator._command(hostile, 42, Path(temp) / "job")
            self.assertIn(hostile, command)
            self.assertNotIn("shell=True", command)


if __name__ == "__main__":
    unittest.main()
