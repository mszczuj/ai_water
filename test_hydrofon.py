import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import call, patch

import hydrofon
import ollama_backend as backend


class HydrofonTests(unittest.TestCase):
    def test_amixer_command(self):
        with patch.object(hydrofon.subprocess, "run") as run:
            hydrofon.hydrofon(True)
            hydrofon.hydrofon(False)
        self.assertEqual(run.call_args_list, [
            call(["amixer", "-q", "-c", "1", "sset", "Direct Monitor", state],
                 check=True, timeout=5)
            for state in ("Mono", "Off")
        ])

    def test_overlapping_streams(self):
        controller = backend.HydrofonController()
        with patch.object(backend, "hydrofon") as switch:
            self.assertTrue(controller.begin_stream())
            self.assertTrue(controller.begin_stream())
            controller.end_stream()
            switch.assert_called_once_with(True)
            controller.end_stream()
            self.assertEqual(switch.call_args_list, [call(True), call(False)])

    def test_enable_failure_does_not_leave_active_stream(self):
        controller = backend.HydrofonController()
        with patch.object(backend, "hydrofon", side_effect=OSError("amixer")):
            with redirect_stderr(io.StringIO()):
                self.assertFalse(controller.begin_stream())
        self.assertEqual(controller.active_streams, 0)

    def test_close_turns_off(self):
        controller = backend.HydrofonController()
        with patch.object(backend, "hydrofon") as switch:
            controller.begin_stream()
            controller.close()
            self.assertEqual(switch.call_args_list, [call(True), call(False)])
        self.assertEqual(controller.active_streams, 0)

    def test_console_stream_cleanup_and_no_hardware(self):
        for controller in (None, backend.HydrofonController()):
            for data in (b'{"message":{"content":"OK"}}\n', b'invalid\n'):
                with self.subTest(controller=controller, data=data):
                    with patch.object(backend, "urlopen", return_value=io.BytesIO(data)):
                        with patch.object(backend, "hydrofon") as switch:
                            with redirect_stdout(io.StringIO()):
                                if data == b'invalid\n':
                                    with self.assertRaises(RuntimeError):
                                        self.chat(controller)
                                else:
                                    self.assertEqual(self.chat(controller), ("", "OK"))
                            expected = [] if controller is None else [call(True), call(False)]
                            self.assertEqual(switch.call_args_list, expected)

    @staticmethod
    def chat(controller):
        return backend.ollama_chat(
            backend.DEFAULT_HOST, backend.DEFAULT_MODEL, [], True, 1,
            backend.Console(False), controller,
        )

    def test_hardware_flags_are_exclusive(self):
        with patch("sys.argv", ["backend", "--hydrofon-enabled"]):
            self.assertTrue(backend.parse_args().hydrofon_enabled)
        with patch("sys.argv", ["backend", "--hydrofon-enabled", "--servo-enabled"]):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                backend.parse_args()
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
