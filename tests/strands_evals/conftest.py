import _thread
import signal

import pytest


class Interrupted(BaseException):
    """What SIGINT raises under the `sigint` fixture."""


class _Sigint:
    error = Interrupted

    def __init__(self):
        self.active = True

    def send(self):
        """Flag SIGINT for the main thread, as `_thread.interrupt_main()` does; a no-op once the test is over."""
        if self.active:
            _thread.interrupt_main()


@pytest.fixture
def sigint():
    """While the test runs, SIGINT raises `sigint.error` instead of `KeyboardInterrupt`, so one that arrives late
    fails only this test instead of stopping the session. `sigint.send()` flags it without waking a blocked wait,
    so the main thread sees it the next time it runs, on every platform."""
    sender = _Sigint()

    def handler(signum, frame):
        raise Interrupted

    previous = signal.signal(signal.SIGINT, handler)
    try:
        yield sender
    finally:
        sender.active = False
        signal.signal(signal.SIGINT, previous)
