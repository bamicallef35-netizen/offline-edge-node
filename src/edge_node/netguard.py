"""Air-gap enforcement.

Inside ``with netguard():`` any attempt to open an outbound connection, or
to resolve a hostname, raises ``NetworkAttempt``. That turns "this runs
offline" from a promise into something a test can check: if a dependency
tries to phone home, the demo crashes instead of quietly leaking.

It patches the process-wide socket entry points (``socket.socket.connect``,
``connect_ex``, ``socket.create_connection`` and ``socket.getaddrinfo``) and
restores them on exit. Every common HTTP client — ``urllib``, ``requests``,
``httpx`` — bottoms out in those, so they are all caught. It does not guard
against a native extension that opens sockets from C without going through
Python's ``socket`` module; nothing in this repository's default path does.
"""

from __future__ import annotations

import contextlib
import socket
from collections.abc import Iterator
from typing import Any

_LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost"}


class NetworkAttempt(RuntimeError):
    """Raised when code inside ``netguard`` tries to reach the network."""


def _host_of(address: Any) -> str | None:
    if isinstance(address, tuple) and address and isinstance(address[0], str):
        return address[0]
    if isinstance(address, str):
        return address
    return None


@contextlib.contextmanager
def netguard(*, allow_localhost: bool = False) -> Iterator[None]:
    """Block outbound network access for the duration of the block.

    ``allow_localhost=True`` permits loopback connections only (for talking
    to a local ``llama-server`` on 127.0.0.1); everything else still raises.
    """
    orig_connect = socket.socket.connect
    orig_connect_ex = socket.socket.connect_ex
    orig_create_connection = socket.create_connection
    orig_getaddrinfo = socket.getaddrinfo

    def _permitted(address: Any) -> bool:
        return allow_localhost and _host_of(address) in _LOCAL_HOSTS

    def guarded_connect(self: socket.socket, address: Any) -> None:
        if _permitted(address):
            return orig_connect(self, address)
        raise NetworkAttempt(f"outbound connect to {address!r} attempted inside netguard")

    def guarded_connect_ex(self: socket.socket, address: Any) -> int:
        if _permitted(address):
            return orig_connect_ex(self, address)
        raise NetworkAttempt(f"outbound connect_ex to {address!r} attempted inside netguard")

    def guarded_create_connection(address: Any, *args: Any, **kwargs: Any) -> socket.socket:
        if _permitted(address):
            return orig_create_connection(address, *args, **kwargs)
        raise NetworkAttempt(f"outbound connection to {address!r} attempted inside netguard")

    def guarded_getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
        if allow_localhost and host in _LOCAL_HOSTS:
            return orig_getaddrinfo(host, *args, **kwargs)
        raise NetworkAttempt(f"DNS lookup of {host!r} attempted inside netguard")

    socket.socket.connect = guarded_connect  # type: ignore[method-assign]
    socket.socket.connect_ex = guarded_connect_ex  # type: ignore[method-assign]
    socket.create_connection = guarded_create_connection
    socket.getaddrinfo = guarded_getaddrinfo
    try:
        yield
    finally:
        socket.socket.connect = orig_connect  # type: ignore[method-assign]
        socket.socket.connect_ex = orig_connect_ex  # type: ignore[method-assign]
        socket.create_connection = orig_create_connection
        socket.getaddrinfo = orig_getaddrinfo
