"""The signature test: inside the guard, nothing reaches the network."""

import socket
import urllib.request

import pytest

from edge_node import NetworkAttempt, netguard


def test_create_connection_is_blocked() -> None:
    with netguard(), pytest.raises(NetworkAttempt):
        socket.create_connection(("example.com", 80), timeout=1)


def test_raw_socket_connect_is_blocked() -> None:
    with netguard(), pytest.raises(NetworkAttempt):
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("1.1.1.1", 53))


def test_dns_lookup_is_blocked() -> None:
    with netguard(), pytest.raises(NetworkAttempt):
        socket.getaddrinfo("example.com", 443)


def test_urllib_cannot_phone_home() -> None:
    with netguard(), pytest.raises(NetworkAttempt):
        urllib.request.urlopen("http://example.com", timeout=1)


def test_localhost_is_blocked_by_default_but_can_be_allowed() -> None:
    with netguard(), pytest.raises(NetworkAttempt):
        socket.getaddrinfo("localhost", 8080)
    with netguard(allow_localhost=True):
        assert socket.getaddrinfo("localhost", 8080)      # resolves without raising
        with pytest.raises(NetworkAttempt):
            socket.getaddrinfo("example.com", 80)         # the outside world still does not


def test_the_guard_is_lifted_afterwards() -> None:
    before = (socket.socket.connect, socket.create_connection, socket.getaddrinfo)
    with netguard():
        assert socket.create_connection is not before[1]
    assert (socket.socket.connect, socket.create_connection, socket.getaddrinfo) == before
