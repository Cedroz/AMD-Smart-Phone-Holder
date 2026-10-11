"""Bounded JSON-line transport over a local socket."""
import json
import socket
import time

VERSION = 1
MAX_MESSAGE = 4 * 1024 * 1024


class JsonSocket:
    def __init__(self, sock):
        self.sock = sock
        self.buffer = bytearray()

    def send(self, value, timeout=5):
        data = json.dumps(value, allow_nan=False, separators=(',', ':')).encode() + b'\n'
        if len(data) > MAX_MESSAGE:
            raise ValueError('IPC message exceeds 4 MiB')
        self.sock.settimeout(timeout)
        self.sock.sendall(data)

    def receive(self, timeout=5):
        deadline = time.monotonic() + timeout
        while True:
            end = self.buffer.find(b'\n')
            if end >= 0:
                if end + 1 > MAX_MESSAGE:
                    raise ValueError('IPC message exceeds 4 MiB')
                data = bytes(self.buffer[:end])
                del self.buffer[:end + 1]
                value = json.loads(data, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
                if not isinstance(value, dict):
                    raise ValueError('IPC envelope must be an object')
                return value
            if len(self.buffer) >= MAX_MESSAGE:
                raise ValueError('IPC message exceeds 4 MiB')
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('IPC deadline expired')
            self.sock.settimeout(remaining)
            data = self.sock.recv(min(65536, MAX_MESSAGE - len(self.buffer)))
            if not data:
                raise EOFError('Vision worker connection closed')
            self.buffer.extend(data)

    def close(self):
        self.sock.close()
