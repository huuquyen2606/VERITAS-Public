import hashlib


def sha256_bytes(bytez: bytes) -> str:
    h = hashlib.sha256()
    h.update(bytez)
    return h.hexdigest()
