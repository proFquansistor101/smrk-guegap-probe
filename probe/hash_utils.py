# hash utilities
import hashlib

def sha256_str(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()
