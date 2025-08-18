from nacl.signing import SigningKey, VerifyKey
from nacl.encoding import HexEncoder


def generate_keypair():
	sk = SigningKey.generate()
	pk = sk.verify_key
	return sk.encode(encoder=HexEncoder).decode(), pk.encode(encoder=HexEncoder).decode()


def sign(sk_hex: str, message: bytes) -> str:
	sk = SigningKey(sk_hex, encoder=HexEncoder)
	signed = sk.sign(message)
	return signed.signature.hex()


def verify(pk_hex: str, message: bytes, signature_hex: str) -> bool:
	vk = VerifyKey(pk_hex, encoder=HexEncoder)
	try:
		vk.verify(message, bytes.fromhex(signature_hex))
		return True
	except Exception:
		return False
