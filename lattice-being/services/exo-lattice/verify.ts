import nacl from "tweetnacl";

export function verify(pkHex: string, payload: string, sigHex: string) {
	const pk = Buffer.from(pkHex, "hex");
	const sig = Buffer.from(sigHex, "hex");
	return nacl.sign.detached.verify(Buffer.from(payload), sig, pk);
}
