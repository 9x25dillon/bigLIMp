import express from "express";
import { connect, StringCodec } from "nats";
import { verify } from "./verify";

const app = express();
app.use(express.json());

const sc = StringCodec();
let nc: any;

(async () => {
	nc = await connect({ servers: process.env.NATS_URL || "nats://nats:4222" });
	console.log("Exo-Lattice connected to NATS");
})();

app.post("/exchange", async (req, res) => {
	const { envelope, signature, publicKey } = req.body || {};
	const payload = JSON.stringify(envelope || {});


	if (!publicKey || !signature || !verify(publicKey, payload, signature)) {
		return res.status(400).json({ ok: false, error: "bad signature" });
	}
	await nc.publish("lb.exo.bridge", sc.encode(payload));
	res.json({ ok: true });
});


app.listen(8080, () => console.log("Exo-Lattice listening on 8080"));

app.listen(8080, () => console.log("Exo-Lattice listening on 8080"));

