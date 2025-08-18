import express from "express";
import { connect, StringCodec } from "nats";

const app = express();
app.use(express.json());

const sc = StringCodec();
let nc: any;

(async () => {
	nc = await connect({ servers: process.env.NATS_URL || "nats://nats:4222" });
	console.log("Exo-Lattice connected to NATS");
})();

app.post("/exchange", async (req, res) => {
	await nc.publish("lb.exo.bridge", sc.encode(JSON.stringify(req.body)));
	res.json({ ok: true });
});

app.listen(8080, () => console.log("Exo-Lattice listening on 8080"));
