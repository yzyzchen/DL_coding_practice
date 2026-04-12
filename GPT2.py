import numpy as np

def gen_text(prompt: str, n_tokens_to_generate: int = 40):
	# Your code here
	encoder, hparams, params = load_encoder_hparams_and_params()
	tok_id = encoder.encode(prompt)
	gen = []
	for _ in range(n_tokens_to_generate):
		logits = gpt2(tok_id, hparams, params)[-1]
		next_token_id = int(np.argmax(logits))
		tok_id.append(next_token_id)
		gen.append(next_token_id)
	return encoder.decode(gen)

def gpt2(tok_id, hparams, params):
	nheads = hparams["n_head"]
	n = len(tok_id)
	emb = params["wte"][tok_id] + params["wpe"][:n]
	for block_params in params["blocks"]:
		pred = block(emb, block_params, nheads)
		pred = layernorm(pred, params["ln_f"]["g"], params["ln_f"]["b"])
	logits = pred @ params["wte"].T
	return logits

def softmax(z):
	exp_z = np.exp(z - np.max(z, axis = -1, keepdims = True))
	return exp_z / np.sum(exp_z, axis = -1, keepdims = True)

def layernorm(z, g, b):
	mean = np.mean(z, axis = -1, keepdims = True)
	var = np.var(z, axis = -1, keepdims = True)
	normalized = (z - mean) / np.sqrt(var + 1e-6)
	return g * normalized + b

def GELU(z):
	return 0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z ** 3)))

def linear(z, c):
	return z @ c["w"] + c["b"]

def FFN(z, c_fc, c_proj):
	return GELU(linear(z, c_fc)) @ c_proj["w"] + c_proj["b"]

def attention(Q, K, V, mask):
	score = Q @ K.T / np.sqrt(Q.shape[-1]) + mask
	attn_weight = softmax(score)
	return attn_weight @ V

def MHA(x, attn_parmas, nheads):
	W, b = attn_parmas["c_attn"].values()
	Wo, bo = attn_parmas["c_proj"].values()
	qkv = x @ W + b
	Q, K, V = np.split(qkv, 3, axis=-1) 
	n, d = Q.shape
	d_head = d // nheads
	mask = np.triu(np.ones((n, n)), k=1)
	mask = np.where(mask == 1, -np.inf, 0)
	ans = []
	for i in range(0, d, d_head):
		Qi = Q[:, i:i+d_head]
		Ki = K[:, i:i+d_head]
		Vi = V[:, i:i+d_head]
		post_attn = attention(Qi, Ki, Vi, mask)
		ans.append(post_attn)
	ans = np.concatenate(ans, axis = -1)
	return ans @ Wo + bo

def block(x, block_params, nheads):
	x1 = x + MHA(layernorm(x, **block_params["ln_1"]), block_params["attn"], nheads)
	x2 = x1 + FFN(layernorm(x1, **block_params["ln_2"]), block_params["mlp"]["c_fc"], block_params["mlp"]["c_proj"])
	return x2

def load_encoder_hparams_and_params(model_size: str = "124M", models_dir: str = "models"):
	class DummyBPE:
		def __init__(self):
			self.encoder_dict = {"hello": 1, "world": 2, "<UNK>": 0}

		def encode(self, text: str):
			tokens = text.strip().split()
			return [self.encoder_dict.get(token, self.encoder_dict["<UNK>"]) for token in tokens]

		def decode(self, token_ids: list):
			reversed_dict = {v: k for k, v in self.encoder_dict.items()}
			return " ".join([reversed_dict.get(tok_id, "<UNK>") for tok_id in token_ids])

	hparams = {
		"n_ctx": 1024,
		"n_head": 2
	}

	params = {
		"wte": np.random.rand(3, 10),
		"wpe": np.random.rand(1024, 10),
		"blocks": [{
			"mlp": {
				"c_fc": {"w": np.random.rand(10, 20), "b": np.random.rand(20)},
				"c_proj": {"w": np.random.rand(20, 10), "b": np.random.rand(10)}
			},
			"attn": {
				"c_attn": {"w": np.random.rand(10, 30), "b": np.random.rand(30)},
				"c_proj": {"w": np.random.rand(10, 10), "b": np.random.rand(10)}
			},
			"ln_1": {"g": np.ones(10), "b": np.zeros(10)},
			"ln_2": {"g": np.ones(10), "b": np.zeros(10)},
		}],
		"ln_f": {
			"g": np.ones(10),
			"b": np.zeros(10),
		}
	}

	encoder = DummyBPE()
	return encoder, hparams, params

np.random.seed(42)
print(gen_text("hello", n_tokens_to_generate=5))