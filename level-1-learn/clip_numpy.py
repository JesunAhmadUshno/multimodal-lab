"""NumPy CLIP-style dual encoder for P22 Learn.

Synthetic teaching data only: coloured shapes rendered as arrays,
each paired with a short caption token sequence. This is not public
data and no result from this module is a public-data result.

Everything is hand-written: forward pass, contrastive (InfoNCE) loss
in both directions, backpropagation, and an Adam optimiser. NumPy is
the only dependency. All randomness comes from fixed seeds.
"""
import numpy as np

MASTER_SEED = 42
IMAGE_SIZE = 16
EMBED_DIM = 32
HIDDEN_DIM = 128
TEXT_DIM = 24
TEMPERATURE = 0.1

COLOURS = ["red", "green", "blue", "yellow"]
COLOUR_RGB = {
    "red": (0.90, 0.15, 0.15),
    "green": (0.15, 0.80, 0.25),
    "blue": (0.20, 0.35, 0.95),
    "yellow": (0.90, 0.80, 0.15),
}
SHAPES = ["circle", "square", "triangle"]
VOCAB = ["<pad>", "a", "red", "green", "blue", "yellow",
         "circle", "square", "triangle"]
TOKEN = {word: i for i, word in enumerate(VOCAB)}
BACKGROUND = 0.05


def render_image(colour, shape, rng):
    """Render one synthetic image: one coloured shape on dark ground."""
    img = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), BACKGROUND, dtype=np.float64)
    cx = 7.5 + int(rng.integers(-3, 4))
    cy = 7.5 + int(rng.integers(-3, 4))
    size = int(rng.integers(4, 6))
    rgb = np.array(COLOUR_RGB[colour], dtype=np.float64)
    ys, xs = np.mgrid[0:IMAGE_SIZE, 0:IMAGE_SIZE]
    if shape == "circle":
        mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= size ** 2
    elif shape == "square":
        mask = (np.abs(xs - cx) <= size) & (np.abs(ys - cy) <= size)
    elif shape == "triangle":
        in_rows = (ys >= cy - size) & (ys <= cy + size)
        half = np.where(in_rows, (ys - (cy - size)) / 2.0, -1.0)
        mask = in_rows & (np.abs(xs - cx) <= half)
    else:
        raise ValueError("unknown shape: " + shape)
    img[mask] = rgb
    return img


def caption_tokens(colour, shape):
    return np.array([TOKEN["a"], TOKEN[colour], TOKEN[shape]],
                    dtype=np.int64)


def make_pairs(n_per_combo, seed_offset):
    """Build paired synthetic data, deterministic in seed_offset.

    Returns images (N, 768), tokens (N, 3), combo ids (N,) where a
    combo id is colour_index * 3 + shape_index over the 12 combos.
    """
    images, tokens, combos = [], [], []
    idx = 0
    for ci, colour in enumerate(COLOURS):
        for si, shape in enumerate(SHAPES):
            for _ in range(n_per_combo):
                rng = np.random.default_rng(seed_offset + idx)
                images.append(render_image(colour, shape, rng).reshape(-1))
                tokens.append(caption_tokens(colour, shape))
                combos.append(ci * len(SHAPES) + si)
                idx += 1
    return (np.stack(images), np.stack(tokens), np.array(combos))


def init_params(seed=MASTER_SEED):
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.normal(0.0, np.sqrt(2.0 / 768), (768, HIDDEN_DIM)),
        "b1": np.zeros(HIDDEN_DIM),
        "W2": rng.normal(0.0, np.sqrt(2.0 / HIDDEN_DIM),
                         (HIDDEN_DIM, EMBED_DIM)),
        "b2": np.zeros(EMBED_DIM),
        "E": rng.normal(0.0, 0.5, (len(VOCAB), TEXT_DIM)),
        "Wt": rng.normal(0.0, np.sqrt(2.0 / TEXT_DIM),
                         (TEXT_DIM, EMBED_DIM)),
        "bt": np.zeros(EMBED_DIM),
    }


def _normalise_rows(z):
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    return z / norms, norms


def encode_image(params, images):
    h_pre = images @ params["W1"] + params["b1"]
    h = np.maximum(h_pre, 0.0)
    z_pre = h @ params["W2"] + params["b2"]
    z, _ = _normalise_rows(z_pre)
    return z


def encode_text(params, tokens):
    emb = params["E"][tokens]
    mask = (tokens != TOKEN["<pad>"]).astype(np.float64)
    counts = mask.sum(axis=1, keepdims=True)
    pooled = (emb * mask[:, :, None]).sum(axis=1) / counts
    z_pre = pooled @ params["Wt"] + params["bt"]
    z, _ = _normalise_rows(z_pre)
    return z


def contrastive_loss(img_emb, txt_emb, temperature=TEMPERATURE):
    """Two-direction InfoNCE over matched pairs. Pure function."""
    n = img_emb.shape[0]
    logits = (img_emb @ txt_emb.T) / temperature
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    row_softmax = exp / exp.sum(axis=1, keepdims=True)
    ce_row = -np.log(row_softmax[np.arange(n), np.arange(n)]).mean()
    logits_t = (txt_emb @ img_emb.T) / temperature
    logits_t = logits_t - logits_t.max(axis=1, keepdims=True)
    exp_t = np.exp(logits_t)
    col_softmax = exp_t / exp_t.sum(axis=1, keepdims=True)
    ce_col = -np.log(col_softmax[np.arange(n), np.arange(n)]).mean()
    return float(0.5 * (ce_row + ce_col))


def loss_and_grads(params, images, tokens, temperature=TEMPERATURE):
    n = images.shape[0]
    # Image forward, keeping intermediates.
    h_pre = images @ params["W1"] + params["b1"]
    h = np.maximum(h_pre, 0.0)
    img_pre = h @ params["W2"] + params["b2"]
    img_emb, img_norms = _normalise_rows(img_pre)
    # Text forward, keeping intermediates.
    emb = params["E"][tokens]
    mask = (tokens != TOKEN["<pad>"]).astype(np.float64)
    counts = mask.sum(axis=1, keepdims=True)
    pooled = (emb * mask[:, :, None]).sum(axis=1) / counts
    txt_pre = pooled @ params["Wt"] + params["bt"]
    txt_emb, txt_norms = _normalise_rows(txt_pre)

    loss = contrastive_loss(img_emb, txt_emb, temperature)

    # Gradient through the two softmax cross entropies.
    logits = (img_emb @ txt_emb.T) / temperature
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    p_row = exp / exp.sum(axis=1, keepdims=True)
    shifted_c = logits - logits.max(axis=0, keepdims=True)
    exp_c = np.exp(shifted_c)
    p_col = exp_c / exp_c.sum(axis=0, keepdims=True)
    eye = np.eye(n)
    dlogits = 0.5 * ((p_row - eye) + (p_col - eye)) / n
    d_img_emb = (dlogits @ txt_emb) / temperature
    d_txt_emb = (dlogits.T @ img_emb) / temperature

    # Back through row normalisation: dz_pre = (dz - z * <dz, z>) / norm
    d_img_pre = (d_img_emb - img_emb * (d_img_emb * img_emb).sum(
        axis=1, keepdims=True)) / img_norms
    d_txt_pre = (d_txt_emb - txt_emb * (d_txt_emb * txt_emb).sum(
        axis=1, keepdims=True)) / txt_norms

    grads = {}
    grads["W2"] = h.T @ d_img_pre
    grads["b2"] = d_img_pre.sum(axis=0)
    dh = (d_img_pre @ params["W2"].T) * (h_pre > 0)
    grads["W1"] = images.T @ dh
    grads["b1"] = dh.sum(axis=0)

    grads["Wt"] = pooled.T @ d_txt_pre
    grads["bt"] = d_txt_pre.sum(axis=0)
    d_pooled = d_txt_pre @ params["Wt"].T
    d_emb = (d_pooled / counts)[:, None, :] * mask[:, :, None]
    d_e = np.zeros_like(params["E"])
    np.add.at(d_e, tokens.reshape(-1),
              d_emb.reshape(-1, TEXT_DIM))
    grads["E"] = d_e
    return loss, grads


class Adam:
    def __init__(self, params, lr=0.01, beta1=0.9, beta2=0.999,
                 eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for k in params:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * grads[k] ** 2
            m_hat = self.m[k] / (1 - self.b1 ** self.t)
            v_hat = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return params


def train(params, images, tokens, epochs=2500, lr=0.01):
    """Minibatch training, one pair per distinct caption per step.

    Batches are built by grouping rows by their exact token
    sequence and cycling through each group in order, so every
    batch holds unique captions and the InfoNCE target is well
    posed (see the design revision in LEARN-SPEC.md). The epochs
    argument counts steps. Returns (params, first_loss, last_loss),
    where the losses are the minibatch losses at the first and last
    step, each measured before that step's update.
    """
    groups = {}
    order = []
    for i in range(tokens.shape[0]):
        key = tuple(tokens[i].tolist())
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(i)
    group_idx = [groups[key] for key in order]
    opt = Adam(params, lr=lr)
    first_loss = None
    last_loss = None
    for step in range(epochs):
        batch = np.array([g[step % len(g)] for g in group_idx])
        loss, grads = loss_and_grads(params, images[batch],
                                     tokens[batch])
        if first_loss is None:
            first_loss = loss
        last_loss = loss
        opt.step(params, grads)
    return params, first_loss, last_loss


def recall_at_k(img_emb, txt_emb, combos, k, direction):
    """Combination-match recall: a hit when any top-k item shares the
    query combination. The rule is fixed in LEARN-SPEC.md."""
    sims = img_emb @ txt_emb.T
    if direction == "text_to_image":
        sims = sims.T
    order = np.argsort(-sims, axis=1)
    hits = 0
    for i in range(sims.shape[0]):
        top = order[i, :k]
        if np.any(combos[top] == combos[i]):
            hits += 1
    return hits / sims.shape[0]


def evaluate(params, images, tokens, combos):
    img_emb = encode_image(params, images)
    txt_emb = encode_text(params, tokens)
    return {
        "image_to_text_recall_at_1": recall_at_k(
            img_emb, txt_emb, combos, 1, "image_to_text"),
        "image_to_text_recall_at_5": recall_at_k(
            img_emb, txt_emb, combos, 5, "image_to_text"),
        "text_to_image_recall_at_1": recall_at_k(
            img_emb, txt_emb, combos, 1, "text_to_image"),
        "text_to_image_recall_at_5": recall_at_k(
            img_emb, txt_emb, combos, 5, "text_to_image"),
    }
