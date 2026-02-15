import * as tf from "@tensorflow/tfjs"
import '@tensorflow/tfjs-backend-webgpu'
tf.setBackend('webgpu').catch(() => tf.setBackend('webgl'))

// ── Types ──────────────────────────────────────────────────────────────

export type ModelOpts = { n_embd?: number; n_head?: number; n_layer?: number; block_size?: number; batch?: number };

export type HeadTrace = { scores: number[]; weights: number[]; out: number[]; };

export type LayerTrace = { r1: number[]; norm1: number[]; q: number[]; k: number[]; v: number[]; cached_keys: number[][]; cached_values: number[][]; heads: HeadTrace[]; concat: number[]; wo: number[]; res1: number[]; r2: number[]; norm2: number[]; fc1: number[]; relu: number[]; fc2: number[]; res2: number[]; };

export type Trace = { token_id: number; pos_id: number; embedding: number[]; norm0: number[]; layers: LayerTrace[]; logits: number[]; probs: number[]; };

// ── Helpers ────────────────────────────────────────────────────────────

/** Xavier-ish init */
const initWeight = (shape: number[], std = 0.08): tf.Variable => tf.variable(tf.randomNormal(shape, 0, std));

/** x @ W^T  — works on [B, T, C] × [out, C] → [B, T, out] */
const linear = (x: tf.Tensor3D, w: tf.Variable): tf.Tensor3D => {
    const [B, T, C] = x.shape
    const flat = x.reshape([B * T, C])
    const out = tf.matMul(flat, w, false, true)
    return out.reshape([B, T, -1]) as tf.Tensor3D
}

/** RMS-norm (no learnable gain) */
const rmsnorm = (x: tf.Tensor, eps = 1e-5): tf.Tensor => tf.tidy(() => {
    const ms = tf.mean(tf.square(x), -1, true)
    return tf.div(x, tf.sqrt(tf.add(ms, eps)))
})

/** Causal mask: 0 for allowed positions, -1e10 for masked ones. Shape [T, T]. */
const causalMask = (T: number): tf.Tensor2D => tf.tidy(() => {
    const buf = tf.buffer([T, T], "float32")
    for (let i = 0; i < T; i++)
        for (let j = i + 1; j < T; j++) buf.set(-1e10, i, j)
    return buf.toTensor() as tf.Tensor2D
})

// ── Tokenizer ──────────────────────────────────────────────────────────

class CharTokenizer {
    readonly chars: string[]
    readonly stoi: Record<string, number> = {}
    readonly itos: Record<number, string> = {}
    readonly bos: number

    constructor(docs: string[]) {
        this.chars = [...new Set(docs.join(""))].sort()
        this.chars.forEach((c, i) => {
            this.stoi[c] = i
            this.itos[i] = c
        })
        this.bos = this.chars.length // last id = BOS/EOS
    }

    get vocabSize() { return this.chars.length + 1 }
    encode = (doc: string): number[] => [this.bos, ...doc.split("").map((c) => this.stoi[c]), this.bos];
}

// ── Layer weights ──────────────────────────────────────────────────────

interface LayerWeights {
    attn_wq: tf.Variable; attn_wk: tf.Variable; attn_wv: tf.Variable; attn_wo: tf.Variable;
    mlp_fc1: tf.Variable; mlp_fc2: tf.Variable;
}

// ── Model ──────────────────────────────────────────────────────────────

export class MicroGPT {
    // config
    n_embd: number; n_head: number; n_layer: number; block_size: number; head_dim: number; scale: number; vocab_size: number;
    // tokenizer state (public so App.tsx can read them)
    docs: string[]; uchars: string[]; stoi: Record<string, number>; itos: Record<number, string>; bos: number;
    // weights
    wte: tf.Variable; wpe: tf.Variable; lm_head: tf.Variable; layers: LayerWeights[];
    // training
    optimizer: tf.AdamOptimizer; posIdx: tf.Tensor1D; maskCache = new Map<number, tf.Tensor2D>(); tokenizedDocs: number[][] = [];
    step_count = 0;

    // ── Factory ────────────────────────────────────────────────────────

    static fromDocs(docs: string[], opts?: ModelOpts): MicroGPT {
        const tok = new CharTokenizer(docs)
        const m = new MicroGPT(tok.vocabSize, opts)
        m.docs = docs
        m.uchars = tok.chars
        m.stoi = tok.stoi
        m.itos = tok.itos
        m.bos = tok.bos
        m.tokenizedDocs = docs.map((d) => m.tokenize(d))
        return m
    }

    // ── Constructor ────────────────────────────────────────────────────

    constructor(vocabSize: number, opts?: ModelOpts) {
        this.n_embd = opts?.n_embd ?? 16
        this.n_head = opts?.n_head ?? 4
        this.n_layer = opts?.n_layer ?? 1
        this.block_size = opts?.block_size ?? 16
        this.head_dim = Math.floor(this.n_embd / this.n_head)
        this.scale = 1 / Math.sqrt(this.head_dim)
        this.vocab_size = vocabSize

        this.docs = []
        this.uchars = []
        this.stoi = {}
        this.itos = {}
        this.bos = 0

        // embeddings
        this.wte = initWeight([vocabSize, this.n_embd])
        this.wpe = initWeight([this.block_size, this.n_embd])
        this.lm_head = initWeight([vocabSize, this.n_embd])

        // transformer layers
        const E = this.n_embd
        this.layers = Array.from({ length: this.n_layer }, () => ({
            attn_wq: initWeight([E, E]), attn_wk: initWeight([E, E]), attn_wv: initWeight([E, E]), attn_wo: initWeight([E, E]),
            mlp_fc1: initWeight([4 * E, E]), mlp_fc2: initWeight([E, 4 * E]),
        }))

        this.optimizer = tf.train.adam(0.01, 0.85, 0.99, 1e-8)
        this.posIdx = tf.range(0, this.block_size, 1, "int32") as tf.Tensor1D
    }

    // ── Accessors ──────────────────────────────────────────────────────

    get num_params(): number {
        let sum = this.wte.size + this.wpe.size + this.lm_head.size;
        this.layers.forEach(L => sum += L.attn_wq.size + L.attn_wk.size + L.attn_wv.size + L.attn_wo.size + L.mlp_fc1.size + L.mlp_fc2.size);
        return sum;
    }

    get state_dict(): Record<string, tf.Variable> {
        const sd: Record<string, tf.Variable> = { wte: this.wte, wpe: this.wpe, lm_head: this.lm_head }
        this.layers.forEach((L, i) => {
            sd[`layer${i}.attn_wq`] = L.attn_wq
            sd[`layer${i}.attn_wk`] = L.attn_wk
            sd[`layer${i}.attn_wv`] = L.attn_wv
            sd[`layer${i}.attn_wo`] = L.attn_wo
            sd[`layer${i}.mlp_fc1`] = L.mlp_fc1
            sd[`layer${i}.mlp_fc2`] = L.mlp_fc2
        })
        return sd
    }

    tokenize = (doc: string): number[] => [this.bos, ...doc.split("").map((c) => this.stoi[c]), this.bos];

    /** Returns a cached causal mask for the given sequence length. Kept alive across tidy scopes. */
    private getMask(T: number): tf.Tensor2D {
        let m = this.maskCache.get(T)
        if (!m) {
            m = tf.keep(causalMask(T))
            this.maskCache.set(T, m)
        }
        return m
    }

    // ── Forward (training) ─────────────────────────────────────────────

    forward(idx: tf.Tensor2D): tf.Tensor3D {
        const [B, T] = idx.shape
        const pos = this.posIdx.slice([0], [T])

        let x = tf.add(tf.gather(this.wte, idx), tf.gather(this.wpe, pos)) as tf.Tensor3D
        x = rmsnorm(x) as tf.Tensor3D

        const mask = this.getMask(T)

        for (const L of this.layers)
            x = this.transformerBlock(x, L, B, T, mask)

        x = rmsnorm(x) as tf.Tensor3D
        return linear(x, this.lm_head)
    }

    /** Single transformer block: pre-norm attention + pre-norm FFN, both with residuals. */
    private transformerBlock(x: tf.Tensor3D, L: LayerWeights, B: number, T: number, mask: tf.Tensor2D): tf.Tensor3D {
        // ── self-attention ──
        const normed1 = rmsnorm(x) as tf.Tensor3D
        const attnOut = this.multiheadAttention(normed1, L, B, T, mask)
        const proj = linear(attnOut, L.attn_wo)
        x = tf.add(x, proj) as tf.Tensor3D

        // ── feed-forward ──
        const normed2 = rmsnorm(x) as tf.Tensor3D
        const hidden = tf.relu(linear(normed2, L.mlp_fc1))
        const ffnOut = linear(hidden as tf.Tensor3D, L.mlp_fc2)
        return tf.add(x, ffnOut) as tf.Tensor3D
    }

    /** Multi-head scaled dot-product attention. Returns [B, T, n_embd] (pre-projection). */
    private multiheadAttention(x: tf.Tensor3D, L: LayerWeights, B: number, T: number, mask: tf.Tensor2D): tf.Tensor3D {
        const { n_head, head_dim, scale } = this

        const toHeads = (t: tf.Tensor3D) =>
            tf.transpose(t.reshape([B, T, n_head, head_dim]), [0, 2, 1, 3])
        // each: [B, n_head, T, head_dim]

        const qh = toHeads(linear(x, L.attn_wq))
        const kh = toHeads(linear(x, L.attn_wk))
        const vh = toHeads(linear(x, L.attn_wv))

        // [B, n_head, T, T]
        let scores = tf.mul(tf.matMul(qh, kh, false, true), scale)
        scores = tf.add(scores, mask) // broadcast [T,T] over [B,H,T,T]
        const weights = tf.softmax(scores)

        // [B, n_head, T, head_dim] → [B, T, n_embd]
        const attended = tf.matMul(weights, vh)
        const merged = tf.transpose(attended, [0, 2, 1, 3])
        return merged.reshape([B, T, this.n_embd]) as tf.Tensor3D
    }

    // ── Forward with trace (generation / visualization) ────────────────

    forwardTrace(token_id: number, pos_id: number, ctx: number[]): Trace {
        return tf.tidy(() => {
            const tokens = [...ctx, token_id]
            const T = tokens.length
            const inp = tf.tensor2d([tokens], [1, T], "int32")
            const pos = this.posIdx.slice([0], [T])

            // helpers: extract the last-position vector or full matrix for the trace
            const vec = (t: tf.Tensor): number[] =>
                t.slice([0, T - 1, 0], [1, 1, -1]).squeeze().arraySync() as number[]
            const mat = (t: tf.Tensor): number[][] =>
                t.squeeze([0]).arraySync() as number[][]

            let x = tf.add(tf.gather(this.wte, inp), tf.gather(this.wpe, pos)) as tf.Tensor3D
            const embedding = vec(x)

            x = rmsnorm(x) as tf.Tensor3D
            const norm0 = vec(x)

            const mask = this.getMask(T)
            const layerTraces: LayerTrace[] = [];

            for (const L of this.layers) {
                const resid1 = x
                const r1 = vec(x)

                x = rmsnorm(x) as tf.Tensor3D
                const norm1 = vec(x)

                const q = linear(x, L.attn_wq), k = linear(x, L.attn_wk), v = linear(x, L.attn_wv)

                const cached_keys = mat(k), cached_values = mat(v)

                const { n_head, head_dim, scale } = this
                const toHeads = (t: tf.Tensor3D) => tf.transpose(t.reshape([1, T, n_head, head_dim]), [0, 2, 1, 3])

                const qh = toHeads(q), kh = toHeads(k), vh = toHeads(v)

                let scores = tf.mul(tf.matMul(qh, kh, false, true), scale)
                scores = tf.add(scores, mask)
                const weights = tf.softmax(scores)

                const attended = tf.matMul(weights, vh)

                const scoresArr = scores.arraySync() as number[][][][]
                const weightsArr = weights.arraySync() as number[][][][]
                const attendedArr = attended.arraySync() as number[][][][]

                const heads: HeadTrace[] = []
                for (let h = 0; h < n_head; h++)
                    heads.push({ scores: scoresArr[0][h][T - 1], weights: weightsArr[0][h][T - 1], out: attendedArr[0][h][T - 1] })

                const merged = tf.transpose(attended, [0, 2, 1, 3]).reshape([1, T, this.n_embd]) as tf.Tensor3D
                const concat = vec(merged)

                const proj = linear(merged, L.attn_wo)
                const wo = vec(proj)

                x = tf.add(resid1, proj) as tf.Tensor3D
                const res1 = vec(x)

                const resid2 = x
                const r2 = vec(x)

                x = rmsnorm(x) as tf.Tensor3D
                const norm2 = vec(x)

                const preRelu = linear(x, L.mlp_fc1)
                const fc1 = vec(preRelu)

                const postRelu = tf.relu(preRelu)
                const relu = vec(postRelu)

                const ffnOut = linear(postRelu as tf.Tensor3D, L.mlp_fc2)
                const fc2 = vec(ffnOut)

                x = tf.add(resid2, ffnOut) as tf.Tensor3D
                const res2 = vec(x)

                layerTraces.push({
                    r1, norm1,
                    q: vec(q), k: vec(k), v: vec(v),
                    cached_keys, cached_values,
                    heads, concat, wo, res1,
                    r2, norm2, fc1, relu, fc2, res2,
                })
            }

            x = rmsnorm(x) as tf.Tensor3D
            const logits = linear(x, this.lm_head)

            return { token_id, pos_id, embedding, norm0, layers: layerTraces, logits: vec(logits), probs: vec(tf.softmax(logits)) } satisfies Trace
        }) as Trace
    }

    // ── Training ───────────────────────────────────────────────────────

    /** Run a single optimizer step. Returns the loss tensor (not awaited — no GPU sync). */
    private trainStep(batchSize: number): number {
        const { block_size, tokenizedDocs, vocab_size } = this;

        // pick random pre-tokenized docs
        const batchTokens = Array.from({ length: batchSize }, () =>
            tokenizedDocs[Math.floor(Math.random() * tokenizedDocs.length)])

        const lengths = batchTokens.map((t) => Math.min(block_size + 1, t.length))
        const T = Math.max(...lengths) - 1

        const inputBuf: number[] = [], targetBuf: number[] = [], weightBuf: number[] = []

        for (let i = 0; i < batchSize; i++) {
            const toks = batchTokens[i]
            const len = lengths[i]
            const padLen = T - (len - 1)
            for (let j = 0; j < len - 1; j++) inputBuf.push(toks[j])
            for (let j = 0; j < padLen; j++) inputBuf.push(0)
            for (let j = 1; j < len; j++) targetBuf.push(toks[j])
            for (let j = 0; j < padLen; j++) targetBuf.push(0)
            for (let j = 0; j < len - 1; j++) weightBuf.push(1)
            for (let j = 0; j < padLen; j++) weightBuf.push(0)
        }

        this.step_count++;

        const loss = tf.tidy(() => {
            const inputs = tf.tensor2d(inputBuf, [batchSize, T], "int32")
            const targets = tf.tensor1d(targetBuf, "int32")
            const weights = tf.tensor1d(weightBuf, "float32")

            return this.optimizer.minimize(() => {
                const logits = this.forward(inputs)
                const logitsFlat = logits.reshape([batchSize * T, vocab_size])
                const oneHot = tf.oneHot(targets, vocab_size)
                return tf.losses.softmaxCrossEntropy(oneHot, logitsFlat, weights)
            }, true) as tf.Scalar
        })
        return loss.dataSync()[0]
    }

    trainSteps = (b = 8, length = 1) => Array.from({ length }, () => this.trainStep(b)).reduce((a, b) => a + b / length, 0)

    // ── Generation ─────────────────────────────────────────────────────

    private genTokens: number[] = []; private genToken = 0; private genPos = 0; private genChars: string[] = []
    resetGeneration() { this.genTokens = []; this.genToken = this.bos; this.genPos = 0; this.genChars = [] }
    generateStep(): { text: string, trace: Trace | undefined, done: boolean } {
        if (this.genPos === 0) this.genToken = this.bos
        if (this.genTokens.length >= this.block_size) return { text: this.genChars.join(''), trace: undefined, done: true }

        const trace = this.forwardTrace(this.genToken, this.genPos, this.genTokens)

        const probs = trace.probs
        let cum = 0, next = probs.length - 1
        const r = Math.random()
        for (let i = 0; i < probs.length; i++) { cum += probs[i]; if (r < cum) { next = i; break } }

        if (next === this.bos) return { text: this.genChars.join(''), trace, done: true }

        this.genTokens.push(this.genToken)
        this.genChars.push(this.uchars[next])
        this.genToken = next
        this.genPos++
        return { text: this.genChars.join(''), trace, done: false }
    }
}