import * as tf from "@tensorflow/tfjs"
import '@tensorflow/tfjs-backend-webgpu'
tf.setBackend('webgpu').catch(() => tf.setBackend('webgl'))

export type ModelOpts = { n_embd?: number; n_head?: number; n_layer?: number; block_size?: number };

export type HeadTrace = { scores: number[]; weights: number[]; out: number[]; };

export type LayerTrace = { r1: number[]; norm1: number[]; q: number[]; k: number[]; v: number[]; cached_keys: number[][]; cached_values: number[][]; heads: HeadTrace[]; concat: number[]; wo: number[]; res1: number[]; r2: number[]; norm2: number[]; fc1: number[]; relu: number[]; fc2: number[]; res2: number[]; };

export type Trace = { token_id: number; pos_id: number; embedding: number[]; norm0: number[]; layers: LayerTrace[]; logits: number[]; probs: number[]; };

const initWeight = (shape: number[], std = 0.08): tf.Variable => tf.variable(tf.randomNormal(shape, 0, std));

const linear = (x: tf.Tensor3D, w: tf.Variable): tf.Tensor3D =>
    tf.matMul(x.reshape([x.shape[0] * x.shape[1], x.shape[2]]), w, false, true).reshape([x.shape[0], x.shape[1], -1])

const rmsnorm = (x: tf.Tensor, eps = 1e-5): tf.Tensor => tf.div(x, tf.sqrt(tf.add(tf.mean(tf.square(x), -1, true), eps)))

const causalMask = (T: number): tf.Tensor2D => tf.tidy(() => {
    const buf = tf.buffer([T, T], "float32")
    for (let i = 0; i < T; i++)
        for (let j = i + 1; j < T; j++) buf.set(-1e10, i, j)
    return buf.toTensor() as tf.Tensor2D
})

export class CharTokenizer {
    readonly chars: string[]
    readonly bos: number

    constructor(public docs: string[]) {
        this.chars = [...new Set(docs.join(''))].sort()
        this.bos = this.chars.length
    }

    get = (D: string) => [this.bos, ...D.split('').map(a => this.chars.findIndex(c => c == a)), this.bos]
    get vocabSize() { return this.chars.length + 1 }
}

type LayerWeights = { attn_wq: tf.Variable; attn_wk: tf.Variable; attn_wv: tf.Variable; attn_wo: tf.Variable; mlp_fc1: tf.Variable; mlp_fc2: tf.Variable; }

export class MicroGPT {
    // config
    n_embd: number; n_head: number; n_layer: number; block_size: number; head_dim: number; vocab_size: number;
    // weights
    wte: tf.Variable; wpe: tf.Variable; lm_head: tf.Variable; layers: LayerWeights[];
    // training
    optimizer: tf.AdamOptimizer; posIdx: tf.Tensor1D; maskCache = new Map<number, tf.Tensor2D>();
    step_count = 0;

    static fromDocs = (docs: string[], opts?: ModelOpts) => new MicroGPT(new CharTokenizer(docs), opts)

    constructor(public tokenizer: CharTokenizer, opts?: ModelOpts) {
        this.n_embd = opts?.n_embd ?? 16
        this.n_head = opts?.n_head ?? 2
        this.n_layer = opts?.n_layer ?? 1
        this.block_size = opts?.block_size ?? 16

        this.head_dim = Math.floor(this.n_embd / this.n_head)
        this.vocab_size = this.tokenizer.vocabSize

        // embeddings
        this.wte = initWeight([this.vocab_size, this.n_embd])
        this.wpe = initWeight([this.block_size, this.n_embd])
        this.lm_head = initWeight([this.vocab_size, this.n_embd])

        // transformer layers
        const E = this.n_embd
        this.layers = Array.from({ length: this.n_layer }, () => ({
            attn_wq: initWeight([E, E]), attn_wk: initWeight([E, E]), attn_wv: initWeight([E, E]), attn_wo: initWeight([E, E]),
            mlp_fc1: initWeight([4 * E, E]), mlp_fc2: initWeight([E, 4 * E]),
        }))

        this.optimizer = tf.train.adam(0.01, 0.85, 0.99, 1e-8)
        this.posIdx = tf.range(0, this.block_size, 1, "int32") as tf.Tensor1D
    }

    num_params = () => Object.values(this.state_dict).reduce((s, v) => s + v.size, 0)

    get state_dict(): Record<string, tf.Variable> {
        const sd: Record<string, tf.Variable> = { wte: this.wte, wpe: this.wpe, lm_head: this.lm_head }
        this.layers.map((L, i) => Object.entries(L).map(([k, v]) => sd[`layer${i}.${k}`] = v))
        return sd
    }

    /** Returns a cached causal mask for the given sequence length. Kept alive across tidy scopes. */
    private getMask(T: number): tf.Tensor2D {
        let m = this.maskCache.get(T)
        if (m) return m
        m = tf.keep(causalMask(T))
        this.maskCache.set(T, m)
        return m
    }

    forward(idx: tf.Tensor2D): tf.Tensor3D
    forward(idx: tf.Tensor2D, traceIds: [number, number]): Trace
    forward(idx: tf.Tensor2D, traceIds?: [number, number]): tf.Tensor3D | Trace {
        const [B, T] = idx.shape
        const pos = this.posIdx.slice([0], [T])
        const tr = !!traceIds
        const vec = (t: tf.Tensor) => t.slice([0, T - 1, 0], [1, 1, -1]).squeeze().arraySync() as number[]
        const mat = (t: tf.Tensor) => t.squeeze([0]).arraySync() as number[][]

        let x = tf.add(tf.gather(this.wte, idx), tf.gather(this.wpe, pos)) as tf.Tensor3D
        const embedding = tr ? vec(x) : undefined
        x = rmsnorm(x) as tf.Tensor3D
        const norm0 = tr ? vec(x) : undefined

        const mask = this.getMask(T)
        const layerTraces: LayerTrace[] = []

        for (const L of this.layers) {
            const r1 = tr ? vec(x) : undefined
            const block = this.transformerBlock(x, L, B, T, mask)
            x = block.out

            if (tr) {
                const { norm1, attn, proj, mid, norm2, preRelu, postRelu, ffnOut } = block
                const [sa, wa, aa] = [attn.scores, attn.weights, attn.attended].map(t => t.arraySync() as number[][][][])

                layerTraces.push({
                    r1: r1!, norm1: vec(norm1), q: vec(attn.q), k: vec(attn.k), v: vec(attn.v),
                    cached_keys: mat(attn.k), cached_values: mat(attn.v),
                    heads: Array.from({ length: this.n_head }, (_, h) => ({ scores: sa[0][h][T - 1], weights: wa[0][h][T - 1], out: aa[0][h][T - 1] })),
                    concat: vec(attn.out), wo: vec(proj), res1: vec(mid), r2: vec(mid), norm2: vec(norm2),
                    fc1: vec(preRelu), relu: vec(postRelu), fc2: vec(ffnOut), res2: vec(block.out),
                })
            }
        }

        x = rmsnorm(x) as tf.Tensor3D
        const logits = linear(x, this.lm_head)

        if (traceIds) return {
            token_id: traceIds[0], pos_id: traceIds[1], embedding: embedding!, norm0: norm0!,
            layers: layerTraces, logits: vec(logits), probs: vec(tf.softmax(logits)),
        } satisfies Trace

        return logits
    }

    private transformerBlock(x: tf.Tensor3D, L: LayerWeights, B: number, T: number, mask: tf.Tensor2D) {
        const norm1 = rmsnorm(x) as tf.Tensor3D
        const attn = this.attention(norm1, L, B, T, mask)
        const proj = linear(attn.out, L.attn_wo)
        const mid = tf.add(x, proj) as tf.Tensor3D

        const norm2 = rmsnorm(mid) as tf.Tensor3D
        const preRelu = linear(norm2, L.mlp_fc1)
        const postRelu = tf.relu(preRelu)
        const ffnOut = linear(postRelu as tf.Tensor3D, L.mlp_fc2)
        const out = tf.add(mid, ffnOut) as tf.Tensor3D

        return { out, norm1, attn, proj, mid, norm2, preRelu, postRelu, ffnOut }
    }

    private attention(x: tf.Tensor3D, L: LayerWeights, B: number, T: number, mask: tf.Tensor2D) {
        const toHeads = (t: tf.Tensor3D) => tf.transpose(t.reshape([B, T, this.n_head, this.head_dim]), [0, 2, 1, 3])

        const q = linear(x, L.attn_wq), k = linear(x, L.attn_wk), v = linear(x, L.attn_wv)
        const qh = toHeads(q), kh = toHeads(k), vh = toHeads(v)

        let scores = tf.mul(tf.matMul(qh, kh, false, true), 1 / Math.sqrt(this.head_dim))
        scores = tf.add(scores, mask)
        const weights = tf.softmax(scores)
        const attended = tf.matMul(weights, vh)
        const merged = tf.transpose(attended, [0, 2, 1, 3])
        const out = merged.reshape([B, T, this.n_embd]) as tf.Tensor3D

        return { out, q, k, v, scores, weights, attended }
    }

    private trainStep(B: number): number {
        const randDoc = () => this.tokenizer.get(this.tokenizer.docs[Math.floor(Math.random() * this.tokenizer.docs.length)])
        const batch = Array.from({ length: B }, randDoc)
        const lengths = batch.map(t => Math.min(this.block_size + 1, t.length))
        const T = Math.max(...lengths) - 1

        const inputs: number[] = [], targets: number[] = [], weights: number[] = []
        for (let i = 0; i < B; i++) {
            const toks = batch[i], len = lengths[i], pad = T - (len - 1)
            inputs.push(...toks.slice(0, len - 1), ...Array(pad).fill(0))
            targets.push(...toks.slice(1, len), ...Array(pad).fill(0))
            weights.push(...Array(len - 1).fill(1), ...Array(pad).fill(0))
        }

        this.step_count++
        const loss = tf.tidy(() =>
            this.optimizer.minimize(() => {
                const logits = this.forward(tf.tensor2d(inputs, [B, T], "int32")).reshape([B * T, this.vocab_size])
                return tf.losses.softmaxCrossEntropy(tf.oneHot(tf.tensor1d(targets, "int32"), this.vocab_size), logits, tf.tensor1d(weights, "float32"))
            }, true) as tf.Scalar
        )
        return loss.dataSync()[0]
    }

    trainSteps = (n = 10, batch = 20) => Array.from({ length: n }, () => this.trainStep(batch)).reduce((a, b) => a + b / n, 0)

    private genToken: number | undefined; private text: string = ''
    resetGeneration() { this.genToken = this.tokenizer.bos; this.text = '' }
    generateStep(): { text: string, trace: Trace | undefined, done: boolean } {
        if (this.text.length >= this.block_size) return { text: this.text, trace: undefined, done: true }

        const tok = this.genToken || this.tokenizer.bos
        const trace = tf.tidy(() => {
            const tokens = [...this.tokenizer.get(this.text), tok]
            return this.forward(tf.tensor2d([tokens], [1, tokens.length], "int32"), [tok, this.text.length])
        }) as Trace

        const sample = (ps: number[], r = Math.random(), sum = 0) => ps.map(p => sum += p).findIndex(p => p > r)
        const next = sample(trace.probs)

        if (next == this.tokenizer.bos) return { text: this.text, trace, done: true }
        this.text += this.tokenizer.chars[next]
        this.genToken = next
        return { text: this.text, trace, done: false }
    }
}