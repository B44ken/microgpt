import * as tf from '@tensorflow/tfjs'

type StateDict = Record<string, tf.Variable>

// returns flat array of .data from a weight matrix (first 16 rows flattened)
const flatWeights = (sd: StateDict, key: string, maxRows = 16, maxCols = 64): number[] | undefined => {
    const t = sd[key]
    if (!t) return undefined

    const rows = Math.min(t.shape[0] || 0, maxRows)
    const cols = Math.min(t.shape[1] || 0, maxCols)
    const res: number[] = []

    const data = t.dataSync() as Float32Array
    const fullCols = t.shape[1] || 0

    for (let i = 0; i < rows; i++)
        for (let j = 0; j < cols; j++)
            res.push(data[i * fullCols + j])
    return res
}

const valColor = (v: number, min: number, max: number) => {
    if (v >= 0) {
        const t = max > 0 ? v / max : 0
        return `rgb(${187 * t}, ${204 * t}, ${255 * t})`
    }
    const t = min < 0 ? v / min : 0
    return `rgb(${238 * t}, ${68 * t}, ${68 * t})`
}

const WeightGrid = ({ data, label, rows = 16, cols = 16 }: { data: number[], label: string, rows?: number, cols?: number }) => {
    const min = Math.min(...data, -1), max = Math.max(...data, 1)
    return <div className='mt-2'>
        <div className='text-xs opacity-75 mb-1'>{label} [{rows}×{cols}]</div>
        <div style={{ display: 'grid', gridTemplateColumns: `repeat(${cols}, 6px)`, gap: 0 }}>
            {data.map((v, i) => <div key={i} title={v.toFixed(4)} style={{ width: 6, height: 6, background: valColor(v, min, max) }} />)}
        </div>
    </div>
}

type Entry = { text: string, io: string, weights?: string[] }

const explain: Record<string, Entry> = {
    'input': {
        io: 'in: int [1] → out: int [1]',
        text: 'the input is 0-26, where 0-25 are letters and 26 is special, meaning beginning/end',
    }, 'embed': {
        text: 'W_token[i] is a 16-dim vector encoding the character at position i. W_pos[i] is a 16-dim vector encoding the position of the character at position i. adding them produces a [16]-dim vector encoding both identity and position.',
        io: 'in: int [1] → out: float [16]',
        weights: ['wte', 'wpe'],
    }, 'norm0': {
        text: 'a rescale step. see faq for details.',
        io: 'in: float [16] → out: float [16]',
        weights: [],
    }, 'transformer': {
        text: 'the transformer block is the core building block. it contains multi-head self-attention (to mix information across positions) followed by a feed-forward mlp (to process each position independently). residual connections around both let gradients flow directly backward.',
        io: 'in: float [16] → out: float [16]',
    }, 'r1': {
        text: 'save a copy of x before attention. after attention transforms x, we add this copy back (residual connection). this lets the model preserve information that might be lost during attention.',
        io: 'in: float [16] → out: float [16] (identity copy)',
        weights: [],
    }, 'norm1': {
        text: 'a rescale step. see faq for details.',
        io: 'in: float [16] → out: float [16]',
        weights: [],
    }, 'qkv': {
        text: 'three parallel linear projections produce query, key, and value vectors. you can think about this as defining "looking for" (Q), what it "contains" (K), and what information it "provides" (V). for example, where Q · K is high, Q[i] is looking for and has successfully found whatever K[i] is. then, V[i] is the information itself.',
        io: 'in: float [16] → out: 3× float [16]',
        weights: ['layer0.attn_wq', 'layer0.attn_wk', 'layer0.attn_wv'],
    }, 'q': {
        text: 'query projection: W_Q · x. represents "what am I looking for?" split into 4 heads of [4] dims each. compared against cached keys via scaled dot-product to determine attention scores.',
        io: 'in: float [16] → out: float [16]',
        weights: ['layer0.attn_wq'],
    }, 'k': {
        text: 'key projection: W_K · x. represents "what do I contain?" stored in the KV cache so future positions can attend back to this one. each head uses a [4]-dim slice.',
        io: 'in: float [16] → out: float [16]',
        weights: ['layer0.attn_wk'],
    }, 'v': {
        text: 'value projection: W_V · x. represents "what information do i give?" once attention decides how much to attend to each position, we weighted sum all the V\'s to produce the output.',
        io: 'in: float [16] → out: float [16]',
        weights: ['layer0.attn_wv'],
    }, 'kv_cache': {
        text: 'the KV cache stores all previous keys and values. at position i, we have i key vectors and i value vectors. this is what makes autoregressive generation O(n) per step instead of O(n²) — we reuse previous computations.', io: 'in: float [16] (new k,v) → out: float [i × 16] (all cached)',
    }, 'cache_k': {
        text: 'the key cache accumulates key vectors from all previous positions. after generating i tokens, this is an [i × 16] matrix. each head\'s [4]-dim slice is compared against the current query to compute attention scores.', io: 'in: float [16] → appended to: float [i × 16]',
    }, 'cache_v': {
        text: 'the value cache accumulates value vectors from all previous positions. after generating i tokens, these cached values are weighted and summed to produce the attention output.', io: 'in: float [16] → appended to: float [i × 16]',
    }, 'heads': {
        text: 'multi-head attention splits the [16]-dim space into 4 independent [4]-dim heads. each head can learn different patterns (e.g., one might attend to recent tokens, another to similar characters). results are concatenated back to [16].', io: 'in: q[16], k_cache[i×16], v_cache[i×16] → out: float [16]',
    }, 'head0': { text: 'head 0 operates on dims [0:3]. it has its own independent pattern, learning which previous positions are relevant for this 4-dimensional subspace.', io: 'in: q[0:3], k_cache[i×4], v_cache[i×4] → out: float [4]' },
    'head1': { text: 'head 1 operates on dims [4:7]. independent from head 0, it can specialize in a different type of relationship between positions.', io: 'in: q[4:7], k_cache[i×4], v_cache[i×4] → out: float [4]' },
    'head2': { text: 'head 2 operates on dims [8:11]. a third independent pattern over its own subspace of the embedding.', io: 'in: q[8:11], k_cache[i×4], v_cache[i×4] → out: float [4]' },
    'head3': { text: 'head 3 operates on dims [12:15]. the fourth and final head, covering the last 4 dimensions of the embedding space.', io: 'in: q[12:15], k_cache[i×4], v_cache[i×4] → out: float [4]' },
    'scores0': { text: 'raw dot products between this head\'s query slice and all cached key slices, scaled by 1/√d. higher scores mean stronger similarity between the current query and a past key.', io: 'in: q[4], k_cache[i×4] → out: float [i]' },
    'scores1': { text: 'raw dot products between this head\'s query slice and all cached key slices, scaled by 1/√d. higher scores mean stronger similarity between the current query and a past key.', io: 'in: q[4], k_cache[i×4] → out: float [i]' },
    'scores2': { text: 'raw dot products between this head\'s query slice and all cached key slices, scaled by 1/√d. higher scores mean stronger similarity between the current query and a past key.', io: 'in: q[4], k_cache[i×4] → out: float [i]' },
    'scores3': { text: 'raw dot products between this head\'s query slice and all cached key slices, scaled by 1/√d. higher scores mean stronger similarity between the current query and a past key.', io: 'in: q[4], k_cache[i×4] → out: float [i]' },
    'weights0': { text: 'softmax normalizes the raw scores into a probability distribution over all positions. each weight is between 0 and 1, and they sum to 1. this determines how much each past position contributes to the output.', io: 'in: float [i] → out: float [i] (sums to 1)' },
    'weights1': { text: 'softmax normalizes the raw scores into a probability distribution over all positions. each weight is between 0 and 1, and they sum to 1. this determines how much each past position contributes to the output.', io: 'in: float [i] → out: float [i] (sums to 1)' },
    'weights2': { text: 'softmax normalizes the raw scores into a probability distribution over all positions. each weight is between 0 and 1, and they sum to 1. this determines how much each past position contributes to the output.', io: 'in: float [i] → out: float [i] (sums to 1)' },
    'weights3': { text: 'softmax normalizes the raw scores into a probability distribution over all positions. each weight is between 0 and 1, and they sum to 1. this determines how much each past position contributes to the output.', io: 'in: float [i] → out: float [i] (sums to 1)' },
    'attn_out0': { text: 'weighted sum of cached value vectors using the attention weights. this is the "output" of this attention head - a [4]-dim vector that combines information from all attended positions.', io: 'in: weights[i], v_cache[i×4] → out: float [4]' },
    'attn_out1': { text: 'weighted sum of cached value vectors using the attention weights. this is the "output" of this attention head - a [4]-dim vector that combines information from all attended positions.', io: 'in: weights[i], v_cache[i×4] → out: float [4]' },
    'attn_out2': { text: 'weighted sum of cached value vectors using the attention weights. this is the "output" of this attention head - a [4]-dim vector that combines information from all attended positions.', io: 'in: weights[i], v_cache[i×4] → out: float [4]' },
    'attn_out3': { text: 'weighted sum of cached value vectors using the attention weights. this is the "output" of this attention head - a [4]-dim vector that combines information from all attended positions.', io: 'in: weights[i], v_cache[i×4] → out: float [4]' },
    'wo': {
        text: 'the output projection W_O mixes the concatenated head outputs back together. this is the only place where information from different heads can interact. the result is a [16]-dim vector.', io: 'in: float [16] (concat heads) → out: float [16]', weights: ['layer0.attn_wo'],
    }, 'residual1': {
        text: 'the first residual block: add the attention output back to the saved R₁, then save the result as R₂ for the next residual. skip connections let gradients flow directly backward through the network.', io: 'in: attn_out[16] + R₁[16] → out: float [16]',
    }, 'res1': {
        text: 'first residual connection: x = attention_output + R₁. this creates a skip connection so gradients flow directly backward. without residuals, deep transformers are nearly impossible to train.', io: 'in: float [16] + float [16] → out: float [16]',
    }, 'r2': {
        text: 'save x again before the MLP. same idea as R₁ - after the MLP transforms x, we add this copy back so pre-MLP information is preserved.', io: 'in: float [16] → out: float [16] (identity copy)',
    }, 'norm2': {
        text: 'RMSnorm before the feed-forward MLP. ensures the MLP receives well-scaled inputs regardless of how much attention amplified or reduced certain dimensions.', io: 'in: float [16] → out: float [16]',
    }, 'mlp': {
        text: 'a 2-layer feed-forward network applied independently to each position. it first projects up to [64] dims (4× expansion), applies ReLU nonlinearity, then projects back to [16]. this is where "thinking" happens - attention routes information, MLP processes it.', io: 'in: float [16] → out: float [16]', weights: ['layer0.mlp_fc1', 'layer0.mlp_fc2'],
    }, 'fc1': {
        text: 'the first MLP layer projects from [16] to [64] dimensions (4× expansion). this expansion gives the network more capacity to learn complex nonlinear transformations.', io: 'in: float [16] → out: float [64]', weights: ['layer0.mlp_fc1'],
    }, 'relu': {
        text: 'ReLU(x) = max(0, x). this activation function zeroes out negative values, introducing the nonlinearity that makes the network capable of learning complex functions. without it, stacking linear layers would just produce another linear layer.', io: 'in: float [64] → out: float [64] (no learned weights)',
    }, 'fc2': {
        text: 'the second MLP layer projects back from [64] to [16] dimensions. together with fc1, this forms a bottleneck: expand → nonlinearity → compress, which is thought to let the network learn diverse features in the expanded space.', io: 'in: float [64] → out: float [16]', weights: ['layer0.mlp_fc2'],
    }, 'res2': {
        text: 'second residual connection: x = MLP_output + R₂. now x contains contributions from the original input, attention, and the MLP. each residual lets the model learn incremental refinements.', io: 'in: float [16] + float [16] → out: float [16]',
    }, 'logits': {
        text: 'the language model head W_lm projects from [16] to [vocab_size] unnormalized log-probabilities. Each logit corresponds to one character. Higher values mean the model predicts that character is more likely next.', io: 'in: float [16] → out: float [vocab_size]', weights: ['lm_head'],
    }, 'probs': {
        text: 'softmax converts logits to a proper probability distribution: P(token_i) = exp(logit_i) / Σ exp(logit_j). The result sums to 1.0. During generation, we sample from this distribution to pick the next token.', io: 'in: float [vocab_size] → out: float [vocab_size] (sums to 1)',
    }, 'output': {
        text: 'sample one token from the probability distribution. We draw a random number and walk through the cumulative distribution until we find the chosen token. The sampled token becomes the input for the next step of generation.', io: 'in: float [vocab_size] → out: int [1] (token index)',
    }
}

const faq: { q: string, a: string }[] = [{
    q: 'why 16 dims, 4 heads, 64 in the mlp?',
    a: 'hyperparameters chosen largely arbitrarily, trading speed for intelligence. but in general, ideas are represented in a few dimensions (16 here, 10k ish for modern llms) to be our space of \'meaning\'. we want a handful of attention heads to focus on different aspects of the input, and a larger multilayer perceptron to learn more nonlinear relationships between things.',
}, {
    q: 'what does a cell in the weight heatmap mean?',
    a: 'it\'s a matrix entry. many of our operations are like W . x, where x is the dims=16 vector mentioned earlier, and W does some sort of learned transformation as part of its "thinking" process.'
}, {
    q: 'how does training change the weights?',
    a: 'we attempt to generate the next character of a name, and then compute the entropy of our result to determine how wrong we were. we then use gradient descent (backpropagation) to adjust the weights to reduce the error. over time, the model learns to generate names that are more likely to be in the training data.',
}, {
    q: 'what is attention?',
    a: 'it decides which characters matter! the current character has \'what am i looking for?\' (Q). each previous character has \'what do i contain?\' (K) and a value (V). for each character i, we compute something like Σᵢ (Q ⋅ Kᵢ) ⋅ Vᵢ, so that Q ≈ Kᵢ means the model focuses on that Vᵢ. it allows for non-local rules like \'i before e except after c\'.',
}, {
    q: 'what is RMSnorm?',
    a: 'RMSnorm(x) = x / √(mean(x²)). it rescales the vector so its root-mean-square equals 1. this prevents "activation drift" where values grow or shrink too far away from each other.',
}, {
    q: 'why scale attention scores by 1/√d?',
    a: 'similarly to RMSnorm, high-dimensional dot products can grow very large very quickly, so we need to divide by that dimension to keep things in check. otherwise, one vector index could dominate the softmax and we\'d never learn anything else.'
}, {
    q: 'why not always pick the highest probability token?',
    a: '1. it leads to robotic, repetitive output. 2. we want some randomness so that we can fully explore the space of possible names every once in a while',
}, {
    q: 'what is it actually learning?',
    a: 'at this scale, not much... but it does learn some simple patterns like common starting letters and consonant-vowel alternation.',
}, {
    q: 'what would more layers do?',
    a: 'more layers means more sequential processing steps. layer 1 might learn common pairs like "ie", "qu", ..., while layer 2 might learn higher-level word structure, and so on. modern llms have a few dozen layers. determining what they all do is an active area of research.',
}, {
    q: 'why do residual connections matter so much?',
    a: 'if we connect layer 1 directly to layer 2, and 2 directly to 3, and so on... what we\'ve done in a sense is a "highway" where gradients flow directly forwards and backwards. they also let each layer learn a small delta rather than reconstructing the entire representation.',
}, {
    q: 'how does this compare to chatgpt?',
    a: 'it\s not that different, just much bigger. 100+ billion parameters, 1000+ dimensions, dozens of transformer layers, etc. but there are some subtleties, like how chatgpt trains on some human feedback on top of just loss. and instead of characters, it predicts \'tokens\', which could be syllables or common full words.'
}]

export default function Explainer({ clicked, stateDict, onBack }: { clicked: string, stateDict?: StateDict, onBack?: () => void }) {
    const entry = clicked ? explain[clicked] : undefined

    if (!entry) return <div className='mt-4 text-left'>
        <h3>questions</h3>
        {faq.map(({ q, a }, i) => <details key={i}> <summary>{q}</summary> <p>{a}</p> </details>)} <br />
        <p className='text-sm opacity-75'>source on <a href="https://github.com/b44ken/microgpt">github</a>. inspired by <a href="https://karpathy.github.io/2026/02/12/microgpt/">karpathy's</a> microgpt.</p>
    </div>

    return <div className='text-left'>
        <button className='text-xs opacity-75 hover:opacity-100 mb-2' onClick={onBack}>back</button>
        <h2 className='font-bold'>{clicked}</h2>
        <p className='text-sm opacity-75 mt-1'>{entry.text}</p>
        <p className='text-xs font-mono opacity-75 mt-2'>{entry.io}</p>
        {stateDict && entry.weights?.map(key => {
            const data = flatWeights(stateDict, key)
            if (!data) return null
            const w = stateDict[key]
            return <WeightGrid key={key} data={data} label={key} rows={Math.min(w.shape[0] || 0, 16)} cols={Math.min(w.shape[1] || 0, 64)} />
        })}
    </div>
}