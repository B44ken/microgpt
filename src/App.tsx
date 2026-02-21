import { useState, useEffect, useRef, useCallback } from 'react'
import * as tf from '@tensorflow/tfjs'
import { MicroGPT, type Trace, type ModelOpts } from './microgpt'
import Architecture from './arch'
import Explainer from './explain'
import names from '../data/firstnames.json'

const defaults: Record<string, number> = { n_embd: 16, n_head: 2, n_layer: 1, block_size: 16 }

const tutorial = [
  "hello! this is a GPT, a neural network that generates text one character at a time. this one has a dataset of names it'll learn to copy. by default, it spits out random text. click generate...",
  "see? total nonsense. it hasn't learned anything yet. click train...",
  "the loss should go down a little bit... (STEP/500)",
  "nice. now stop training and generate again...",
  "much better! click any node in the diagram to learn how it works",
]

export default function App() {
  const [mode, setMode] = useState<'idle' | 'train' | 'gen'>('idle')
  const [run, setRun] = useState({ step: 0, loss: 0, output: '', trace: undefined as Trace | undefined })
  const [explain, setExplain] = useState('')
  const [opts, setOpts] = useState(defaults)
  const [collapsed, setCollapsed] = useState(false)
  const [sidebar, setSidebar] = useState(false)
  const [tut, setTut] = useState(0)
  const model = useRef<MicroGPT | null>(null)
  const prev = useRef(mode)

  const reset = async (o: Record<string, number>) => {
    setMode('idle'); setOpts(o); setRun({ step: 0, loss: 0, output: '', trace: undefined })
    model.current = await tf.ready().then(() => MicroGPT.fromDocs(names as string[], o))
  }

  const train = useCallback(() => {
    const m = model.current
    m && setRun(r => ({ ...r, loss: m.trainSteps(1000), step: m.step_count }))
  }, [])

  const gen = useCallback(() => {
    const { text, trace, done } = model.current?.generateStep() ?? { text: '', trace: undefined, done: true }
    setRun(r => ({ ...r, output: text, trace }))
    if (done) setMode('idle')
  }, [])

  useEffect(() => {
    let active = true
    const loop = async () => {
      if (!active) return
      if (mode === 'gen') { gen(); setTimeout(() => active && loop(), 500) }
      if (mode === 'train') { await train(); if (active) requestAnimationFrame(loop) }
    }
    loop()
    return () => { active = false }
  }, [mode, train, gen])

  // tutorial auto-advance
  useEffect(() => {
    if (tut < 0) return
    if (tut === 0 && mode === 'gen') setTut(1)
    if (tut === 1 && mode === 'train') setTut(2)
    if (tut === 2 && run.step >= 490) { setMode('idle'); setTut(3) }
    if (tut === 3 && mode === 'gen' && prev.current !== 'gen') setTut(4)
    if (tut === 4 && explain !== '') setTut(-1)
    prev.current = mode
  }, [tut, mode, run.step, explain])

  const toggleGen = () => {
    if (mode === 'gen') return setMode('idle')
    model.current?.resetGeneration()
    setRun(r => ({ ...r, output: '' }))
    setMode('gen')
  }

  useEffect(() => void setTimeout(reset, 0, defaults), [])

  const Opt = ({ label, k }: { label: string, k: keyof ModelOpts }) =>
    <label className='opt flex justify-between'> {label} <input type='number' value={opts[k]} onChange={e => reset({ ...opts, [k]: +e.target.value })} /> </label>

  return <>
    <a href='#' className="menu-toggle" onClick={() => setSidebar(!sidebar)}>{sidebar ? 'x' : 'menu'}</a>

    <nav className={sidebar ? 'open' : ''}>
      <h1>microgpt</h1>

      {tut >= 0 && <div className="tutorial">
        {tutorial[tut].replace('STEP', `${run.step}`)}
        <a className='mt-2 text-xs opacity-75 inline-block' href="#" onClick={() => setTut(-1)}>skip tutorial</a>
      </div>}
      <div className="btn-row">
        <button onClick={() => setMode(mode === 'train' ? 'idle' : 'train')}>{mode === 'train' ? 'stop' : 'train'}</button>
        <button onClick={toggleGen}>{mode === 'gen' ? 'stop' : 'generate'}</button>
        <button onClick={() => reset(opts)}>reset</button>
      </div>
      <div className="output">{run.output}_</div>
      <div className="metrics"> <p>step {run.step}</p> <p>loss {run.loss.toFixed(2)}</p> </div>

      <Explainer clicked={explain} stateDict={model.current?.state_dict} onBack={() => setExplain('')} />

      <div className="grow"></div>

      <div className='*:mt-1'>
        <h3>parameters</h3>
        <Opt label='dimension' k='n_embd' /> <Opt label='layers' k='n_layer' /> <Opt label='heads' k='n_head' />
        <label className='flex justify-between'> collapse <input type='checkbox' checked={collapsed} onChange={e => setCollapsed(e.target.checked)} /> </label>
        <div className='opacity-75 text-right'> total params: {Math.floor((model.current?.num_params() || 0) / 1000)}k </div>
      </div>
    </nav >

    <main> <Architecture trace={run.trace} onClickNode={i => { setExplain(i); setSidebar(true) }} cfg={opts as Required<ModelOpts>} collapsed={collapsed} /> </main>
  </>
}