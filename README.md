# CarRacing REINFORCE

A REINFORCE (policy-gradient) reinforcement learning agent trained to drive in
Gymnasium's `CarRacing-v3` environment, starting from raw pixel observations.

## What's in here

- `policy.py` — small CNN policy network (image in, action probabilities out)
- `train.py` — single-environment REINFORCE training loop (CPU)
- `train_gpu.py` — parallelized training: multiple game copies stepped across
  CPU cores at once, with the policy network batched on GPU
- `reward_wrapper.py` / `train_phase2.py` — an experimental second training
  phase that adds a speed bonus/penalty on top of the base reward, to nudge
  an already-successful driver toward finishing faster
- `eval.py` — headless evaluation: run N episodes, report scores/completion
- `watch.py` — pop open a window and watch a trained checkpoint drive live
- `checkpoint.pt` — first trained model (300 episodes, single-environment CPU training)
- `checkpoint_gpu.pt` — final trained model (25,200 episodes, parallel GPU training)
- `training_progress_final.png` — full training curve

## Results

Trained across several stages, from 300 episodes (CPU, single environment) up
to 25,200 episodes (GPU + 10 parallel environments). Final performance
plateaued around a **~780-790 average score**, with a best single lap of
**~930** (CarRacing is considered "solved" around a 900+ average over 100
trials, per Gymnasium's benchmark).

Notably, the final evaluation episodes consistently ran out of the
per-episode decision budget without crashing — suggesting the agent may be
capable of higher scores with a longer time allowance per episode, rather
than needing purely more training.

## Running it

```bash
pip install torch gymnasium[box2d] numpy matplotlib pygame
python train_gpu.py    # train from scratch (or resume from checkpoint_gpu.pt)
python eval.py 10       # headless evaluation
python watch.py 3 checkpoint_gpu.pt   # watch it drive
```
