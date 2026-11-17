# CarRacing REINFORCE

A REINFORCE (policy-gradient) agent that learns to drive Gymnasium's `CarRacing-v3`
from raw pixels. No critic, no replay buffer, no frame stacking: one small CNN, one
frame in, one of five actions out.

## Result

Scored on the standard benchmark: 100 tracks, the environment's own 1000-frame limit,
track *k* generated from seed *k* so every run sees the same 100 tracks.

| Checkpoint | Actions | Mean score | 95% interval | Median | Laps finished | Scored 900+ |
|---|---|---:|---:|---:|---:|---:|
| `checkpoint_gpu.pt` (25,200 episodes) | greedy (most likely action) | **666** | 630 to 703 | 719 | 2 / 100 | 2 / 100 |
| `checkpoint_gpu.pt` (25,200 episodes) | sampled, as in training | **752** | 720 to 783 | 826 | 3 / 100 | 3 / 100 |
| `checkpoint.pt` (300 episodes) | greedy | -44 | -52 to -36 | -63 | 0 / 100 | 0 / 100 |

CarRacing counts as solved at a 900 average over 100 tracks. **This agent does not
solve it.** It learns to stay on the road and covers most of a lap (77% of track tiles
greedy, 85% sampled), but it drives too slowly to finish inside the frame limit on
all but a handful of tracks.

Three things the numbers say:

- **Training-time averages flatter the agent.** The training log's rolling average
  ended around 790. That figure comes from unseeded tracks and from a policy that
  changes every update. The same final checkpoint, same sampled actions, on 100
  fixed tracks scores 752.
- **The agent is worse when it stops rolling dice.** Picking the most likely action
  costs 85 points on the same tracks (paired difference, standard error 22; sampled
  is better on 66 of 100). Greedy gets stuck: 24 tracks score under 500 against 8
  when sampling. The network never sees speed (single frame), and the randomness
  seems to be what gets it moving again.
- **The time limit is not the problem; speed is.** 96 of 100 greedy episodes end at
  the frame limit. That limit is the benchmark's, not this repo's, and the score
  already charges 0.1 per frame, so a longer episode would not rescue the average.
  The five laps that did finish (two greedy, three sampled) took 877 to 985 of the
  1000 frames.

An earlier version of this README reported "a ~780-790 average" (the training-log
figure) and a "best lap of ~930" (the best of 25,200 training episodes), and read the
time-outs as a sign the agent needed a longer allowance. The table above replaces
those claims. Per-track results are in `eval_greedy.json`, `eval_sample.json` and
`eval_greedy_300ep.json`.

## What is here

- `policy.py`: the CNN policy (96x96 image in, action logits out)
- `train.py`: single-environment REINFORCE on CPU (produced `checkpoint.pt`)
- `train_gpu.py`: 10 environments stepped in parallel, policy batched on GPU
  (produced `checkpoint_gpu.pt`)
- `eval.py`: the seeded benchmark above
- `watch.py`: open a window and watch a checkpoint drive
- `reward_wrapper.py`, `train_phase2.py`: an experimental second phase that pays a
  bonus for finishing a lap quickly. Not run to completion; no checkpoint is shipped.
- `tests/`: checks on the return calculation, the reward wrapper, and that a seeded
  evaluation episode is reproducible
- `training*.log`, `reward_log*.json`, `training_progress*.png`: the original training
  record (sampled actions, unseeded tracks)

## Running it

```bash
pip install -r requirements.txt

python eval.py                         # 100 seeded tracks, greedy   (~30 min on one core)
python eval.py --mode sample           # same tracks, sampled actions
python eval.py --episodes 10 --checkpoint checkpoint.pt
python watch.py 3 checkpoint_gpu.pt    # watch it drive
python train_gpu.py                    # continue training from checkpoint_gpu.pt
pytest -q
```

## Known limits

- Single-frame input, so the policy cannot see its own speed. Frame stacking is the
  obvious next change and the likeliest route past the speed ceiling.
- Plain REINFORCE with a normalised-return baseline: high variance, and 25,200
  episodes (about 18 hours on an RTX 4070) to get here.
- `train_gpu.py` used to standardise returns over padding steps as well as played
  ones. That is fixed (and tested), but the shipped checkpoint was trained before
  the fix. Episodes rarely end early, so the effect on it was small.
- Resuming training reloads the weights but starts the Adam optimiser fresh.
