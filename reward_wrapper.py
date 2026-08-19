"""
Adds a speed bonus/penalty on top of the base CarRacing reward, and handles
frame-skipping internally so "one decision" always means the same thing
everywhere in the pipeline (matches FRAME_SKIP used in phase 1 training/eval).

The bonus only applies when the car genuinely completes the whole track (all
tiles visited) -- not on a crash or a timeout. Finishing faster than "par"
(set from how long the bot took on average during its passing test) earns
extra points; finishing slower loses a few. Crashing/not finishing is
untouched -- this only fine-tunes behavior AFTER it already knows how to
finish, nudging it toward doing so more efficiently.
"""
import gymnasium as gym

FRAME_SKIP = 4
BONUS_PER_DECISION = 2.0   # points gained/lost per decision-step faster/slower than par
MAX_BONUS_MAGNITUDE = 150  # cap so this never overwhelms the base ~900+ completion reward


class CarRacingSpeedBonus(gym.Wrapper):
    def __init__(self, env, par_decisions):
        super().__init__(env)
        self.par_decisions = par_decisions
        self.decision_count = 0

    def reset(self, **kwargs):
        self.decision_count = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        info = {}
        for _ in range(FRAME_SKIP):
            obs, r, terminated, truncated, info = self.env.step(action)
            total_reward += r
            if terminated or truncated:
                break
        self.decision_count += 1

        if terminated:
            core = self.env.unwrapped
            crashed = core.tile_visited_count < len(core.track)
            if not crashed:
                diff = self.par_decisions - self.decision_count
                bonus = max(-MAX_BONUS_MAGNITUDE, min(MAX_BONUS_MAGNITUDE, diff * BONUS_PER_DECISION))
                total_reward += bonus
                info["speed_bonus"] = bonus
                info["decisions_taken"] = self.decision_count

        return obs, total_reward, terminated, truncated, info
