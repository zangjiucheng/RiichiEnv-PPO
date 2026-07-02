# PPO Training Metrics Reference

Reference for the metrics logged by `riichienv_ml.trainers.ppo.run_ppo_training`
(console log + W&B). ⭐ = check regularly during a run. 🔍 = diagnostic, only
needed when something looks wrong.

## Training quality (from `PPOLearner.update()`, once per PPO step)

| Metric | W&B key | Meaning | How to read it |
|---|---|---|---|
| ⭐ Total loss | `loss` | Weighted sum: `policy_loss + value_coef*value_loss - entropy_coef*entropy + kl_ref_loss` | Should oscillate or flatten out, not blow up or go NaN |
| ⭐ Policy loss | `policy_loss` (`pi`) | PPO clipped surrogate loss | Near 0 means the current policy is close to what was sampled; a large negative value means this update made the policy noticeably worse (uncommon, worth investigating) |
| ⭐ Value loss | `value_loss` (`v`) | MSE between the value head's prediction and the actual return | Persistently high means the value function isn't learning well, which degrades the GAE advantage estimates |
| ⭐ Entropy | `entropy` (`ent`) | Entropy of the policy distribution -- higher means more exploration | Gradual decline is normal convergence; a sharp early collapse toward 0 suggests insufficient exploration / premature convergence to a local policy |
| ⭐ Explained variance | `explained_variance` (`ev`) | Fraction of return variance the value function explains (1 = perfect, 0 = no better than guessing the mean) | **One of the most important metrics to watch.** Persistently low (<0.5) or negative means the value function isn't learning, which drags down the whole PPO update |
| ⭐ Approx KL | `approx_kl` (`kl`) | Approximate KL divergence between the new and old policy | Much larger than `ppo_clip` means a single update moved the policy too far -- a sign of instability |
| 🔍 Clip fraction | `clip_frac` (`clip`) | Fraction of samples that triggered the PPO clip | Near 1 means most of the update is being clipped away (little effective learning signal); near 0 means the clip isn't doing anything (could tighten it) |
| 🔍 KL vs teacher | `kl_ref` | KL divergence between the current policy and the frozen teacher (the `load_model` checkpoint) | Expected to ramp up gradually per `alpha_kl_warmup_steps`; don't confuse this with `approx_kl` above (that's old-vs-new policy, this is current-vs-teacher) |
| 🔍 Ratio mean/std/max | `ratio/mean`, `ratio/std`, `ratio/max` | Policy probability ratio `exp(new_logp - old_logp)` | A max frequently far from 1 (e.g. >2 or <0.5) means this batch of data is already fairly off-policy |
| 🔍 KL max | `kl/max` | Per-sample max KL (not the mean) | Useful for spotting individual outlier samples when the mean looks fine but the max is large |
| 🔍 Grad norm | `grad_norm` | Gradient norm before clipping | Frequently hitting `max_grad_norm` (0.5) suggests the learning rate may be too high |
| ⭐ Advantage/return stats | `adv/raw_mean`, `adv/raw_std`, `return/mean`, `return/std` | Raw (pre-normalization) advantage/return distribution | Mainly used to catch the GRP reward shaping going off the rails (e.g. std suddenly exploding) |

## Rollout stats (from the collected games)

| Metric | W&B key | Meaning | How to read it |
|---|---|---|---|
| ⭐ Mean reward | `rollout/reward_mean` (`rew`) | End-of-hanchan reward based on final rank (1st: +10, 2nd: +4, 3rd: -4, 4th: -10) | **The most direct "is it getting stronger" signal** -- should trend upward over training |
| ⭐ Mean rank | `rollout/rank_mean` (`rank`) | Hero seat's average placement (1-4, lower is better) | Read alongside reward; for 4p the theoretical average is 2.5, so meaningfully below that indicates outperforming the baseline |
| 🔍 Per-kyoku reward | `rollout/kyoku_reward_mean` (`k_rew`) | GRP model's potential-based reward for each individual round (not the whole hanchan) | Naturally noisy (±1.3 is normal) -- don't judge from a single value, watch the long-run trend |
| 🔍 Kyoku length | `rollout/kyoku_length_mean` (`k_len`) | Average number of actions per round | Sanity check that games aren't running abnormally long/short |

## Timing / profiling metrics

Added to debug why a training step was taking ~9.5 minutes (root cause:
cuDNN disabled by default -- see `RIICHIENV_DISABLE_CUDNN` in the README).

| Metric | W&B key | Meaning | How to read it |
|---|---|---|---|
| 🔍 Collection time | `time/collect_s` | Wall-clock for all rollout workers to finish this batch in parallel | Currently ~2 min; mainly useful for spotting a straggler worker |
| 🔍 Batch prep time | `time/prep_s` | Time to `np.concatenate` the collected transitions into one batch | Always small (<1s), not worth watching |
| ⭐ PPO update time | `time/update_s` | Total wall-clock for one full PPO update (all epochs) | **This is where the ~5.6 of the original 9.5 min/step was hiding** -- the metric that led to finding the cuDNN issue |
| 🔍 Data transfer | `time/data_transfer_s` | Time to move the rollout batch onto the GPU | Always small, not a bottleneck |
| 🔍 Teacher forward | `time/teacher_fwd_s` | Forward-pass time through the frozen teacher (KL regularization) | 0 for the first update (KL warmup hasn't started), then grows gradually |
| ⭐ Main forward+backward | `time/main_fwd_bwd_s` | Forward + backward + optimizer.step for the main model | **336.7s with cuDNN disabled vs. 14.5s with it enabled** -- the concrete evidence for the fix |
| 🔍 Weight sync | logged as `t_sync` (console only, not pushed to W&B) | Time to broadcast updated weights to all rollout workers via `ray.put` | Consistently ~0.1-0.2s, not a bottleneck |

## Evaluation metrics (every `eval_interval` steps, vs. a fixed baseline)

| Metric | W&B key | Meaning |
|---|---|---|
| ⭐ Eval reward | `eval/reward` | Mean reward using the greedy (non-sampling) policy against the baseline |
| ⭐ Eval rank | `eval/rank` ± `eval/rank_se` | Mean rank (and standard error) from the same evaluation |

## Day-to-day checklist

Watch `ev`, `approx_kl`, `rollout/reward_mean`/`rollout/rank_mean`, and
`eval/reward`/`eval/rank`. Ignore the `time/*` metrics unless training
suddenly feels slower than usual.
