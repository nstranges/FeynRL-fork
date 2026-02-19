# FAQ

## Why not run rollout engines fully in parallel (continuous generation) while training runs?

You're right that overlapping rollout and training can leave some GPU capacity on the table in certain configurations. The reason this framework doesn't default to "always-on" rollout is mainly about data off-policyness and algorithmic bottlenecks, not just throughput.

1. **On-policy methods don't benefit much from stale, continuously generated data.** Most practical RL post-training recipes for large models are effectively on-policy (or close to it) and rely on mechanisms like PPO-style clipping (or related constraints) to stay stable when the policy changes. If a rollout engine keeps generating while the policy is being updated, a growing fraction of those samples quickly become off-policy. Once the divergence is large, clipping/constraints tend to squash the update signal and many samples contribute little to no useful gradient. In other words, "more generations" is not automatically "more learning" if those generations are produced by a policy that is already out of date relative to the current optimizer state.

2. **The limiting factor is usually algorithm reliability, not raw rollout speed.** In practice, the hard part of RL for large models isn't only that generation is expensive, it's that training can be brittle: rewards can be sparse, noisy, or misspecified; credit assignment is hard; small implementation details matter; and optimization can destabilize easily. If the underlying method is not reliably improving the policy when it should, increasing rollout throughput often just increases complexity (queues, buffering, off-policy correction, synchronization) without improving outcomes.

This does not imply that system throughput is unimportant or can be ignored. It emphasizes that RL itself has many fundamental challenges, and system optimization pays off most once the algorithm is in a healthy place. That said, we do plan to adopt proven patterns from other works as long as we can do it without much sacrificing the core goals of this repo.

## Other frameworks include many rollout-engine system improvements. Why don't you include them?

We try to include recent improvements as much as possible, especially with regard to the rollout engine—and this is one of the reasons we open-sourced the repo.

That said, some of these improvements have only a marginal impact on performance, but add significant complexity to the pipeline (see response to previous question). In cases like that, we avoid including them by default. However, we are open to PRs that improve system throughput without sacrificing the core goals of this repo.

## There are differences between your implementation of methods like GRPO. Why is that the case?

That is correct. RL training is sensitive to small implementation details, and some important details are often overlooked when applying RL to large models, unlike classic RL settings such as games. As a result, in some places LeanRL makes deliberate choices to improve stability and performance, even if that means it does not match a specific reference implementation line for line.

When the differences are intentional, we document them and name variants explicitly. For example, you may see SGRPO, which indicates a GRPO style method with stability focused implementation choices and some clear changes from the original work.

## I found a bug. What should I do?

That is wonderful. Please open a GitHub issue with steps to reproduce, expected behavior, and actual behavior. If you can include logs, config files, or a minimal script, it will help a lot. Pull requests are also welcome, and we will review them as quickly as possible.

## We have a new work where we propose a method that can improve results significantly. Would you be open to adding our method to the repo?

Absolutely. We're open to contributions that improve the repo. Please open a GitHub issue and include enough context (paper link, a short summary, expected gains, and how to reproduce) so others can follow along and review it. Also, please make sure your code is clean and closely follows the repo structure. If you prefer to discuss privately first, you can email Rasool.

## We have a new rollout engine that can significantly improve rollout throughput. Would you be open to adding our rollout engine to the repo?

Of course. Please open a PR, or if you prefer to discuss it first, you can email Rasool.

## I have a few research ideas and want guidance. Can you help?

We can try. If you are comfortable sharing your idea publicly, open a GitHub issue and include enough context for others to follow along. If you prefer to discuss privately, you can email Rasool. Contact details are available on [Rasool's website](https://rasoolfa.github.io/).
