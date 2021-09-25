---
toc: true
layout: post
description: A collection of related topics about code-level optimization tricks in DRL, which dramatically changes the results.
categories: [DRL]
title: Making your Deep RL matters.
---

A collection of implementation tricks, hyperparameter sensitivity, and others in Deep RL which I gave a presentation in my research group.

Author: Long M. Luu, contact: minhlong9413@gmail.com or Discord AerysS#5558.

"Your ResNet, batchnorm, and very deep networks don't work here." - Andrej Karpathy

![Untitled](Making%20your%20Deep%20RL%20matters%20c01bfd91a2dc4cef8c405348e5a7d7dc/Untitled.png)

# References

All codebases are released. Just use CatalyzeX.

[Deep Reinforcement Learning that Matters](https://arxiv.org/abs/1709.06560)

[How Many Random Seeds? Statistical Power Analysis in Deep Reinforcement Learning Experiments](https://arxiv.org/abs/1806.08295)

[Reproducibility of Benchmarked Deep Reinforcement Learning Tasks for Continuous Control](https://arxiv.org/abs/1708.04133)

[Implementation Matters in Deep RL: A Case Study on PPO and TRPO](https://openreview.net/forum?id=r1etN1rtPB)

[Reproducibility of Benchmarked Deep Reinforcement Learning Tasks for Continuous Control](https://arxiv.org/abs/1708.04133)

[What Matters for On-Policy Deep Actor-Critic Methods? A Large-Scale...](https://openreview.net/forum?id=nIAxjsniDzg)

[An Empirical Model of Large-Batch Training](https://arxiv.org/abs/1812.06162)

[Deep Reinforcement Learning at the Edge of the Statistical Precipice](https://arxiv.org/abs/2108.13264)

[http://joschu.net/docs/nuts-and-bolts.pdf](http://joschu.net/docs/nuts-and-bolts.pdf)

[http://amid.fish/reproducing-deep-rl](http://amid.fish/reproducing-deep-rl)

[https://www.alexirpan.com/2018/02/14/rl-hard.html](https://www.alexirpan.com/2018/02/14/rl-hard.html)

[https://openai.com/blog/science-of-ai/](https://openai.com/blog/science-of-ai/)

[https://costa.sh/blog-the-32-implementation-details-of-ppo.html](https://costa.sh/blog-the-32-implementation-details-of-ppo.html)

# Why it matters?

- Reproducibility, **especially in Deep RL**, is hard (sources above).
- **Multiple factors** affect the results: random seed, hyperparameters, code-level optimizations.
- It is common to report the **best of N** results, which makes misleading claims.

![Untitled](Making%20your%20Deep%20RL%20matters%20c01bfd91a2dc4cef8c405348e5a7d7dc/Untitled%201.png)

Agarwal et al., 2021

# Statistical Power Analysis (Colas et al., 2017 paper)

Consider two algorithms below. The name of the algorithm is not important. The mean and 95% confidence interval are averaged over **5 seeds**. Our concern: is algorithm 1 better than 2?

![Untitled](Making%20your%20Deep%20RL%20matters%20c01bfd91a2dc4cef8c405348e5a7d7dc/Untitled%202.png)

The measure of performance: the average cumulated reward over last 100 evaluation episodes. It seems like Algo 1 is better than Algo 2.

## Statistical problem

The performance can be modeled as a *random variable* $X$. Running this algorithm results in $x^i$. Run for N times, we obtain a statistical sample $x = (x^1,...,x^N)$.

A random variable is characterized by its mean $\mu$ and standard deviation $\sigma$. The real values are unknown, so we compute the unbiased estimations $\overline{x}=\sum_{i=1}^n x^i$ and $s \approx \sqrt{\frac{\sum_{i+1}^N (x^i - \overline{x})^2}{N-1}}$. The larger the $N$, the more confident one can be in the estimations.

Here, two algorithms with respective performances $X_1$ and $X_2$ are compared. If they follow normal distributions, then $X_{\text{diff}} = X_1 - X_2$ also follows a normal distribution with parameters $\sigma_{\text{diff}} = \sqrt{(\sigma_1^2 + \sigma_2^2)}$ and $\mu_{\text{diff}} = \mu_1 - \mu_2$. 

In this case, the estimator of the mean of $X_{\text{diff}}$ is $\overline{x}_{\text{diff}} = \overline{x}_1 - \overline{x_2}$ and the estimator of $\sigma_{\text{diff}}$ is $s_{\text{diff}} = \sqrt{s_1^2 + s_2^2}$. The effect size $\epsilon$ can be defined as $\epsilon = \mu_1 - \mu_2$.

Testing $\epsilon$ between two algorithms is mathematically equivalent to testing a difference between $\mu_{\text{diff}}$ and 0.

## Difference test

We define the null hypothesis $H_0$and the alternate hypothesis $H_a$ using the two-tail case:

- $H_0: \mu_{\text{diff}} = 0$
- $H_a: \mu_{\text{diff}} \neq 0$

When we have an assumption about which algorithm performs better (say, Algo 1), we can use the one-tail version:

- $H_0: \mu_{\text{diff}} \leq 0$
- $H_a: \mu_{\text{diff}} > 0$

At first, **we assume the null hypothesis**. Once a sample $x_{\text{diff}}$ is sampled from $X_{\text{diff}}$, we can estimate the probability $p$ (called $p$-value) of observing the data as *extreme **(*$\overline{x}_{\text{diff}}$ is far from 0)**, under the null hypothesis assumption. $p$-value answers the question:

How probable is it to observe this sample or a more extreme one, given that there is no true difference in the performances of both algorithms?

We can rewrite it for the one-tail case:

$$p\text{-value} = P(X_{\text{diff}} \geq \overline{x}_{\text{diff}} | H_0)$$

For the two-tail case:

$$p\text{-value} = \begin{cases} & P(X_{\text{diff}} \geq \overline{x}_{\text{diff}} | H_0), \overline{x}_{\text{diff}} > 0 \\
& P(X_{\text{diff}} \leq \overline{x}_{\text{diff}} | H_0), \overline{x}_{\text{diff}} \leq 0
\end{cases}$$

When this probability becomes really **low**, it means that **it is highly improbable that two algorithms with no performance difference produced the collected** (Algo 1 is different from Algo 2)**.**

A difference is called significant at level $\alpha$ when $p$-value < $\alpha$ in the one-tail case, and $\alpha/2$ in the two-tail case. Usually $\alpha=0.05$. However, $\alpha = 0.05$ also means there is a 5% chance we conclude it **wrong**.

**In the paper by Colas et al., the authors also suggest an alternate way to test the difference using 95% confidence intervals (95% CIs). However, I will only focus on the t-test for now.**

## Statistical testing

![Untitled](Making%20your%20Deep%20RL%20matters%20c01bfd91a2dc4cef8c405348e5a7d7dc/Untitled%203.png)

Type-I error: false positive. **Rejects $H_0$ when it is true.**

Type-II error: false negative. **Accepts $H_0$ when it is false.**

## The t-test and Welch's t-test

The t-test assumes that the variances of both algorithms are **equal**, while Welch's t-test assumes they are **unequal**. Both tests are equivalent when the std are equal.

The t-test assumes the following:

- The scale of data measurements must be continuous and ordinal (can be ranked). This is the case in RL.
- Data is obtained by collecting a representative sample from the population.
This seem reasonable in RL.
- Measurements are independent from one another. This seems reasonable
in RL.
- Data is normally-distributed, or at least bell-shaped.

We then compute the $t$-statistic and the degree of freedom $\nu$ using the following equations:

![Untitled](Making%20your%20Deep%20RL%20matters%20c01bfd91a2dc4cef8c405348e5a7d7dc/Untitled%204.png)

where $x_{\text{diff}} = x_1 - x_2$; $s_1, s_2$ is the empirical standard deviations of the two samples and $N_1, N_2$ are their sizes (which we assume $N_1 = N_2 = N$).

A figure to make sense of these concepts:

![Untitled](Making%20your%20Deep%20RL%20matters%20c01bfd91a2dc4cef8c405348e5a7d7dc/Untitled%205.png)

$H_0$ assumes $\mu_{\text{diff}} = 0$, so the distribution is centered on 0. $H_a$ assumes a positive difference $\mu_{\text{diff}} = \epsilon$, so the distribution is shifted by the t-value corresponding to $\epsilon$, $t_\epsilon$. We consider the one-tail case, and test for the positive difference.

Using the computed t-statistic and $\nu$, we can compute the $\alpha$ value. $**\alpha$ is enough to declare statistical significance**. With modern software work, we can directly compute t-statistic and $\alpha$ without worrying about $\nu$.

## Back to the problem

![Untitled](Making%20your%20Deep%20RL%20matters%20c01bfd91a2dc4cef8c405348e5a7d7dc/Untitled%202.png)

The measure of performance: the average cumulated reward over last 100 evaluation episodes. It seems like Algo 1 is better than Algo 2. The p-value returned is 0.031, which is lower than $\alpha = 0.05$. 

However, **they are the same algorithm: DDPG**. They have the same set of parameters, are evaluated on the same environment (so there is no implementation tricks involved), and are averaged over 5 seeds each.

## Estimate the type-I error.

Experiment: same algorithm, runs for $N = [2, 21]$.

![Untitled](Making%20your%20Deep%20RL%20matters%20c01bfd91a2dc4cef8c405348e5a7d7dc/Untitled%206.png)

So in practice, $N=\{5, 10\}$ works well.

# Deep Reinforcement Learning that matters

Empirical results from Henderson et al., 2017 paper.

## Reward Scaling

- **Idea**: Multiply the reward by some scalar: $r = \sigma r$.
- **Why it matters**: this affects action-value function based method like DDPG.

![Untitled](Making%20your%20Deep%20RL%20matters%20c01bfd91a2dc4cef8c405348e5a7d7dc/Untitled%207.png)

![Untitled](Making%20your%20Deep%20RL%20matters%20c01bfd91a2dc4cef8c405348e5a7d7dc/Untitled%208.png)

## Random seeds and trials

- **Idea**: run multiple runs with different random seeds
- **Why it matters**: environment stochasticity or stochasticity in the learning process, e.g. random weight initialization, Q-value initialization.

![Untitled](Making%20your%20Deep%20RL%20matters%20c01bfd91a2dc4cef8c405348e5a7d7dc/Untitled%209.png)

Additional result from Islam et al., 2017 paper:

![Untitled](Making%20your%20Deep%20RL%20matters%20c01bfd91a2dc4cef8c405348e5a7d7dc/Untitled%2010.png)

![Untitled](Making%20your%20Deep%20RL%20matters%20c01bfd91a2dc4cef8c405348e5a7d7dc/Untitled%2011.png)

## Environment variables

- Idea: different environment can affect the performance
- Why it matters: although the reward can be high, it can learn undesirable policy.

In this figure, HalfCheetah has stable dynamics. Hopper does not have stable dynamics. **Swimmer has a local optima.**

![Untitled](Making%20your%20Deep%20RL%20matters%20c01bfd91a2dc4cef8c405348e5a7d7dc/Untitled%2012.png)

[https://www.youtube.com/watch?v=lKpUQYjgm80](https://www.youtube.com/watch?v=lKpUQYjgm80)

[https://www.youtube.com/watch?v=ghCo7ERx6qo](https://www.youtube.com/watch?v=ghCo7ERx6qo)

CoastRunners **does not directly reward the player’s progression around the course**, instead the player earns higher scores by **hitting targets laid out along the route**. We assumed the score the player earned would reflect the informal goal of finishing the race, so we included the game in an internal benchmark designed to measure the performance of reinforcement learning systems on racing games. However, it turned out that the targets were laid out in such a way that the reinforcement learning agent could gain a high score without having to finish the course. This led to some unexpected behavior when we trained an RL agent to play the game.

[https://www.youtube.com/watch?v=tlOIHko8ySg&t=1s](https://www.youtube.com/watch?v=tlOIHko8ySg&t=1s)

## Implementation tricks

- Idea: code-level optimization tricks like advantage normalization, n-steps TD return.
- Why it matters: it drastically changes the result.

Consider: 

- Set 1: **TRPO** from TRPO codebase (Schulman 2015), from PPO codebase(Schulman 2017), and rllib Tensorflow (Duan 2016) codebases.
- Set 2: **DDPG** rllab Theano (Duan 2016), rllabplusplus (Gu 2016), OpenAI Baselines (Plapper 2017).

![Untitled](Making%20your%20Deep%20RL%20matters%20c01bfd91a2dc4cef8c405348e5a7d7dc/Untitled%2013.png)

![Untitled](Making%20your%20Deep%20RL%20matters%20c01bfd91a2dc4cef8c405348e5a7d7dc/Untitled%2014.png)

# Implementation Matters in Deep RL (Engstorm et al., 2020 paper)

Different code-level optimization tricks can lead to (dramatically) **different** results.

Specifically, PPO implementation contains these optimizations that are not (or barely) described in the original paper:

1. Value function clipping. PPO originally suggests fitting the value network via regression to target values: $L^V = (V_{\theta_t} - V_{targ})^2$. However, the implementation in OpenAI Baselines fits the network with a PPO-like objective:

$$L^V = \max \left[ (V_{\theta_t} - V_{targ})^2, (clip(V_{\theta_t}, V_{\theta_{t-1}} - \epsilon, V_{\theta_{t-1}} + \epsilon) - V_{targ})^2 \right]$$

1. Reward scaling. Rewards are divided through by the std of a rolling discounted sum of the rewards.
2. Orthogonal initialization and layer scaling.
3. Adam learning rate annealing.
4. Reward clipping: [-5, 5] or [-10, 10].
5. Observation normalization: states are normalized to mean-zero, variance-one vectors before training.
6. Observation clipping: like reward.
7. Hyperbolic tan (tanh) activations.
8. Global gradient clipping: global $\ell_2$-norm less than 0.5.

![Untitled](Making%20your%20Deep%20RL%20matters%20c01bfd91a2dc4cef8c405348e5a7d7dc/Untitled%2015.png)

The authors then consider a PPO-M (minimal) variant, that **does not use** these optimization tricks, alongside PPO and TRPO, and the TRPO+ variant that uses PPO tricks.

![Untitled](Making%20your%20Deep%20RL%20matters%20c01bfd91a2dc4cef8c405348e5a7d7dc/Untitled%2016.png)

## Results comparing 4 algorithms

Define:

![Untitled](Making%20your%20Deep%20RL%20matters%20c01bfd91a2dc4cef8c405348e5a7d7dc/Untitled%2017.png)

AAI measures the **maximal effect of switching algorithms**, and ACLI measures the **maximal effect of adding tricks.**

We have:

![Untitled](Making%20your%20Deep%20RL%20matters%20c01bfd91a2dc4cef8c405348e5a7d7dc/Untitled%2018.png)

# How to tune hyperparameters?

Note: this tuning guide depends on **empirical results**, not theoretical.

### Network Architecture

From Henderson et al., 2017 paper:

Investigate three common architectures: (64, 64), (100, 50, 25) and (400, 30), activation: tanh, ReLU, Leaky ReLU.

![Untitled](Making%20your%20Deep%20RL%20matters%20c01bfd91a2dc4cef8c405348e5a7d7dc/Untitled%2019.png)

![Untitled](Making%20your%20Deep%20RL%20matters%20c01bfd91a2dc4cef8c405348e5a7d7dc/Untitled%2020.png)

![Untitled](Making%20your%20Deep%20RL%20matters%20c01bfd91a2dc4cef8c405348e5a7d7dc/Untitled%2021.png)

From Islam et al., 2017 paper: (purpose: to **reproduce** results)

![Untitled](Making%20your%20Deep%20RL%20matters%20c01bfd91a2dc4cef8c405348e5a7d7dc/Untitled%2022.png)

![Untitled](Making%20your%20Deep%20RL%20matters%20c01bfd91a2dc4cef8c405348e5a7d7dc/Untitled%2023.png)

![Untitled](Making%20your%20Deep%20RL%20matters%20c01bfd91a2dc4cef8c405348e5a7d7dc/Untitled%2024.png)

![Untitled](Making%20your%20Deep%20RL%20matters%20c01bfd91a2dc4cef8c405348e5a7d7dc/Untitled%2025.png)

### Batch Size

From Islam et al., 2017 paper

![Untitled](Making%20your%20Deep%20RL%20matters%20c01bfd91a2dc4cef8c405348e5a7d7dc/Untitled%2026.png)

![Untitled](Making%20your%20Deep%20RL%20matters%20c01bfd91a2dc4cef8c405348e5a7d7dc/Untitled%2027.png)

![Untitled](Making%20your%20Deep%20RL%20matters%20c01bfd91a2dc4cef8c405348e5a7d7dc/Untitled%2028.png)

### Findings from Andrychowicz et al., 2021 paper

Train **250k agents** in 5 continuous control environments. Each choice is run for 3 random seeds, but the reported results are based on the performance of hundreds of runs.

![Untitled](Making%20your%20Deep%20RL%20matters%20c01bfd91a2dc4cef8c405348e5a7d7dc/Untitled%2029.png)

![Untitled](Making%20your%20Deep%20RL%20matters%20c01bfd91a2dc4cef8c405348e5a7d7dc/Untitled%2030.png)

![Untitled](Making%20your%20Deep%20RL%20matters%20c01bfd91a2dc4cef8c405348e5a7d7dc/Untitled%2031.png)

![Untitled](Making%20your%20Deep%20RL%20matters%20c01bfd91a2dc4cef8c405348e5a7d7dc/Untitled%2032.png)

![Untitled](Making%20your%20Deep%20RL%20matters%20c01bfd91a2dc4cef8c405348e5a7d7dc/Untitled%2033.png)

![Untitled](Making%20your%20Deep%20RL%20matters%20c01bfd91a2dc4cef8c405348e5a7d7dc/Untitled%2034.png)

![Untitled](Making%20your%20Deep%20RL%20matters%20c01bfd91a2dc4cef8c405348e5a7d7dc/Untitled%2035.png)

![Untitled](Making%20your%20Deep%20RL%20matters%20c01bfd91a2dc4cef8c405348e5a7d7dc/Untitled%2036.png)

![Untitled](Making%20your%20Deep%20RL%20matters%20c01bfd91a2dc4cef8c405348e5a7d7dc/Untitled%2037.png)

![Untitled](Making%20your%20Deep%20RL%20matters%20c01bfd91a2dc4cef8c405348e5a7d7dc/Untitled%2038.png)

![Untitled](Making%20your%20Deep%20RL%20matters%20c01bfd91a2dc4cef8c405348e5a7d7dc/Untitled%2039.png)

![Untitled](Making%20your%20Deep%20RL%20matters%20c01bfd91a2dc4cef8c405348e5a7d7dc/Untitled%2040.png)

![Untitled](Making%20your%20Deep%20RL%20matters%20c01bfd91a2dc4cef8c405348e5a7d7dc/Untitled%2041.png)

![Untitled](Making%20your%20Deep%20RL%20matters%20c01bfd91a2dc4cef8c405348e5a7d7dc/Untitled%2042.png)

![Untitled](Making%20your%20Deep%20RL%20matters%20c01bfd91a2dc4cef8c405348e5a7d7dc/Untitled%2043.png)

![Untitled](Making%20your%20Deep%20RL%20matters%20c01bfd91a2dc4cef8c405348e5a7d7dc/Untitled%2044.png)

![Untitled](Making%20your%20Deep%20RL%20matters%20c01bfd91a2dc4cef8c405348e5a7d7dc/Untitled%2045.png)

# Suggested method to compare algorithms

Pseudocode thanks to James MacGlashan (Senior Research Scientist - Sony AI).

```python
trials = 10
neval = 20

alg1_performance = []
for _ in range(trials):
    p = train_alg1(10000)  # train for 10,000 time steps, get policy
    eval_performances = []
    for _ in range(neval):
        episode = run_episode(p)
        avg_episode_reward = compute_avg_reward(episode)
        eval_performances.append(avg_episode_reward)
    trial_performance = np.mean(eval_performances)
    alg1_performance.append(trial_performance)

alg2_performance = []
for _ in range(trials):
    p = train_alg2(10000)  # train for 10,000 time steps, get policy
    eval_performances = []
    for _ in range(neval):
        episode = run_episode(p)
        avg_reward = compute_avg_reward(episode)
        eval_performances.append(avg_reward)
    trial_performance = np.mean(eval_performances)
    alg2_performance.append(trial_performance)

p_value = t_test(alg1_performance, alg2_performance)
```

# Suggestions and Conclusion

> In general, however, the most important step to reproducibility is to **report all** hyperparameters, implementation details, experimental setup, and evaluation methods for both baseline comparison methods and novel work. Without the publication of implementations and related details, wasted effort on reproducing state-of-the-art works will plague the community and slow down progress.

![Untitled](Making%20your%20Deep%20RL%20matters%20c01bfd91a2dc4cef8c405348e5a7d7dc/Untitled%2046.png)

> Overall, our results highlight the necessity of designing deep RL methods in a modular manner. When building algorithms, we should understand precisely how each component impacts agent training—both in terms of overall performance and underlying algorithmic behavior. It is impossible to properly attribute successes and failures in the  complicated systems that make up deep RL methods without such diligence. More broadly, our findings suggest that developing an RL toolkit will require moving beyond the current **benchmark-driven** evaluation model to a more fine-grained understanding of deep RL methods.

Note: Stable-Baselines3 has all these recommendations implemented by default.