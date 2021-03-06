---
toc: true
use_math: true
layout: post
description: An example using LoL to explain to a friend about black box in RL
categories: [DRL, shitpost]
title: "You should not treat RL as a black box: an example using League of Legends"
---

Recently I tried to explain a struggle in a project to a friend, who had zero knowledge 
in Reinforcement Learning, that was: my friend set up a suitable action space for an agent to choose to "neglect" a chest in a gridworld, and she thought: "if the agent is smart enough, it should learn how to neglect these chests if necessary." The key word here is "if necessary": she did **not** know what cases should be considered "necessary". In short, she did not know the optimal policy and under what conditions would make the policy optimal. Say, for example, if you have 10 steps left in the gridworld, and you can both get the chest and the goal in 10 steps, why should you neglect the chest, assume that it gives bonus reward and the number of steps left does not contribute to the reward? Transfer to our problem, in this case, she wonders why the agent does not neglect the chest (expectation), but she does not know how many steps are left, and where is the agent on the gridworld (reality).

Now the thing is, how should I explain this struggle to a friend *who has zero knowledge in RL*. He does not know everything RL: the definitions of states, actions, rewards, optimal policies, etc. Actually, it is easier to give the [OpenAI blog post about faulty reward functions](https://openai.com/blog/faulty-reward-functions/), and tells that the struggle is similar to this, but I do not think it is a good explanation: it is still very abstract. Saying that your problem B is similar to problem A does not clearly explain B. I need a better example.

Luckily, simple explanation is one of my specializations. We both play League of Legends, and we all know the famous [Faker's Zed vs Ryu's Zed](https://www.youtube.com/watch?v=ZPCfoCVCx3U) highlight. This move is widely considered to be the most mind-blowing moment in LoL history, even up to now (so if you don't understand what happened, don't worry I don't too). One of the factors that helped Faker to do this highlight was he was facing Ryu - a top player in a professional match, and he was in a bad situation: it was hard, thus these actions were necessary to survive.

Now, instead of the above situation, consider that a player who plays Zed now faces with a champion who **cannot** resist and has weak defense. Moreover, assume that this Zed has a lot of items that he can kill the champion almost instantly. In this case, is Faker's talent level necessary? Do you need to perform complex combo? From my point of view and my friends', we all agree that although it sounds badass, it is not necessary to be Faker to kill that champion.

We then go back to the "if necessary" case: if the situation is simple like the example I just mentioned, the agent does not have to "neglect" the chest - it can simply get both. However, if the situation is like Faker vs Ryu, then Faker's talent level is necessary. In my friend's case, she did **not** know the situation, and she has been wondering about why the agent does not behave as expected for weeks. My answer is simple: "what actions are expected, and why they are expected?". Sadly, she refuses to listen, so the project has been in that loop ever since.

From this experience, I think that not knowing a good policy in RL is a dangerous thing: you do not know how good the agent is right now, which can make your evaluation bias. No, randomness is not a solid baseline, especially if the environment requires a sequence of steps. Before diving into RL, make sure you know these things so the bug will be a lot easier to detect. Finally, I would like to insert a message from user `hackeronsteriod`, who works with AI at Google:
> Also, I would suggest not treating RL like a black box.  You should have an idea of what a good policy is in your environment, what actions you expect it to do, what reward you expect it to achieve, etc. That will be more reliable to tell if you actually have converged to a good policy.


