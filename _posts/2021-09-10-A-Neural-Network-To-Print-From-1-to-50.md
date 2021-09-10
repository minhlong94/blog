---
toc: true
layout: post
description: A minimal example of using markdown with fastpages.
categories: [shitpost]
title: A Neural Network to print from 1 to 50
---
# A Neural Network to print from 1 to 50
Recently, a user named `logo` asked a question on the [RL Discord](https://discord.com/invite/xhfNqQv) server:
```
how to print numbers from 1 to 50 in python?
```
Little did I know, I was about to engage in one of the funniest conversation.
## Pure Python approach
```
for i in range(50):
    print(i)
```
A very simple and straightforward solution. Wait, why can't we just `print(1,2,3,4,5,6,7,8,9,10...,50)`?

"That's too long" - user Ariel. "it's better to do `exec(f"print({','.join(str(i) for i in range(1,51))})")`"

Ok but?
```python
i = 1
while True:
    print(i)
    i += 1
    if i > 50:
        break
```

Best method. Straight forward:
```python
print(1)
print(2)
print(3)
.
.
.
.
print(50)
```

"next research question is how to do it with functional approach"? Oh right.
```python
def getprint():
    return print
getprint()(1)
getprint()(2)
.
.
.
.
getprint()(50)
```

Or better: `[print(i) for i in range(1,51)]`

User `James` replied to the functional approach:
```python
def recursive_print(start, end):
    def recursive_count_str(i):
        return f"{i}\n{recursive_count_str(i+1)}" if i < end else f"{i}"
    print(recursive_count_str(start))


recursive_print(1, 50)
```

Ok, can we do better?
```python
class PrintFactory:
    def __init__(self, number):
        self.number = number
        
    def print_number(self):
        for i in range(self.number):
            print(i)

def print50():
    print_factory = PrintFactory(50)
    print_factory.print_number()

print50()
```
- by OrganicPitaChips.

"I feel a strong yin yang struggle right now between my shitposting nature and the whole "admin" thing" - Ariel.

```python
class ObjectOrientedPrintFactory:
    def __init__(self, number):
        self.number = number
        self.child_factory = None
        if number > 0:
            self.child_factory = ObjectOrientedPrintFactory(number - 1)
        
    def print_number(self):
        if self.child_factory:
            self.child_factory.print_number()
        print(self.number)

def print50():
    print_factory = ObjectOrientedPrintFactory(50)
    print_factory.print_number()

print50()
```
- hackeronsteriods.


Yeah this is from [this link](https://github.com/EnterpriseQualityCoding/FizzBuzzEnterpriseEdition/tree/uinverse/src/main/java/com/seriouscompany/business/java/fizzbuzz/packagenamingpackage/impl/factories)

## Neural Network to print from 1 to 50
"I like how nobody actually suggested training a neural network" - Ariel
"Challenge accepted" - James
```python
import tensorflow as tf
import numpy as np


class IncrementAutoregressive(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self._inc_weight = tf.Variable(np.random.normal())
        self.terminal_logits_layer = tf.keras.layers.Dense(2)

    def call(self, inputs, **kwargs):
        new_inputs = inputs + self._inc_weight
        terminal_logits = self.terminal_logits_layer(new_inputs)
        return new_inputs, terminal_logits

    def rollout(self, start_val):
        next_num = tf.constant([[start_val]], dtype=tf.float32)
        stop = tf.constant(False)
        result = [next_num]
        while not stop:
            next_num, stop_logits = self(next_num)
            stop = tf.reduce_all(tf.argmax(stop_logits, axis=-1) == 1)
            result.append(next_num)
        return tf.concat(result, axis=0)


def main() -> None:
    layer = IncrementAutoregressive()
    xs = tf.reshape(tf.range(1, 50, dtype=tf.float32), (49, 1))
    ys = xs + 1.0
    stops = tf.concat([tf.fill((48,), 0), [1]], axis=0)
    opt = tf.keras.optimizers.Adam(0.01)
    loss1 = tf.keras.losses.MeanSquaredError()
    loss2 = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    @tf.function
    def step():
        with tf.GradientTape() as tape:
            pred, pred_stops = layer(xs)
            pred_loss = loss1(ys, pred)
            pred_stops_loss = loss2(stops, pred_stops)
            total_loss = pred_loss + pred_stops_loss
        grad = tape.gradient(total_loss, layer.trainable_weights)
        opt.apply_gradients(zip(grad, layer.trainable_weights))
        return total_loss

    for _ in range(10000):
        loss = step()
        print(loss)

    rollout = layer.rollout(1.0)
    print(tf.round(rollout))


if __name__ == "__main__":
    main()
```

"Please add TPU support I need this to support big data".
"100k epochs ensures the right output".
"When I run it locally I usually get 1-50, and somtimes 1-51. Dat bias term learning is apparently rough".


"but what if I want to use Reinf Learning, we would surely need a PrintGymEnv"?

OK.
```python
import numpy as np
import gym
from gym.spaces import Box

class PrintEnv(gym.Env):
    def __init__(self, start: int = 1, end: int = 50):
        self.start = start
        self.end = end
        self.counter = start
        
        
        self.observation_space = Box(1, 50, (1,), dtype=np.int32)
        self.action_space = Box(1, 50, (1,), dtype=np.int32)
        
    def reset(self):
        self.counter = self.start
        return self.counter
    
    def step(self, action: int):
        if action == self.counter:
            reward = 1.
            self.counter += 1
        else:
            reward = 0.
            
        if self.counter >= self.end:
            done = True
        else:
            done = False
            
        return self.counter, reward, done, {}
        
    def render(self, mode="human"):
        print(self.counter)
```
- Ariel.

I think we need a PyTorch version, no?
```python
import torch
from torch import nn, Tensor

class IncrementAutoregressive(nn.Module):
    def __init__(self):
        super().__init__()
        self.inc_weight = nn.Parameter(torch.normal(0., 1., (1,), requires_grad=True))
        self.terminal_logits_layer = nn.Linear(1, 2)

    def forward(self, inputs):
        new_inputs = inputs + self.inc_weight
        terminal_logits = self.terminal_logits_layer(new_inputs)
        return new_inputs, terminal_logits

def rollout(model: IncrementAutoregressive, start_val: int) -> Tensor:
    next_num = torch.tensor([start_val])
    stop = False
    result = [next_num]
    i = 0
    while not stop:
        next_num, stop_logits = layer(next_num)
        stop = stop_logits.argmax().item()
        result.append(next_num.detach())
        i += 1

    return torch.cat(result).round()

def train(layer: nn.Module):

    xs = torch.arange(1, 50).view((49, 1)).to(torch.float32)
    ys = xs + 1.0
    stops = torch.zeros_like(xs).view((49,)).to(torch.long)
    stops[-1] = 1.

    opt = torch.optim.Adam(layer.parameters(), 0.05)

    val_loss = nn.MSELoss()
    stop_loss = nn.CrossEntropyLoss()

    for t in range(10000):
        pred, pred_stops = layer(xs)

        loss = val_loss(pred, ys) + stop_loss(pred_stops, stops)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if t % 1000 == 0:
            print(f"Loss at step {t}: {loss.item()}")

layer = IncrementAutoregressive()
train(layer)
result = rollout(layer, 1.)
print(result)
```
- Ariel

## State-of-the-art algorithm to print from 1 to 50
The council consists of Ariel and me.
```python
Another solution, inspired by the State-Of-The-Art sorting algorithm SleepSort, using advanced concurrent programming techniques:
from threading import Thread
import time

def wait_and_print(n: int):
    time.sleep(n)
    print(n)

threads = [Thread(target=wait_and_print, args=(i,)) for i in range(1, 50)]

for thread in threads:
    thread.start()

for thread in threads:
    thread.join()
```

"Let's be language agnostic" - OrganicPitaChips:
```
>+++++++++++[-<+++++>] # initialize 55??? at first cell
>++++++++++<<[->+>-[>+>>]>[+[-<+>]>+>>]<<<<<<]>>[-]>>>++++++++++<[->-[>+>>]>[+[-
<+>]>+>>]<<<<<]>[-]>>[>++++++[-<++++++++>]<.<<+>+>[-]]<[<[->-<]++++++[->++++++++
<]>.[-]]<<++++++[-<++++++++>]<.[-]<<[-<+>]
```
