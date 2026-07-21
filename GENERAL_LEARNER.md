AI must not edit this file.

## Long-term Goal

Make an AGI agent with these characteristics:
- Asynchronous - humans don't function in lockstep with their task environments
- Has a generalized interface - humans have senses and interact with the world through their bodies
- Learns generally, fast, consistently, robustly, and in real-time
- Doesn't forget things - can master task A, then task B, then task C, and task A and B are still mastered
- Scales to its capacity - given how many parameters it has, it will learn how to optimally solve all tasks it is presented with within that capacity
- Latency resistant - humans can adapt to different latency situations, although generally at degraded performance
- Tunable reaction time - for testing purposes via fair competition with humans
- Generalizes into reality - it doesn't matter if the model is playing a game or controlling a real-world robot
- Applies cross-cutting experience - humans can improve at some tasks indirectly by learning others
- Has long-term perfect recall - I have a theory that a model can have a hard data base (knowledge not baked into weights) that it can learn to reference for long-term recall
- Experiences the world and learns in serial, is always learning 'online', doesn't depend on pretraining and being frozen
- Can learn from direct experience, or by observation
- Can optionally learn in serial for massively in-parallel knowledge consumption (not sure if possible)

## Current Goal

Create a learner that can master cartpole FAST and robustly. It should be able to learn cartpole before your eyes in realtime, on the order of a minute or two. It should be stable, so once it reaches a high score, it should not degrade anymore.

This learner should be made with the goal of generalizing to other games.

The next game planned is Lunar Lander.

## Testing Constraints

Tests must be fast, and smart. If a learner is showing extreme instability in 2 minutes, there is absolutely no reason to run a 20 minute horizon test, don't waste time.

Tests must be stochastic, locking one seed and doing a test on that is not proof of robustness. You must show results over multiple seeds.
