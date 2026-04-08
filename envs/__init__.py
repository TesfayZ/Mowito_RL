"""Custom Gymnasium environments for cart-pole swing-up tasks."""

from gymnasium.envs.registration import register

register(
    id="CartPoleSwingUp-v0",
    entry_point="envs.cartpole_swingup:CartPoleSwingUpEnv",
    max_episode_steps=500,
)

register(
    id="DoubleCartPoleSwingUp-v0",
    entry_point="envs.double_cartpole_swingup:DoubleCartPoleSwingUpEnv",
    max_episode_steps=1000,
)
