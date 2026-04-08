"""
Cart-Pole Swing-Up Environment

A cart moves on a 1D track. A pole (pendulum) is attached to the cart via
a free hinge. The pole starts hanging DOWN and must be swung UP to the
vertical position and balanced there.

Physics derived from Lagrangian mechanics for an underactuated cart-pole system.
Integration uses 4th-order Runge-Kutta for accuracy.

State: [x, x_dot, theta, theta_dot]
    x: cart position
    theta: pole angle (0 = UP, pi = DOWN)

Observation: [x, sin(theta), cos(theta), x_dot, theta_dot]
    sin/cos representation avoids angle discontinuity

Action: continuous force on cart in [-1, 1], scaled by force_mag
"""

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.error import DependencyNotInstalled


class CartPoleSwingUpEnv(gym.Env):
    """Cart-Pole Swing-Up: swing a pendulum from DOWN to UP and balance it."""

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self, render_mode=None):
        super().__init__()

        # Physical parameters
        self.gravity = 9.81
        self.cart_mass = 1.0
        self.pole_mass = 0.1
        self.total_mass = self.cart_mass + self.pole_mass
        self.pole_length = 0.5  # half-length of pole
        self.pole_mass_length = self.pole_mass * self.pole_length
        self.force_mag = 10.0
        self.friction = 0.1  # cart friction coefficient

        # Simulation parameters
        self.dt = 0.02  # timestep
        self.n_substeps = 2  # RK4 substeps per step

        # Track limits
        self.x_threshold = 2.4

        # Observation: [x, sin(theta), cos(theta), x_dot, theta_dot]
        high = np.array([
            self.x_threshold,      # x (episode terminates at ±x_threshold)
            1.0,                   # sin(theta)
            1.0,                   # cos(theta)
            10.0,                  # x_dot (empirical bound)
            4 * np.pi,             # theta_dot (empirical bound)
        ], dtype=np.float32)

        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

        # Rendering
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.screen_width = 600
        self.screen_height = 400
        self.isopen = True

        # State: [x, x_dot, theta, theta_dot]
        self.state = None

    def _dynamics(self, state, force):
        """Compute state derivatives using cart-pole equations of motion.

        Equations derived from Lagrangian mechanics:
            (M + m) * x_ddot + m * l * theta_ddot * cos(theta) - m * l * theta_dot^2 * sin(theta) = F - friction * x_dot
            m * l * x_ddot * cos(theta) + m * l^2 * theta_ddot - m * g * l * sin(theta) = 0

        Solving for x_ddot and theta_ddot:
        """
        x, x_dot, theta, theta_dot = state
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)

        ml = self.pole_mass_length
        total_m = self.total_mass
        l = self.pole_length

        # Denominator (determinant of mass matrix)
        denom = total_m - self.pole_mass * cos_theta ** 2

        # Angular acceleration
        theta_ddot = (
            (total_m * self.gravity * sin_theta
             - cos_theta * (force - self.friction * x_dot + ml * theta_dot ** 2 * sin_theta))
            / (l * denom)
        )

        # Cart acceleration
        x_ddot = (
            (force - self.friction * x_dot + ml * theta_dot ** 2 * sin_theta
             - ml * theta_ddot * cos_theta)
            / total_m
        )

        return np.array([x_dot, x_ddot, theta_dot, theta_ddot])

    def _rk4_step(self, state, force, dt):
        """4th-order Runge-Kutta integration step."""
        k1 = self._dynamics(state, force)
        k2 = self._dynamics(state + 0.5 * dt * k1, force)
        k3 = self._dynamics(state + 0.5 * dt * k2, force)
        k4 = self._dynamics(state + dt * k3, force)
        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def step(self, action):
        assert self.state is not None, "Call reset() before step()"

        force = float(np.clip(action[0], -1.0, 1.0)) * self.force_mag

        # Integrate physics with substeps for stability
        sub_dt = self.dt / self.n_substeps
        for _ in range(self.n_substeps):
            self.state = self._rk4_step(self.state, force, sub_dt)

        # Normalize theta to [-pi, pi]
        self.state[2] = ((self.state[2] + math.pi) % (2 * math.pi)) - math.pi

        obs = self._get_obs()
        reward = self._compute_reward(action)
        terminated = self._is_terminated()
        truncated = False
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        x, x_dot, theta, theta_dot = self.state
        return np.array([
            x,
            math.sin(theta),
            math.cos(theta),
            x_dot,
            theta_dot,
        ], dtype=np.float32)

    def _compute_reward(self, action):
        """Dense reward encouraging upright pole and centered cart.

        Primary: cosine-based uprightness reward (0 when down, 1 when up)
        Secondary: small penalties for cart displacement and control effort
        """
        x, x_dot, theta, theta_dot = self.state

        # Uprightness: cos(theta) = 1 when up (theta=0), -1 when down (theta=pi)
        upright = (math.cos(theta) + 1.0) / 2.0  # normalized to [0, 1]

        # Cart centering penalty (small, don't discourage swing-up motion)
        cart_penalty = 0.01 * x ** 2

        # Control effort penalty
        control_penalty = 0.001 * float(action[0]) ** 2

        # Angular velocity penalty near upright (encourages stillness at top)
        # Smooth activation via upright**4 avoids reward discontinuity at threshold
        velocity_penalty = 0.002 * upright ** 4 * theta_dot ** 2

        reward = upright - cart_penalty - control_penalty - velocity_penalty
        return float(reward)

    def _is_terminated(self):
        x = self.state[0]
        not_finite = not np.isfinite(self.state).all()
        cart_out = abs(x) > self.x_threshold
        return not_finite or cart_out

    def _get_info(self):
        x, x_dot, theta, theta_dot = self.state
        cos_theta = math.cos(theta)
        return {
            "upright": (cos_theta + 1.0) / 2.0,
            "cart_position": x,
            "pole_angle_deg": math.degrees(theta),
            "is_balanced": abs(theta) < 0.2 and abs(theta_dot) < 1.0,
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Default: pole starts DOWN (theta = pi) with small noise
        noise = 0.01
        if options and "random_init" in options:
            # For evaluation: random initial angles
            theta_init = self.np_random.uniform(-math.pi, math.pi)
        else:
            theta_init = math.pi + self.np_random.uniform(-noise, noise)

        self.state = np.array([
            self.np_random.uniform(-0.05, 0.05),   # x: near center
            self.np_random.uniform(-0.01, 0.01),    # x_dot: near zero
            theta_init,                              # theta: DOWN
            self.np_random.uniform(-0.01, 0.01),    # theta_dot: near zero
        ])

        obs = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return obs, info

    def render(self):
        if self.render_mode is None:
            return None

        try:
            import pygame
        except ImportError:
            raise DependencyNotInstalled("pygame is required for rendering")

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:
                self.screen = pygame.Surface(
                    (self.screen_width, self.screen_height)
                )
            self.clock = pygame.time.Clock()

        x, x_dot, theta, theta_dot = self.state

        # World-to-screen scaling
        world_width = self.x_threshold * 2.0
        scale = self.screen_width / world_width
        cart_y = self.screen_height * 0.6  # cart vertical position
        pole_pixel_len = self.pole_length * 2 * scale  # full pole length in pixels

        self.screen.fill((255, 255, 255))

        # Draw track
        pygame.draw.line(
            self.screen, (0, 0, 0),
            (0, int(cart_y + 20)), (self.screen_width, int(cart_y + 20)), 2
        )

        # Cart position in pixels
        cart_x = int(x * scale + self.screen_width / 2)

        # Draw cart
        cart_width, cart_height = 50, 30
        cart_rect = pygame.Rect(
            cart_x - cart_width // 2,
            int(cart_y) - cart_height // 2,
            cart_width, cart_height
        )
        pygame.draw.rect(self.screen, (50, 50, 50), cart_rect)

        # Draw pole (theta=0 is UP, positive theta is clockwise)
        pole_end_x = cart_x + int(pole_pixel_len * math.sin(theta))
        pole_end_y = int(cart_y) - int(pole_pixel_len * math.cos(theta))

        pygame.draw.line(
            self.screen, (0, 128, 200),
            (cart_x, int(cart_y)),
            (pole_end_x, pole_end_y), 6
        )

        # Draw pole tip
        pygame.draw.circle(self.screen, (200, 0, 0), (pole_end_x, pole_end_y), 8)

        # Draw pivot
        pygame.draw.circle(self.screen, (0, 0, 0), (cart_x, int(cart_y)), 5)

        # Draw uprightness indicator
        upright = (math.cos(theta) + 1.0) / 2.0
        bar_width = int(200 * upright)
        color = (int(255 * (1 - upright)), int(255 * upright), 0)
        pygame.draw.rect(self.screen, color, pygame.Rect(10, 10, bar_width, 15))
        pygame.draw.rect(self.screen, (0, 0, 0), pygame.Rect(10, 10, 200, 15), 1)

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.isopen = False
