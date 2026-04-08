"""
Double Cart-Pole Swing-Up Environment

A cart moves on a 1D track. Two identical poles (pendulums) are connected
in series: pole 1 attaches to the cart, pole 2 attaches to the tip of pole 1.
Both poles start hanging DOWN and must be swung UP to vertical and balanced.

Physics derived from Lagrangian mechanics for a double pendulum on a cart.
Integration uses 4th-order Runge-Kutta for accuracy.

State: [x, x_dot, theta1, theta1_dot, theta2, theta2_dot]
    x: cart position
    theta1: first pole angle (0 = UP, pi = DOWN)
    theta2: second pole angle relative to vertical (0 = UP, pi = DOWN)

Observation: [x, sin(t1), cos(t1), sin(t2), cos(t2), x_dot, t1_dot, t2_dot]
Action: continuous force on cart in [-1, 1], scaled by force_mag
"""

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.error import DependencyNotInstalled


class DoubleCartPoleSwingUpEnv(gym.Env):
    """Double Cart-Pole Swing-Up: swing two pendulums UP and balance them."""

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self, render_mode=None):
        super().__init__()

        # Physical parameters (both poles identical)
        self.gravity = 9.81
        self.cart_mass = 1.0
        self.pole_mass_1 = 0.1
        self.pole_mass_2 = 0.1
        self.pole_length_1 = 0.5  # half-length of pole 1
        self.pole_length_2 = 0.5  # half-length of pole 2
        self.force_mag = 10.0
        self.friction = 0.1

        # Simulation parameters
        self.dt = 0.02
        self.n_substeps = 4  # more substeps for double pendulum stability

        # Track limits
        self.x_threshold = 2.4

        # Observation: [x, sin(t1), cos(t1), sin(t2), cos(t2), x_dot, t1_dot, t2_dot]
        high = np.array([
            self.x_threshold,       # x (episode terminates at ±x_threshold)
            1.0, 1.0,              # sin/cos theta1
            1.0, 1.0,              # sin/cos theta2
            10.0,                  # x_dot (empirical bound)
            8 * np.pi,             # theta1_dot (empirical bound, chaotic dynamics)
            8 * np.pi,             # theta2_dot (empirical bound, chaotic dynamics)
        ], dtype=np.float32)

        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

        # Rendering
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.screen_width = 600
        self.screen_height = 500
        self.isopen = True

        self.state = None

    def _dynamics(self, state, force):
        """Compute state derivatives for double pendulum on cart.

        Uses the Euler-Lagrange equations for a cart with two serial pendulums.
        Mass matrix M(q) * q_ddot = f(q, q_dot, u)
        Solved via numpy linear algebra.
        """
        x, x_dot, t1, t1_dot, t2, t2_dot = state

        m0 = self.cart_mass
        m1 = self.pole_mass_1
        m2 = self.pole_mass_2
        l1 = self.pole_length_1 * 2  # full length
        l2 = self.pole_length_2 * 2  # full length
        lc1 = self.pole_length_1     # distance to center of mass
        lc2 = self.pole_length_2
        g = self.gravity

        # Moments of inertia (thin rod about center: (1/12)*m*l^2)
        I1 = (1.0 / 12.0) * m1 * l1 ** 2
        I2 = (1.0 / 12.0) * m2 * l2 ** 2

        s1, c1 = math.sin(t1), math.cos(t1)
        s2, c2 = math.sin(t2), math.cos(t2)
        s12 = math.sin(t1 - t2)
        c12 = math.cos(t1 - t2)

        # Mass matrix M (3x3) for [x_ddot, t1_ddot, t2_ddot]
        M = np.zeros((3, 3))
        M[0, 0] = m0 + m1 + m2
        M[0, 1] = (m1 * lc1 + m2 * l1) * c1
        M[0, 2] = m2 * lc2 * c2
        M[1, 0] = M[0, 1]
        M[1, 1] = m1 * lc1 ** 2 + m2 * l1 ** 2 + I1
        M[1, 2] = m2 * l1 * lc2 * c12
        M[2, 0] = M[0, 2]
        M[2, 1] = M[1, 2]
        M[2, 2] = m2 * lc2 ** 2 + I2

        # Coriolis / centrifugal / gravity forces
        f = np.zeros(3)
        f[0] = (force - self.friction * x_dot
                + (m1 * lc1 + m2 * l1) * t1_dot ** 2 * s1
                + m2 * lc2 * t2_dot ** 2 * s2)
        f[1] = ((m1 * lc1 + m2 * l1) * g * s1
                - m2 * l1 * lc2 * t2_dot ** 2 * s12)
        f[2] = (m2 * lc2 * g * s2
                + m2 * l1 * lc2 * t1_dot ** 2 * s12)

        # Solve M * q_ddot = f
        try:
            q_ddot = np.linalg.solve(M, f)
        except np.linalg.LinAlgError:
            q_ddot = np.zeros(3)

        return np.array([x_dot, q_ddot[0], t1_dot, q_ddot[1], t2_dot, q_ddot[2]])

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

        sub_dt = self.dt / self.n_substeps
        for _ in range(self.n_substeps):
            self.state = self._rk4_step(self.state, force, sub_dt)

        # Normalize angles to [-pi, pi]
        self.state[2] = ((self.state[2] + math.pi) % (2 * math.pi)) - math.pi
        self.state[4] = ((self.state[4] + math.pi) % (2 * math.pi)) - math.pi

        obs = self._get_obs()
        reward = self._compute_reward(action)
        terminated = self._is_terminated()
        truncated = False
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        x, x_dot, t1, t1_dot, t2, t2_dot = self.state
        return np.array([
            x,
            math.sin(t1), math.cos(t1),
            math.sin(t2), math.cos(t2),
            x_dot, t1_dot, t2_dot,
        ], dtype=np.float32)

    def _compute_reward(self, action):
        """Dense reward based on tip height and individual pole uprightness."""
        x, x_dot, t1, t1_dot, t2, t2_dot = self.state

        # Individual pole uprightness (0 when down, 1 when up)
        upright1 = (math.cos(t1) + 1.0) / 2.0
        upright2 = (math.cos(t2) + 1.0) / 2.0

        # Tip height: y position of the second pole's tip
        l1 = self.pole_length_1 * 2  # full pole length
        l2 = self.pole_length_2 * 2
        tip_y = l1 * math.cos(t1) + l2 * math.cos(t2)
        max_height = l1 + l2
        tip_height_normalized = (tip_y + max_height) / (2.0 * max_height)  # [0, 1]

        # Combined uprightness: both poles must be up
        uprightness = 0.4 * upright1 + 0.4 * upright2 + 0.2 * tip_height_normalized

        # Cart centering penalty
        cart_penalty = 0.005 * x ** 2

        # Control penalty
        control_penalty = 0.001 * float(action[0]) ** 2

        # Angular velocity penalty near upright
        # Smooth activation via uprightness products avoids reward discontinuity
        upright_factor = (upright1 * upright2) ** 2
        vel_penalty = 0.001 * upright_factor * (t1_dot ** 2 + t2_dot ** 2)

        reward = uprightness - cart_penalty - control_penalty - vel_penalty
        return float(reward)

    def _is_terminated(self):
        x = self.state[0]
        not_finite = not np.isfinite(self.state).all()
        cart_out = abs(x) > self.x_threshold
        return not_finite or cart_out

    def _get_info(self):
        x, x_dot, t1, t1_dot, t2, t2_dot = self.state
        l1 = self.pole_length_1 * 2
        l2 = self.pole_length_2 * 2
        tip_y = l1 * math.cos(t1) + l2 * math.cos(t2)
        max_height = l1 + l2
        return {
            "upright1": (math.cos(t1) + 1.0) / 2.0,
            "upright2": (math.cos(t2) + 1.0) / 2.0,
            "tip_height": tip_y,
            "tip_height_normalized": (tip_y + max_height) / (2.0 * max_height),
            "cart_position": x,
            "is_balanced": (
                abs(t1) < 0.2 and abs(t2) < 0.2
                and abs(t1_dot) < 1.0 and abs(t2_dot) < 1.0
            ),
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        noise = 0.01
        if options and "random_init" in options:
            t1_init = self.np_random.uniform(-math.pi, math.pi)
            t2_init = self.np_random.uniform(-math.pi, math.pi)
        else:
            t1_init = math.pi + self.np_random.uniform(-noise, noise)
            t2_init = math.pi + self.np_random.uniform(-noise, noise)

        self.state = np.array([
            self.np_random.uniform(-0.05, 0.05),
            self.np_random.uniform(-0.01, 0.01),
            t1_init,
            self.np_random.uniform(-0.01, 0.01),
            t2_init,
            self.np_random.uniform(-0.01, 0.01),
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

        x, x_dot, t1, t1_dot, t2, t2_dot = self.state

        world_width = self.x_threshold * 2.0
        scale = self.screen_width / world_width
        cart_y = self.screen_height * 0.65
        pole_scale = scale  # pixels per meter

        l1_px = self.pole_length_1 * 2 * pole_scale
        l2_px = self.pole_length_2 * 2 * pole_scale

        self.screen.fill((255, 255, 255))

        # Track
        pygame.draw.line(
            self.screen, (0, 0, 0),
            (0, int(cart_y + 20)), (self.screen_width, int(cart_y + 20)), 2
        )

        cart_x = int(x * scale + self.screen_width / 2)

        # Cart
        cart_w, cart_h = 50, 30
        cart_rect = pygame.Rect(
            cart_x - cart_w // 2, int(cart_y) - cart_h // 2, cart_w, cart_h
        )
        pygame.draw.rect(self.screen, (50, 50, 50), cart_rect)

        # Pole 1
        p1_end_x = cart_x + int(l1_px * math.sin(t1))
        p1_end_y = int(cart_y) - int(l1_px * math.cos(t1))
        pygame.draw.line(
            self.screen, (0, 128, 200),
            (cart_x, int(cart_y)), (p1_end_x, p1_end_y), 6
        )

        # Pole 2
        p2_end_x = p1_end_x + int(l2_px * math.sin(t2))
        p2_end_y = p1_end_y - int(l2_px * math.cos(t2))
        pygame.draw.line(
            self.screen, (200, 100, 0),
            (p1_end_x, p1_end_y), (p2_end_x, p2_end_y), 6
        )

        # Joints and tip
        pygame.draw.circle(self.screen, (0, 0, 0), (cart_x, int(cart_y)), 5)
        pygame.draw.circle(self.screen, (0, 0, 0), (p1_end_x, p1_end_y), 5)
        pygame.draw.circle(self.screen, (200, 0, 0), (p2_end_x, p2_end_y), 8)

        # Uprightness indicator
        u1 = (math.cos(t1) + 1.0) / 2.0
        u2 = (math.cos(t2) + 1.0) / 2.0
        combined = (u1 + u2) / 2.0
        bar_w = int(200 * combined)
        color = (int(255 * (1 - combined)), int(255 * combined), 0)
        pygame.draw.rect(self.screen, color, pygame.Rect(10, 10, bar_w, 15))
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
