"""
Classic Snake environment.

Layout:  10×10 interior playfield surrounded by a 1-cell border wall.
         Total display: 12×12 cells at 28 px/cell → 336×336 px image.
Task:    Move the green snake to eat the red food without hitting walls or itself.
Oracle:  BFS to food; falls back to flood-fill survival when no direct path exists.
"""
from __future__ import annotations

import io
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from PIL import Image, ImageDraw

# ── geometry ──────────────────────────────────────────────────────────────────
GRID_INNER = 10       # playable cells per axis (positions 1..10)
DISPLAY    = 12       # total cells rendered per axis (inner + 1 border on each side)
CELL       = 28       # px per cell → 12 × 28 = 336
IMAGE_SIZE = DISPLAY * CELL   # 336

# ── palette ────────────────────────────────────────────────────────────────────
_BG         = (17,  17,  17)
_BORDER     = (45,  45,  55)
_EMPTY      = (17,  17,  17)
_BODY       = (34, 197,  94)    # #22c55e
_HEAD       = (134, 239, 172)   # #86efac
_FOOD       = (239,  68,  68)   # #ef4444
_FOOD_SHINE = (254, 202, 202)

# ── actions ────────────────────────────────────────────────────────────────────
ACTIONS   = ["UP", "DOWN", "LEFT", "RIGHT", "NONE"]
_DELTA    = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}
_OPPOSITE = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}

PROMPT = (
    "You are playing the classic Snake game. "
    "Move the green snake to eat the red food and grow. "
    "Avoid hitting the gray walls or your own body or the game ends. "
    "Reply with exactly one word: UP, DOWN, LEFT, RIGHT, or NONE (to continue in your current direction)."
)

CONTINUE_PROMPT = PROMPT  # each frame is self-contained; same instruction throughout


def compute_episode_reward(env: "SnakeEnv") -> float:
    """Sparse reward: raw score (food eaten). GRPO normalises within each group."""
    return float(env.score)


@dataclass
class SnakeEnv:
    grid_inner: int = GRID_INNER

    snake:      list[tuple[int, int]] = field(default_factory=list)
    food:       tuple[int, int]       = (0, 0)
    direction:  str                   = "RIGHT"
    score:      int                   = 0
    step_count: int                   = 0
    done:       bool                  = False

    # ── construction ──────────────────────────────────────────────────────────
    def reset(self, rng: Optional[random.Random] = None) -> "SnakeEnv":
        rng = rng or random.Random()
        mid = self.grid_inner // 2 + 1
        self.snake      = [(mid, mid), (mid - 1, mid), (mid - 2, mid)]
        self.direction  = "RIGHT"
        self.score      = 0
        self.step_count = 0
        self.done       = False
        self.food       = self._place_food(rng)
        return self

    def _place_food(self, rng: random.Random) -> tuple[int, int]:
        occupied = set(self.snake)
        free = [
            (x, y)
            for x in range(1, self.grid_inner + 1)
            for y in range(1, self.grid_inner + 1)
            if (x, y) not in occupied
        ]
        return rng.choice(free)

    @classmethod
    def from_state(cls, state: dict) -> "SnakeEnv":
        env            = cls(grid_inner=state["grid_inner"])
        env.snake      = [tuple(p) for p in state["snake"]]
        env.food       = tuple(state["food"])
        env.direction  = state["direction"]
        env.score      = state["score"]
        env.step_count = state.get("step_count", 0)
        env.done       = state.get("done", False)
        return env

    # ── helpers ───────────────────────────────────────────────────────────────
    def _valid(self, pos: tuple[int, int]) -> bool:
        x, y = pos
        return 1 <= x <= self.grid_inner and 1 <= y <= self.grid_inner

    def _bfs_to_food(self) -> Optional[str]:
        """BFS from head toward food. Returns first action or None."""
        head     = self.snake[0]
        body_set = set(self.snake[1:])   # tail vacates each step

        queue: deque[tuple[tuple[int, int], str]] = deque()
        visited = {head}

        for action, (dx, dy) in _DELTA.items():
            if action == _OPPOSITE.get(self.direction):
                continue
            npos = (head[0] + dx, head[1] + dy)
            if npos == self.food:
                return action
            if self._valid(npos) and npos not in body_set:
                visited.add(npos)
                queue.append((npos, action))

        while queue:
            pos, first = queue.popleft()
            for _, (dx, dy) in _DELTA.items():
                npos = (pos[0] + dx, pos[1] + dy)
                if npos == self.food:
                    return first
                if self._valid(npos) and npos not in visited and npos not in body_set:
                    visited.add(npos)
                    queue.append((npos, first))
        return None

    def _flood_size(self, start: tuple[int, int], blocked: set) -> int:
        """Count cells reachable from start avoiding blocked cells."""
        visited = {start}
        queue   = deque([start])
        while queue:
            pos = queue.popleft()
            for dx, dy in _DELTA.values():
                npos = (pos[0] + dx, pos[1] + dy)
                if self._valid(npos) and npos not in visited and npos not in blocked:
                    visited.add(npos)
                    queue.append(npos)
        return len(visited)

    # ── oracle ────────────────────────────────────────────────────────────────
    def oracle_action(self, rng: Optional[random.Random] = None) -> str:
        # 1) Direct BFS path to food
        action = self._bfs_to_food()
        if action is not None:
            return action

        # 2) Survival: pick the move that maximises reachable empty space
        head     = self.snake[0]
        body_set = set(self.snake[1:])
        best_action: Optional[str] = None
        best_flood  = -1

        for action, (dx, dy) in _DELTA.items():
            if action == _OPPOSITE.get(self.direction):
                continue
            npos = (head[0] + dx, head[1] + dy)
            if not self._valid(npos) or npos in body_set:
                continue
            flood = self._flood_size(npos, body_set | {npos})
            if flood > best_flood:
                best_flood  = flood
                best_action = action

        return best_action or self.direction

    # ── dynamics ──────────────────────────────────────────────────────────────
    def step(self, action: str, rng: Optional[random.Random] = None) -> tuple[float, bool]:
        """
        Advance one step.  Returns (reward, done):
            +1.0   food eaten
            -1.0   death (wall or self-collision)
             0.0   normal move
        """
        rng = rng or random.Random()
        if self.done:
            return 0.0, True

        self.step_count += 1

        # NONE, reversal, or unknown token: continue in current direction
        if action == "NONE" or action not in _DELTA or action == _OPPOSITE.get(self.direction):
            action = self.direction

        self.direction = action
        dx, dy = _DELTA[action]
        new_head = (self.snake[0][0] + dx, self.snake[0][1] + dy)

        # Collision check
        if not self._valid(new_head) or new_head in set(self.snake[:-1]):
            self.done = True
            return -1.0, True

        self.snake = [new_head] + self.snake

        if new_head == self.food:
            self.score += 1
            free = [
                (x, y)
                for x in range(1, self.grid_inner + 1)
                for y in range(1, self.grid_inner + 1)
                if (x, y) not in set(self.snake)
            ]
            if not free:
                self.done = True
                return 1.0, True
            self.food = rng.choice(free)
            return 1.0, False

        self.snake.pop()
        return 0.0, False

    # ── rendering ─────────────────────────────────────────────────────────────
    def render(self) -> Image.Image:
        img       = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), _BG)
        draw      = ImageDraw.Draw(img)
        snake_set = set(self.snake)
        head      = self.snake[0] if self.snake else None

        for dy in range(DISPLAY):
            for dx in range(DISPLAY):
                x0 = dx * CELL
                y0 = dy * CELL
                x1 = x0 + CELL - 1
                y1 = y0 + CELL - 1

                is_border = (dx == 0 or dx == DISPLAY - 1 or
                             dy == 0 or dy == DISPLAY - 1)
                if is_border:
                    draw.rectangle([x0, y0, x1, y1], fill=_BORDER)
                    # subtle brick texture
                    if dy % 2 == 0:
                        draw.line([(x0, y0 + CELL // 2), (x1, y0 + CELL // 2)],
                                  fill=(35, 35, 45), width=1)
                    continue

                # display coords 1..10 map to game coords 1..10
                pos = (dx, dy)
                draw.rectangle([x0, y0, x1, y1], fill=_EMPTY)

                if pos == head:
                    draw.rectangle([x0 + 2, y0 + 2, x1 - 2, y1 - 2], fill=_HEAD)
                    eye_off = CELL // 5
                    eye_r   = max(2, CELL // 10)
                    for ex, ey in [(x0 + eye_off, y0 + eye_off),
                                   (x1 - eye_off, y0 + eye_off)]:
                        draw.ellipse([ex - eye_r, ey - eye_r,
                                      ex + eye_r, ey + eye_r], fill=_BG)

                elif pos in snake_set:
                    draw.rectangle([x0 + 3, y0 + 3, x1 - 3, y1 - 3], fill=_BODY)

                elif pos == self.food:
                    margin = CELL // 5
                    draw.ellipse([x0 + margin, y0 + margin,
                                  x1 - margin, y1 - margin], fill=_FOOD)
                    shine_r = max(2, CELL // 10)
                    sx = x0 + margin + shine_r
                    sy = y0 + margin + shine_r
                    draw.ellipse([sx - shine_r, sy - shine_r,
                                  sx + shine_r, sy + shine_r], fill=_FOOD_SHINE)

        return img

    # ── serialization ─────────────────────────────────────────────────────────
    def to_bytes(self) -> bytes:
        buf = io.BytesIO()
        self.render().save(buf, format="PNG")
        return buf.getvalue()

    def get_state(self) -> dict:
        return {
            "grid_inner": self.grid_inner,
            "snake":      [list(p) for p in self.snake],
            "food":       list(self.food),
            "direction":  self.direction,
            "score":      self.score,
            "step_count": self.step_count,
        }
