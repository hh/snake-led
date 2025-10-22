#!/usr/bin/env python3
"""
Run a tiny snake game across two Framework LED Matrix modules.

Use arrow keys to steer (q to quit).  By default the left matrix is /dev/ttyACM0
and the right is /dev/ttyACM1; override with --left/--right.

Multiple food dots spawn at once; blinking dots are bonus snacks worth extra
points and additional body segments. Watch for 2x2 powerups that double your
snake's length!

For quick sanity checks without a terminal, pass --demo N to run N automated
frames without the curses UI.
""" 

import argparse
import curses
import random
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterable, List, Sequence, Tuple

import serial  # type: ignore

# Protocol constants
FWK_MAGIC = (0x32, 0xAC)
CMD_ANIMATE = 0x04
CMD_STAGE_GREY_COL = 0x07
CMD_DRAW_BUFFER = 0x08

BOARD_WIDTH = 18
BOARD_HEIGHT = 34
MODULE_COLS = 9

SNAKE_BRIGHTNESS_LOW = 0x20  # faint but visible brightness for second player
SNAKE_BRIGHTNESS_HIGH = 0xFF
FOOD_BRIGHTNESS = 0x80
BONUS_BRIGHTNESS = 0xFF
POWERUP_BRIGHTNESS = 0xC0

DEFAULT_TICK = 0.18
TARGET_FOOD_COUNT = 3
BONUS_CHANCE = 0.25
BONUS_VALUE = 3
BONUS_BLINK_PERIOD = 0.25
POWERUP_CHANCE = 0.15  # Chance to spawn a 2x2 powerup


@dataclass
class Module:
    name: str
    port: serial.Serial

    def write(self, payload: Sequence[int]) -> None:
        self.port.write(bytes(payload))

    def send_command(self, command: int, *payload: int) -> None:
        data = list(FWK_MAGIC) + [command] + [int(p) & 0xFF for p in payload]
        self.write(data)


def stage_columns(module: Module, columns: Sequence[Sequence[int]]) -> None:
    for display_col, values in enumerate(columns):
        if len(values) != BOARD_HEIGHT:
            raise ValueError("column length must equal board height")
        device_col = (MODULE_COLS - 1) - display_col  # firmware counts from the right
        module.write(
            list(FWK_MAGIC)
            + [CMD_STAGE_GREY_COL, device_col]
            + [int(v) & 0xFF for v in values]
        )
    module.write(list(FWK_MAGIC) + [CMD_DRAW_BUFFER, 0x00])


def clear_module(module: Module) -> None:
    blank = [0x00] * BOARD_HEIGHT
    stage_columns(module, [blank for _ in range(MODULE_COLS)])


@dataclass
class Food:
    x: int
    y: int
    value: int = 1
    brightness: int = FOOD_BRIGHTNESS
    blink_period: float = 0.0
    created_at: float = field(default_factory=time.monotonic)
    is_powerup: bool = False  # True for 2x2 powerup that doubles snake length

    @property
    def pos(self) -> Tuple[int, int]:
        return (self.x, self.y)

    @property
    def positions(self) -> List[Tuple[int, int]]:
        """Returns all positions occupied by this food item."""
        if self.is_powerup:
            # 2x2 powerup
            return [
                (self.x, self.y),
                ((self.x + 1) % BOARD_WIDTH, self.y),
                (self.x, (self.y + 1) % BOARD_HEIGHT),
                ((self.x + 1) % BOARD_WIDTH, (self.y + 1) % BOARD_HEIGHT),
            ]
        else:
            return [(self.x, self.y)]


@dataclass
class SnakeState:
    name: str
    segments: List[Tuple[int, int]]
    direction: Tuple[int, int]
    pending_growth: int = 0
    score: int = 0
    brightness: int = SNAKE_BRIGHTNESS_HIGH

    @property
    def head(self) -> Tuple[int, int]:
        return self.segments[-1]


def render_board(
    snakes: Sequence[SnakeState],
    foods: Sequence[Food],
    now: float,
) -> Tuple[List[List[int]], List[List[int]]]:
    left = [[0x00 for _ in range(BOARD_HEIGHT)] for _ in range(MODULE_COLS)]
    right = [[0x00 for _ in range(BOARD_HEIGHT)] for _ in range(MODULE_COLS)]

    for snake in snakes:
        for x, y in snake.segments:
            cols = left if x < MODULE_COLS else right
            idx = x if x < MODULE_COLS else x - MODULE_COLS
            cols[idx][y] = max(cols[idx][y], snake.brightness)

    for food in foods:
        brightness = food.brightness
        if food.blink_period > 0:
            phase = int(((now - food.created_at) / food.blink_period)) % 2
            brightness = brightness if phase == 0 else 0
        # Render all positions occupied by this food item
        for fx, fy in food.positions:
            cols = left if fx < MODULE_COLS else right
            idx = fx if fx < MODULE_COLS else fx - MODULE_COLS
            cols[idx][fy] = max(brightness, cols[idx][fy])

    return left, right


def random_free_cell(occupied: Sequence[Tuple[int, int]]) -> Tuple[int, int]:
    occupied_set = set(occupied)
    if len(occupied_set) >= BOARD_WIDTH * BOARD_HEIGHT:
        raise RuntimeError("board is full")
    while True:
        candidate = (random.randrange(BOARD_WIDTH), random.randrange(BOARD_HEIGHT))
        if candidate not in occupied_set:
            return candidate


@contextmanager
def open_module(path: str, label: str) -> Iterable[Module]:
    port = serial.Serial(path, 115200, timeout=0, write_timeout=0)
    module = Module(label, port)
    # Ensure any previous animation/scrolling state is cleared so the game can
    # take exclusive control of the framebuffer updates.
    try:
        module.send_command(CMD_ANIMATE, 0x00)
    except Exception:
        # If this fails we still try to continue; worst-case the module keeps
        # whatever state it had, but the game loop will overwrite the pixels.
        pass
    try:
        yield module
    finally:
        try:
            clear_module(module)
        except Exception:
            pass
        port.close()


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Snake across two Framework LED Matrices")
    parser.add_argument("--left", default="/dev/ttyACM0", help="Serial device for left matrix")
    parser.add_argument("--right", default="/dev/ttyACM1", help="Serial device for right matrix")
    parser.add_argument("--speed", type=float, default=DEFAULT_TICK,
                        help="Seconds between frames (default: %(default)s)")
    parser.add_argument("--demo", type=int, default=0,
                        help="Run without curses UI for N steps (for testing/CI)")
    return parser.parse_args(argv)


def create_initial_snakes() -> List[SnakeState]:
    mid_y = BOARD_HEIGHT // 2
    player_one_dir = (1, 0)
    player_two_dir = (-1, 0)

    def build_segments(head: Tuple[int, int], direction: Tuple[int, int], length: int = 3) -> List[Tuple[int, int]]:
        dx, dy = direction
        segments: List[Tuple[int, int]] = []
        for idx in range(length):
            offset = length - idx - 1
            x = (head[0] - dx * offset) % BOARD_WIDTH
            y = (head[1] - dy * offset) % BOARD_HEIGHT
            segments.append((x, y))
        return segments

    row_gap = max(2, BOARD_HEIGHT // 6)
    player_one_row = max(1, mid_y - row_gap)
    player_two_row = min(BOARD_HEIGHT - 2, mid_y + row_gap)
    player_one_head = (BOARD_WIDTH // 3, player_one_row)
    player_two_head = (BOARD_WIDTH - 3, player_two_row)

    player_one = SnakeState(
        name="P1",
        segments=build_segments(player_one_head, player_one_dir),
        direction=player_one_dir,
        brightness=SNAKE_BRIGHTNESS_HIGH,
    )
    player_two = SnakeState(
        name="P2",
        segments=build_segments(player_two_head, player_two_dir),
        direction=player_two_dir,
        brightness=SNAKE_BRIGHTNESS_LOW,
    )
    return [player_one, player_two]


def spawn_food(occupied: Sequence[Tuple[int, int]]) -> Food:
    occupied_set = set(occupied)

    # Determine food type
    roll = random.random()
    if roll < POWERUP_CHANCE:
        # Spawn 2x2 powerup - need to find a position where all 4 pixels are free
        max_attempts = 100
        for _ in range(max_attempts):
            x = random.randrange(BOARD_WIDTH)
            y = random.randrange(BOARD_HEIGHT)
            powerup_positions = [
                (x, y),
                ((x + 1) % BOARD_WIDTH, y),
                (x, (y + 1) % BOARD_HEIGHT),
                ((x + 1) % BOARD_WIDTH, (y + 1) % BOARD_HEIGHT),
            ]
            if all(p not in occupied_set for p in powerup_positions):
                return Food(
                    x,
                    y,
                    value=0,  # Value will be calculated when eaten
                    brightness=POWERUP_BRIGHTNESS,
                    blink_period=0.0,
                    is_powerup=True,
                )
        # If we can't place a 2x2 powerup, fall through to regular food

    # Regular or bonus food
    pos = random_free_cell(occupied)
    if random.random() < BONUS_CHANCE:
        return Food(
            pos[0],
            pos[1],
            value=BONUS_VALUE,
            brightness=BONUS_BRIGHTNESS,
            blink_period=BONUS_BLINK_PERIOD,
        )
    return Food(pos[0], pos[1])


def ensure_food_supply(snakes: Sequence[SnakeState], foods: List[Food]) -> None:
    occupied = {pos for snake in snakes for pos in snake.segments}
    # Include all positions from all food items (including multi-pixel powerups)
    for food in foods:
        occupied.update(food.positions)
    while len(foods) < TARGET_FOOD_COUNT:
        new_food = spawn_food(list(occupied))
        foods.append(new_food)
        occupied.update(new_food.positions)


def advance_snakes(snakes: List[SnakeState], foods: List[Food]) -> Tuple[bool, str]:
    new_heads: List[Tuple[int, int]] = []
    for snake in snakes:
        head_x, head_y = snake.head
        new_head = ((head_x + snake.direction[0]) % BOARD_WIDTH,
                    (head_y + snake.direction[1]) % BOARD_HEIGHT)
        new_heads.append(new_head)

    if len(set(new_heads)) < len(new_heads):
        return True, "Head-on collision!"

    for idx, snake in enumerate(snakes):
        new_head = new_heads[idx]
        if new_head in snake.segments:
            return True, f"{snake.name} ran into itself!"
        other_segments = {
            pos
            for other_idx, other in enumerate(snakes)
            if other_idx != idx
            for pos in other.segments
        }
        if new_head in other_segments:
            return True, f"{snake.name} crashed into another snake!"

    for idx, snake in enumerate(snakes):
        new_head = new_heads[idx]
        snake.segments.append(new_head)
        # Check if snake head hits any position of any food item
        eaten_idx = next((i for i, food in enumerate(foods) if new_head in food.positions), None)
        if eaten_idx is not None:
            eaten = foods.pop(eaten_idx)
            if eaten.is_powerup:
                # Powerup doubles the snake's current length
                current_length = len(snake.segments)
                growth_amount = current_length
                snake.pending_growth += growth_amount
                snake.score += 10  # Powerup gives 10 points
            else:
                snake.pending_growth += eaten.value
                snake.score += eaten.value
        if snake.pending_growth > 0:
            snake.pending_growth -= 1
        else:
            snake.segments.pop(0)

    ensure_food_supply(snakes, foods)
    return False, ""


def run_demo(left: Module, right: Module, steps: int, tick: float) -> None:
    snakes = create_initial_snakes()
    foods: List[Food] = []
    ensure_food_supply(snakes, foods)

    for step in range(max(1, steps)):
        if step % 7 == 0:
            for snake in snakes:
                snake.direction = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])

        crashed, _ = advance_snakes(snakes, foods)
        if crashed:
            snakes = create_initial_snakes()
            foods.clear()
            ensure_food_supply(snakes, foods)
            continue

        now = time.monotonic()
        left_cols, right_cols = render_board(snakes, foods, now)
        stage_columns(left, left_cols)
        stage_columns(right, right_cols)
        time.sleep(tick)


def game_loop(stdscr: "curses._CursesWindow", left: Module, right: Module, tick: float) -> None:
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(40)

    snakes = create_initial_snakes()
    foods: List[Food] = []
    ensure_food_supply(snakes, foods)
    last_tick = time.monotonic()

    player_controls = [
        {
            curses.KEY_UP: (0, -1),
            curses.KEY_DOWN: (0, 1),
            curses.KEY_LEFT: (1, 0),
            curses.KEY_RIGHT: (-1, 0),
        },
        {
            ord("."): (0, -1),
            ord(">"): (0, -1),
            ord("e"): (0, 1),
            ord("E"): (0, 1),
            ord("o"): (1, 0),
            ord("O"): (1, 0),
            ord("u"): (-1, 0),
            ord("U"): (-1, 0),
        },
    ]

    def set_direction(target: SnakeState, new_dir: Tuple[int, int]) -> None:
        if (target.direction[0] * -1, target.direction[1] * -1) == new_dir:
            return
        target.direction = new_dir

    while True:
        key = stdscr.getch()
        if key in (ord("q"), ord("Q")):
            break
        if key != -1:
            if key in player_controls[0]:
                set_direction(snakes[0], player_controls[0][key])
            elif key in player_controls[1]:
                set_direction(snakes[1], player_controls[1][key])

        now = time.monotonic()
        if now - last_tick < tick:
            continue
        last_tick = now

        crashed, reason = advance_snakes(snakes, foods)
        frame_time = time.monotonic()
        left_cols, right_cols = render_board(snakes, foods, frame_time)
        stage_columns(left, left_cols)
        stage_columns(right, right_cols)

        stdscr.erase()
        stdscr.addstr(0, 0, "Snake on dual Framework LED Matrices")
        stdscr.addstr(
            1,
            0,
            f"P1 (arrows) Score: {snakes[0].score}  Length: {len(snakes[0].segments)}",
        )
        stdscr.addstr(
            2,
            0,
            f"P2 (. up, e down, o left, u right) Score: {snakes[1].score}  Length: {len(snakes[1].segments)}",
        )
        stdscr.addstr(
            3,
            0,
            f"Food items: {len(foods)}  Quit: q",
        )
        stdscr.addstr(
            4,
            0,
            "Bonus dots blink. 2x2 powerups double your snake!",
        )
        if crashed:
            stdscr.addstr(5, 0, f"{reason} Press any key to restart.")
        stdscr.refresh()

        if crashed:
            stdscr.nodelay(False)
            stdscr.getch()
            stdscr.nodelay(True)
            snakes = create_initial_snakes()
            foods.clear()
            ensure_food_supply(snakes, foods)
            clear_module(left)
            clear_module(right)
            last_tick = time.monotonic()


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    tick = args.speed if args.speed > 0 else DEFAULT_TICK

    try:
        with open_module(args.left, "left") as left, open_module(args.right, "right") as right:
            clear_module(left)
            clear_module(right)
            if args.demo:
                run_demo(left, right, args.demo, tick)
            else:
                def wrapped(stdscr: "curses._CursesWindow") -> None:
                    game_loop(stdscr, left, right, tick)

                curses.wrapper(wrapped)
    except serial.SerialException as exc:
        print(f"Failed to open device: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
