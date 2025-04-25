import gym
from gym import spaces
import numpy as np
import pyglet
from pyglet.gl import *
from pyglet.window import key
import math
import torch

import random

import threading

fps = 1/60.0

doorOptions = {
    'A': {'position': None, 'probability': 0.75, 'reward': 10, 'symbol': 'A'},
    'B': {'position': None, 'probability': 0.25, 'reward': 10, 'symbol': 'B'},
    'C': {'position': None, 'probability': 0.75, 'reward': 3, 'symbol': 'C'},
    'D': {'position': None, 'probability': 0.25, 'reward': 3, 'symbol': 'D'},
}

doorPairs = [
    ['A', 'B'],
    ['C', 'D'],
]

# A simple maze layout.
MAZE = [
    "#######",
    "#     #",
    "#     #",
    "#     #",
    "#     #",
    "#A###B#",
]

# Final room layout with good outcome
GOOD_ROOM = [
    "#######"
    '  # #  ',
    '  # #  ',
    '  # #  ',
    '  # #  ',
    "### ###",
    "#     #",
    "# !!! #",
    "#     #",
    "#######"
]

# Final room layout with bad outcome
BAD_ROOM = [
    "#######",
    '  # #  ',
    '  # #  ',
    '  # #  ',
    '  # #  ',
    "### ###",
    "#     #",
    "#     #",
    "#     #",
    "#######"
]


def threadWorker(wrappers: list['GameWrapper'], actions: list[list[bool]], results: list, index: int):
    results[index] = [w.step(a) for w, a in zip(wrappers, actions)]


class multiGames:

    def __init__(self, width: int, height: int, batches: int = 10, size: int = 50):
        self.batches = batches
        self.size = size

        self.games = []

        for i in range(batches):
            self.games.append([GameWrapper(width, height, False)
                              for _ in range(size)])

    def step(self, inputs: list[list[bool]]) -> tuple[bytes, int, bool]:
        threads = []

        results = [None for _ in range(self.batches)]

        for i in range(self.batches):
            threadWorker(
                self.games[i],
                inputs[i * self.size: i * self.size + self.size],
                results,
                i,
            )

        # for i in range(self.batches):
        #     t = threading.Thread(
        #         target=threadWorker,
        #         args=(
        #             self.games[i],
        #             inputs[i * self.size: i * self.size + self.size],
        #             results,
        #             i,
        #         ),
        #     )
        #     threads.append(t)
        #     t.start()

        # for t in threads:
        #     t.join()

        # flatten
        return [item for sublist in results for item in sublist]


class MazeGame(pyglet.window.Window):
    def __init__(self, width, height, visible=True):
        super(MazeGame, self).__init__(
            width=width,
            height=height,
            caption="Maze Navigation Game Debug",
            resizable=False,
            visible=visible
        )

        self.fixed_width = width
        self.fixed_height = height

        # Enable depth testing so that closer objects obscure further ones.
        glEnable(GL_DEPTH_TEST)

        # Background color
        glClearColor(0.5, 0.7, 1.0, 1.0)

        # Capture the mouse to make it invisible and lock it to the center of the window.
        # self.set_exclusive_mouse(True)

        self.speed = 2.0    # units per second
        self.rotation_speed = 180.0  # degrees per second

        self.reset()

        self.keys = key.KeyStateHandler()
        self.push_handlers(self.keys)

    def reset(self):
        '''Reset the maze'''
        # Adjust starting position to be more central.
        self.x = 3.5  # starting X position
        self.z = 1.5  # starting Z position
        self.y = 0.5  # camera height
        self.angle = 180.0    # degrees; 0 means facing down the negative Z direction

        self.maze: list[str] = MAZE

        self.selectedDoor: dict = None

        # Doors with unique symbols and their target rooms
        selected = random.choice(doorPairs)
        rotated = random.choice([True, False])

        # Select the first and second door. Rotate them if needed
        self.doors: dict[dict] = {
            'A': doorOptions[selected[rotated]],
            'B': doorOptions[selected[not rotated]],
        }

        prob = self.doors[random.choice(['A', 'B'])]['probability']
        isInverse = random.random() <= prob

        # Initialize door positions
        for row in range(len(self.maze)):
            for col in range(len(self.maze[row])):
                if self.maze[row][col] in self.doors:
                    self.doors[self.maze[row][col]]['position'] = (col, row)

                    # pre cook probability to show possible good/bad outcome

                    self.doors[self.maze[row][col]
                               ]['signal'] = isInverse

    def on_resize(self, width, height):
        # Enforce fixed size
        if width != self.fixed_width or height != self.fixed_height:
            self.set_size(self.fixed_width, self.fixed_height)

        # Use framebuffer size in case of high-DPI displays.
        fb_width, fb_height = self.get_framebuffer_size()
        glViewport(0, 0, fb_width, fb_height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(65.0, fb_width / float(fb_height), 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)
        return pyglet.event.EVENT_HANDLED

    def update(self, dt):
        dx, dz = 0.0, 0.0

        if self.keys[key.W]:
            dx += math.sin(math.radians(self.angle))
            dz += -math.cos(math.radians(self.angle))
        if self.keys[key.S]:
            dx -= math.sin(math.radians(self.angle))
            dz -= -math.cos(math.radians(self.angle))
        if self.keys[key.A]:
            dx += math.sin(math.radians(self.angle - 90))
            dz += -math.cos(math.radians(self.angle - 90))
        if self.keys[key.D]:
            dx += math.sin(math.radians(self.angle + 90))
            dz += -math.cos(math.radians(self.angle + 90))

        dist = math.hypot(dx, dz)
        if dist > 0:
            dx = dx / dist * self.speed * dt
            dz = dz / dist * self.speed * dt

            new_x = self.x + dx
            new_z = self.z + dz
            if not self.collides(new_x, new_z):
                self.x = new_x
                self.z = new_z

        if self.keys[key.LEFT]:
            self.angle += self.rotation_speed * dt
        if self.keys[key.RIGHT]:
            self.angle -= self.rotation_speed * dt

        if self.keys[key.ESCAPE]:
            pyglet.app.exit()

        if self.keys[key.R]:
            self.reset()

        # Check if the player is in a door
        if self.maze == MAZE:
            for door_symbol, door_data in self.doors.items():
                door_col, door_row = door_data['position']
                # Check if the player is inside the door
                if abs(int(self.x) - door_col) <= 1 and abs(int(self.z) - door_row) <= 1:
                    # Teleport the player
                    self.x = 3.5
                    self.z = 1.5
                    self.angle = 180.0  # Reset angle to face the corridor

                    # Change the maze to the target room of the door, depending on the probability
                    ch = np.random.choice(
                        [True, False],
                        p=[0.9, 0.1],
                        size=1,
                    ) ^ door_data['signal']  # switch if signal is on/off

                    self.maze = GOOD_ROOM if ch else BAD_ROOM
                    self.selectedDoor = door_data
                    break

    def collides(self, new_x, new_z):
        col = int(new_x)
        row = int(new_z)
        if row < 0 or row >= len(self.maze) or col < 0 or col >= len(self.maze[0]):
            return True
        if self.maze[row][col] == '#' or self.maze[row][col] == '!':
            return True
        if self.maze[row][col] in self.doors:  # Doors are interactable, not collidable
            return False
        return False

    def on_draw(self):
        self.clear()
        glLoadIdentity()

        # Set up the camera view.
        direction_x = math.sin(math.radians(self.angle))
        direction_z = -math.cos(math.radians(self.angle))
        gluLookAt(self.x, self.y, self.z,
                  self.x + direction_x, self.y, self.z + direction_z,
                  0, 1, 0)

        # Draw the maze walls
        for row in range(len(self.maze)):
            for col in range(len(self.maze[row])):

                if self.maze[row][col] == '!':
                    # Draw the reward in the Good room
                    self.draw_cube(
                        col,
                        row,
                        height=1.0,
                        color=(0.93, 0.75, 0.015)
                    )
                elif self.maze[row][col] == '#':
                    # draw a walled cube
                    self.draw_cube(
                        col,
                        row,
                        height=2.0,
                    )
                elif self.maze[row][col] != ' ':
                    # draw a smaller cube i guess?
                    self.draw_cube(
                        col,
                        row,
                        y=1.0,
                        height=2.0,
                    )
                # Draw the penalty in the Bad room

        # Draw doors with the same color and add symbols
        if self.maze == MAZE:
            for door_symbol, door_data in self.doors.items():
                door_col, door_row = door_data['position']
                self.draw_cube(
                    door_col,
                    door_row,
                    height=1.0,
                    color=(0.1, 0.1, 0.1),
                )
                self.draw_door_symbol(
                    door_data['symbol'],
                    door_col,
                    door_row,
                    door_data['signal'],
                )

        self.draw_floor(color=(0.3, 0.3, 0.3))

    def draw_cube(self, col, row, y=0.0, size=1.0, height=1.0, color=(0.7, 0.5, 0.2), ):
        x = col
        z = row
        glColor3f(*color)

        glBegin(GL_QUADS)
        # Front face
        glVertex3f(x, y, z)
        glVertex3f(x+size, y, z)
        glVertex3f(x+size, height, z)
        glVertex3f(x, height, z)

        # Back face
        glVertex3f(x, y, z+size)
        glVertex3f(x+size, y, z+size)
        glVertex3f(x+size, height, z+size)
        glVertex3f(x, height, z+size)

        # Left face
        glVertex3f(x, y, z)
        glVertex3f(x, y, z+size)
        glVertex3f(x, height, z+size)
        glVertex3f(x, height, z)

        # Right face
        glVertex3f(x+size, y, z)
        glVertex3f(x+size, y, z+size)
        glVertex3f(x+size, height, z+size)
        glVertex3f(x+size, height, z)

        # Top face
        glVertex3f(x, height, z)
        glVertex3f(x+size, height, z)
        glVertex3f(x+size, height, z+size)
        glVertex3f(x, height, z+size)

        # Bottom face
        glVertex3f(x, y, z)
        glVertex3f(x+size, y, z)
        glVertex3f(x+size, y, z+size)
        glVertex3f(x, y, z+size)
        glEnd()

    def draw_floor(self, color=(0.3, 0.3, 0.3)):
        glColor3f(*color)
        glBegin(GL_QUADS)
        glVertex3f(0, 0, 0)
        glVertex3f(len(self.maze[0]), 0, 0)
        glVertex3f(len(self.maze[0]), 0, len(self.maze))
        glVertex3f(0, 0, len(self.maze))
        glEnd()

    def draw_door_symbol(self, door_symbol, col, row, signal=True):
        """
        Draws a symbol on the door to differentiate between doors.
        """
        # Calculate the 3D position of the door's center
        x = col + 0.5  # Center of the door
        z = row - 0.01  # Slightly lower than the door height to avoid z-fighting
        y = 1.5  # Height of the cross on the door

        size = 0.5  # Size of the symbol

        glColor3f(1.0, 1.0, 1.0)  # White color for visibility
        glLineWidth(3.0)  # Set line width for better visibility

        glBegin(GL_LINES)

        # Rotate the door symbol based on the symbol
        match door_symbol:
            case 'A':
                if signal:
                    glColor3f(0.0, 1.0, 0.0)  # Green for normal outcome
                else:
                    glColor3f(1.0, 0.0, 0.0)  # Red for bad outcome

                # Draw an X symbol for the door
                # Bottom left of the X
                glVertex3f(x - size / 2, y - size / 2, z)
                glVertex3f(x + size / 2, y + size / 2, z)  # Top right of the X

                # Bottom right of the X
                glVertex3f(x + size / 2, y - size / 2, z)
                glVertex3f(x - size / 2, y + size / 2, z)  # Top left of the X

            case 'B':
                if signal:
                    glColor3f(0.0, 1.0, 0.0)  # Green for normal outcome
                else:
                    glColor3f(1.0, 0.0, 0.0)  # Red for bad outcome

                # Draw a cross symbol for the door
                glVertex3f(x, y - size / 2, z)  # Bottom of the vertical line
                glVertex3f(x, y + size / 2, z)  # Top of the vertical line

                glVertex3f(x - size / 2, y, z)  # Left of the horizontal line
                glVertex3f(x + size / 2, y, z)  # Right of the horizontal line

            case 'C':
                if signal:
                    glColor3f(0.0, 1.0, 0.0)  # Green for normal outcome
                else:
                    glColor3f(1.0, 0.0, 0.0)  # Red for bad outcome

                # Draw an box symbol for the door

                # top
                glVertex3f(x - size / 2, y + size / 2, z)
                glVertex3f(x + size / 2, y + size / 2, z)

                # right
                glVertex3f(x + size / 2, y - size / 2, z)
                glVertex3f(x + size / 2, y + size / 2, z)

                # bottom
                glVertex3f(x - size / 2, y - size / 2, z)
                glVertex3f(x + size / 2, y - size / 2, z)

                # left
                glVertex3f(x - size / 2, y - size / 2, z)
                glVertex3f(x - size / 2, y + size / 2, z)

            case 'D':
                if signal:
                    glColor3f(0.0, 1.0, 0.0)  # Green for normal outcome
                else:
                    glColor3f(1.0, 0.0, 0.0)  # Red for bad outcome

                # Draw a diamond symbol for the door
                # top
                glVertex3f(x + size / 2, y, z)
                glVertex3f(x, y + size / 2, z)

                # right
                glVertex3f(x, y + size / 2, z)
                glVertex3f(x - size / 2, y, z)

                # bottom
                glVertex3f(x - size / 2, y, z)
                glVertex3f(x, y - size / 2, z)

                # left
                glVertex3f(x, y - size / 2, z)
                glVertex3f(x + size / 2, y, z)

        glEnd()

    def get_screenshot(self, save=False, name='screenshot.png'):
        """
        Capture a screenshot of the current frame.
        """
        self.dispatch_event('on_draw')

        if save:
            pyglet.image.get_buffer_manager().get_color_buffer().save(name)

        # Get the buffer and convert to numpy array
        buffer = pyglet.image.get_buffer_manager().get_color_buffer()
        image_data = buffer.get_image_data()
        width, height = buffer.width, buffer.height
        pitch = -(image_data.width * len(image_data.format))  # Flip vertically

        # Raw image bytes to numpy array
        img_data = np.frombuffer(image_data.get_data(
            image_data.format, -pitch), dtype=np.uint8)
        img_data = img_data.reshape(
            (height, width, len(image_data.format)))  # (H, W, C)

        # Drop the alpha channel if it exists
        if img_data.shape[2] == 4:
            img_data = img_data[:, :, :3]

        # Convert to PyTorch tensor and flip vertically
        tensor = torch.from_numpy(img_data).permute(
            2, 0, 1).flip(1)  # Shape: (C, H, W), flip Y-axis

        # Normalize to [0, 1]
        tensor = tensor.float() / 255.0

        # Reshape to (1, C, H, W)
        # tensor = tensor.unsqueeze(0)

        return tensor


class GameWrapper:
    width: int
    height: int

    showWindow: bool

    def __init__(self, width: int, height: int, showWindow: bool = True):
        self.width = width
        self.height = height
        self.showWindow = showWindow

        self.game = MazeGame(width=width, height=height, visible=showWindow)

        self.reward = 0

    def step(self, actions: list[bool]) -> tuple[bytes, int, bool]:
        '''
        step Step through an iteration. In game terms "Tick"

        Args:
            actions (list[bool]): A list of the 6 actions the player must take. Only works if showWindow is false

        Raises:
            IndexError: If list does not contain the proper amount of items

        Returns:
            tuple[bytes, int, bool]: The (image, reward, completed?) states that are used to train the AI.
        '''
        # if not self.showWindow:
        if len(actions) != 6:
            raise IndexError(
                f'actions does not have the correct amount of actions. Expected 6, got {len(actions)}'
            )

        action_keys = {
            key.W: actions[0],
            key.S: actions[1],
            key.A: actions[2],
            key.D: actions[3],
            key.LEFT: actions[4],
            key.RIGHT: actions[5],
        }

        self.game.keys.update(action_keys)

        self._run_scheduler()

        # Get the image of the game
        image = self.game.get_screenshot()

        # mientras mas tiempo se pase sin moverse
        self.reward -= 0.01 if not (actions[0]
                                    or actions[1]
                                    or actions[2]
                                    or actions[3]) else 0
        done = False

        if self.game.maze != MAZE:
            # Reward for reaching next room
            self.reward += 0.001
            if self.game.z > 4:
                done = True
                if self.game.maze == GOOD_ROOM:
                    # Reward for good room
                    self.reward += self.game.selectedDoor['reward']
                elif self.game.maze == BAD_ROOM:
                    # Penalty for bad room
                    self.reward -= -self.game.selectedDoor['reward']

        return image, self.reward, done

    def reset(self):
        """
        Reset the game to its initial state
        """
        self.game.reset()
        self.reward = 0

        return self.game.get_screenshot(), 0, False

    def close(self):
        self.game.close()
        self.game = None

    def _run_scheduler(self):
        pyglet.clock.tick()

        self.game.switch_to()
        self.game.dispatch_events()

        self.game.on_draw()

        self.game.flip()
