import gym
from gym import spaces
import numpy as np
import pyglet
from pyglet.gl import *
from pyglet.window import key
import math

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

class MazeGame(pyglet.window.Window):
    def __init__(self, width, height):
        super(MazeGame, self).__init__(width=width, height=height, caption="Maze Navigation Game Debug", resizable=True)
        
        # Enable depth testing so that closer objects obscure further ones.
        glEnable(GL_DEPTH_TEST)
        
        # Background color
        glClearColor(0.5, 0.7, 1.0, 1.0)
        
        # Capture the mouse to make it invisible and lock it to the center of the window.
        # self.set_exclusive_mouse(True)
        
        # Adjust starting position to be more central.
        self.x = 3.5  # starting X position
        self.z = 1.5  # starting Z position
        self.y = 0.5  # camera height
        self.angle = 180.0    # degrees; 0 means facing down the negative Z direction
        
        self.speed = 2.0    # units per second
        self.rotation_speed = 180.0  # degrees per second
        
        self.maze = MAZE
        self.keys = key.KeyStateHandler()
        self.push_handlers(self.keys)
        
        pyglet.clock.schedule_interval(self.update, 1/60.0)

        # Doors with unique symbols and their target rooms
        self.doors = {
            'A': {'position': None, 'probability': 0.75},
            'B': {'position': None, 'probability': 0.25},
        }

        # Initialize door positions
        for row in range(len(self.maze)):
            for col in range(len(self.maze[row])):
                if self.maze[row][col] in self.doors:
                    self.doors[self.maze[row][col]]['position'] = (col, row)
    
    def on_resize(self, width, height):
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
            self.x = 3.5
            self.z = 1.5
            self.angle = 180.0
            self.maze = MAZE

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
                    ch = np.random.choice([1, 0], p=[door_data['probability'], 1 - door_data['probability']], size=1)

                    self.maze = GOOD_ROOM if ch == 1 else BAD_ROOM
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
                # Draw the reward in the Good room
                if self.maze[row][col] == '!':
                    self.draw_cube(col, row, height=1.0, color=(0.93, 0.75, 0.015))
                elif self.maze[row][col] == '#':
                    self.draw_cube(col, row, height=2.0)
                elif self.maze[row][col] != ' ':
                    self.draw_cube(col, row, y=1.0, height=2.0)
                # Draw the penalty in the Bad room

        # Draw doors with the same color and add symbols
        if self.maze == MAZE:
            for door_symbol, door_data in self.doors.items():
                door_col, door_row = door_data['position']
                self.draw_cube(door_col, door_row, height=1.0, color=(0.1, 0.1, 0.1))
                self.draw_door_symbol(door_symbol, door_col, door_row)

        self.draw_floor(color=(0.3, 0.3, 0.3))

    def draw_cube(self, col, row, y=0.0, size=1.0, height=1.0, color=(0.7, 0.2, 0.2)):
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

    def draw_door_symbol(self, door_symbol, col, row):
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
        if door_symbol == 'A':
            #glColor3f(0.0, 1.0, 0.0)  # Green for good outcome

            # Draw an X symbol for the door
            glVertex3f(x - size / 2, y - size / 2, z)  # Bottom left of the X
            glVertex3f(x + size / 2, y + size / 2, z)  # Top right of the X

            glVertex3f(x + size / 2, y - size / 2, z)  # Bottom right of the X
            glVertex3f(x - size / 2, y + size / 2, z)  # Top left of the X

        elif door_symbol == 'B':
            #glColor3f(1.0, 0.0, 0.0)  # Red for bad outcome

            # Draw a cross symbol for the door
            glVertex3f(x, y - size / 2, z)  # Bottom of the vertical line
            glVertex3f(x, y + size / 2, z)  # Top of the vertical line

            glVertex3f(x - size / 2, y, z)  # Left of the horizontal line
            glVertex3f(x + size / 2, y, z)  # Right of the horizontal line

        glEnd()
    
    def get_screenshot(self):
        """
        Capture a screenshot of the current frame.
        """
        self.dispatch_event('on_draw')
        pyglet.image.get_buffer_manager().get_color_buffer().save('screenshot.png')
        return pyglet.image.load('screenshot.png').get_image_data().get_data('RGB', self.width * 3)

    
class MazeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    """
    Custom RL environment for the Pyglet-based maze game.
    """
    def __init__(self, render_mode="human"):
        super(MazeEnv, self).__init__()

        self.width = 256
        self.height = 256
        self.render_mode = render_mode

        # Define the action space: [0: Forward, 1: Backward, 2: Left, 3: Right, 4: Turn left, 5: Turn right]
        self.action_space = spaces.Discrete(6)

        # Define the observation space: RGB image of the environment
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8)

        # Initialize the Pyglet game only if in "human" mode
        self.game = MazeGame(width=self.width, height=self.height) if render_mode == "human" else None
        self.window = self.game if render_mode == "human" else None

        # Store the initial state
        self.state = self.render(mode=self.render_mode)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset the game state
        self.game.x = 3.5
        self.game.z = 1.5
        self.game.angle = 180.0
        self.game.maze = MAZE

        # Return the initial observation
        self.state = self.render(mode=self.render_mode)
        return self.state, {}

    def step(self, action):
        """
        Take a step in the environment based on the action.
        """
        # Temporarily store the action keys
        action_keys = {key.W: False, key.S: False, key.A: False, key.D: False, key.LEFT: False, key.RIGHT: False}

        if action == 0:
            action_keys[key.W] = True
        elif action == 1:
            action_keys[key.S] = True
        elif action == 2:
            action_keys[key.A] = True
        elif action == 3:
            action_keys[key.D] = True
        elif action == 4:
            action_keys[key.LEFT] = True
        elif action == 5:
            action_keys[key.RIGHT] = True

        # Apply the actions to the game
        self.game.keys.update(action_keys)
        self.game.update(1/60.0)  # Assume 60 FPS physics update

        # Reset keys after update to prevent sticky input
        self.game.keys = {key.W: False, key.S: False, key.A: False, key.D: False, key.LEFT: False, key.RIGHT: False, key.ESCAPE: False, key.R: False}

        # Get the new state
        self.state = self.render(mode=self.render_mode)

        # Reward system
        reward = -0.1  # Small penalty for each step
        done = False

        if self.game.maze != MAZE and self.game.z > 6:
            done = True
            if self.game.maze == GOOD_ROOM:
                reward = 10  # Reward for good room
            elif self.game.maze == BAD_ROOM:
                reward = -10  # Penalty for bad room

        return self.state, reward, done, False, {}

    def render(self, mode="human"):
        if mode == "human":
            if self.window is None:
                self.window = MazeGame(width=self.width, height=self.height) # Ensure window is created if switched to human mode
            self.window.switch_to()
            self.window.dispatch_events()
            self.window.dispatch_event("on_draw")
            self.window.flip()
        buffer = pyglet.image.get_buffer_manager().get_color_buffer()
        data = buffer.get_image_data().get_data("RGB", self.width * 3)

        return np.frombuffer(data, dtype=np.uint8).reshape((self.height, self.width, 3))

    def close(self):
        """
        Close the environment and Pyglet window.
        """
        if self.window:
            self.window.close()
            self.window = None