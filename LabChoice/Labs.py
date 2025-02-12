from LabChoice import Choices

import numpy as np
import pyglet

gameSize = 64

class ReturnedState:
    img: np.ndarray
    previousReward: float

class GameState:
    selectedChoice: int
    choices: list[Choices.ChoiceObject]
    previousReward: float

    def __init__(self, choices: list[Choices.ChoiceObject]):
        self.selectedChoice = 0
        self.choices = choices
        self.previousReward = 0

    def selectChoice(self):
        print(f'{self.selectedChoice} selected!')
        self.previousReward, self.choices = self.choices[self.selectedChoice].getReward()
        self.selectedChoice = 0

    def getState(self) -> ReturnedState:
        toReturn = ReturnedState()

        toReturn.previousReward = self.previousReward
        toReturn.img = np.array([0])

        return toReturn
    
    def draw(self, offsetX, offsetY):
        print(f'{self.selectedChoice} is currently selected')
        # bkg
        # TODO dynamically set color
        if self.selectedChoice == 1:
            pyglet.shapes.Rectangle(
                offsetX * gameSize,
                offsetY * gameSize,
                gameSize,
                gameSize, 
                (217, 119, 87),
            ).draw()
        else: 
            pyglet.shapes.Rectangle(
                offsetX * gameSize,
                offsetY * gameSize,
                gameSize,
                gameSize,
                (174, 117, 162),
            ).draw()

        # Wall
        if self.selectedChoice == 0:
            pyglet.shapes.Rectangle(
                offsetX * gameSize,
                offsetY * gameSize,
                gameSize*0.1,
                gameSize,
                (0,0,0),
            ).draw()

        if self.selectedChoice == len(self.choices) - 1:
            pyglet.shapes.Rectangle(
                offsetX * gameSize + gameSize*0.9,
                offsetY * gameSize,
                gameSize*0.1,
                gameSize,
                (0, 0, 0),
            ).draw()

        # door
        pyglet.shapes.Rectangle(
            offsetX * gameSize + gameSize*0.3,
            offsetY * gameSize,
            gameSize*0.4,
            gameSize*0.7,
            (52, 22, 7),
        ).draw()

        # image
        # TODO



class Lab:
    _valid: bool
    _games: int
    _states: list[GameState]
    _initialChoices: list[Choices.ChoiceObject]
    _falloff: float


    def __init__(self, choices: list[Choices.ChoiceObject], falloff: float = 0.1) -> None:
        self._valid = False
        self._initialChoices = choices
        self._falloff = falloff

        self.setNumberOfGames(1)

    def setNumberOfGames(self, size) -> None:
        self._valid = False

        self._games = size
        self._states = [GameState(self._initialChoices)
                        for i in range(self._games)]


    def draw(self):
        pyglet.clock.tick()

        self.window.switch_to()
        self.window.dispatch_events()
        
        for state in self._states:
            state.draw(0, 0)
        
        self.window.flip()

    def startEngine(self) -> list[ReturnedState]:
        self.window = pyglet.window.Window(width=gameSize, height=gameSize)

        self.draw()

        self._valid = True

        toReturn: list[ReturnedState]
        toReturn = []

        for i in range(self._games):
            toReturn.append(self._states[i].getState())
        
        return toReturn

    def iterate(self, actions: list[int]) -> list[ReturnedState]:
        if len(actions) != self._games:
            raise ValueError(
                f'Incorrect amount of actions. {len(actions)} given, {self._games} required!')
        
        if not self._valid:
            raise RuntimeError('startEngine must be called before running an iteration.')
        
        toReturn: list[ReturnedState]
        toReturn = []

        for i in range(self._games):
            if actions[i] == 0:
                self._states[i].selectChoice()
            else:
                self._states[i].previousReward -= self._falloff
                self._states[i].selectedChoice = actions[i] - 1

        self.draw()

        for i in range(self._games):
            toReturn.append(self._states[i].getState())

        return toReturn
    