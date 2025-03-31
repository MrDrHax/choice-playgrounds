

from LabChoice import Choices, Labs


if __name__ == '__main__':
    import random
    import time

    class ChoiceA(Choices.ChoiceObject):
        def getReward(self):
            return random.choices([0, 10], weights=[25, 75])[0], choices

        def image(self) -> str:
            return 'test1'

        def name(self) -> str:
            return 'A'

    class ChoiceB(Choices.ChoiceObject):
        def getReward(self):
            return random.choices([0, 10], weights=[75, 25])[0], choices

        def image(self) -> str:
            return 'test2'

        def name(self) -> str:
            return 'B'

    choices = [ChoiceA(), ChoiceB()]

    test = Labs.Lab(choices)

    test.setNumberOfGames(1)

    test.startEngine()

    while True:
        chosen = random.choice([0, 1, 2])
        print(chosen)
        test.iterate([chosen])
        time.sleep(0.01)
