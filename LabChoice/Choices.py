from abc import ABC, abstractmethod

class ChoiceObject(ABC):

    @abstractmethod
    def getReward(self) -> tuple[float, list['ChoiceObject'] | None]:
        '''
        getReward The reward the model should get if it gets chosen.

        Returns:
            float: the reward (if any) for the choice.
            list['ChoiceObject']: the next possible choices. If None, game should be considered finished.
        '''

    @abstractmethod
    def image(self) -> str:
        '''
        image Image to use for the door.

        Returns:
            str: The name of the image, loaded in the image atlas.
        '''

    @abstractmethod
    def name(self) -> str:
        '''
        name The name of the door.

        Returns:
            str: The name.
        ''' 
