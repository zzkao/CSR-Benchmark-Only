from abc import ABC, abstractmethod
from state import *

class BaseAgent(ABC):
    """
    The minimum necessary to have an agent interact with the environment.
    """

    def __init__(self):
        pass

    @abstractmethod
    def step(self, environment_history: list[State]) -> Action:
        pass