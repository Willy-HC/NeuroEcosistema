
from dataclasses import dataclass

from neuroecosistema.agents.base_agent import BaseAgent


@dataclass
class Prey(BaseAgent):
    """Representa una presa dentro del ecosistema."""

    speed: float = 1.8
    radius: int = 5
    color: tuple[int, int, int] = (80, 180, 255)