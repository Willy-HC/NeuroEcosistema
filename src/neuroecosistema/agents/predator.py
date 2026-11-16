
from dataclasses import dataclass

from neuroecosistema.agents.base_agent import BaseAgent


@dataclass
class Predator(BaseAgent):
    """Representa un depredador dentro del ecosistema."""

    speed: float = 1.5
    radius: int = 8
    color: tuple[int, int, int] = (230, 75, 65)