from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ...models import ParkingLayout


LayoutAlgorithm = Callable[[ParkingLayout], ParkingLayout]


@dataclass(frozen=True)
class ScenePromptConfig:
    roleDefinition: str
    geometricRules: str
    requiredElements: List[str]
    exampleJSON: str


@dataclass
class SceneDefinition:
    id: str
    name: str
    description: str
    promptConfig: ScenePromptConfig
    styles: Dict[str, Any] = field(default_factory=dict)
    zOrder: List[str] = field(default_factory=list)
    elementNormalization: Dict[str, str] = field(default_factory=dict)
    postProcessAlgorithms: Optional[List[LayoutAlgorithm]] = None

