from dataclasses import dataclass

@dataclass(frozen=True)
class Feature:
  command: str
  arguments: list[str]
