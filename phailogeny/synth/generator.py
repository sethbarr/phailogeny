"""Synthetic estate generator for validation of the analysis pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class SyntheticAgent:
    """Minimal synthetic agent definition used in validation tests."""

    role: str
    prompt: str
    tools: tuple[str, ...]


class SyntheticEstateGenerator:
    """Generate a small synthetic estate with roles and simple planted relations."""

    def __init__(self, roles: int = 20, agents_per_role: int = 10, seed: int = 7) -> None:
        self.roles = roles
        self.agents_per_role = agents_per_role
        self.seed = seed

    def generate(self) -> list[SyntheticAgent]:
        """Generate a synthetic estate with role-specific prompts and tools."""
        agents: list[SyntheticAgent] = []
        for role_index in range(self.roles):
            role_name = f"role-{role_index}"
            for agent_index in range(self.agents_per_role):
                prompt = f"You are a {role_name}. Focus on task {agent_index}."
                tools = ("read_file", "write_file") if agent_index % 2 == 0 else ("read_file", "grep")
                agents.append(SyntheticAgent(role=role_name, prompt=prompt, tools=tools))
        return agents

    def write_truth(self, path: str, agents: list[SyntheticAgent]) -> None:
        """Write minimal true ancestry metadata for the synthetic estate."""
        payload = {
            "roles": [agent.role for agent in agents],
            "agents": [
                {
                    "role": agent.role,
                    "prompt": agent.prompt,
                    "tools": list(agent.tools),
                }
                for agent in agents
            ],
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
