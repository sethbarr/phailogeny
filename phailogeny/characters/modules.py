"""Module extraction for agent prompt and capability descriptions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentModule:
    """A single semantic module extracted from an agent profile."""

    module_id: str
    agent_id: str
    kind: str
    label: str
    content: str
    reference: str | None = None


def _split_sections(text: str, heading: str = "heading") -> list[str]:
    """Split a prompt body into sections by markdown headings or paragraph blocks."""
    blocks: list[str] = []
    current: list[str] = []
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("#") and current:
            blocks.append("\n".join(current).strip())
            current = []
        if stripped:
            current.append(raw_line)
    if current:
        blocks.append("\n".join(current).strip())

    if not blocks and text.strip():
        return [text.strip()]

    if heading == "paragraph":
        paragraphs: list[str] = []
        buffer: list[str] = []
        for block in blocks:
            for paragraph in block.split("\n\n"):
                clean = paragraph.strip()
                if clean:
                    paragraphs.append(clean)
        return paragraphs

    return [block for block in blocks if block.strip()]


def split_agent_modules(agent: dict[str, object]) -> list[AgentModule]:
    """Split an agent record into prompt sections and tool modules."""
    modules: list[AgentModule] = []
    agent_id = str(agent.get("agent_id") or agent.get("source_path") or "unknown")

    prompt = str(agent.get("prompt") or "")
    for index, section in enumerate(_split_sections(prompt)):
        modules.append(
            AgentModule(
                module_id=f"{agent_id}:prompt:{index}",
                agent_id=agent_id,
                kind="prompt_section",
                label=f"section-{index}",
                content=section,
                reference=None,
            )
        )

    for index, tool_name in enumerate(agent.get("tools") or []):
        value = str(tool_name)
        modules.append(
            AgentModule(
                module_id=f"{agent_id}:tool:{index}",
                agent_id=agent_id,
                kind="tool",
                label=value,
                content=value,
                reference=value,
            )
        )

    return modules
