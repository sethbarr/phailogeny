#!/usr/bin/env bash
# Shallow-clone the public subagent collections used in the analysis into data/raw/.
set -euo pipefail
cd "$(dirname "$0")/../data/raw" 2>/dev/null || { mkdir -p "$(dirname "$0")/../data/raw"; cd "$(dirname "$0")/../data/raw"; }
while read -r repo dir; do
  [ -d "$dir" ] || git clone --depth 1 -q "https://github.com/$repo.git" "$dir"
  echo "$dir $(git -C "$dir" rev-parse --short HEAD)"
done <<LIST
wshobson/agents wshobson
VoltAgent/awesome-claude-code-subagents voltagent
lst97/claude-code-sub-agents lst97
VoltAgent/awesome-codex-subagents voltagent-codex
davepoon/buildwithclaude buildwithclaude
0xfurai/claude-code-subagents 0xfurai
contains-studio/agents contains-studio
0xSteph/pentest-ai-agents pentest
vijaythecoder/awesome-claude-agents vijaythecoder
glittercowboy/taches-cc-resources taches
LIST
