"""code/runtime/

Minimal runtime that executes the debate flows (declarations/flows/abalone_*_debate_flow.json) with local Ollama models.
It is a temporary stand-in for infer.py, which belongs to someone else and is not modified.

- problem_statement.py: CSV row -> task and English problem statement (dataset values for the tool agents, original
  units for the article experts).
- ollama_chat.py: async client for Ollama /api/chat with tools and structured output.
- agent_runner.py: one agent turn: prompt rendering, MCP tool loop, call budgets, output validation, citation check.
- debate.py: rounds, claim and challenge ids, who speaks, agreement check, debate_state and debate_ledger.
- flow_runner.py: walks a flow graph, starts one MCP server per node, keeps the run's blackboard and logs.
- run_debate.py: command line (debate per CSV row, or one question to one article expert).
"""
