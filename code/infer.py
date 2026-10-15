"""infer.py

Main entrypoint for machine learning task execution and flow orchestration.

Responsibilities:
- Ingest the user prompt or ML problem specification (dataset paths, target variable, evaluation metric, constraints).
- Initialize the shared blackboard (blackboard.md) by writing the initial problem specification into the problem_spec section.
- Load and parse the conversational flow graph defined in declarations/flows/.
- Orchestrate the agent sequence according to graph edges, handling turns, cycles, and safety guards.
- Provide each active agent with its relevant prior context, execute its tools, and persist findings to the blackboard.
- Guide execution toward the terminal agent (Consolidator), who reviews the accumulated evidence and produces the final decision.
"""
