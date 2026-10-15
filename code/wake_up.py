"""wake_up.py

System bootstrap and agent registration module ("wake-up").

Responsibilities:
- Load environment variables from .env (API keys, execution timeouts, sandbox configurations).
- Scan and validate declarations in declarations/ (agents, tools, conversation flows).
- Pair each declared tool schema with its corresponding executable function in code/tools/.
- Instantiate agent pools by injecting their role, LLM configuration, tools, and blackboard permissions.
- Ensure runtime dependencies and sandbox environments are ready before infer.py starts execution.
"""
