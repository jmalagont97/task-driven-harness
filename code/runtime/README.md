# Debate runtime

A minimal runtime that runs the Abalone debate flows with local Ollama models. It is a temporary stand-in for
`code/infer.py` (problem ingestion, blackboard, graph orchestration), which is not modified.

## Quick start

Run from the repository root, with Ollama running and the article index built
(`code/tools/article_rag/README.md`).

```bash
.venv/bin/python -m pip install -r requirements-agents-ollama.txt

# The English problem statement, without calling any model
.venv/bin/python code/runtime/run_debate.py --input evals/abalone_debate/smoke_sex.csv --render-only

# One debate per CSV row (no sex column -> sex debate; no number_of_rings column -> age debate)
.venv/bin/python code/runtime/run_debate.py --input evals/abalone_debate/smoke_sex.csv --model gemma4:e4b

# Rows of the full dataset: --task hides that column and keeps it as the truth
.venv/bin/python code/runtime/run_debate.py --input artifacts/datasets/abalone_dataset/train/abalone_dataset.csv --task age --rows 0 5

# One standalone question to one article expert
.venv/bin/python code/runtime/run_debate.py --only article_mcshane_1992_agent --question "¿Se forman los anillos una vez al año?"
```

`--model` sets the Ollama model of every agent for the run (`gemma4:e4b` for development, `gemma4:31b` for the final
evaluation); without it each declaration's `model.model_name` is used. `--think/--no-think` overrides
`runtime.adapters.ollama.think`, and `--num-ctx` overrides `runtime.adapters.ollama.num_ctx`.

`gemma4:31b` does not fit a 32 GB GPU with 8 parallel requests of 32768 tokens. It fits whole with
`OLLAMA_NUM_PARALLEL=4` and `--num-ctx 20480`, which is above the largest request of the e4b evaluation (about 17,000
tokens).

### Forced-class sex debate

`declarations/flows/abalone_sex_forced_debate_flow.json` is the sex debate in which the tool agents
(`abalone_sex_forced_agent_{a,b}`) and the arbiter (`abalone_debate_forced_consolidator_agent`) must answer M, F or
I: `adult_undetermined` is not in their schemas. For an adult they pick the larger of P(M) and P(F), and
`mf_ambiguous` lowers the reliability instead of withholding the class. If the arbiter fails, the runtime fallback
picks the class with the largest summed probability over the debate's tool results. Run it with
`--flow declarations/flows/abalone_sex_forced_debate_flow.json`, or `--forced-sex` in
`evals/abalone_debate/run_debate_eval.py`.

## The debate

Each task has a panel of six participants, each with its own MCP server:

| Participant | Server | Sees |
|---|---|---|
| `abalone_{task}_agent_a` | `code/agent_mcp_server.py` | its two tools (one ML, one expert) |
| `abalone_{task}_agent_b` | `code/agent_mcp_server.py` | two other tools |
| `article_{nash_1994, prince_1988a, prince_1988b, mcshane_1992}_agent` | `code/article_mcp_server.py` | `search_article` and `get_section` of one article |

Rounds are passes around the flow's ring (A, B, then the four experts):

1. **Opinions.** Tool agents call their two tools and give a position with claims; experts search their article and
   give claims with `chunk_id`s, or abstain when the article does not cover the specimen (they do not speak again).
2. **Challenges.** Everyone who did not abstain confirms or challenges other participants' claims (at most 3 reviews,
   always with evidence).
3. **Replies.** Only the challenged participants speak: `defend`, `concede` or `clarify`; nothing new may be added.
4. **Extra cycle** (rounds 4 and 5), only when the tool agents still disagree and a defended challenge involves them.

A participant with nothing to do in a round passes without a model call. Agreement is computed by the runtime from
the tool agents' positions (sex: same maturity and call; age: within 1 ring with overlapping 80% intervals). The
arbiter (`abalone_debate_consolidator_agent`) reads the `debate_ledger` (every claim, who challenged it, why, the
reply and its status) and the participants' sections, and writes the final answer with the weight it gave each
participant.

## A run folder

`artifacts/runs/<timestamp>-<task>-row<n>/` (not versioned):

| File | Content |
|---|---|
| `problem.md` | The English problem statement |
| `blackboard.md` | The run's blackboard (the repository's `blackboard.md` is never written) |
| `debate_state.json` | Positions, expectations, abstentions, the ledger and the round events |
| `final_decision.json` | The arbiter's answer, the ledger per participant, the truth when known, totals |
| `transcript.jsonl` | Every model call (tokens, seconds), tool call, edge, skip and turn output |
| `mcp_<node>.log` | Each MCP server's stderr: tool calls, scores, scope violations |

## How it fits the harness schema

- Each node's MCP server is the `mcp_servers` entry with the node's id as key.
- `debate_state` and `debate_ledger` are blackboard sections written by the runtime, as `infer.py` writes
  `problem_spec`; participants read `problem_spec` and `debate_state`, the arbiter reads everything.
- A loop edge is taken while its `continue_condition` holds and `cycle_count < max_cycle_iterations`; when no edge
  applies, or `global_max_turns`/`max_visits` is exhausted, the run goes to the terminal node (`fallback_target`).
- The template's `{{output_format}}` placeholder gets a short note; the full JSON Schema is sent with the final
  instruction and as Ollama's `format`, so tool-calling exchanges carry a shorter prompt.
- Every agent with `can_use_rag: true` may cite only `chunk_id`s its own tools returned in the run (one corrective
  retry, then invalid citations are removed).

## Speed on this machine

Ollama runs on the RTX 5090 (32 GB). Measured on 2026-10-23: `gemma4:e4b` writes about 96-178 tokens/s and reads
about 13,000 prompt tokens/s; `gemma4:31b` writes about 35 tokens/s. The Ollama service sets
`OLLAMA_MAX_LOADED_MODELS=2` and `OLLAMA_NUM_PARALLEL=1`, so gemma4 and `embeddinggemma` stay loaded together and a
search does not evict the chat model. Before the GPU driver was fixed everything ran on CPU at 7.9 tokens/s, with
turns of 1.5-3 minutes.

## Tests

```bash
.venv/bin/python -m unittest tests.test_debate_protocol tests.test_problem_statement tests.test_agent_scoped_server -v
```

`tests/test_debate_protocol.py` runs a whole sex debate with a scripted model and fake tool gateways, so it needs no
model server.
