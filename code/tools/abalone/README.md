# Abalone tools

Eight read-only tools that estimate the **sex** (M = male, F = female, I = infant) or the **age** (ring count; age in
years ≈ rings + 1.5) of an abalone from its shell measurements. For each task there are two ML models and two expert
tools (rules or simple statistics), so an agent can compare a statistical opinion with one a biologist can check by hand.
All eight are declared in `declarations/tools/abalone_*.json` and served over MCP by `code/mcp_server.py`.

| Tool | Task | Kind | Method | Hold-out result |
|---|---|---|---|---|
| `abalone_sex_random_forest` | sex | ML | Random Forest, 500 trees | accuracy 54.4 %, infant vs adult 82.3 % |
| `abalone_sex_logistic_regression` | sex | ML | Multinomial logistic regression | accuracy 55.7 %, infant vs adult 80.0 % |
| `abalone_sex_maturity_rules` | sex | expert | 2 rules on shell_weight, diameter, rings; abstains on M/F | infant vs adult 81.2 % |
| `abalone_sex_bivariate_profile` | sex | expert | Bivariate normal profiles of (diameter, rings) + Bayes' rule | accuracy 52.7 %, infant vs adult 79.0 % |
| `abalone_age_gradient_boosting` | age | ML | HistGradientBoosting on log1p(rings) + cross-conformal interval | MAE 1.51, R² 0.55, 80 % coverage 80.1 % |
| `abalone_age_ridge_regression` | age | ML | RidgeCV + cross-conformal interval | MAE 1.65, R² 0.40, 80 % coverage 79.6 % |
| `abalone_age_allometric_formula` | age | expert | rings = a · shell_weight^b per sex | MAE 1.75, R² 0.41, 80 % coverage 82.6 % |
| `abalone_age_shell_weight_lookup` | age | expert | shell_weight decile × sex → median, P10–P90 | MAE 1.74, R² 0.34, 80 % coverage 84.9 % |

Measured on a frozen stratified 80/20 split (seed 42; 3,340 train / 835 test rows; the 2 rows with `height = 0` are
excluded). Male vs female is near chance for every tool (51.7–55.1 % among adults, against 54 % for always answering
"M"), so the sex tools report the infant-vs-adult decision first and flag ambiguous M/F calls.

## Quick start

Run everything from the repository root.

```bash
# 1. Dependencies (adds the MCP SDK on top of requirements.txt)
.venv/bin/python -m pip install -r requirements-abalone-tools.txt

# 2. Train: freezes the split, fits the models, estimates the expert parameters, writes the artifacts (~10 s)
.venv/bin/python code/tools/abalone/train_abalone_tools.py

# 3. Test (32 tests, including a real MCP client talking to the server over stdio)
.venv/bin/python -m unittest discover -s tests -v

# 4. Check the declarations, or serve the tools over MCP (stdio)
.venv/bin/python code/mcp_server.py --server abalone_tools --check
.venv/bin/python code/mcp_server.py --server abalone_tools
```

`.mcp.json` registers the server for MCP clients that read it (for example Claude Code); the client asks before
starting it. To try the tools by hand, point the MCP Inspector at the same command (the server defaults to
`--server abalone_tools`): `npx @modelcontextprotocol/inspector .venv/bin/python code/mcp_server.py`.

Calling a tool from Python without MCP:

```python
import sys; sys.path.insert(0, "code")
import tool_registry
tools = tool_registry.load_tool_declarations(server="abalone_tools")
tool_registry.invoke(tools["abalone_age_shell_weight_lookup"], {"specimens": [{
    "length": 0.475, "diameter": 0.365, "height": 0.13, "whole_weight": 0.4805,
    "shucked_weight": 0.1905, "viscera_weight": 0.114, "shell_weight": 0.1475, "sex": "F"}]})
```

## Contract

- **Input**: `{"specimens": [...]}` with 1–50 specimens. Every measurement is in dataset units, the original
  millimetres or grams divided by 200 (95 mm → 0.475). Sex tools need `number_of_rings`; age tools need `sex`
  (the two expert age tools also accept `"unknown"`). Values that look like millimetres or grams are rejected with a
  hint, so the calling LLM can fix them.
- **Output**: the repository envelope `{ok, tool_id, version, result, flags, warnings, error, disclaimer}`.
  `result.predictions` has one entry per specimen; `result.model_card` carries the hold-out metrics read from
  `metrics.json`. A failure is `ok: false` with `error.type` in `input_error | artifact_missing | runtime_error`;
  executors never raise.
- **MCP mapping**: `name` → tool name, `mcp.title` → title, `prompt_context` → description, `input_schema` /
  `output_schema` → inputSchema / outputSchema, `mcp.annotations` → annotations (read-only, idempotent, closed world).
  Results carry the envelope both as `structuredContent` and as serialized text; tool failures come back with
  `isError: true`, and only an unknown tool name is a JSON-RPC error.

## Files

| Path | Role |
|---|---|
| `code/tools/abalone/common.py` | Paths, input validation, feature frames, envelope, artifact loading, frozen split |
| `code/tools/abalone/sex_models.py`, `sex_experts.py` | The four sex executors |
| `code/tools/abalone/age_models.py`, `age_experts.py` | The four age executors |
| `code/tools/abalone/train_abalone_tools.py` | Offline training and evaluation |
| `code/tool_registry.py` | Loads and validates declarations, resolves entrypoints, invokes executors |
| `code/mcp_server.py` | Generic MCP server driven by the declarations |
| `declarations/tools/abalone_*.json` | The eight tool declarations (single source of each contract) |
| `declarations/agents/abalone_*_agent.json` | Sex specialist, age specialist and read-only consolidator |
| `declarations/flows/abalone_tools_flow.json` | Flow graph: sex agent → age agent (when needed) → consolidator |
| `tests/test_abalone_tools.py` | Tests for all of the above |
| `requirements-abalone-tools.txt`, `.mcp.json` | Extra dependencies and MCP client registration |
| `artifacts/datasets/abalone_dataset/splits.json` | Frozen split (generated) |
| `artifacts/models/abalone/` | Models, `expert_params.json`, `metrics.json` (generated) |

`artifacts/*` is git-ignored, so the split and the models are not versioned. Training is deterministic: running the
training script reproduces them, and `splits.json` records the dataset hash so a changed CSV is detected.

## Design notes

- **Declarations are the source of truth.** Paths, thresholds (the maturity rules live in `config.rules`), limits and
  descriptions are read from the JSON files. The training script reads the same declarations, so artifacts land where
  the tools look for them. Editing `config.rules` without retraining is detected: the tool flags `rule_stats_stale`.
- **No training at call time.** Tools only load artifacts, which keeps them deterministic, read-only and free of
  leakage. The test split is only used for the final evaluation.
- **Why `code/` is not imported as a package.** Python's standard library has a module named `code`, so
  `import code.tools` fails. Entrypoints are written as file paths (`code/tools/abalone/sex_models.py::predict_random_forest`)
  and resolved relative to `code/`.
- **Calibration.** The plan calibrates a sex model when its expected calibration error exceeds 0.05. Measured
  out-of-fold on the train split, the Random Forest scores 0.042 and the logistic regression 0.013, so neither is
  calibrated; the training script applies the rule automatically and records the figures in `metrics.json`.
- **Intervals.** ML age tools use a cross-conformal 80 % interval (out-of-fold absolute residuals, finite-sample
  quantile). Quantile-loss boosting was tested during planning and covered only 67 %.

## Differences from the plan

- The declaration loader lives in the new `code/tool_registry.py` instead of `wake_up.py`, and the extra dependencies
  in the new `requirements-abalone-tools.txt` instead of `requirements.txt`, so no existing file was modified.
- The flow declares its two extra blackboard sections (`blackboard_sections`) instead of editing `blackboard.md`.
- Added the `maturity_uncertain` flag (p_adult within 0.1 of 0.5), the `"unknown"` sex for the two expert age tools,
  and `sparse_cell` / `empty_cell_fallback` / `out_of_profile` / `tail_region` flags described in each declaration.
- The MCP SDK installed is 2.3.0, whose low-level server no longer validates schemas; inputs are validated by the
  executors (with unit hints) and outputs by `mcp_server.py` before they reach the client.
- Not done because it would modify existing files: updating conclusion 7.2 of the EDA notebook (its 70–75 % ceiling
  for sex is not reached; measured accuracy is about 55 %) and versioning the dataset through `.gitignore`.

## Removing everything

Nothing outside these paths was changed, so deleting them restores the repository exactly:

```bash
rm -r code/tools/abalone artifacts/models/abalone
rm code/tool_registry.py code/mcp_server.py requirements-abalone-tools.txt .mcp.json
rm declarations/tools/abalone_*.json declarations/agents/abalone_*.json declarations/flows/abalone_tools_flow.json
rm artifacts/datasets/abalone_dataset/splits.json
rm tests/test_abalone_tools.py && rm -rf tests/__pycache__ && rmdir tests   # rmdir only succeeds if nothing else lives there
# optional: .venv/bin/python -m pip uninstall mcp mcp-types
```
