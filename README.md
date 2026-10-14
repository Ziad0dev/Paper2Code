# Paper2Code
an implementation of the paper https://arxiv.org/pdf/2504.17192

*Paper2Code: Automating Code Generation from Scientific Papers in Machine Learning* (PaperCoder) turns an ML paper into a code repository with three LLM stages, which `paper2code.py` implements on Claude Opus 5.5:

1. **Planning:** an overall plan, the architecture (file list in dependency order), and a `config.yaml` with every hyperparameter, returned as schema-validated JSON (structured outputs). Ambiguities in the paper are listed explicitly.
2. **Analysis:** a detailed implementation spec per file, citing sections and equations.
3. **Coding:** each file written in dependency order, seeing the plan, its spec and all code written so far.

```sh
pip install anthropic                  # or: nix-shell
export ANTHROPIC_API_KEY=...           # or `ant auth login`
python paper2code.py 1706.03762 -o out/transformer        # arXiv id; PDF read natively
python paper2code.py paper.pdf --plan-only               # just plan.json / config.yaml
python paper2code.py paper.md --effort xhigh
python -m pytest tests                                    # pipeline tests with a fake client, no API calls
```

Output: `plan.json`, `analysis.md`, `config.yaml`, `requirements.txt`, the source files and a README listing the unclear points.

Implementation notes:
- The paper goes in once as the first content block with `cache_control`, so the ~2N+1 calls for an N-file repo read it from the prompt cache. Token usage, including cached reads, is printed at the end.
- Every call streams (long code files), checks `stop_reason` for `refusal` / `max_tokens`, and opts into server-side refusal fallbacks (`fallbacks: "default"`, beta `server-side-fallback-2026-07-01`); remove `fallbacks`/`betas` in `ask()` if you'd rather see refusals directly.
- Cost scales with paper length × number of files; `--plan-only` is a cheap first look.

Not yet run against the live API from this machine (no credentials here); the pipeline is covered by the fake-client tests and the request shape was checked against `anthropic` 1.8.0.
