#!/usr/bin/env python3
"""Paper2Code: turn a machine-learning paper into a code repository.

Implements the three-stage pipeline of "Paper2Code: Automating Code Generation
from Scientific Papers in Machine Learning" (Seo et al., arXiv:2504.17192):

  1. Planning  - overall plan, architecture (file list), dependency-ordered logic
                 analysis and a config file, as structured JSON
  2. Analysis  - a detailed implementation spec for every file
  3. Coding    - each file written in dependency order, seeing the plan, its spec
                 and the code already written

    python paper2code.py 1706.03762                 # arXiv id (downloads the PDF)
    python paper2code.py paper.pdf -o out/attention
    python paper2code.py paper.md --plan-only

Uses Claude Opus 5.5 via the Anthropic SDK. The paper is sent once as a cached
document block, so every later call reuses it from the prompt cache.
"""
import argparse
import base64
import json
import re
import sys
import urllib.request
from pathlib import Path

import anthropic

MODEL = "claude-opus-5-5"
# On a safety-classifier decline the API re-runs the request on Anthropic's
# recommended fallback model instead of returning a refusal.
FALLBACK_BETA = "server-side-fallback-2026-07-01"

SYSTEM = """You are an expert machine learning researcher and engineer who reproduces papers as clean, \
runnable code repositories. Be faithful to the paper: use the architectures, losses, hyperparameters and \
evaluation protocol it describes. Where the paper leaves something unspecified, choose a sensible default \
and mark it with a `# NOTE: not specified in the paper` comment rather than inventing claims."""

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "overview": {"type": "string", "description": "What the paper proposes and what the repository will reproduce"},
        "components": {"type": "array", "items": {"type": "string"}, "description": "Methods, datasets, metrics, experiments"},
        "files": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "purpose": {"type": "string"},
                    "depends_on": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["path", "purpose", "depends_on"],
                "additionalProperties": False,
            },
            "description": "Every source file, in the order they should be written (dependencies first)",
        },
        "config_yaml": {"type": "string", "description": "config.yaml with all hyperparameters from the paper"},
        "requirements": {"type": "array", "items": {"type": "string"}},
        "unclear_points": {"type": "array", "items": {"type": "string"}, "description": "Ambiguities in the paper"},
    },
    "required": ["overview", "components", "files", "config_yaml", "requirements", "unclear_points"],
    "additionalProperties": False,
}


class RefusedError(RuntimeError):
    pass


def load_paper(source):
    """A user content block for the paper: PDF document (arXiv id, URL or file) or text."""
    if re.fullmatch(r"\d{4}\.\d{4,5}(v\d+)?", source):
        source = f"https://arxiv.org/pdf/{source}"
    if source.startswith("http"):
        req = urllib.request.Request(source, headers={"User-Agent": "paper2code/0.1"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = resp.read()
    else:
        data = Path(source).read_bytes()
    if data[:4] == b"%PDF":
        return {"type": "document", "source": {"type": "base64", "media_type": "application/pdf",
                                               "data": base64.standard_b64encode(data).decode()},
                "cache_control": {"type": "ephemeral"}}
    return {"type": "text", "text": "<paper>\n" + data.decode("utf-8", errors="replace") + "\n</paper>",
            "cache_control": {"type": "ephemeral"}}


class PaperCoder:
    def __init__(self, paper_block, client=None, effort="high", log=print):
        self.client = client or anthropic.Anthropic()
        self.paper = paper_block
        self.effort = effort
        self.log = log
        self.usage = {"input": 0, "output": 0, "cache_read": 0}

    def ask(self, prompt, json_schema=None, max_tokens=64000):
        """One call with the paper as the cached prefix. Returns the response text."""
        output_config = {"effort": self.effort}
        if json_schema:
            output_config["format"] = {"type": "json_schema", "schema": json_schema}
        with self.client.beta.messages.stream(
            model=MODEL,
            max_tokens=max_tokens,
            betas=[FALLBACK_BETA],
            fallbacks="default",
            system=SYSTEM,
            output_config=output_config,
            messages=[{"role": "user", "content": [self.paper, {"type": "text", "text": prompt}]}],
        ) as stream:
            message = stream.get_final_message()
        self.usage["input"] += message.usage.input_tokens
        self.usage["output"] += message.usage.output_tokens
        self.usage["cache_read"] += message.usage.cache_read_input_tokens or 0
        if message.stop_reason == "refusal":
            raise RefusedError(getattr(message.stop_details, "explanation", None) or "request declined")
        if message.stop_reason == "max_tokens":
            self.log("  warning: response hit max_tokens and may be truncated")
        return "".join(b.text for b in message.content if b.type == "text")

    def plan(self):
        self.log("[1/3] planning")
        text = self.ask(
            "Read the paper and plan a repository that reproduces its method and main experiments. "
            "Design a small, modular file layout (typically 4-10 Python files: data loading, model, "
            "training, evaluation, utilities, a main entry point). List files so that every file comes "
            "after the files it imports. Put every hyperparameter in config.yaml.",
            json_schema=PLAN_SCHEMA,
        )
        return json.loads(text)

    def analyze(self, plan, file):
        return self.ask(
            f"Repository plan:\n{json.dumps(plan, indent=1)}\n\n"
            f"Write the detailed implementation specification for `{file['path']}` ({file['purpose']}). "
            "Cover: classes and functions with signatures, the exact equations and algorithm steps from the "
            "paper (cite section/equation numbers), which config.yaml keys it reads, tensor shapes, and how it "
            "interacts with the other files. Specification only, no code."
        )

    def code(self, plan, file, spec, written):
        context = "\n\n".join(f"### {path}\n```python\n{src}\n```" for path, src in written.items() if path.endswith(".py"))
        text = self.ask(
            f"Repository plan:\n{json.dumps(plan, indent=1)}\n\n"
            f"Files written so far:\n{context or '(none yet)'}\n\n"
            f"Specification for `{file['path']}`:\n{spec}\n\n"
            f"Write the complete contents of `{file['path']}`. It must import correctly from the files "
            "above and read hyperparameters from config.yaml. Output only the file contents in a single "
            "fenced code block."
        )
        return strip_fences(text)

    def run(self, out_dir, plan_only=False):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        plan = self.plan()
        (out / "plan.json").write_text(json.dumps(plan, indent=2))
        (out / "config.yaml").write_text(plan["config_yaml"])
        (out / "requirements.txt").write_text("\n".join(plan["requirements"]) + "\n")
        self.log(f"  {len(plan['files'])} files: " + ", ".join(f["path"] for f in plan["files"]))
        if plan_only:
            return plan

        self.log("[2/3] analysis")
        specs = {}
        for f in plan["files"]:
            self.log(f"  {f['path']}")
            specs[f["path"]] = self.analyze(plan, f)
        (out / "analysis.md").write_text("\n\n".join(f"# {p}\n\n{s}" for p, s in specs.items()))

        self.log("[3/3] coding")
        written = {"config.yaml": plan["config_yaml"]}
        for f in plan["files"]:
            self.log(f"  {f['path']}")
            src = self.code(plan, f, specs[f["path"]], written)
            path = out / f["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(src)
            written[f["path"]] = src

        (out / "README.md").write_text(readme(plan))
        self.log(f"done: {out}  (tokens in {self.usage['input']:,} / cached {self.usage['cache_read']:,} / out {self.usage['output']:,})")
        return plan


def strip_fences(text):
    m = re.search(r"```[\w.+-]*\n(.*?)```", text, re.DOTALL)
    return (m.group(1) if m else text).rstrip() + "\n"


def readme(plan):
    lines = ["# Generated reproduction", "", plan["overview"], "", "## Files"]
    lines += [f"- `{f['path']}`: {f['purpose']}" for f in plan["files"]]
    if plan["unclear_points"]:
        lines += ["", "## Points the paper leaves unclear"] + [f"- {u}" for u in plan["unclear_points"]]
    lines += ["", "Generated by Paper2Code. Review before trusting any results."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paper", help="arXiv id, PDF URL, PDF file or text/markdown file")
    parser.add_argument("-o", "--out", default="generated")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    args = parser.parse_args()
    try:
        PaperCoder(load_paper(args.paper), effort=args.effort).run(args.out, args.plan_only)
    except RefusedError as e:
        sys.exit(f"the model declined: {e}")
    except (anthropic.AuthenticationError, TypeError) as e:
        if isinstance(e, TypeError) and "authentication" not in str(e):
            raise
        sys.exit("no valid Anthropic credentials: set ANTHROPIC_API_KEY or run `ant auth login`")
    except anthropic.RateLimitError:
        sys.exit("rate limited; try again later")
    except anthropic.APIStatusError as e:
        sys.exit(f"API error {e.status_code}: {e.message}")
    except anthropic.APIConnectionError:
        sys.exit("could not reach the Anthropic API")


if __name__ == "__main__":
    main()
