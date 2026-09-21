import ast
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import typer

import models  # noqa: F401  (imports every module in models/, filling the registry)
from models.base import Policy
from models.registry import UserError, families, family_of, get_model, models_in
from src.eval import evaluate

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT /"runs"/"best_models.json"

app = typer.Typer(help="Train, evaluate and play Flappy Bird policies.",no_args_is_help=True, add_completion=True)

# ---------- best-model manifest ----------

def _read_manifest() -> dict:
    if not MANIFEST.is_file():
        return {}
    try:
        return json.loads(MANIFEST.read_text())
    except json.JSONDecodeError as e:
        raise UserError(f"{MANIFEST.name} is not valid JSON: {e}")


def _manifest_key(target: str) -> str:
    """Canonical manifest key: lowercase for families, registry casing for full names."""
    if "." in target:
        return get_model(target).registry_name
    if target.lower() not in families():
        raise UserError(f"Unknown family '{target}'. Available: {', '.join(families())}")
    return target.lower()


def _best_checkpoint(target: str) -> Path:
    key = _manifest_key(target)
    entry = _read_manifest().get(key)
    if not entry or "checkpoint" not in entry:
        raise UserError(
            f"No best model registered for '{key}' in {MANIFEST.name}.\n"
            f"Register one with `flappy-bird set-best {key} <checkpoint>`, or pass --checkpoint."
        )
    path = ROOT / entry["checkpoint"]  # absolute paths in the manifest also work
    if not path.is_file():
        raise UserError(f"{MANIFEST.name} points '{key}' at {path}, which doesn't exist.")
    return path


# ---------- helpers ----------

def _parse_kwargs(items: List[str]) -> dict:
    out = {}
    for item in items:
        key, sep, val = item.partition("=")
        if not sep:
            raise UserError(f"--kwarg expects KEY=VALUE, got '{item}'")
        try:
            out[key] = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            out[key] = val
    return out


def _run(target, episodes, render, population_size, threshold, seed, device:str, kwarg,checkpoint=None):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    if checkpoint is None:
        checkpoint = _best_checkpoint(target)
    cls = get_model(target) if "." in target else Policy
    policy = cls.load(checkpoint, device, **_parse_kwargs(kwarg)).eval()

    name = policy.registry_name
    ok = name.lower() == target.lower() if "." in target else family_of(name) == target.lower()
    if not ok:
        raise UserError(f"Checkpoint holds '{name}', which doesn't match '{target}'.")

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    def progress(ep):
        typer.echo(f"episode {ep['episode']:3d} | steps={ep['steps']:5d} | "
                   f"mean_score={ep['mean_score']:6.2f} | max_score={ep['max_score']:3d}", err=True)

    return evaluate(policy, num_episodes=episodes, population_size=population_size,
                    render=render, threshold=threshold, device=device, on_episode=progress,show_score=True)


def _summary(r: dict) -> str:
    return (f"{r['model']}\n"
            f"mean_score {r['mean_score']:.2f} ± {r['std_score']:.2f} | "
            f"best {r['max_score']} | episodes {r['num_episodes']}")


TARGET = typer.Argument(..., help="Family (e.g. `gru`, uses its best checkpoint) or full name (`gru.GRU2`).")


# ---------- commands ----------

@app.command()
def play(
    target: str = TARGET,
    episodes: int = typer.Option(5, "--episodes", "-n", min=1),
    render: bool = typer.Option(True, "--render/--no-render"),
    population_size: int = typer.Option(1, "--population-size", "-p", min=1),
    threshold: float = typer.Option(0.5, "--threshold", "--thr"),
    seed: Optional[int] = typer.Option(None, "--seed"),
    device: str = typer.Option("auto", "--device", help="auto | cpu | cuda"),
    kwarg: List[str] = typer.Option([], "--kwarg", help="Constructor arg for legacy checkpoints, e.g. K=4 (repeatable)."),
):
    """Watch a policy play (rendered by default)."""
    result = _run(target, episodes, render, population_size, threshold, seed, device, kwarg)
    typer.echo(_summary(result))


@app.command("eval")
def eval_(
    target: str = TARGET,
    checkpoint: Optional[Path] = typer.Option(None, "--checkpoint", "-c"),
    episodes: int = typer.Option(50, "--episodes", "-n", min=1),
    render: bool = typer.Option(False, "--render/--no-render"),
    population_size: int = typer.Option(1, "--population-size", "-p", min=1),
    threshold: float = typer.Option(0.5, "--threshold", "--thr"),
    seed: Optional[int] = typer.Option(None, "--seed"),
    device: str = typer.Option("auto", "--device"),
    kwarg: List[str] = typer.Option([], "--kwarg"),
    as_json: bool = typer.Option(False, "--json", help="Print machine-readable results to stdout."),
):
    """Evaluate a policy (headless by default) and report metrics."""
    result = _run(target, episodes, render, population_size, threshold, seed, device, kwarg, checkpoint=checkpoint)
    typer.echo(json.dumps(result, indent=2) if as_json else _summary(result))


@app.command("list")
def list_():
    """List registered models and the best checkpoints in best_models.json."""
    manifest = _read_manifest()
    for family in families():
        entry = manifest.get(family)
        best = entry.get("checkpoint", "?") if entry else "(none)"
        typer.echo(f"{family}  [best: {best}]")
        for name in models_in(family):
            extra = f"  [best: {manifest[name].get('checkpoint', '?')}]" if name in manifest else ""
            typer.echo(f"  {name}{extra}")


@app.command("set-best")
def set_best(
    target: str = typer.Argument(..., help="Family (`gru`) or full model name (`gru.GRU2`)."),
    checkpoint: Path = typer.Argument(..., help="Checkpoint to register as the best one."),
):
    """Register a checkpoint as the best model in best_models.json."""
    key = _manifest_key(target)
    policy = Policy.load(checkpoint)  # validates the file and reads which model it holds
    
    name = policy.registry_name
    ok = name.lower() == target.lower() if "." in target else family_of(name) == target.lower()
    if not ok:
        raise UserError(f"Checkpoint holds '{name}', which doesn't match '{target}'.")

    resolved = checkpoint.resolve()
    try:
        stored = resolved.relative_to(ROOT)
    except ValueError:
        stored = resolved  # outside the repo: keep the absolute path

    entry = {"checkpoint": stored.as_posix()}
    if policy.metrics:
        entry["metrics"] = policy.metrics

    manifest = _read_manifest()
    manifest[key] = entry
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    typer.echo(f"Registered {stored.as_posix()} as best '{key}' ({policy.registry_name})")


# @app.command()
# def train(
#     model: str = typer.Argument(..., help="Full model name, e.g. gru.GRU1"),
#     config: Optional[Path] = typer.Option(None, "--config", "-c", help="YAML config file."),
# ):
#     """Train a model (shared training loop is not implemented yet)."""
#     get_model(model)  # validate the name early
#     from src.train.common import train as run_train
#     try:
#         run_train(model, config)
#     except NotImplementedError as e:
#         raise UserError(str(e))


@app.command()
def convert(
    source: Path = typer.Argument(..., help="Legacy checkpoint (bare state_dict)."),
    model: str = typer.Argument(..., help="Full model name it belongs to, e.g. mlp.MLP_Temp1"),
    output: Path = typer.Option(..., "--output", "-o"),
    kwarg: List[str] = typer.Option([], "--kwarg", help="Constructor args, e.g. K=4 (repeatable)."),
):
    """Upgrade an old checkpoint to the self-describing format."""
    get_model(model).load(source, **_parse_kwargs(kwarg)).save(output)
    typer.echo(f"Wrote {output}")



if __name__ == "__main__":
    app()