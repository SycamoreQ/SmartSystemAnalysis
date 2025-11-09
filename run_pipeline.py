#!/usr/bin/env python3
import os, sys, argparse, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODEL = ROOT / "model"
PLOTS = ROOT / "plots"

def run_step(cmd, env=None, cwd=None):
    print(f"\n==> Running: {' '.join(str(c) for c in cmd)}")
    if cwd: print(f"    cwd: {cwd}")
    subprocess.run(cmd, check=True, env=env, cwd=cwd)

def must_exist(path: Path, hint: str):
    if not path.exists():
        parent = path.parent
        listing = "\n".join(sorted(p.name for p in parent.glob('*'))) if parent.exists() else "<parent missing>"
        raise FileNotFoundError(f"Missing: {path}\nHint: {hint}\nParent listing ({parent}):\n{listing}")

def find_one(candidates):
    for c in candidates:
        if c.exists(): return c.resolve()
    hits = []
    for name in {p.name for p in candidates}:
        hits += list(ROOT.rglob(name))
    if hits: return hits[0].resolve()
    return candidates[0]

def main():
    print(f"RUN_PIPELINE_FILE={Path(__file__).resolve()}")
    print(f"PROJECT_ROOT={ROOT}")
    ap = argparse.ArgumentParser(description="GBDT→DBN reliability pipeline")
    ap.add_argument("--fd", default=os.environ.get("FD", "FD001"))
    ap.add_argument("--steps", nargs="+", default=["all"],
                    choices=["all","preprocess","train","weibull","infer-weibull","plots"])
    args = ap.parse_args()
    env = os.environ.copy(); env["FD"] = args.fd

    preprocess = find_one([DATA/"NASA.py", DATA/"preprocess.py"])
    trainer    = find_one([MODEL/"GBDT.py", MODEL/"train_monitor.py"])
    weibull    = find_one([MODEL/"weibull_trans.py", MODEL/"weibull_transition.py"])
    infer_w    = find_one([MODEL/"infer.py", MODEL/"build_dbn_and_infer_weibull.py"])

    must_exist(preprocess, "Put preprocessing script in data/")
    must_exist(trainer,   "Put trainer script in model/")
    must_exist(weibull,   "Put Weibull script in model/")
    must_exist(infer_w,   "Put inference script in model/")

    print(f"USING_PREPROCESS={preprocess}")
    print(f"USING_TRAINER={trainer}")
    print(f"USING_WEIBULL={weibull}")
    print(f"USING_INFER_WEIBULL={infer_w}")

    steps = set(args.steps)
    if "all" in steps: steps = {"preprocess","train","weibull","infer-weibull"}

    if "preprocess" in steps:   run_step([sys.executable, str(preprocess)], env=env, cwd=str(ROOT))
    if "train"      in steps:   run_step([sys.executable, str(trainer)],   env=env, cwd=str(ROOT))
    if "weibull"    in steps:   run_step([sys.executable, str(weibull)],   env=env, cwd=str(ROOT))
    if "infer-weibull" in steps:run_step([sys.executable, str(infer_w)],   env=env, cwd=str(ROOT))

    print("\nPipeline finished.")

if __name__ == "__main__":
    sys.exit(main())
