import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
from harness import run_coding_task
async def run(ctx):
    return await run_coding_task(ctx)
