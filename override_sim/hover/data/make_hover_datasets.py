from pathlib import Path

from override_sim.config.schemas import SimConfig
from override_sim.config.worlds import build_world_cfg
from override_sim.sim.generator import Simulator

OUT_DIR = Path("override_sim/hover/data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

base_cfg = SimConfig(target_override_rate=0.25)
world_names = ["W0_baseline", "W_U_dominant", "W_B_dominant"]

for world_name in world_names:
    cfg = build_world_cfg(base_cfg, world_name)
    df = Simulator(cfg).run(return_latents=False, hover_mode=True)

    out_path = OUT_DIR / f"{world_name}.csv"
    df.to_csv(out_path, index=False)

    print(f"Saved {world_name} -> {out_path} | shape={df.shape} | override_rate={df['override'].mean():.3f}")