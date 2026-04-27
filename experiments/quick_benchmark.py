from override_sim.config.schemas import SimConfig
from override_sim.config.worlds import build_world_cfg, get_world_description
from override_sim.sim.generator import Simulator
from override_sim.features.blocks import FeatureBuilder
from override_sim.features.leakage import LeakageGuard
from override_sim.models.baselines import GBDTOverrideModel , LogisticOverrideModel  # tiny 2-layer GBDT
from override_sim.eval.metrics import compute_metrics
import pandas as pd

worlds = ["W_B_dominant", "W_U_dominant"]
cfg_base = SimConfig(target_override_rate=0.25)
fb = FeatureBuilder()

for w in worlds:
    cfg = build_world_cfg(cfg_base, w)
    print(f"\n=== {w} – {get_world_description(w)} ===")
    df = Simulator(cfg).run()
    X = fb.build(df)
    LeakageGuard().assert_no_forbidden_columns(X)
    y = df["override"].to_numpy()

    model = GBDTOverrideModel(max_depth=2, n_estimators=80, learning_rate=0.08)
    model.fit(X, y)

    metrics = compute_metrics(y, model.predict_proba(X))
    print({k: round(v, 3) for k, v in metrics.items()})