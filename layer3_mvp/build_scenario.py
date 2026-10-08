#!/usr/bin/env python
"""Turn a scenario file (who walks from where to where) into a Habitat-GS dataset.

For every actor it runs GAMMA through habitat-gs/tools_gs/generate_trajectory.py to
produce a driver .pkl, then writes a scene dataset whose scene instance lists all
actors as gaussian_avatars. Existing drivers are reused unless --force is given.
"""
import argparse, itertools, json, math, os, pickle, re, shutil, subprocess, sys
import numpy as np

ROOT = "/home/24singhk/Layer3_Prototype"
GS = ROOT + "/gs_scene01"
# Body shape the existing avatars were driven with (kept identical for all actors).
BETAS = [1.764, 0.400, 0.979, 2.241, 1.868, -0.977, 0.950, -0.151, -0.103, 0.411]
AVATAR_DEFAULTS = {"offset_y": 1.512, "scale": 1.15}


def run_gamma(actor, navmesh, out_path, log_path, max_seconds, seed):
    cmd = [sys.executable, "tools_gs/generate_trajectory.py", "--navmesh", navmesh, "--output", out_path,
           "--start", *[str(v) for v in actor["start"]]]
    for via in actor.get("via", []):
        cmd += ["--via", *[str(v) for v in via]]
    cmd += ["--end", *[str(v) for v in actor["goal"]],
            "--smpl-model-path", ROOT + "/GAMMA-release/body_models/smplx",
            "--gamma-root", ROOT + "/GAMMA-release", "--body-model-path", ROOT + "/GAMMA-release/body_models",
            "--gender", "female", "--betas", *[str(b) for b in BETAS], "--random-seed", str(seed),
            "--length", str(int(round(max_seconds / 0.25)))]
    with open(log_path, "w") as log:
        rc = subprocess.call(cmd, cwd=ROOT + "/habitat-gs", stdout=log, stderr=subprocess.STDOUT)
    if rc != 0 or not os.path.exists(out_path):
        return None
    m = re.search(r"distance_scale=([0-9.]+)", open(log_path).read())
    return float(m.group(1)) if m else 1.0


def generate_driver(actor, navmesh, out_path, log_path, max_seconds, attempts=5):
    """Run GAMMA; retry with other seeds when the walk did not really cover the route.

    generate_trajectory.py stretches the motion to fit the planned path, so a scale far
    from 1 means sliding feet (the walker wandered or stalled). Keeps the best attempt.
    """
    best = None
    for k in range(attempts):
        seed = int(actor.get("seed", 0)) + 101 * k
        tmp = out_path + ".try"
        scale = run_gamma(actor, navmesh, tmp, log_path + (".%d" % k if k else ""), max_seconds, seed)
        if scale is None:
            continue
        err = abs(math.log(scale))
        if best is None or err < best[0]:
            best = (err, scale, seed); os.replace(tmp, out_path)
        if 0.88 <= scale <= 1.15:
            break
    if os.path.exists(out_path + ".try"):
        os.remove(out_path + ".try")
    if best is None:
        raise RuntimeError("trajectory generation failed for %s (see %s)" % (actor["actor_id"], log_path))
    return best[1], best[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--force", action="store_true", help="Regenerate drivers that already exist.")
    args = ap.parse_args()
    sc = json.load(open(args.scenario))
    scene = sc.get("scene", "scene01")
    gs = os.path.join(ROOT, sc.get("scene_root", "gs_scene01"))   # scene folder holding train/<scene>/
    out = os.path.join(gs, sc["scenario"])
    for sub in ("stages", "scenes", "trajectories", "logs"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    navmesh = "%s/train/%s/%s.navmesh" % (gs, scene, scene)

    avatars, resolved = [], []
    for actor in sc["actors"]:
        aid = actor["actor_id"]
        if actor.get("action", "walk") != "walk":
            raise ValueError("%s: only the 'walk' action is supported so far" % aid)
        driver = os.path.join(out, "trajectories", aid + ".driver.pkl")
        if args.force or not os.path.exists(driver):
            print("generating", aid, "...", flush=True)
            scale, used = generate_driver(actor, navmesh, driver, os.path.join(out, "logs", aid + ".log"), float(sc.get("max_walk_seconds", 25)))
            print("  fit %.2f (seed %d)%s" % (scale, used, "" if 0.88 <= scale <= 1.15 else "  <-- POOR WALK, feet will slide; change this route"), flush=True)
        d = pickle.load(open(driver, "rb"))
        duration = len(d["transl"]) / float(d["fps"])
        t0 = float(actor.get("start_time_s", 0.0))
        entry = {"name": aid, "canonical_gaussians": "%s/avatars/%s/canonical_gs.npz" % (ROOT, actor.get("avatar", "avatar2")),
                 "driver": driver, "smpl_model_path": GS + "/avatars/smplx", "smpl_type": "smplx", "time_begin": t0}
        entry.update(AVATAR_DEFAULTS)
        entry.update({k: actor[k] for k in ("offset_y", "scale") if k in actor})
        avatars.append(entry)
        resolved.append({"actor_id": aid, "action": "walk", "avatar": actor.get("avatar", "avatar2"), "driver": driver,
                         "start_time_s": t0, "duration_s": round(duration, 3), "start": actor["start"], "goal": actor["goal"]})
        print("%-9s %-8s %.1fs  %d frames" % (aid, actor.get("avatar", "avatar2"), duration, len(d["transl"])))

    # Actors do not avoid each other yet: report the closest approach so a clash can be fixed in the scenario.
    tracks = {r["actor_id"]: (r["start_time_s"], np.asarray(pickle.load(open(r["driver"], "rb"))["transl"])[:, [0, 2]]) for r in resolved}
    worst = None
    for a, b in itertools.combinations(tracks, 2):
        if tracks[a][0] != tracks[b][0]:
            continue
        n = min(len(tracks[a][1]), len(tracks[b][1]))
        gap = np.linalg.norm(tracks[a][1][:n] - tracks[b][1][:n], axis=1)
        if worst is None or gap.min() < worst[0]:
            worst = (float(gap.min()), a, b, int(gap.argmin()) / 40.0)
    if worst:
        print("closest approach: %.2fm between %s and %s at t=%.1fs%s" % (*worst, "  <-- TOO CLOSE, change a route" if worst[0] < 0.8 else ""))

    json.dump({"render_asset": "../../train/%s/%s.gs.ply" % (scene, scene), "render_asset_type": "gaussian_splatting", "units_to_meters": 1.0,
               "orient_up": [0, 1, 0], "orient_front": [0, 0, -1], "frustum_culling": False, "light_setup": "default"},
              open(os.path.join(out, "stages", scene + ".stage_config.json"), "w"), indent=2)
    json.dump({"stages": {"paths": {".json": ["stages"]}}, "scene_instances": {"paths": {".json": ["scenes"]}},
               "navmesh_instances": {scene: "../train/%s/%s.navmesh" % (scene, scene)}},
              open(os.path.join(out, sc["scenario"] + ".scene_dataset_config.json"), "w"), indent=2)
    json.dump({"stage_instance": {"template_name": scene}, "navmesh_instance": scene, "gaussian_avatars": avatars},
              open(os.path.join(out, "scenes", scene + ".scene_instance.json"), "w"), indent=2)
    json.dump({"scenario": sc["scenario"], "scene": scene, "actors": resolved},
              open(os.path.join(out, "scenario_resolved.json"), "w"), indent=2)
    print("dataset:", os.path.join(out, sc["scenario"] + ".scene_dataset_config.json"))


if __name__ == "__main__":
    main()
