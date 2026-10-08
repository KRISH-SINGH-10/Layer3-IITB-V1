#!/usr/bin/env python
"""Convert a scene-spec JSON (the pipeline's universal scenario format) into a Layer 3 scenario.

Reads twin_id, layer_c, layer_b, drone and output; validates every actor position against the
twin's navmesh; writes layer3_mvp/scenarios/<scene_id>.json and prints the build and render commands.
Spec frame: x = scene X, y = -scene Z, z up, metres.
"""
import argparse, json, math, os, sys
import numpy as np
import habitat_sim

ROOT = "/home/24singhk/Layer3_Prototype"
CLOTHING_TO_AVATAR = {"casual_maroon_shirt": "avatar2", "green_dress": "avatar1"}   # the two avatars that exist today
SNAP_LIMIT_M = 1.5


class PlacementError(ValueError):
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    args = ap.parse_args()
    spec = json.load(open(args.spec))
    twin = spec["twin_id"]; root = "gs_" + twin; fps = float(spec.get("output", {}).get("fps", 30))
    pf = habitat_sim.PathFinder(); pf.load_nav_mesh("%s/%s/train/%s/%s.navmesh" % (ROOT, root, twin, twin))
    objects = {o["object_id"]: o for o in spec.get("layer_b", [])}

    def to_scene(xy, what):
        p = np.array([xy[0], 0.0, -xy[1]], dtype=np.float32); q = np.array(pf.snap_point(p))
        if np.isnan(q).any() or math.hypot(q[0] - p[0], q[2] - p[2]) > SNAP_LIMIT_M:
            raise PlacementError("%s %s is not on the walkable area" % (what, list(xy)))
        return [round(float(v), 3) for v in q]

    actors, notes = [], []
    for i, a in enumerate(spec.get("layer_c", [])):
        aid = a["actor_id"]; start = to_scene(a["start_pos"], aid + " start_pos")
        target = (a.get("interaction") or {}).get("target_object")
        if a.get("goal_pos") is not None:
            goal = to_scene(a["goal_pos"], aid + " goal_pos")
        elif target in objects:                       # stop about 2 m short of the object, on the actor's side
            o = np.array(objects[target]["position"][:2], dtype=float); s_ = np.array(a["start_pos"], dtype=float)
            goal = to_scene((o + 2.0 * (s_ - o) / max(np.linalg.norm(s_ - o), 1e-6)).tolist(), aid + " approach point for " + target)
        else:
            raise PlacementError("%s needs goal_pos or interaction.target_object: the prompt alone does not give a destination" % aid)
        sp = habitat_sim.ShortestPath(); sp.requested_start = np.array(start, dtype=np.float32); sp.requested_end = np.array(goal, dtype=np.float32)
        if not pf.find_path(sp):
            raise PlacementError("%s: no walkable route from start to goal" % aid)
        if a.get("speed", "normal") != "normal": notes.append("%s: speed '%s' requested, walking pace generated (only walking exists so far)" % (aid, a["speed"]))
        clothing = (a.get("appearance") or {}).get("clothing")
        if clothing not in CLOTHING_TO_AVATAR: notes.append("%s: clothing '%s' has no avatar, using avatar2" % (aid, clothing))
        actors.append({"actor_id": aid, "action": "walk", "avatar": CLOTHING_TO_AVATAR.get(clothing, "avatar2"), "start": start, "goal": goal,
                       "start_time_s": round(float(a.get("start_frame") or 0) / fps, 3), "seed": i + 1, "prompt": a.get("prompt"),
                       "anomaly": bool(a.get("anomaly", False)), "anomaly_type": a.get("anomaly_type"), "severity": a.get("severity"),
                       "route_length_m": round(float(sp.geodesic_distance), 2)})
        print("%s: %s -> %s  route %.1f m" % (aid, start, goal, sp.geodesic_distance))

    longest = max(a["route_length_m"] for a in actors)
    sc = {"scenario": spec["scene_id"], "scene": twin, "scene_root": root, "source_spec": os.path.abspath(args.spec),
          "max_walk_seconds": max(25, int(math.ceil(longest / 0.8)) + 5), "actors": actors}
    out = os.path.join(ROOT, "layer3_mvp", "scenarios", spec["scene_id"] + ".json")
    json.dump(sc, open(out, "w"), indent=2)
    for n in notes: print("note:", n)
    d = spec.get("drone", {}); pos = (d.get("trajectory") or {}).get("position"); res = (d.get("sensor") or {}).get("resolution", [1920, 1080])
    cam = "" if pos is None else " --elevated-cam %.2f %.2f %.2f" % (pos[0], float(d.get("altitude", 25.0)), -pos[1])
    py = ROOT + "/miniconda3/envs/habitat-gs/bin/python"
    print("scenario:", out)
    print("build :  %s layer3_mvp/build_scenario.py --scenario %s" % (py, out))
    print("render:  %s layer3_mvp/render_clip.py --dataset %s/%s/%s/%s.scene_dataset_config.json --fps %g --width %d --height %d%s" % (
        py, ROOT, root, spec["scene_id"], spec["scene_id"], fps, res[0], res[1], cam))


if __name__ == "__main__":
    try:
        main()
    except PlacementError as e:
        print("PlacementError:", e); sys.exit(2)
