#!/usr/bin/env python
"""Write a scenario file automatically: N people on separate routes in any scene.

Picks the busiest open part of the scene's navmesh, then start/goal pairs of a
given walking length that stay apart from each other. Edit the result by hand
afterwards if you want specific places.
"""
import argparse, itertools, json, math, os
import numpy as np
import habitat_sim

ROOT = "/home/24singhk/Layer3_Prototype"
SPEED = 0.95  # m/s, typical GAMMA walking speed, used only to predict near-misses


def track(points, n=2400, dt=0.025):
    pts = np.array(points)[:, [0, 2]]
    cum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    s = np.minimum(np.arange(n) * dt * SPEED, cum[-1])
    return np.stack([np.interp(s, cum, pts[:, 0]), np.interp(s, cum, pts[:, 1])], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--name", required=True, help="Scenario name; also the dataset/output folder name.")
    ap.add_argument("--actors", type=int, default=6)
    ap.add_argument("--route-length", type=float, default=25.0, help="Target walking distance in metres.")
    ap.add_argument("--radius", type=float, default=0.0, help="Size of the area to use; 0 = route length.")
    ap.add_argument("--center", type=float, nargs=3, default=None, help="Centre of the area (x y z); default = automatic.")
    ap.add_argument("--avatars", default="avatar1,avatar2,avatar3,avatar4,avatar6,avatar8")
    ap.add_argument("--min-gap", type=float, default=2.2, help="Predicted closest approach allowed between two people (m).")
    ap.add_argument("--max-height-diff", type=float, default=0.5, help="Keep routes within this height band (m) of the centre.")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    pf = habitat_sim.PathFinder()
    pf.load_nav_mesh("%s/gs_scene01/train/%s/%s.navmesh" % (ROOT, args.scene, args.scene))
    pf.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    radius = args.radius or args.route_length
    cloud = np.array([pf.get_random_navigable_point() for _ in range(6000)])
    isl = np.array([pf.get_island(p) for p in cloud]); main_island = int(np.bincount(isl).argmax()); cloud = cloud[isl == main_island]
    if args.center is None:
        # the point with the most walkable area around it at a similar height
        cand = cloud[rng.choice(len(cloud), size=min(400, len(cloud)), replace=False)]
        score = [int(((np.linalg.norm((cloud - c)[:, [0, 2]], axis=1) < radius) & (np.abs(cloud[:, 1] - c[1]) < args.max_height_diff)).sum()) for c in cand]
        center = cand[int(np.argmax(score))]
    else:
        center = np.array(pf.snap_point(np.array(args.center, dtype=np.float32)))
    pts = []
    for _ in range(12000):
        p = pf.get_random_navigable_point_near(center.astype(np.float32), radius, 300)
        if np.isnan(p).any() or pf.get_island(p) != main_island or abs(p[1] - center[1]) > args.max_height_diff:
            continue
        if pf.distance_to_closest_obstacle(p, 2.0) < 0.9:
            continue
        pts.append(np.array(p))
    pts = np.array(pts)
    print("centre %s  candidate points %d" % (np.round(center, 1).tolist(), len(pts)))

    lo, hi = 0.9 * args.route_length, 1.12 * args.route_length
    routes, tries = [], 0
    while len(routes) < args.actors and tries < 80000:
        tries += 1
        a, b = pts[rng.integers(len(pts))], pts[rng.integers(len(pts))]
        sp = habitat_sim.ShortestPath(); sp.requested_start = a; sp.requested_end = b
        if not pf.find_path(sp) or not (lo <= sp.geodesic_distance <= hi):
            continue
        if np.abs(np.array(sp.points)[:, 1] - center[1]).max() > args.max_height_diff:
            continue
        tr = track(sp.points)
        if any(np.linalg.norm(r["start"] - a) < 4.0 or np.linalg.norm(r["goal"] - b) < 4.0 or
               np.linalg.norm(r["track"] - tr, axis=1).min() < args.min_gap for r in routes):
            continue
        routes.append({"start": a, "goal": b, "track": tr, "length": sp.geodesic_distance})
    if len(routes) < args.actors:
        raise SystemExit("only found %d of %d routes; try a shorter --route-length or smaller --min-gap" % (len(routes), args.actors))

    avatars = args.avatars.split(",")
    actors = []
    for i, r in enumerate(routes):
        print("person_%d  %.1f m  %s -> %s" % (i + 1, r["length"], np.round(r["start"], 1).tolist(), np.round(r["goal"], 1).tolist()))
        actors.append({"actor_id": "person_%d" % (i + 1), "action": "walk", "avatar": avatars[i % len(avatars)],
                       "start": [round(float(v), 2) for v in r["start"]], "goal": [round(float(v), 2) for v in r["goal"]],
                       "start_time_s": 0.0, "seed": args.seed * 10 + i})
    for a, b in itertools.combinations(range(len(routes)), 2):
        d = np.linalg.norm(routes[a]["track"] - routes[b]["track"], axis=1)
        if d.min() < 3.5:
            print("  predicted pass: person_%d / person_%d  %.1f m" % (a + 1, b + 1, d.min()))
    out = os.path.join(ROOT, "layer3_mvp", "scenarios", args.name + ".json")
    json.dump({"scenario": args.name, "scene": args.scene, "description": "%d people walking about %.0f m each in %s (auto-generated routes)." % (
        args.actors, args.route_length, args.scene), "max_walk_seconds": int(math.ceil(args.route_length / 0.6)), "actors": actors}, open(out, "w"), indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
