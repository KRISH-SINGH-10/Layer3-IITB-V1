#!/usr/bin/env python
"""Onboard a raw 3DGS point_cloud.ply as a Habitat-GS scene for Layer 3.

Steps: fit the ground plane, rotate so up = +Y, scale to metres, recentre, write
<name>.gs.ply, derive a walkable area from the splats (ground coverage minus
obstacles), bake a navmesh, list car-sized objects, and write scene_info.json
plus a labelled top-down preview.
"""
import argparse, json, math, os, sys
import numpy as np
from scipy import ndimage
from plyfile import PlyData, PlyElement
from PIL import Image, ImageDraw

ROOT = "/home/24singhk/Layer3_Prototype"
sys.path.insert(0, ROOT + "/habitat-gs/tools_gs")
import rotate_gs


def fit_ground(xyz, op):
    solid = op > 0.5
    span = np.percentile(xyz[solid], 90, 0) - np.percentile(xyz[solid], 10, 0)
    core = solid & (np.abs(xyz - np.median(xyz[solid], 0)) < 4 * span).all(1)
    rng = np.random.default_rng(0)
    S = xyz[core][rng.choice(core.sum(), min(60000, core.sum()), replace=False)]
    thr = 0.004 * np.linalg.norm(span); best = (0, None)
    for _ in range(800):
        a, b, c = S[rng.choice(len(S), 3, replace=False)]
        n = np.cross(b - a, c - a); L = np.linalg.norm(n)
        if L < 1e-9: continue
        n /= L; cnt = int((np.abs((S - a) @ n) < thr).sum())
        if cnt > best[0]: best = (cnt, n, float(a @ n))
    _, n, off = best
    inl = np.abs(S @ n - off) < thr                      # refine on inliers
    c0 = S[inl].mean(0); _, _, vt = np.linalg.svd(S[inl] - c0, full_matrices=False)
    n2 = vt[2] * np.sign(vt[2] @ n); off2 = float(c0 @ n2)
    if np.median(S @ n2 - off2) < 0: n2, off2 = -n2, -off2   # most of the scene sits above the ground
    return n2, off2, best[0] / len(S)


def disk(r_cells):
    r = int(math.ceil(r_cells)); y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r_cells * r_cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--metres-per-unit", type=float, required=True)
    ap.add_argument("--cell", type=float, default=0.25, help="Walkable-grid cell size in metres.")
    args = ap.parse_args()
    s = args.metres_per_unit
    out = os.path.join(ROOT, "gs_" + args.name); tr = os.path.join(out, "train", args.name)
    os.makedirs(tr, exist_ok=True); os.makedirs(os.path.join(out, "dynamic_nav", "stages"), exist_ok=True)

    v = PlyData.read(args.ply)["vertex"].data
    xyz = np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float64)
    op = 1 / (1 + np.exp(-v["opacity"].astype(np.float64)))
    up, off, frac = fit_ground(xyz, op)
    a = np.cross(up, [1.0, 0, 0]); a /= np.linalg.norm(a); b = np.cross(a, up)
    R = np.stack([b, up, np.cross(b, up)], 0)
    print("ground plane: up %s  inliers %.0f%%  det(R) %.3f" % (np.round(up, 3).tolist(), 100 * frac, np.linalg.det(R)))

    tmp = os.path.join(tr, "_rotated.ply")
    rotate_gs.rotate_ply(args.ply, tmp, R, rotate_gs.matrix_to_quaternion_wxyz(R), {})
    pd = PlyData.read(tmp); w = pd["vertex"].data.copy()
    q = np.column_stack([w["x"], w["y"], w["z"]]).astype(np.float64); q[:, 1] -= off
    tall = (op > 0.5) & (q[:, 1] > 0.5 * np.percentile(q[op > 0.5][:, 1], 99.5))
    cx, cz = (np.median(q[tall][:, 0]), np.median(q[tall][:, 2])) if tall.sum() > 200 else (np.median(q[:, 0]), np.median(q[:, 2]))
    q[:, 0] -= cx; q[:, 2] -= cz; q *= s
    w["x"], w["y"], w["z"] = q[:, 0].astype(np.float32), q[:, 1].astype(np.float32), q[:, 2].astype(np.float32)
    for k in ("scale_0", "scale_1", "scale_2"): w[k] = (w[k] + math.log(s)).astype(np.float32)
    gs_path = os.path.join(tr, args.name + ".gs.ply")
    PlyData([PlyElement.describe(w, "vertex")], text=False).write(gs_path); os.remove(tmp)
    scale_m = np.exp(np.column_stack([w["scale_0"], w["scale_1"], w["scale_2"]]).astype(np.float64)).max(1)
    solid = op > 0.4
    lo = np.percentile(q[solid], 0.5, 0); hi = np.percentile(q[solid], 99.5, 0)
    print("scene (metres): x [%.1f, %.1f]  z [%.1f, %.1f]  tallest %.1f m  -> %s" % (lo[0], hi[0], lo[2], hi[2], np.percentile(q[solid][:, 1], 99.5), gs_path))

    # ---- walkable grid
    res = args.cell; x0, z0 = lo[0] - 3, lo[2] - 3
    W = int((hi[0] + 3 - x0) / res) + 1; H = int((hi[2] + 3 - z0) / res) + 1
    def cells(mask):
        u = ((q[mask, 0] - x0) / res).astype(int); vv = ((q[mask, 2] - z0) / res).astype(int)
        ok = (u >= 0) & (u < W) & (vv >= 0) & (vv < H)
        return vv[ok], u[ok]
    ground = np.zeros((H, W), bool)
    gmask = (op > 0.2) & (np.abs(q[:, 1]) < 0.3)
    for r_lo, r_hi in ((0, 0.375), (0.375, 0.75), (0.75, 1.5), (1.5, 1e9)):
        m = gmask & (2 * scale_m >= r_lo) & (2 * scale_m < r_hi)
        layer = np.zeros((H, W), bool); layer[cells(m)] = True
        ground |= ndimage.binary_dilation(layer, structure=disk(min(max(r_lo, res), 2.5) / res))
    ground = ndimage.binary_closing(ground, structure=disk(1.0 / res))
    cnt = np.zeros((H, W), np.int32); np.add.at(cnt, cells((op > 0.4) & (q[:, 1] > 0.35) & (q[:, 1] < 2.3)), 1)
    obstacle = ndimage.binary_dilation(cnt >= 2, structure=disk(0.5 / res))
    walk = ground & ~obstacle
    lab, k = ndimage.label(walk); sizes = ndimage.sum(walk, lab, range(1, k + 1)) * res * res
    walk = np.isin(lab, [i + 1 for i, a_ in enumerate(sizes) if a_ >= 30.0])
    print("walkable: %.0f m2 in %d region(s) (largest %.0f m2); ground coverage %.0f m2; obstacles %.0f m2" % (
        walk.sum() * res * res, int((sizes >= 30).sum()), sizes.max() if k else 0, ground.sum() * res * res, obstacle.sum() * res * res))

    # ---- mesh (row-merged quads at y = 0) and navmesh
    verts, faces = [], []
    for r in range(H):
        row = walk[r]; c = 0
        while c < W:
            if row[c]:
                c1 = c
                while c1 < W and row[c1]: c1 += 1
                xa, xb, za, zb = x0 + c * res, x0 + c1 * res, z0 + r * res, z0 + (r + 1) * res
                i = len(verts) + 1; verts += [(xa, 0, za), (xb, 0, za), (xb, 0, zb), (xa, 0, zb)]; faces += [(i, i + 3, i + 2), (i, i + 2, i + 1)]
                c = c1
            else: c += 1
    obj = os.path.join(tr, args.name + ".walkable.obj")
    with open(obj, "w") as f:
        f.write("".join("v %.3f %.3f %.3f\n" % p for p in verts) + "".join("f %d %d %d\n" % t for t in faces))
    import habitat_sim
    sc = habitat_sim.SimulatorConfiguration(); sc.scene_id = obj; sc.create_renderer = False; sc.load_semantic_mesh = False
    sim = habitat_sim.Simulator(habitat_sim.Configuration(sc, [habitat_sim.agent.AgentConfiguration()]))
    ns = habitat_sim.NavMeshSettings(); ns.set_defaults(); ns.agent_radius = 0.1; ns.agent_height = 1.7; ns.cell_size = 0.1; ns.cell_height = 0.1; ns.agent_max_climb = 0.2
    ok = sim.recompute_navmesh(sim.pathfinder, ns)
    nav = os.path.join(tr, args.name + ".navmesh")
    if ok: sim.pathfinder.save_nav_mesh(nav)
    print("navmesh: built %s  navigable area %.0f m2 -> %s" % (ok, sim.pathfinder.navigable_area if ok else 0, nav)); sim.close()

    # ---- car-sized objects (candidate Layer B entries)
    m = (op > 0.4) & (q[:, 1] > 0.25) & (q[:, 1] < 3.0)
    P = q[m]; g = 0.17; iu = ((P[:, 0] - x0) / g).astype(int); iv = ((P[:, 2] - z0) / g).astype(int)
    occ = np.zeros((iv.max() + 1, iu.max() + 1), bool); occ[iv, iu] = True
    lab2, k2 = ndimage.label(ndimage.binary_closing(occ), structure=np.ones((3, 3))); ids = lab2[iv, iu]
    objects = []
    for c in range(1, k2 + 1):
        S = P[ids == c]
        if len(S) < 150: continue
        xy = S[:, [0, 2]] - S[:, [0, 2]].mean(0); ew, ev = np.linalg.eigh(np.cov(xy.T)); pr = xy @ ev
        L = np.percentile(pr[:, 1], 98) - np.percentile(pr[:, 1], 2); Wd = np.percentile(pr[:, 0], 98) - np.percentile(pr[:, 0], 2); Hh = np.percentile(S[:, 1], 97)
        if 2.2 < L < 7.0 and 0.9 < Wd < 3.0 and 0.6 < Hh < 2.6 and 1.6 < L / Wd < 3.4:
            objects.append({"object_id": "vehicle_%02d" % (len(objects) + 1), "label": "parked_vehicle", "position": [round(float(S[:, 0].mean()), 2), 0.0, round(float(S[:, 2].mean()), 2)],
                            "size_lwh": [round(float(L), 2), round(float(Wd), 2), round(float(Hh), 2)], "yaw_deg": round(math.degrees(math.atan2(ev[1, 1], ev[0, 1])), 1)})
    print("car-sized objects found:", len(objects))

    json.dump({"render_asset": "../../train/%s/%s.gs.ply" % (args.name, args.name), "render_asset_type": "gaussian_splatting", "units_to_meters": 1.0,
               "orient_up": [0, 1, 0], "orient_front": [0, 0, -1], "frustum_culling": False, "light_setup": "default"},
              open(os.path.join(out, "dynamic_nav", "stages", args.name + ".stage_config.json"), "w"), indent=2)
    info = {"twin_id": args.name, "source_ply": args.ply, "gaussians": int(len(w)), "metres_per_unit": s, "up_in_source": up.tolist(), "rotation_source_to_scene": R.tolist(),
            "ground_offset_units": off, "recentre_units": [float(cx), float(cz)], "frame": "Y up, metres, origin on the ground under the tallest structure",
            "bounds_m": {"x": [float(lo[0]), float(hi[0])], "z": [float(lo[2]), float(hi[2])], "tallest": float(np.percentile(q[solid][:, 1], 99.5))},
            "walkable_grid": {"origin_xz": [float(x0), float(z0)], "cell_m": res, "shape_hw": [H, W]}, "objects": objects}
    json.dump(info, open(os.path.join(out, "scene_info.json"), "w"), indent=2)
    np.save(os.path.join(out, "walkable_mask.npy"), walk)

    # ---- labelled top-down preview
    px = 0.08; PW = int((hi[0] + 3 - x0) / px); PH = int((hi[2] + 3 - z0) / px)
    rgb = np.clip(0.5 + 0.28209479 * np.column_stack([w["f_dc_0"], w["f_dc_1"], w["f_dc_2"]]), 0, 1)
    mm = (op > 0.25) & (q[:, 1] < 6.0); order = np.argsort(q[mm][:, 1])
    uu = ((q[mm][:, 0] - x0) / px).astype(int).clip(0, PW - 1); vv = ((q[mm][:, 2] - z0) / px).astype(int).clip(0, PH - 1)
    img = np.full((PH, PW, 3), 18, np.uint8)
    for du in (0, 1):
        for dv in (0, 1): img[(vv[order] + dv).clip(0, PH - 1), (uu[order] + du).clip(0, PW - 1)] = (rgb[mm][order] * 255).astype(np.uint8)
    wm = np.asarray(Image.fromarray(walk).resize((PW, PH), Image.NEAREST))
    img[wm] = (0.6 * img[wm] + 0.4 * np.array([40, 200, 90])).astype(np.uint8)
    im = Image.fromarray(img); dr = ImageDraw.Draw(im)
    for k in range(int(math.ceil(x0 / 10)) * 10, int(hi[0] + 3), 10): dr.line([((k - x0) / px, 0), ((k - x0) / px, PH)], fill=(70, 110, 160)); dr.text(((k - x0) / px + 3, 3), "x=%d" % k, fill=(150, 200, 255))
    for k in range(int(math.ceil(z0 / 10)) * 10, int(hi[2] + 3), 10): dr.line([(0, (k - z0) / px), (PW, (k - z0) / px)], fill=(70, 110, 160)); dr.text((3, (k - z0) / px + 3), "z=%d" % k, fill=(150, 200, 255))
    for o in objects: dr.text(((o["position"][0] - x0) / px, (o["position"][2] - z0) / px), o["object_id"][-2:], fill=(255, 220, 0))
    im.save(os.path.join(out, "topdown_walkable.jpg"), quality=85)
    print("wrote", os.path.join(out, "scene_info.json"), "and topdown_walkable.jpg", im.size)


if __name__ == "__main__":
    main()
