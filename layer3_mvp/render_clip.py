#!/usr/bin/env python
"""Render a clip of the Gaussian avatars walking in a Habitat-GS scene.

Works for any dataset whose scene instance lists gaussian_avatars (one or many).
Per camera view it writes rgb.mp4, rgb_annotated.mp4, depth_vis.mp4 and
annotations.json (camera pose plus each actor's position and 2D box per frame).
With --rerun it also streams RGB, depth, boxes and a navmesh top-down map with
the actors' positions to a running Rerun gRPC proxy.
"""
import argparse, json, math, os, pickle
import numpy as np
import habitat_sim
import imageio
import cv2
from habitat_sim.utils.common import quat_from_angle_axis, quat_to_magnum

ROOT = "/home/24singhk/Layer3_Prototype"
HFOV = 90.0


def quat_to_matrix(q):
    m = quat_to_magnum(q).to_matrix()
    return np.array([[m[c][r] for c in range(3)] for r in range(3)], dtype=np.float64)


def look_at_angles(cam, target):
    yaw = math.atan2(-(target[0] - cam[0]), -(target[2] - cam[2]))
    pitch = math.atan2(target[1] - cam[1], math.hypot(target[0] - cam[0], target[2] - cam[2]))
    return yaw, pitch


def angles_to_matrix(yaw, pitch):
    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    return ry @ rx


def project(points, cam_pos, cam_rot, width, height):
    """World points -> pixel (u, v) and depth along the camera's -z axis."""
    f = (width / 2.0) / math.tan(math.radians(HFOV) / 2.0)
    pc = (np.asarray(points, dtype=np.float64) - cam_pos) @ cam_rot
    depth = -pc[:, 2]
    safe = np.maximum(depth, 1e-6)
    u = width / 2.0 + f * pc[:, 0] / safe
    v = height / 2.0 - f * pc[:, 1] / safe
    return u, v, depth, f


def actor_box(capsules, cam_pos, cam_rot, width, height):
    if capsules is None or len(capsules) == 0:
        return None
    pts = np.concatenate([capsules[:, :3], capsules[:, 3:6]], 0)
    rad = np.concatenate([capsules[:, 6], capsules[:, 6]], 0)
    u, v, depth, f = project(pts, cam_pos, cam_rot, width, height)
    ok = depth > 0.2
    if not ok.any():
        return None
    r_px = f * rad[ok] / depth[ok]
    x0, x1 = float((u[ok] - r_px).min()), float((u[ok] + r_px).max())
    y0, y1 = float((v[ok] - r_px).min()), float((v[ok] + r_px).max())
    if x1 < 0 or y1 < 0 or x0 >= width or y0 >= height:
        return None
    return [max(0.0, x0), max(0.0, y0), min(width - 1.0, x1), min(height - 1.0, y1)]


def seen_fraction(points, depth_img, cam_pos, cam_rot, width, height, tol):
    """Fraction of world points that are in frame and not hidden behind nearer geometry."""
    u, v, d, _ = project(points, cam_pos, cam_rot, width, height)
    ok = (d > 0.2) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    if not ok.any():
        return 0.0
    r = depth_img[np.clip(np.round(v[ok]).astype(int), 0, height - 1), np.clip(np.round(u[ok]).astype(int), 0, width - 1)]
    return float(((r <= 0.0) | (r >= d[ok] - tol)).sum()) / len(points)


PALETTE = [(0, 220, 90), (255, 90, 90), (80, 160, 255), (255, 200, 0), (220, 90, 255), (0, 220, 220), (255, 140, 0), (160, 255, 120)]


def draw_box(img, box, color):
    x0, y0, x1, y1 = [int(round(b)) for b in box]
    t = max(2, img.shape[1] // 480)
    img[y0:y0 + t, x0:x1 + 1] = color
    img[max(y1 - t + 1, 0):y1 + 1, x0:x1 + 1] = color
    img[y0:y1 + 1, x0:x0 + t] = color
    img[y0:y1 + 1, max(x1 - t + 1, 0):x1 + 1] = color


def depth_to_color(depth, max_m):
    """Depth in metres -> RGB heat map (near = warm, far = cool, no hit = black)."""
    d = np.clip(depth / max_m, 0.0, 1.0)
    img = cv2.applyColorMap((255 - d * 255).astype(np.uint8), cv2.COLORMAP_TURBO)[..., ::-1].copy()
    img[depth <= 0.0] = 0
    return img


class TopDown:
    """Navmesh rasterised from above; x -> right, z -> down."""

    def __init__(self, pathfinder, meters_per_pixel=0.2, margin=2.0):
        lo, hi = pathfinder.get_bounds()
        self.mpp = meters_per_pixel
        self.origin = np.array([lo[0] - margin, lo[2] - margin], dtype=np.float64)
        w = int(math.ceil((hi[0] - lo[0] + 2 * margin) / self.mpp))
        h = int(math.ceil((hi[2] - lo[2] + 2 * margin) / self.mpp))
        verts = np.asarray(pathfinder.build_navmesh_vertices(), dtype=np.float64).reshape(-1, 3)
        tris = self.to_px(verts).reshape(-1, 3, 2).astype(np.int32)
        self.image = np.full((h, w, 3), 28, np.uint8)
        cv2.fillPoly(self.image, list(tris), (205, 205, 205))

    def to_px(self, pts):
        pts = np.asarray(pts, dtype=np.float64)
        return (pts[..., [0, 2]] - self.origin) / self.mpp


def load_actors(dataset, scene):
    """Actor ids, driver files and durations from the dataset's scene instance."""
    base = os.path.join(os.path.dirname(dataset), "scenes")
    inst = json.load(open(os.path.join(base, scene + ".scene_instance.json")))
    actors = []
    for k, av in enumerate(inst.get("gaussian_avatars", [])):
        path = av["driver"] if os.path.isabs(av["driver"]) else os.path.normpath(os.path.join(base, av["driver"]))
        drv = pickle.load(open(path, "rb"))
        tr = np.asarray(drv["transl"], dtype=np.float64)
        t0 = float(av.get("time_begin", 0.0))
        actors.append({"actor_id": av.get("name", "person_%d" % (k + 1)), "transl": tr, "time_begin": t0,
                       "time_end": t0 + len(tr) / float(drv["fps"]),
                       "avatar": os.path.basename(os.path.dirname(av["canonical_gaussians"]))})
    return actors


def set_camera(agent, cam, target):
    yaw, pitch = look_at_angles(cam, target)
    st = habitat_sim.AgentState()
    st.position = np.array(cam, dtype=np.float32)
    st.rotation = quat_from_angle_axis(yaw, np.array([0, 1.0, 0])) * quat_from_angle_axis(pitch, np.array([1.0, 0, 0]))
    agent.set_state(st)


def pick_cameras(actors, sim, agent, width, height):
    """Choose a ground and an elevated viewpoint.

    Candidates on rings around the walking area are scored by how many path
    endpoints they keep in frame, then test-rendered: views showing large dark
    voids (parts of the scene the reconstruction does not cover) are penalised.
    """
    pf = sim.pathfinder
    ends = np.array([p for a in actors for p in (a["transl"][0], a["transl"][-1])]) + np.array([0, 1.0, 0])
    center = ends.mean(0)
    samples = np.concatenate([a["transl"][:: max(1, len(a["transl"]) // 12)] for a in actors]) + np.array([0, 1.0, 0])
    extent = float(np.linalg.norm((samples - center)[:, [0, 2]], axis=1).max())

    def best_view(height_above, radii, need_navigable):
        best = None
        for radius in radii:
            for k in range(24):
                ang = 2 * math.pi * k / 24
                pos = center + np.array([radius * math.sin(ang), 0.0, radius * math.cos(ang)])
                if need_navigable:
                    snapped = np.array(pf.snap_point(pos.astype(np.float32)))
                    if np.isnan(snapped).any() or np.linalg.norm((snapped - pos)[[0, 2]]) > 0.5:
                        continue
                    if pf.distance_to_closest_obstacle(snapped, 2.0) < 0.8:
                        continue
                    cam = snapped + np.array([0, height_above, 0])
                else:
                    cam = np.array([pos[0], center[1] - 1.0 + height_above, pos[2]])
                set_camera(agent, cam, center)
                yaw, pitch = look_at_angles(cam, center)
                obs = sim.get_sensor_observations()
                rgb = obs["color_sensor"][..., :3]
                depth = np.asarray(obs["depth_sensor"], dtype=np.float32)
                seen = seen_fraction(samples, depth, cam, angles_to_matrix(yaw, pitch), width, height, 1.0)
                dark = float((rgb.max(-1) < 35).mean())
                near = float(((depth > 0) & (depth < 2.5)).mean())  # something right in front of the lens
                score = 10.0 * seen - 0.03 * radius - 30.0 * dark - 6.0 * near
                if best is None or score > best[0]:
                    best = (score, cam, seen)
        if best is None:
            raise RuntimeError("no usable camera position found")
        return best

    _, ground, dark_g = best_view(1.7, tuple(max(10.0, extent * f) for f in (0.7, 0.9, 1.1, 1.3)), True)
    # Drone-style view: high and fairly steep, which is where a drone-captured reconstruction is cleanest.
    _, elevated, dark_e = max([best_view(max(10.0, extent * h), tuple(max(4.0, extent * f) for f in (0.3, 0.6, 0.9)), False)
                               for h in (0.9, 1.2)] +
                              [best_view(h, tuple(max(8.0, extent * f) for f in (0.6, 0.9, 1.2)), False) for h in (5.0, 8.0)],
                              key=lambda b: b[0])
    print("cameras: ground %s (paths visible %.0f%%)  elevated %s (paths visible %.0f%%)" % (
        np.round(ground, 1).tolist(), 100 * dark_g, np.round(elevated, 1).tolist(), 100 * dark_e))
    return {"ground": {"pos": ground, "target": center}, "elevated": {"pos": elevated, "target": center}}


def main():
    global HFOV
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=ROOT + "/gs_scene01/dynamic_nav/dynamic_nav.scene_dataset_config.json")
    ap.add_argument("--scene", default="scene01")
    ap.add_argument("--out", default=None, help="Default: layer3_mvp/outputs/<dataset folder name>")
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--duration", type=float, default=0.0, help="Seconds; 0 = until the first actor finishes.")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--hfov", type=float, default=70.0, help="Horizontal field of view in degrees.")
    ap.add_argument("--jpeg-quality", type=int, default=88, help="JPEG quality of the frames streamed to Rerun.")
    ap.add_argument("--depth-max", type=float, default=40.0, help="Depth display range in metres.")
    ap.add_argument("--ground-cam", type=float, nargs=3, default=None, help="Override ground camera position x y z.")
    ap.add_argument("--elevated-cam", type=float, nargs=3, default=None, help="Override elevated camera position x y z.")
    ap.add_argument("--follow", default="person_1", help="Actor a close tracking camera follows; 'none' to disable.")
    ap.add_argument("--stream-width", type=int, default=1280,
                    help="Width of the wide views sent to Rerun (the web viewer holds about 2 GB); 0 = full size. Saved videos are always full size.")
    ap.add_argument("--rerun", default=None, help="e.g. rerun+http://127.0.0.1:9876/proxy")
    ap.add_argument("--rerun-app", default=None)
    args = ap.parse_args()
    HFOV = args.hfov
    tag = os.path.basename(os.path.dirname(args.dataset))
    out_root = args.out or os.path.join(ROOT, "layer3_mvp", "outputs", tag)

    actors = load_actors(args.dataset, args.scene)
    duration = args.duration or min(a["time_end"] for a in actors)

    s = habitat_sim.utils.settings.default_sim_settings.copy()
    s.update({"scene": args.scene, "scene_dataset_config_file": args.dataset, "default_agent_navmesh": False,
              "enable_physics": False, "sensor_height": 0.0, "width": args.width, "height": args.height,
              "depth_sensor": True, "hfov": args.hfov})
    sim = habitat_sim.Simulator(habitat_sim.utils.settings.make_cfg(s))
    agent = sim.get_agent(0)
    mgr = sim._gaussian_avatar_manager
    avatars = list(getattr(mgr, "avatars", None) or getattr(mgr, "_avatars", [])) if mgr is not None else []
    if len(avatars) != len(actors):
        raise RuntimeError("scene instance lists %d actors but %d avatars loaded" % (len(actors), len(avatars)))
    print("actors:", len(avatars), "| duration %.2fs at %.0f fps" % (duration, args.fps))
    views = pick_cameras(actors, sim, agent, args.width, args.height)
    for name, override in (("ground", args.ground_cam), ("elevated", args.elevated_cam)):
        if override:
            views[name]["pos"] = np.array(override, dtype=np.float64)
    topdown = TopDown(sim.pathfinder)
    ids = [a["actor_id"] for a in actors]
    if args.follow != "none":
        views["follow"] = {"track": ids.index(args.follow) if args.follow in ids else 0}

    def actor_center(k):
        caps = np.asarray(avatars[k]._proxy_capsules, dtype=np.float64)
        return ((caps[:, :3] + caps[:, 3:6]) / 2.0).mean(0), caps

    def camera_pose():
        ss = agent.get_state().sensor_states["color_sensor"]
        return np.array(ss.position, dtype=np.float64), quat_to_matrix(ss.rotation)

    def follow_offset(k):
        """Fixed offset for a low chase camera: the direction around the actor that keeps them in
        clear view for most of the walk (planters and pillars block eye-level side views)."""
        tr = actors[k]["transl"]
        d = tr[-1] - tr[0]; d[1] = 0.0; d /= np.linalg.norm(d)
        side = np.array([d[2], 0.0, -d[0]])
        times = np.linspace(0.5, duration - 0.5, 14)
        best = None
        for deg in range(0, 360, 45):
            ang = math.radians(deg)
            off = 5.0 * (math.cos(ang) * d + math.sin(ang) * side) + np.array([0, 2.2, 0])
            seen = []
            for t in times:
                sim.gaussian_time = float(t); sim._update_gaussian_avatars()
                c, caps = actor_center(k)
                set_camera(agent, c + off, c)
                depth = np.asarray(sim.get_sensor_observations()["depth_sensor"], dtype=np.float32)
                seen.append(seen_fraction((caps[:, :3] + caps[:, 3:6]) / 2.0, depth, *camera_pose(), args.width, args.height, 0.6))
            seen = np.array(seen)
            score = float(seen.mean()) - 0.5 * float((seen < 0.3).mean()) + (0.05 if deg in (45, 315) else 0.0)
            if best is None or score > best[0]:
                best = (score, off)
        return best[1]

    rr = None
    if args.rerun:
        import rerun as rr
        import rerun.blueprint as rrb
        rr.init(args.rerun_app or "layer3-" + tag)
        rr.connect_grpc(args.rerun)
        rr.send_blueprint(rrb.Blueprint(
            rrb.Grid(
                rrb.Spatial2DView(origin="ground/rgb", name="Ground camera"),
                rrb.Spatial2DView(origin="elevated/rgb", name="Elevated camera"),
                *([rrb.Spatial2DView(origin="follow/rgb", name="Follow camera (%s)" % ids[views["follow"]["track"]])] if "follow" in views else []),
                rrb.Spatial2DView(origin="ground/depth", name="Depth (ground camera)"),
                rrb.Spatial2DView(origin="bev/map", name="Navmesh (top-down)"),
                grid_columns=3 if "follow" in views else 2,
            ),
            collapse_panels=True,
        ))
        rr.log("bev/map", rr.Image(topdown.image), static=True)
        rr.log("bev/map/planned_paths", rr.LineStrips2D([topdown.to_px(a["transl"][::10]) for a in actors],
                                                        colors=[PALETTE[k % len(PALETTE)] for k in range(len(actors))]), static=True)
        cams = np.array([v["pos"] for v in views.values() if "pos" in v])
        rr.log("bev/map/cameras", rr.Points2D(topdown.to_px(cams), colors=[[255, 255, 255]], radii=[5.0],
                                              labels=["%s camera" % k for k, v in views.items() if "pos" in v], show_labels=False), static=True)

    n_frames = int(round(duration * args.fps))
    for vi, (vname, view) in enumerate(views.items()):
        track = view.get("track")
        if track is None:
            set_camera(agent, view["pos"], view["target"])
            cam_pos, cam_rot = camera_pose()
        else:
            offset = follow_offset(track); smooth = None

        out_dir = os.path.join(out_root, vname); os.makedirs(out_dir, exist_ok=True)
        writers = {n: imageio.get_writer(os.path.join(out_dir, n + ".mp4"), fps=args.fps, quality=9, macro_block_size=None)
                   for n in ("rgb", "rgb_annotated", "depth_vis")}
        frames_meta = []; n_boxes = 0; min_gap = 1e9
        for i in range(n_frames):
            t = i / args.fps
            sim.gaussian_time = t
            sim._update_gaussian_avatars()
            if track is not None:
                c, _ = actor_center(track)
                smooth = c if smooth is None else 0.8 * smooth + 0.2 * c
                set_camera(agent, smooth + offset, smooth)
                cam_pos, cam_rot = camera_pose()
            obs = sim.get_sensor_observations()
            rgb = obs["color_sensor"][..., :3].copy()
            depth = np.asarray(obs["depth_sensor"], dtype=np.float32)
            depth_vis = depth_to_color(depth, args.depth_max)
            ann = rgb.copy(); rec = []; centers = []; boxes = []; box_ids = []
            for k, av in enumerate(avatars):
                caps = np.asarray(av._proxy_capsules, dtype=np.float64)
                visible = len(caps) > 0 and actors[k]["time_begin"] <= t <= actors[k]["time_end"]
                box = actor_box(caps, cam_pos, cam_rot, args.width, args.height) if visible else None
                root = ((caps[:, :3] + caps[:, 3:6]) / 2.0).mean(0) if visible else None
                vis = seen_fraction((caps[:, :3] + caps[:, 3:6]) / 2.0, depth, cam_pos, cam_rot, args.width, args.height, 0.6) if box is not None else 0.0
                rec.append({"actor_id": ids[k], "action": "walk", "avatar": actors[k]["avatar"],
                            "center_world": root.tolist() if root is not None else None, "bbox_xyxy": box,
                            "visibility": round(vis, 3), "occluded": bool(box is not None and vis < 0.3)})
                if root is not None:
                    centers.append((k, root))
                if box is not None and vis >= 0.3:
                    draw_box(ann, box, PALETTE[k % len(PALETTE)]); boxes.append(box); box_ids.append(k); n_boxes += 1
            for a in range(len(centers)):
                for b in range(a + 1, len(centers)):
                    min_gap = min(min_gap, float(np.linalg.norm((centers[a][1] - centers[b][1])[[0, 2]])))
            frames_meta.append({"frame": i, "time_s": round(t, 4), "actors": rec})
            if track is not None:
                frames_meta[-1]["camera_position"] = cam_pos.tolist(); frames_meta[-1]["camera_rotation_world_from_cam"] = cam_rot.tolist()
            writers["rgb"].append_data(rgb); writers["rgb_annotated"].append_data(ann); writers["depth_vis"].append_data(depth_vis)
            if rr is not None:
                rr.set_time("frame", sequence=i); rr.set_time("clip_time", duration=t)
                sw = args.width if (track is not None or not args.stream_width) else min(args.stream_width, args.width)
                k_s = sw / float(args.width)
                img = rgb if sw == args.width else cv2.resize(rgb, (sw, int(round(args.height * k_s))), interpolation=cv2.INTER_AREA)
                rr.log("%s/rgb" % vname, rr.Image(img).compress(jpeg_quality=args.jpeg_quality))
                if boxes:
                    rr.log("%s/rgb/actors" % vname, rr.Boxes2D(array=np.array(boxes) * k_s, array_format=rr.Box2DFormat.XYXY,
                                                                labels=[ids[k] for k in box_ids],
                                                                colors=[PALETTE[k % len(PALETTE)] for k in box_ids]))
                if vi == 0:
                    rr.log("%s/depth" % vname, rr.Image(cv2.resize(depth_vis, (640, int(round(640.0 * args.height / args.width))), interpolation=cv2.INTER_AREA)).compress(jpeg_quality=75))
                    if centers:
                        rr.log("bev/map/actors", rr.Points2D(topdown.to_px(np.array([c for _, c in centers])), radii=[6.0],
                                                             labels=[ids[k] for k, _ in centers],
                                                             colors=[PALETTE[k % len(PALETTE)] for k, _ in centers]))
        for w in writers.values():
            w.close()
        meta = {"scene": args.scene, "view": vname, "fps": args.fps, "width": args.width, "height": args.height, "hfov_deg": HFOV,
                "camera": "moving (per-frame pose in frames)" if track is not None else "fixed", "camera_position": cam_pos.tolist(), "camera_rotation_world_from_cam": cam_rot.tolist(), "frames": frames_meta}
        json.dump(meta, open(os.path.join(out_dir, "annotations.json"), "w"))
        print("view %-8s frames %d  boxes %d  closest pair %.2fm  cam %s -> %s" % (
            vname, n_frames, n_boxes, min_gap if min_gap < 1e9 else -1, np.round(cam_pos, 1).tolist(), out_dir))
    sim.close()


if __name__ == "__main__":
    main()
