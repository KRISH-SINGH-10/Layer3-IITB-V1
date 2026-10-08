#!/usr/bin/env python
"""Publish an already rendered clip (layer3_mvp/outputs/<name>) to the Rerun viewer.

Sends the saved MP4s as video (small, full resolution) plus the boxes and the
navmesh map from the annotation files. No GPU rendering; takes a few seconds.
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import habitat_sim
import render_clip as rc
import rerun as rr
import rerun.blueprint as rrb

ROOT = "/home/24singhk/Layer3_Prototype"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True, help="Folder under layer3_mvp/outputs, e.g. plaza_six_long")
    ap.add_argument("--dataset", default=None, help="Default: gs_scene01/<clip>/<clip>.scene_dataset_config.json")
    ap.add_argument("--scene", default="scene01")
    ap.add_argument("--rerun", default="rerun+http://127.0.0.1:9876/proxy")
    args = ap.parse_args()
    clip_dir = args.clip if os.path.isabs(args.clip) else os.path.join(ROOT, "layer3_mvp", "outputs", args.clip)
    tag = os.path.basename(clip_dir.rstrip("/"))
    dataset = args.dataset or os.path.join(ROOT, "gs_scene01", tag, tag + ".scene_dataset_config.json")
    views = [v for v in ("ground", "elevated", "follow") if os.path.exists(os.path.join(clip_dir, v, "rgb.mp4"))]
    ann = {v: json.load(open(os.path.join(clip_dir, v, "annotations.json"))) for v in views}
    args.scene = ann[views[0]].get("scene", args.scene)
    actors = rc.load_actors(dataset, args.scene)
    ids = [a["actor_id"] for a in actors]
    color = {aid: rc.PALETTE[k % len(rc.PALETTE)] for k, aid in enumerate(ids)}

    rr.init("layer3-" + tag)
    rr.connect_grpc(args.rerun)
    panels = [rrb.Spatial2DView(origin="%s/rgb" % v, name="%s camera" % v.capitalize()) for v in views]
    panels += [rrb.Spatial2DView(origin="ground/depth", name="Depth (ground camera)"),
               rrb.Spatial2DView(origin="bev/map", name="Navmesh (top-down)")]
    rr.send_blueprint(rrb.Blueprint(rrb.Grid(*panels, grid_columns=3 if len(panels) > 4 else 2), collapse_panels=True))

    def log_video(entity, path, n_frames, fps):
        asset = rr.AssetVideo(path=path)
        rr.log(entity, asset, static=True)
        stamps = asset.read_frame_timestamps_nanos()[:n_frames]
        n = len(stamps)
        rr.send_columns(entity,
                        indexes=[rr.TimeColumn("frame", sequence=np.arange(n)), rr.TimeColumn("clip_time", duration=np.arange(n) / fps)],
                        columns=rr.VideoFrameReference.columns_nanos(stamps))
        return n

    for v in views:
        frames = ann[v]["frames"]; fps = float(ann[v]["fps"])
        n = log_video("%s/rgb" % v, os.path.join(clip_dir, v, "rgb.mp4"), len(frames), fps)
        for fr in frames[:n]:
            rr.set_time("frame", sequence=fr["frame"]); rr.set_time("clip_time", duration=fr["time_s"])
            shown = [a for a in fr["actors"] if a["bbox_xyxy"] and not a.get("occluded", False)]
            if shown:
                rr.log("%s/rgb/actors" % v, rr.Boxes2D(array=np.array([a["bbox_xyxy"] for a in shown]), array_format=rr.Box2DFormat.XYXY,
                                                       labels=[a["actor_id"] for a in shown], colors=[color[a["actor_id"]] for a in shown]))
        print("view %-8s %d frames as video" % (v, n))

    g = ann["ground"]; fps = float(g["fps"])
    log_video("ground/depth", os.path.join(clip_dir, "ground", "depth_vis.mp4"), len(g["frames"]), fps)
    nav = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(dataset))), "train", args.scene, args.scene + ".navmesh")   # scene folder holds train/<scene>/
    if not os.path.exists(nav):
        sys.exit("navmesh not found: " + nav)
    pf = habitat_sim.PathFinder(); pf.load_nav_mesh(nav)
    top = rc.TopDown(pf)
    rr.log("bev/map", rr.Image(top.image), static=True)
    rr.log("bev/map/planned_paths", rr.LineStrips2D([top.to_px(a["transl"][::10]) for a in actors],
                                                    colors=[rc.PALETTE[k % len(rc.PALETTE)] for k in range(len(actors))]), static=True)
    for fr in g["frames"]:
        pts = [(a["actor_id"], a["center_world"]) for a in fr["actors"] if a["center_world"]]
        if pts:
            rr.set_time("frame", sequence=fr["frame"]); rr.set_time("clip_time", duration=fr["time_s"])
            rr.log("bev/map/actors", rr.Points2D(top.to_px(np.array([p for _, p in pts])), radii=[6.0],
                                                 labels=[i for i, _ in pts], colors=[color[i] for i, _ in pts]))
    print("published layer3-%s (%d views, %.1fs)" % (tag, len(views), len(g["frames"]) / fps))


if __name__ == "__main__":
    main()
