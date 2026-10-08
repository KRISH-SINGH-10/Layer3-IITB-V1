# Layer3-IITB V1 — Scene-Conditioned Human Motion for Drone Digital Twins

Layer 3 of a five-layer synthetic drone-data pipeline. It places moving human actors into a
3D Gaussian Splatting (3DGS) digital twin, renders them from ground, drone and chase cameras,
and writes per-frame annotations.

| Layer | Module | Role |
|---|---|---|
| 1 | Scene reconstruction | Drone video to a 3DGS digital twin |
| 2 | Layered Object Representation | Static background (A), semi-static objects (B), dynamic actors (C) |
| **3** | **Human motion synthesis (this repo)** | **Walking SMPL-X actors placed and rendered in the twin** |
| 4 | Generative augmentation | Lighting, weather and domain adaptation |
| 5 | Drone-view rendering and export | Final frames and COCO / MOT annotations |

## What it does

1. **Onboards a raw 3DGS scene.** Fits the ground plane, rotates the scene upright, scales it to
   metres, derives the walkable area from the splats and bakes a navmesh.
2. **Reads a scenario.** Either the pipeline's scene-spec JSON (`layer_c` actors) or a short
   scenario file listing who walks from where to where.
3. **Generates motion.** Plans a route on the navmesh and synthesises an SMPL-X walking motion
   along it with GAMMA.
4. **Renders.** Animated Gaussian avatars are drawn inside the scene with Habitat-GS from three
   cameras: ground, elevated (drone) and a chase camera that follows one actor.
5. **Annotates.** Each frame records every actor's world position, 2D bounding box, a visibility
   score and an occlusion flag.

## Results

| Scene | Clip | Actors | Length |
|---|---|---|---|
| Airfield with control tower (own Layer 1 scan) | `samples/airfield_tower_walk_001` | 1 | 24 s, 716 frames at 30 fps |
| Plaza (Habitat-GS sample scene) | `samples/plaza_six` | 6 | 9.5 s, 189 frames at 20 fps |
| Promenade (Habitat-GS sample scene) | `samples/promenade_six` | 6 | auto-generated routes |

All clips are rendered at 1920x1080. Each sample folder holds annotated videos, the matching
annotation files and the resolved scenario. The airfield folder also holds the generated SMPL-X
motion (`actor_001_smplx_motion.pkl`) and a depth video.

## Repository layout

```
layer3_mvp/
  onboard_scene.py      raw 3DGS PLY -> upright metric scene + navmesh + object list
  spec_to_scenario.py   scene-spec JSON -> validated Layer 3 scenario
  auto_scenario.py      generate N non-clashing routes automatically
  build_scenario.py     scenario -> GAMMA motions + Habitat-GS dataset
  render_clip.py        dataset -> videos, depth, annotations (optionally live to Rerun)
  replay_clip.py        publish a rendered clip to the Rerun web viewer as video
  scenarios/            scenario files
  specs/                scene-spec example
patches/                build fix for the Habitat-GS CUDA rasterizer
scenes/twin01/          airfield scene: source PLY, scene info, walkable map, navmesh
samples/                example outputs
```

## Requirements

Not included in this repository; obtain each from its own source and licence.

| Component | Used for |
|---|---|
| [Habitat-GS](https://github.com/zju3dv/habitat-gs) (Habitat-Sim 0.3.3 with Gaussian Splatting) | Scene and avatar rendering, navmesh |
| [GAMMA](https://github.com/yz-cnsdqz/GAMMA-release) and its pretrained checkpoints | Locomotion synthesis |
| [SMPL-X](https://smpl-x.is.tue.mpg.de/) body models and VPoser | Body model |
| Gaussian avatars (`canonical_gs.npz`) | Actor appearance |
| Python packages: `numpy`, `scipy`, `plyfile`, `imageio`, `imageio-ffmpeg`, `opencv-python`, `Pillow`, `rerun-sdk` | Tooling |

Developed on an NVIDIA DGX (B200, CUDA 12.8, Ubuntu 24.04). The scripts assume the project root
`/home/24singhk/Layer3_Prototype`; change `ROOT` at the top of each script for another machine.

### Build fix for Habitat-GS

With GLM 1.x, `nvcc` silently drops GLM vector maths inside the rasterizer kernels unless
`--expt-relaxed-constexpr` is set. The symptom is a Gaussian scene that renders almost black with
no error. Apply the patch before building:

```bash
cd habitat-gs
git apply ../patches/habitat-gs-rasterizer-build-fix.patch
HABITAT_WITH_CUDA=ON HABITAT_WITH_BULLET=OFF pip install .
```

Habitat-Sim also needs the environment's EGL library on the loader path:

```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
```

## Usage

### 1. Onboard a scene

```bash
python layer3_mvp/onboard_scene.py --ply scenes/twin01/point_cloud.ply --name twin01 --metres-per-unit 4.2
```

Writes `gs_twin01/` with the upright metric scene, navmesh, `scene_info.json` and a labelled
top-down preview. `--metres-per-unit` is the real-world size of one unit in the source PLY. For
the airfield it was estimated at 4.2 from the dimensions of 14 parked cars.

### 2. Describe the scenario

From the pipeline's scene-spec format:

```bash
python layer3_mvp/spec_to_scenario.py --spec layer3_mvp/specs/airfield_tower_walk_001.json
```

Every actor position is checked against the walkable area; an invalid position stops with a
`PlacementError`. The command prints the build and render commands for the next steps.

Or generate routes automatically:

```bash
python layer3_mvp/auto_scenario.py --scene scene01 --name my_scene --actors 6 --route-length 25
```

Or write a scenario by hand (`layer3_mvp/scenarios/plaza_six.json`):

```json
{
  "actor_id": "person_1",
  "action": "walk",
  "avatar": "avatar2",
  "start": [42.67, -0.26, -24.39],
  "goal": [53.08, -0.34, -27.10],
  "start_time_s": 0.0,
  "seed": 1
}
```

### 3. Generate the motions

```bash
python layer3_mvp/build_scenario.py --scenario layer3_mvp/scenarios/plaza_six.json
```

Runs GAMMA once per actor, writes the Habitat-GS dataset and reports the closest approach between
any two actors. Existing motions are reused unless `--force` is given.

### 4. Render

```bash
python layer3_mvp/render_clip.py --dataset gs_scene01/plaza_six/plaza_six.scene_dataset_config.json --fps 30
```

Outputs under `layer3_mvp/outputs/<scenario>/<view>/`:

| File | Content |
|---|---|
| `rgb.mp4` | Clean frames |
| `rgb_annotated.mp4` | Frames with actor boxes |
| `depth_vis.mp4` | Depth as a colour map |
| `annotations.json` | Camera pose and per-frame actor records |

Cameras are chosen by test-rendering candidate positions and scoring line of sight to the walking
paths. Override them with `--ground-cam x y z`, `--elevated-cam x y z` and `--follow <actor_id>`.

### 5. View

```bash
python layer3_mvp/replay_clip.py --clip plaza_six
```

Publishes the clip to a running Rerun server with five panels: ground, elevated, follow, depth and
a top-down navmesh map with the actors' positions.

## Annotation format

```json
{
  "frame": 120,
  "time_s": 4.0,
  "actors": [
    {
      "actor_id": "person_1",
      "action": "walk",
      "avatar": "avatar2",
      "center_world": [48.1, 0.72, -25.9],
      "bbox_xyxy": [912.4, 388.0, 968.9, 540.2],
      "visibility": 0.96,
      "occluded": false
    }
  ]
}
```

Boxes are projected from the actor's body capsules. Visibility is the fraction of the body with a
clear line of sight in the rendered depth map. The follow view also stores the camera pose per
frame.

## Scene-spec extensions

The example in `layer3_mvp/specs/` follows the pipeline's scene-spec schema, with two additions:

- `layer_c[].goal_pos` — the actor's destination. The base schema gives a start position and a
  text prompt but no destination.
- `layer_b[].existing_in_twin` and `size_lwh` — objects detected in the scan, as opposed to
  objects placed from an asset library.

Spec coordinates are `x = scene X`, `y = -scene Z`, with `z` up, in metres.

## Limitations

- Walking only. Running, loitering and object interactions need other motion sources.
- Actors do not avoid each other; the builder reports the closest approach so routes can be changed.
- The scale of a raw scan is estimated when the source has no metric scale.
- Depth has holes on large flat ground surfaces.
- Lighting and weather fields in the scene-spec are carried through but not applied here.
- Views far from the original capture path show reconstruction artefacts.
