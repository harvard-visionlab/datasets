# ProcTHOR routes → slipstream lossless stores

Scripted camera routes through ProcTHOR-10k houses, rendered with the procthor-walks camera (AI2-THOR 5.0.0,
CloudRendering, 160×120, VFOV 73.74°, height 1.454 m, physics paused, agent invisible), in furnished and empty
versions of each house on identical paths. Datasets `procthor-routes-objects` / `procthor-routes-empty`
(registry `datasets/_configs/procthor_routes.py`, card `datasets/cards/procthor-routes.md`).

## Modules

| module | env | what |
| --- | --- | --- |
| `thor.py` | render | renderer port of Rupert Tawiah-Quashie's `prj_habitat/scripts/generate_procthor_data.py` (camera, coordinates, empty house, agent-invisibility check) |
| `plan.py` | render | route planner on the house's reachable grid: goal / turn / outback / loop / brownian / brownian_inertia{80,90,95} (90 piloted, not built); `metrics()` per window |
| `pilot.py` | render | plan + render windows for a set of houses, both cells; `--shard k/n`, resumable (house.json written last) |
| `viz.py` | render | trajectory sheets and GIFs |
| `check_render.py` | both | render-equivalence gate vs the procthor-walks lossless stores (`dump` datasets env, `compare` render env; `--sequential`, `--gen-root`) |
| `build.py` | datasets | `store` (lossless store + verify), `index` (splits/v1 + subsets/*.parquet), `pair` (cross-cell check) |
| `publish.py` | datasets | lab_storage master + private S3 |

Route kinds map to subset names in `build.ROUTE`: goal→goal_directed, turn→full_turn, outback→out_and_back. The kind's
index in `plan.TYPES` seeds the planner rng (`[house, type index, T, k, try]`): append new kinds, never reorder.

## Render environment

Kempner A100 (`--partition kempner --account kempner_alvarez_lab --exclude holygpu8a19102`: that node can't render).
Rupert's env: `/n/holylabs/alvarez_lab/Users/rtawiahquashie/envs/procthor/bin/python` (ai2thor 5.0.0, numpy 2.2.6,
pillow 12.3.0, scipy) with `.../envs/vulkan-tools/bin` on PATH. House file: `val.json.gz`
(sha256 42986daa2f2a3a84eb1bf1c2cadf83231f9691d0461d8cb5a57eede97683e77e; private copy at
`s3://visionlab-datasets/slipstream-cache/procthor-walks/source/procthor10k-rtq/`).

Why A100 (check_render, 2026-11-27, 6 stored walks × 2 cells × 5 frames vs the lossless stores): nothing is bit-exact,
not even Rupert's unchanged generator re-run (A100: 80 % / 100 % of frames within ±1 grey level objects / empty, a few
isolated pixels up to 130). Our port on A100 (walk rendered in order): 70 % / 97 %, mean |diff| 0.007 / 0.001. Our port
on an RTX 4090 (vesper): 0 % / 13 %, mean |diff| 0.22 / 0.13, 7–10 % of pixels off. Same ai2thor build, env, house
file, frame size, FOV, poses (generator re-run reproduces stored poses exactly).

## Val build (2026-11-27/28)

```bash
# render: 4 A100s x 4 workers, 200 val houses, ~56 min, 49 GB npz
python -m procthor_routes.pilot --houses-root <src> --out <R>/val --houses 0-199 --shard $K/16 --per 5 --lengths 61,121 \
    --types goal,turn,outback,loop,brownian,brownian_inertia80,brownian_inertia95      # cd datasets/prep
# stores (shared CPU job), tables, pairing, publish
python -m datasets.prep.procthor_routes.build store --render <R>/val --split val --cell {objects,empty} --store-root <ST>
python -m datasets.prep.procthor_routes.build index --render <R>/val --split val --tree-root <R>/tree
python -m datasets.prep.procthor_routes.build pair  --store-root <ST> --split val
python -m datasets.prep.procthor_routes.publish --cell {objects,empty} --split val --store-root <ST> --tree-root <R>/tree
```
`<R>` = `/n/netscratch/alvarez_lab/Lab/alvarez/procthor-routes`, `<ST>` = `/n/netscratch/alvarez_lab/Lab/datasets/slipstream`.
Per house: up to 5 windows per (route, length), 40 planner tries each; every try is counted in house.json `attempts`.
