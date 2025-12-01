# 3DGRUT Public API & Usage Guide

This document unifies everything you need to know to script against the repository: Python APIs, CLIs, exporters, GUI hooks, and the playground engine. It is organized by component so you can jump directly to the part you need.

---

## 1. Configuration Model & Data Conventions

- **Hydra/OmegaConf everywhere.** All CLIs resolve their configuration through Hydra. You can override any key via `key=value` arguments, or provide entire config files under `configs/` (see `configs/apps/*.yaml` for presets).
- **Batch layout.** Camera rays, ground-truth pixels, and intrinsics are described by the `Batch` dataclass, while dataset protocols declare the metadata each dataset must expose.

```24:83:threedgrut/datasets/protocols.py
@dataclass
class Batch:
    rays_ori: torch.Tensor  # [B, H, W, 3]
    rays_dir: torch.Tensor
    T_to_world: torch.Tensor
    rgb_gt: Optional[torch.Tensor] = None
    mask: Optional[torch.Tensor] = None
    intrinsics: Optional[list] = None
    intrinsics_OpenCVPinholeCameraModelParameters: Optional[dict] = None
    intrinsics_OpenCVFisheyeCameraModelParameters: Optional[dict] = None


class BoundedMultiViewDataset(Protocol):
    def get_scene_bbox(self) -> tuple[torch.Tensor, torch.Tensor]: ...
    def get_scene_extent(self) -> float: ...
    def get_observer_points(self) -> np.ndarray: ...
    def get_poses(self) -> np.ndarray: ...
    def get_gpu_batch_with_intrinsics(self, batch: dict) -> Batch: ...
```

- **Custom OmegaConf resolvers.** `train.py` registers resolvers such as `int_list` so you can pass lists on the command line. Hydra instantiates `Trainer3DGRUT` automatically:

```16:55:train.py
import hydra
from omegaconf import DictConfig, OmegaConf
from threedgrut.utils.logger import logger

OmegaConf.register_new_resolver("int_list", lambda l: [int(x) for x in l])

@hydra.main(config_path="configs", version_base=None)
def main(conf: DictConfig) -> None:
    logger.info(f"Git hash: {get_git_revision_hash()}")
    logger.info(f"Compiling native code..")
    from threedgrut.trainer import Trainer3DGRUT

    trainer = Trainer3DGRUT(conf)
    trainer.run_training()


if __name__ == "__main__":
    main()
```

**Coordinate system:** everything inside the trainer, datasets, and renderers uses the “right, down, front” convention (documented in `BoundedMultiViewDataset`).

---

## 2. Command-Line Entry Points

| Script | Purpose | Key Flags |
|--------|---------|-----------|
| `python train.py --config-name apps/nerf_synthetic_3dgrt.yaml path=...` | Hydra-driven training loop. | `with_gui`, `with_viser_gui`, `n_iterations`, `resume`, any config key override. |
| `python render.py --checkpoint runs/<scene>/ckpt_last.pt --out-dir outputs/eval` | Offline test-set rendering and metric logging via `Renderer`. | `--path` to override dataset path, `--save-gt/--compute-extra-metrics`. |
| `python playground.py --gs_object runs/<scene>/ckpt_last.pt` | Launch Polyscope GUI playground (3DGRUT renderer + mesh insertions). | `--mesh_assets`, `--envmap_assets`, `--buffer_mode`. |
| `python threedgrut_playground/viser_gui.py --gs_object=...` | Lightweight viser-based remote GUI. | Same as playground + viser-specific toggles. |
| `bash benchmark/*.sh` | Reproduce paper numbers per dataset. | Provide config YAML / results folder arguments. |
| `python -m threedgrut.export.scripts.ply_to_usd input.ply --output_file model.usdz` | Convert pre-existing PLY Gaussians into USDZ for Omniverse / Isaac Sim. | `--apply_normalizing_transform` (in config). |

Hydra allows stacking overrides, e.g.:

```bash
python train.py --config-name apps/colmap_3dgrt.yaml \
  path=data/mipnerf360/bonsai out_dir=runs experiment_name=bonsai \
  dataset.downsample_factor=2 optimizer.type=selective_adam with_gui=True
```

---

## 3. Core Training Stack (`threedgrut`)

### 3.1 `Trainer3DGRUT`

`Trainer3DGRUT` encapsulates dataset loading, model creation, optimization (including densification strategies), logging, GUI updates, validation, evaluation, and export.

```48:136:threedgrut/trainer.py
class Trainer3DGRUT:
    """Trainer for paper: "3D Gaussian Ray Tracing..." """

    @staticmethod
    def create_from_checkpoint(resume: str, conf: DictConfig):
        conf.resume = resume
        conf.import_ingp.enabled = False
        conf.import_ply.enabled = False
        return Trainer3DGRUT(conf)

    @staticmethod
    def create_from_ingp(ply_path: str, conf: DictConfig):
        conf.resume = ""
        conf.import_ingp.enabled = True
        conf.import_ingp.path = ply_path
        conf.import_ply.enabled = False
        return Trainer3DGRUT(conf)

    @staticmethod
    def create_from_ply(ply_path: str, conf: DictConfig):
        conf.resume = ""
        conf.import_ingp.enabled = False
        conf.import_ply.enabled = True
        conf.import_ply.path = ply_path
        return Trainer3DGRUT(conf)
```

Key methods:

- `__init__`: builds dataloaders, initializes scene extents, creates `MixtureOfGaussians`, picks a strategy, configures metrics/logging, and optionally the GUI.
- `run_training()`: loops `run_train_pass` for as many epochs as Hydra computed, interleaving validation checks, GUI updates, and checkpointing.
- `run_validation_pass()`, `get_losses()`, `get_metrics()`: introspection hooks for your own dashboards or automated sweeps.
- `on_training_end()`: exports INGP / PLY / USDZ files and optionally evaluates checkpoints.
- `save_checkpoint()` and exporter logic allow you to rehydrate training later via `create_from_*` helpers.

**Programmatic usage example:**

```python
from omegaconf import OmegaConf
from threedgrut.trainer import Trainer3DGRUT

conf = OmegaConf.load("configs/apps/colmap_3dgrt.yaml")
conf.path = "data/mipnerf360/bonsai"
conf.with_gui = False
trainer = Trainer3DGRUT.create_from_checkpoint("runs/bonsai/ckpt_last.pt", conf)
trainer.run_validation_pass(conf)  # run only validation
```

### 3.2 `Renderer`

`threedgrut.render.Renderer` renders an entire dataset split using a preloaded model or a checkpoint.

```32:138:threedgrut/render.py
class Renderer:
    def __init__(self, model, conf, global_step, out_dir, path="", save_gt=True, writer=None,
                 compute_extra_metrics=True) -> None:
        self.dataset, self.dataloader = self.create_test_dataloader(conf)
        ...

    @classmethod
    def from_checkpoint(cls, checkpoint_path, out_dir, path="", save_gt=True, writer=None,
                        model=None, computes_extra_metrics=True):
        checkpoint = torch.load(checkpoint_path)
        conf = checkpoint["config"]
        if model is None:
            model = MixtureOfGaussians(conf)
            model.init_from_checkpoint(checkpoint)
        model.build_acc()
        return Renderer(model=model, conf=conf, global_step=global_step, out_dir=out_dir, ...)

    def render_all(self):
        criterions = {"psnr": PeakSignalNoiseRatio(data_range=1).to("cuda")}
        ...
        for iteration, batch in enumerate(self.dataloader):
            gpu_batch = self.dataset.get_gpu_batch_with_intrinsics(batch)
            outputs = self.model(gpu_batch)
            torchvision.utils.save_image(...)
            psnr_single_img = criterions["psnr"](outputs["pred_rgb"], gpu_batch.rgb_gt).item()
            ...
        return mean_psnr, std_psnr, mean_inference_time
```

Usage tips:

- Call `Renderer.from_preloaded_model(model, out_dir, path, save_gt)` right after training to re-use GPU memory.
- `compute_extra_metrics=False` is useful if you only care about PSNR but want faster renders.
- You can extend `render_all` to dump depth, opacity, or other buffers from `outputs`.

---

## 4. Dataset Interfaces (`threedgrut.datasets`)

- Dataset factory functions normalize creation of train/val/test splits:

```21:98:threedgrut/datasets/__init__.py
def make(name: str, config, ray_jitter):
    match name:
        case "nerf":
            train_dataset = NeRFDataset(...)
            val_dataset = NeRFDataset(...)
        case "colmap":
            train_dataset = ColmapDataset(...)
            val_dataset = ColmapDataset(...)
        case "scannetpp":
            ...
        case _:
            raise ValueError(...)
    return train_dataset, val_dataset
```

- **`ColmapDataset`**: loads `sparse/0` intrinsics/extrinsics, filters frames by `test_split_interval`, lazily caches per-worker rays, and exposes masks. `__getitem__` returns Python dicts, `get_gpu_batch_with_intrinsics` converts them to `Batch`.

```342:397:threedgrut/datasets/dataset_colmap.py
    def __getitem__(self, idx) -> dict:
        image_data = np.asarray(Image.open(self.image_paths[idx]))
        output_dict = {
            "data": torch.tensor(image_data).unsqueeze(0),
            "pose": torch.tensor(self.poses[idx]).unsqueeze(0),
            "intr": self.get_intrinsics_idx(idx),
        }
        if os.path.exists(mask_path := self.mask_paths[idx]):
            mask = torch.from_numpy(np.array(Image.open(mask_path).convert("L"))).reshape(1, actual_h, actual_w, 1)
            output_dict["mask"] = mask
        return output_dict
```

- **`NeRFDataset`**: reads `{transforms_{split}.json}`, performs Blender → 3DGRUT axis conversion, and caches camera rays on-demand per data loader worker to keep VRAM usage in check.
- **`ScannetppDataset`**: identical API, but tailored to fisheye and multi-sensor data (see file for details).
- **`MultiEpochsDataLoader`** repeats the dataset indefinitely so `n_iterations` can exceed dataset length without Hydra hacks.

```136:166:threedgrut/datasets/utils.py
class MultiEpochsDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)
```

**Example: iterate over COLMAP validation views manually**

```python
from omegaconf import OmegaConf
from threedgrut import datasets

conf = OmegaConf.load("configs/apps/colmap_3dgrt.yaml")
conf.path = "data/mipnerf360/bonsai"
train_ds, val_ds = datasets.make(conf.dataset.type, conf, ray_jitter=None)
sample = val_ds[0]
gpu_batch = val_ds.get_gpu_batch_with_intrinsics(sample)
print(gpu_batch.rgb_gt.shape, gpu_batch.mask is not None)
```

---

## 5. Model & Rendering Internals

### 5.1 `MixtureOfGaussians`

This `torch.nn.Module` hosts Gaussian parameters (position, rotation, scale, density, SH features), manages initialization, optimizers, and communicates with native renderers (`threedgrt_tracer` for OptiX and `threedgut_tracer` for rasterization).

```49:205:threedgrut/model/model.py
class MixtureOfGaussians(torch.nn.Module, ExportableModel):
    def __init__(self, conf, scene_extent=None):
        super().__init__()
        self.positions = torch.nn.Parameter(torch.empty([0, 3]))
        self.rotation = torch.nn.Parameter(torch.empty([0, 4]))
        self.scale = torch.nn.Parameter(torch.empty([0, 3]))
        self.density = torch.nn.Parameter(torch.empty([0, 1]))
        self.features_albedo = torch.nn.Parameter(torch.empty([0, 3]))
        self.features_specular = torch.nn.Parameter(torch.empty([0, specular_dim]))
        ...
        if conf.render.method == "3dgrt":
            self.renderer = threedgrt_tracer.Tracer(conf)
        elif conf.render.method == "3dgut":
            self.renderer = threedgut_tracer.Tracer(conf)
        else:
            raise ValueError(...)
```

Important helpers:

- `init_from_random_point_cloud`, `init_from_colmap`, `init_from_pretrained_point_cloud`, `init_from_checkpoint`, `init_from_ingp`, `init_from_ply`.
- `build_acc(rebuild=True)` rebuilds the BVH and must be called after structural changes.
- `get_model_parameters()` collects everything you need to checkpoint or export.
- SH feature management supports **progressive training** (`feature_dim_increase_interval`, `increase_num_active_features()`).

### 5.2 Backgrounds & Losses

- `threedgrut.model.background.make` returns either `BackgroundColor` (black/white/random) or `SkipBackground`. `BackgroundColor` optionally injects random colors per pixel while training to reduce bias.

```29:95:threedgrut/model/background.py
def make(name: str, config):
    match name:
        case "background-color":
            return BackgroundColor(config=config)
        case "skip-background":
            return SkipBackground(config=config)

class BackgroundColor(BaseBackground):
    def setup(self, **kwargs):
        self.background_color_type = self.config.color
        ...
    def forward(self, ray_to_world, rays_d, rgb, opacity, train: bool):
        if self.background_color_type == "random" and train:
            color = torch.rand_like(rays_d, dtype=torch.float32, device=self.device)
            self.color = color
            rgb = rgb + color * (1.0 - opacity)
        elif self.background_color_type != "black":
            rgb = rgb + self.color * (1.0 - opacity)
        return rgb, opacity
```

- Losses (`threedgrut/model/losses.py`) expose L1/L2/SSIM building blocks, with the trainer toggling them via Hydra config.

### 5.3 Geometry helpers

Functions such as `k_nearest_neighbors`, `nearest_neighbor_dist_cpuKD`, and `safe_normalize` (`threedgrut/model/geometry.py`) feed densification heuristics and exports. They rely on scikit-learn’s KD-Tree and operate on CPU to keep GPU kernels minimal.

---

## 6. Optimization Strategies & Utilities

### 6.1 Strategies

- `BaseStrategy` defines hooks around each training iteration (pre/post backward, post optimizer, gradient buffers).
- `GSStrategy` implements the classic densify/split/prune schedule from the 3D Gaussian Splatting paper, complete with gradient accumulation buffers and density reset hooks.

```26:126:threedgrut/strategy/gs.py
class GSStrategy(BaseStrategy):
    def __init__(self, config, model: MixtureOfGaussians) -> None:
        super().__init__(config=config, model=model)
        self.split_n_gaussians = self.conf.strategy.densify.split.n_gaussians
        self.relative_size_threshold = self.conf.strategy.densify.relative_size_threshold
        ...

    def post_optimizer_step(self, step: int, scene_extent: float, train_dataset, batch=None, writer=None) -> bool:
        scene_updated = False
        if check_step_condition(...):
            self.densify_gaussians(scene_extent=scene_extent)
            scene_updated = True
        if check_step_condition(...):
            self.prune_gaussians_opacity()
            scene_updated = True
        ...
        return scene_updated
```

- `MCMCStrategy` adds Markov Chain Monte Carlo densification (see `threedgrut/strategy/mcmc.py`).

**Selective optimizers:** `SelectiveAdam` (in `threedgrut/optimizers`) accepts visibility masks from the renderer to accelerate convergence.

### 6.2 Logging, Timing, and GUI Hooks

- `threedgrut.utils.logger.RichLogger` wraps `rich` to provide progress bars, tables, and ensures CLI output stays readable. Use `logger.track(...)` to wrap your own loops.
- `threedgrut.utils.timer.ScopedTimer` and `CudaTimer` help profile CPU sections and CUDA kernels:

```29:167:threedgrut/utils/timer.py
@dataclass(slots=True, kw_only=True)
class TimingOptions:
    active: bool = True
    print_enabled: bool = False
    print_details: bool = False
    synchronize: bool = False
    all_results: dict[str, list[float]] = field(default_factory=dict)
    func_print_host: Callable = log.info

class ScopedTimer:
    def __init__(self, name: Optional[str] = None, opts: TimingOptions = timing_options, enabled: bool = True) -> None:
        ...
    def __enter__(self) -> Self:
        assert self.name is not None
        ...
    def __exit__(...):
        self.elapsed = (time.perf_counter_ns() - self.start) / 1000000.0
        self._print_local_summary()
```

- `threedgrut.utils.gui.GUI` (Polyscope) and `threedgrut.utils.viser_gui_util.ViserGUI` provide live visualization. Enable them via Hydra (`with_gui=True` or `with_viser_gui=True`). They pull batches through `model(...)` to render intermediate outputs.

### 6.3 Native Extension Loader

`threedgrut.utils.jit.load` centralizes CUDA extension build flags, ensuring OptiX/RT kernels compile consistently across platforms.

```24:119:threedgrut/utils/jit.py
def load(extra_cflags=None, extra_cuda_cflags=None, extra_ldflags=None, extra_include_paths=None,
         with_cuda=True, verbose=True, *args, **kwargs):
    cflags = ["-DNVDR_TORCH"]
    cuda_cflags = ["-DNVDR_TORCH", "-std=c++17", "--extended-lambda", "--expt-relaxed-constexpr", "-Xcompiler=-fno-strict-aliasing"]
    ...
    return torch.utils.cpp_extension.load(
        extra_cflags=cflags,
        extra_cuda_cflags=cuda_cflags,
        extra_ldflags=ldflags,
        extra_include_paths=include_paths,
        with_cuda=with_cuda,
        verbose=verbose,
        *args,
        **kwargs,
    )
```

---

## 7. Export & Interoperability

All exporters implement the `ModelExporter` interface and operate on `ExportableModel`s (already satisfied by `MixtureOfGaussians`).

```22:71:threedgrut/export/base.py
class ExportableModel(abc.ABC):
    @abc.abstractmethod
    def get_positions(self) -> torch.Tensor: ...
    @abc.abstractmethod
    def get_max_n_features(self) -> int: ...
    ...

class ModelExporter(abc.ABC):
    @abc.abstractmethod
    def export(self, model: ExportableModel, output_path: Path, dataset=None, conf=None, **kwargs) -> None:
        pass
```

- **INGPExporter** writes `*.ingp` files (msgpack + gzip) compatible with Instant-NGP tooling.

```27:87:threedgrut/export/ingp_exporter.py
class INGPExporter(ModelExporter):
    @torch.no_grad()
    def export(self, model: ExportableModel, output_path: Path, dataset=None, conf=None, force_half: bool = False, **kwargs):
        positions = model.get_positions()
        export_dtype = torch.float16 if force_half else positions.dtype
        mogt_config["mog_positions"] = positions.flatten().to(dtype=export_dtype, device="cpu").detach().numpy().tobytes()
        ...
        with gzip.open(output_path, "wb") as f:
            packed = msgpack.packb(mogt_config)
            f.write(packed)
```

- **PLYExporter** emits Gaussian splats in the format popularized by 3DGS (`f_dc_*`, `f_rest_*`, `scale_*`, etc.).

```33:84:threedgrut/export/ply_exporter.py
class PLYExporter(ModelExporter):
    @torch.no_grad()
    def export(self, model: ExportableModel, output_path: Path, dataset=None, conf=None, **kwargs) -> None:
        positions = model.get_positions().detach().cpu().numpy()
        mogt_albedo = model.get_features_albedo().detach().cpu().numpy()
        mogt_specular = model.get_features_specular().detach().cpu().numpy().reshape((num_gaussians, num_speculars, 3))
        ...
        elements = np.empty(num_gaussians, dtype=dtype_full)
        elements[:] = list(map(tuple, attributes))
        PlyData([el]).write(output_path)
```

- **USDZExporter** produces USDZ+NuRec bundles for Omniverse / Isaac Sim, including optional scene normalization.

```36:137:threedgrut/export/usdz_exporter.py
class USDZExporter(ModelExporter):
    @torch.no_grad()
    def export(self, model: ExportableModel, output_path: Path, dataset=None, conf: Dict[str, Any] = None, **kwargs):
        positions = model.get_positions().detach().cpu().numpy()
        rotations = model.get_rotation(preactivation=True).detach().cpu().numpy()
        ...
        template = fill_3dgut_template(**template_params)
        with gzip.GzipFile(fileobj=buffer, mode="wb", compresslevel=0) as f:
            packed = msgpack.packb(template)
            f.write(packed)
        write_to_usdz(output_path, model_file, gauss_usd, default_usd)
```

**Usage pattern:** exports are controlled via Hydra config (`export_ingp.enabled=true`, etc.) or by explicitly instantiating the exporters after training.

---

## 8. Interactive Playground (`threedgrut_playground`)

### 8.1 Engine Primitives

The playground exposes a hybrid renderer that traces Gaussian fields and mesh primitives through OptiX. Core data structures live in `engine.py`.

```56:95:threedgrut_playground/engine.py
@dataclass
class RayPack:
    rays_ori: torch.FloatTensor
    rays_dir: torch.FloatTensor
    pixel_x: Optional[torch.IntTensor] = None
    pixel_y: Optional[torch.IntTensor] = None
    mask: Optional[torch.BoolTensor] = None

@dataclass
class PBRMaterial:
    material_id: int
    diffuse_map: Optional[torch.Tensor] = None
    emissive_map: Optional[torch.Tensor] = None
    metallic_roughness_map: Optional[torch.Tensor] = None
    normal_map: Optional[torch.Tensor] = None
    diffuse_factor: torch.Tensor = None
    emissive_factor: torch.Tensor = None
    metallic_factor: float = 0.0
    roughness_factor: float = 0.0
    alpha_mode: int = 0
    transmission_factor: float = 0.0
    ior: float = 1.0
```

`Primitives`, `Environment`, `SPP`, and `DepthOfField` modules allow the UI (Polyscope, viser, or headless notebooks) to combine Gaussians with ray-traced PBR meshes, glass, mirrors, and HDR environment maps. The same engine powers the playground GUI (`ps_gui.py`) and the Kaolin tutorial headless notebook.

### 8.2 Hybrid Tracer

`threedgrut_playground.tracer.Tracer` bridges the playground engine with the OptiX kernels compiled under `threedgrt_tracer`.

```49:145:threedgrut_playground/tracer.py
class Tracer:
    def __init__(self, conf):
        self.device = "cuda"
        self.conf = conf
        load_playground_plugin(conf)
        self.tracer_wrapper = _playground_plugin.HybridOptixTracer(
            threedgrt_tracer_module_path,
            playground_module_path,
            torch.utils.cpp_extension.CUDA_HOME,
            self.conf.render.pipeline_type,
            ...
        )

    def render_playground(...):
        (pred_rgb, pred_opacity, pred_dist, pred_normals, hits_count) = self.tracer_wrapper.trace_hybrid(
            frame_id,
            poses,
            ray_o,
            ray_d,
            particle_density,
            features,
            sph_degree,
            ...
            envmap,
            envmap_offset,
            max_pbr_bounces,
        )
        return {"pred_rgb": pred_rgb, "pred_opacity": pred_opacity, ...}
```

### 8.3 GUIs & Headless Modes

- `threedgrut_playground/ps_gui.py` hosts the Polyscope application (`Playground` class). It forwards UI changes to `Engine3DGRUT`, pipes rendered frames back into Polyscope textures, and exposes widgets for mesh/material/envmap editing, MSAA/Denoiser combos, camera presets, and trajectory recording.
- `threedgrut_playground/viser_gui.py` exposes almost the same controls inside a viser server (great for remote sessions).
- `threedgrut_playground/headless.ipynb` demonstrates scripting the engine without any GUI.

**Programmatic example (headless render loop):**

```python
from threedgrut_playground.engine import Engine3DGRUT
engine = Engine3DGRUT(gs_object="runs/bonsai/ckpt_last.pt",
                      mesh_assets_folder="threedgrut_playground/assets",
                      default_config="apps/colmap_3dgrt.yaml",
                      envmap_assets_folder="threedgrut_playground/assets")
camera = engine.camera_from_dataset(view_idx=0)
outputs = engine.render_pass(camera, is_first_pass=True)
```

---

## 9. Workflow Playbooks

### 9.1 Train → Evaluate → Export

1. **Train:** `python train.py --config-name apps/colmap_3dgrt.yaml path=data/... out_dir=runs experiment_name=bonsai`
2. **Resume or switch inputs:** `Trainer3DGRUT.create_from_ply("export_last.ply", conf)` rehydrates Gaussians from a point cloud.
3. **Benchmark:** `python render.py --checkpoint runs/bonsai/ckpt_last.pt --out-dir outputs/bonsai_eval`
4. **Export for Omniverse:** set `export_usdz.enabled=true export_usdz.apply_normalizing_transform=true` in the config or call `USDZExporter().export(...)`.

### 9.2 Custom Dataset Loader

If you need a custom sensor, implement the `BoundedMultiViewDataset` protocol and register it inside `threedgrut/datasets/__init__.py`. Point `dataset.type` to your key and Hydra will feed it everywhere (trainer, renderer, exporter, GUI).

### 9.3 Headless Playground Rendering

```python
from viser import ViserClient
from threedgrut_playground.viser_gui import PlaygroundViserApp

app = PlaygroundViserApp(gs_object="runs/bonsai/ckpt_last.pt")
for client in app.server.get_clients().values():
    app.update_render_view(client, force=True)
```

### 9.4 Convert 3DGS PLY → USDZ

```bash
python -m threedgrut.export.scripts.ply_to_usd model.ply \
  --output_file model.usdz \
  --default_config apps/colmap_3dgut.yaml
```

### 9.5 Logging & Profiling

Wrap custom code in `ScopedTimer("my_section")` or set `timing_options.print_enabled=True` in your script to expose timings instantly on stdout. Use `logger.log_table(...)` for structured metrics.

---

## 10. Tips & References

- **Configs to inspect:** `configs/apps/*.yaml` (dataset presets), `configs/strategy/*.yaml` (densification), `configs/render/*.yaml`.
- **Native kernels:** `threedgrt_tracer` (OptiX ray tracing), `threedgut_tracer` (rasterizer), `threedgrut_playground/src` (hybrid path tracer). These are built on-demand via setup scripts.
- **Assets & scripts:** `threedgrut_playground/download_assets.sh` downloads demo meshes + envmaps; `benchmark/` scripts automate evaluation; `attributes`: see `ATTRIBUTIONS.md`.

With these building blocks you can compose training, evaluation, exporting, and interactive visualization pipelines programmatically or via Hydra CLI, and extend the project with new datasets, strategies, or render back-ends without diving blindly into the codebase.
