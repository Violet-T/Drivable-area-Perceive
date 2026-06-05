# Source Layout

Project-owned code is organized by responsibility:

```text
src/
├── YOLOP/        single-frame drivable/free-space inference
├── SEA_RAFT/     SEA-RAFT wrapper and optical-flow-based temporal baseline
├── STGRU/        future trainable temporal fusion module
└── utils/        shared warp, fusion, visualization, dataset, and legacy scripts
```

Third-party repositories are placed inside the module that owns them:

```text
src/YOLOP/external/YOLOP/
src/SEA_RAFT/external/SEA-RAFT/
src/utils/legacy/external/PIDNet/
```

`SEA_RAFT` uses an underscore because Python package names cannot contain `-`.
The module still refers to the SEA-RAFT model in documentation and command arguments.
