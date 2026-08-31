# RAICOM2026 Webots Packages

This branch keeps the previous uploaded package and adds a verified solid-placement-tray package without overwriting it.

## Files

- `raicom2026_webots_package.zip`: previous uploaded package, kept unchanged.
- `raicom2026_webots_package_solid_verified_20260505.zip`: verified package containing the normal arena world and its dependent PROTO files.

## Difference

The new `solid_verified_20260505` package uses `RaicomConvertedPlacementTraySolid.proto` and `meshes/converted/raicom_placement_tray_solid.obj` for the placement trays. The tray visual mesh and collision mesh are the same closed solid holed mesh, so non-slot areas are solid and slots remain through holes. It also includes matching Webots preview icons.

## Local verification

- `worlds/raicom2026_arena.wbt` opens through the normal Webots GUI path.
- Full drop check result: `RAICOM_FULL_DROP_CHECK_OK`.
- Solid tray mesh check: 348 vertices, 554 faces, 0 boundary edges, 0 nonmanifold edges.
