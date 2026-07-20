---
name: npy-metadata-extractor
description: Inspect NumPy .npy arrays used for CT volumes, segmentation masks, and skeletons. Use when asked for an array's shape, dimensions, data type, voxel count, byte size, minimum, maximum, or mean intensity, or when validating an array before another NDE operation.
---

# NumPy Metadata Extractor

Inspect one or more `.npy` arrays without modifying them. Use the bundled script so large arrays are memory-mapped instead of loaded fully into memory.

## Workflow

1. Resolve every requested input relative to the repository root. Do not assume the current working directory is the script directory.
2. Reject missing paths and non-`.npy` files. Never enable pickled-object loading.
3. Run:

   ```bash
   python .agents/skills/npy-metadata-extractor/scripts/extract_metadata.py <file.npy> [<another.npy> ...]
   ```

4. Report the resolved path, shape, number of dimensions, dtype, voxel count, memory size, minimum, maximum, and mean. Preserve `null` values for statistics that cannot be computed, such as an empty array.
5. Call out unexpected conditions, including a non-3D CT volume, incompatible shapes, or a mask whose dtype is not boolean or integer.

## Constraints

- Treat input arrays as read-only.
- Use absolute paths when the working directory is uncertain.
- Do not copy a multi-gigabyte array merely to compute metadata.
- Do not infer physical dimensions or voxel spacing from array shape alone.

## Bundled Script

`scripts/extract_metadata.py` emits machine-readable JSON and exits nonzero if any input is invalid. Pass multiple files in one invocation when comparing arrays.
