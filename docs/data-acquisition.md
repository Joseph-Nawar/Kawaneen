# Phase 2 data acquisition

Kawaneen acquisition is a gated, offline-first workflow. The Phase 1 source registry is the legal policy input; a version-controlled TOML specification is the technical input. Both must authorize an operation before any bytes are copied.

ALARB, ArabiCCR, and the named Saudi MOJ-derived seed are currently permitted only for their explicit local purposes. ALARB is limited to controlled evaluation, parsing, integrity, duplicate, and privacy inspection. ArabiCCR is limited to local research, parsing, evaluation, and inspection. The MOJ-derived seed is limited to local research, parsing, integrity, duplicate, and privacy inspection; it is not authoritative or evaluation-approved. Training, publishing, public display, and public-demo operations are always denied. There is no force or bypass flag.

## Storage and integrity

Raw files are stored under `data/raw/<source>/<version>/`, which is ignored by Git. Files are copied byte-for-byte through a `.partial` file and installed with an atomic replacement. Existing differing files are rejected. Source names, versions, and relative filenames are path-validated; symlink escapes and traversal are rejected.

The ALARB specification pins the Hugging Face dataset repository to revision `e64bfdc867146294a65434c5ca16c2c4c5288ca2` and names the README and official Parquet train/test files. The adapter uses the official Hub client and copies from its cache. ArabiCCR’s canonical acquisition source is Mendeley Data DOI `10.17632/np538c95yy.3`, Version 3. Its recorded method is an authorized manual official download followed by `make data-import-arabiccr FILE=/path/to/ArabiCCR-dataset.csv`; the raw file is an imported local artifact, not a separate source.

Verification checks non-empty bytes, SHA-256, filenames and formats, UTF-8/BOM state, strict CSV headers and row counts, Parquet schema and row counts, schema fingerprints, physical duplicates, exact duplicate rows, and official ALARB train/test overlap. It never changes raw files and does not perform fuzzy or semantic deduplication.

`kawaneen data manifest build <source>` writes deterministic, repository-relative metadata to `data/manifests/`. The lockfile is changed only by this CLI operation. No raw legal text is committed.

Snapshot eligibility is separated into `legal_clearance`, `authorized_for_local_parsing`, `authorized_for_evaluation`, `authorized_for_training`, and `authorized_for_public_display`. Local parsing does not require public-display clearance, but it remains private, source-gated, and subject to the recorded risk decision. Training and public use remain denied unless separately authorized.
