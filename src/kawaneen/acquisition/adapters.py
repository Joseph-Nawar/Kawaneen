"""Source adapters with provider-specific transport and shared safe storage."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from huggingface_hub import hf_hub_download  # pyright: ignore[reportUnknownVariableType]

from kawaneen.acquisition.models import FileDigest, SourceSpecification
from kawaneen.acquisition.storage import StorageError, copy_immutable, source_root


class AdapterError(RuntimeError):
    """Raised when a source adapter cannot complete a permitted operation."""


class HuggingFaceAdapter:
    """Acquire selected files from one pinned Hugging Face dataset revision."""

    def acquire(self, specification: SourceSpecification, raw_root: Path) -> tuple[FileDigest, ...]:
        if specification.provider != "huggingface":
            raise AdapterError("Hugging Face adapter received a non-Hugging Face specification")
        destination_root = source_root(raw_root, specification.source_id, specification.version)
        digests: list[FileDigest] = []
        for expected in specification.files:
            try:
                cached_link = Path(
                    cast(Any, hf_hub_download)(
                        repo_id=specification.identifier,
                        filename=expected.path,
                        revision=specification.revision,
                        repo_type="dataset",
                    )
                )
                cached = cached_link.resolve(strict=True)
            except Exception as exc:
                raise AdapterError(
                    f"Hugging Face download failed for {expected.path}: {exc}"
                ) from exc
            try:
                digests.append(copy_immutable(cached, destination_root, expected.path))
            except StorageError as exc:
                raise AdapterError(
                    f"safe raw installation failed for {expected.path}: {exc}"
                ) from exc
        return tuple(digests)


class LocalFileAdapter:
    """Import one explicitly selected local file without network access."""

    def import_file(
        self,
        specification: SourceSpecification,
        source_file: Path,
        raw_root: Path,
    ) -> tuple[FileDigest, ...]:
        if len(specification.files) != 1:
            raise AdapterError("local import requires a single-file specification")
        expected = specification.files[0]
        if source_file.name != Path(expected.path).name:
            raise AdapterError(f"local filename must be exactly {Path(expected.path).name}")
        destination_root = source_root(raw_root, specification.source_id, specification.version)
        try:
            digest = copy_immutable(source_file, destination_root, expected.path)
        except StorageError as exc:
            raise AdapterError(f"safe local installation failed: {exc}") from exc
        return (digest,)


class MendeleyAdapterUnavailable(AdapterError):
    """The documented public Mendeley route is not stable for unauthenticated download."""


class MendeleyAdapter:
    """Explicitly refuse undocumented browser or expiring Mendeley download URLs."""

    def acquire(self, specification: SourceSpecification, raw_root: Path) -> tuple[FileDigest, ...]:
        raise MendeleyAdapterUnavailable(
            "No stable unauthenticated public Mendeley download route was verified; "
            "use kawaneen data import-local arabiccr --file <ArabiCCR-dataset.csv> "
            "--purpose local_research"
        )
