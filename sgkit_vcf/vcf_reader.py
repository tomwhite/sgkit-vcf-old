import itertools
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional, Sequence, Union

import dask
import numpy as np
import xarray as xr
from cyvcf2 import VCF, Variant

from sgkit.api import DIM_VARIANT, create_genotype_call_dataset
from sgkit.typing import PathType
from sgkit_vcf.utils import chunks

DEFAULT_ALT_NUMBER = 3  # see vcf_read.py in scikit_allel


@contextmanager
def open_vcf(path: PathType) -> VCF:
    """A context manager for opening a VCF file."""
    vcf = VCF(path)
    try:
        yield vcf
    finally:
        vcf.close()


def region_filter(
    variants: Iterator[Variant], region: Optional[str] = None
) -> Iterator[Variant]:
    """Filter out variants that don't start in the given region."""
    if region is None:
        return variants
    else:
        start = get_region_start(region)
        return itertools.filterfalse(lambda v: v.POS < start, variants)


def get_region_start(region: str) -> int:
    """Return the start position of the region string."""
    if ":" not in region:
        return 1
    contig, start_end = region.split(":")
    start, end = start_end.split("-")
    return int(start)


def vcf_to_zarr_sequential(
    input: PathType,
    output: PathType,
    region: Optional[str] = None,
    chunk_length: int = 10_000,
    chunk_width: int = 1_000,
) -> None:

    with open_vcf(input) as vcf:

        alt_number = DEFAULT_ALT_NUMBER

        sample_id = np.array(vcf.samples, dtype=str)
        n_sample = len(sample_id)
        n_allele = alt_number + 1
        n_ploidy = 2

        variant_contig_names = vcf.seqnames

        # Remember max lengths of variable-length strings
        max_variant_id_length = 0
        max_variant_allele_length = 0

        # Iterate through variants in batches of chunk_length

        if region is None:
            variants = vcf
        else:
            variants = vcf(region)

        variant_contig = np.empty(chunk_length, dtype="i1")
        variant_position = np.empty(chunk_length, dtype="i4")
        call_genotype = np.empty((chunk_length, n_sample, n_ploidy), dtype="i1")
        call_genotype_phased = np.empty((chunk_length, n_sample), dtype=bool)

        first_variants_chunk = True
        for variants_chunk in chunks(region_filter(variants, region), chunk_length):

            variant_ids = []
            variant_alleles = []

            for i, variant in enumerate(variants_chunk):
                variant_id = variant.ID if variant.ID is not None else "."
                variant_ids.append(variant_id)
                max_variant_id_length = max(max_variant_id_length, len(variant_id))
                variant_contig[i] = variant_contig_names.index(variant.CHROM)
                variant_position[i] = variant.POS

                alleles = [variant.REF] + variant.ALT
                if len(alleles) > n_allele:
                    alleles = alleles[:n_allele]
                elif len(alleles) < n_allele:
                    alleles = alleles + ([""] * (n_allele - len(alleles)))
                variant_alleles.append(alleles)
                max_variant_allele_length = max(
                    max_variant_allele_length, max(len(x) for x in alleles)
                )

                gt = variant.genotype.array()
                call_genotype[i] = gt[..., 0:-1]
                call_genotype_phased[i] = gt[..., -1]

            # Truncate np arrays (if last chunk is smaller than chunk_length)
            if i + 1 < chunk_length:
                variant_contig = variant_contig[: i + 1]
                variant_position = variant_position[: i + 1]
                call_genotype = call_genotype[: i + 1]
                call_genotype_phased = call_genotype_phased[: i + 1]

            variant_id = np.array(variant_ids, dtype="O")
            variant_id_mask = variant_id == "."
            variant_alleles = np.array(variant_alleles, dtype="O")

            ds: xr.Dataset = create_genotype_call_dataset(
                variant_contig_names=variant_contig_names,
                variant_contig=variant_contig,
                variant_position=variant_position,
                variant_alleles=variant_alleles,
                sample_id=sample_id,
                call_genotype=call_genotype,
                call_genotype_phased=call_genotype_phased,
                variant_id=variant_id,
            )
            ds["variant_id_mask"] = (
                [DIM_VARIANT],
                variant_id_mask,
            )
            ds.attrs["max_variant_id_length"] = max_variant_id_length
            ds.attrs["max_variant_allele_length"] = max_variant_allele_length

            if first_variants_chunk:
                # Enforce uniform chunks in the variants dimension
                # Also chunk in the samples direction
                encoding = dict(
                    call_genotype=dict(chunks=(chunk_length, chunk_width, n_ploidy)),
                    call_genotype_mask=dict(
                        chunks=(chunk_length, chunk_width, n_ploidy)
                    ),
                    call_genotype_phased=dict(chunks=(chunk_length, chunk_width)),
                    variant_allele=dict(chunks=(chunk_length, n_allele)),
                    variant_contig=dict(chunks=(chunk_length,)),
                    variant_id=dict(chunks=(chunk_length,)),
                    variant_id_mask=dict(chunks=(chunk_length,)),
                    variant_position=dict(chunks=(chunk_length,)),
                    sample_id=dict(chunks=(chunk_width,)),
                )

                ds.to_zarr(output, mode="w", encoding=encoding)
                first_variants_chunk = False
            else:
                # Append along the variants dimension
                ds.to_zarr(output, append_dim=DIM_VARIANT)


def vcf_to_dataset(
    input: PathType,
    output: PathType,
    region: Optional[str] = None,
    chunk_length: int = 10_000,
    chunk_width: int = 1_000,
) -> xr.Dataset:
    vcf_to_zarr_sequential(input, output, region, chunk_length, chunk_width)
    ds: xr.Dataset = xr.open_zarr(str(output))  # type: ignore[no-untyped-call]
    return ds


def vcf_to_zarr_parallel(
    input: Union[PathType, Sequence[PathType]],
    output: PathType,
    regions: Union[None, Sequence[str], Sequence[Sequence[str]]],
    chunk_length: int = 10_000,
    chunk_width: int = 1_000,
) -> None:
    """Convert specified regions of a VCF to zarr files, then concat, rechunk, write to zarr"""

    if isinstance(input, str) or isinstance(input, Path):
        # Single input
        inputs: Sequence[PathType] = [input]
        assert regions is not None  # this would just be sequential case
        input_regions: Sequence[Sequence[Optional[str]]] = [regions]  # type: ignore
    else:
        # Multiple inputs
        inputs = input
        if regions is None:
            input_regions = [[None]] * len(inputs)
        else:
            if len(regions) == 0 or isinstance(regions[0], str):
                raise ValueError(
                    f"Multiple input regions must be a sequence of sequence of strings: {regions}"
                )
            input_regions = regions

    assert len(inputs) == len(input_regions)

    tempdir = Path(tempfile.mkdtemp(prefix="vcf_to_zarr_"))

    datasets = []
    parts = []
    for i, input in enumerate(inputs):
        filename = Path(input).name
        for r, region in enumerate(input_regions[i]):
            part = tempdir / filename / f"part-{r}.zarr"
            parts.append(part)
            ds = dask.delayed(vcf_to_dataset)(
                input,
                output=part,
                region=region,
                chunk_length=chunk_length,
                chunk_width=chunk_width,
            )
            datasets.append(ds)
    datasets = dask.compute(*datasets)

    # Ensure Dask task graph is efficient, see https://github.com/dask/dask/issues/5105
    dask.config.set({"optimization.fuse.ave-width": 50})

    # Combine the datasets into one
    ds = xr.concat(datasets, dim="variants", data_vars="minimal")  # type: ignore[no-untyped-call, no-redef]
    ds: xr.Dataset = ds.chunk({"variants": chunk_length, "samples": chunk_width})  # type: ignore

    # Set variable length strings to fixed length ones to avoid xarray/conventions.py:188 warning
    # (Also avoids this issue: https://github.com/pydata/xarray/issues/3476)
    max_variant_id_length = max(ds.attrs["max_variant_id_length"] for ds in datasets)
    max_variant_allele_length = max(
        ds.attrs["max_variant_allele_length"] for ds in datasets
    )
    ds["variant_id"] = ds["variant_id"].astype(f"S{max_variant_id_length}")
    ds["variant_allele"] = ds["variant_allele"].astype(f"S{max_variant_allele_length}")
    del ds.attrs["max_variant_id_length"]
    del ds.attrs["max_variant_allele_length"]

    delayed = ds.to_zarr(output, mode="w", compute=False)
    # delayed.visualize()
    delayed.compute()

    # Delete intermediate files from temporary directory
    shutil.rmtree(tempdir)


def vcf_to_zarr(
    input: Union[PathType, Sequence[PathType]],
    output: PathType,
    *,
    regions: Union[None, Sequence[str], Sequence[Sequence[str]]] = None,
    chunk_length: int = 10_000,
    chunk_width: int = 1_000,
) -> None:
    if (isinstance(input, str) or isinstance(input, Path)) and (
        regions is None or isinstance(regions, str)
    ):
        vcf_to_zarr_sequential(
            input,
            output,
            region=regions,
            chunk_length=chunk_length,
            chunk_width=chunk_width,
        )
    else:
        vcf_to_zarr_parallel(
            input,
            output,
            regions=regions,
            chunk_length=chunk_length,
            chunk_width=chunk_width,
        )


def count_variants(path: PathType, region: Optional[str] = None) -> int:
    """Count the number of variants in a VCF file."""
    with open_vcf(path) as vcf:
        if region is not None:
            vcf = vcf(region)
        count = 0
        for variant in region_filter(vcf, region):
            count = count + 1
        return count
