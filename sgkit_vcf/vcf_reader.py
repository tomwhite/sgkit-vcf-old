import itertools
from contextlib import contextmanager
from typing import Iterator, Optional, Sequence, TypeVar, Union

import dask
import numpy as np
import xarray as xr
from cyvcf2 import VCF, Variant

from sgkit.api import DIM_VARIANT, create_genotype_call_dataset
from sgkit.typing import PathType

DEFAULT_ALT_NUMBER = 3  # see vcf_read.py in scikit_allel

T = TypeVar("T")


@contextmanager
def open_vcf(path: PathType) -> VCF:
    """A context manager for opening a VCF file."""
    vcf = VCF(path)
    try:
        yield vcf
    finally:
        vcf.close()


# https://dev.to/orenovadia/solution-chunked-iterator-python-riddle-3ple
def chunks(iterator: Iterator[T], n: int) -> Iterator[Iterator[T]]:
    """
    Convert an iterator into an iterator of iterators, where the inner iterators
    each return `n` items, except the last, which may return fewer.
    """

    for first in iterator:  # take one item out (exits loop if `iterator` is empty)
        rest_of_chunk = itertools.islice(iterator, 0, n - 1)
        yield itertools.chain([first], rest_of_chunk)  # concatenate the first item back


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
) -> None:

    with open_vcf(input) as vcf:

        alt_number = DEFAULT_ALT_NUMBER

        sample_id = np.array(vcf.samples, dtype=str)
        n_sample = len(sample_id)
        n_allele = alt_number + 1
        n_ploidy = 2

        variant_contig_names = vcf.seqnames

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
                variant_ids.append(variant.ID if variant.ID is not None else ".")
                variant_contig[i] = variant_contig_names.index(variant.CHROM)
                variant_position[i] = variant.POS

                alleles = [variant.REF] + variant.ALT
                if len(alleles) > n_allele:
                    alleles = alleles[:n_allele]
                elif len(alleles) < n_allele:
                    alleles = alleles + ([""] * (n_allele - len(alleles)))
                variant_alleles.append(alleles)

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

            if first_variants_chunk:
                # Enforce uniform chunks in the variants dimension
                encoding = dict(
                    call_genotype=dict(chunks=(chunk_length, n_sample, n_ploidy)),
                    call_genotype_mask=dict(chunks=(chunk_length, n_sample, n_ploidy)),
                    call_genotype_phased=dict(chunks=(chunk_length, n_sample)),
                    variant_allele=dict(chunks=(chunk_length, n_allele)),
                    variant_contig=dict(chunks=(chunk_length,)),
                    variant_id=dict(chunks=(chunk_length,)),
                    variant_id_mask=dict(chunks=(chunk_length,)),
                    variant_position=dict(chunks=(chunk_length,)),
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
) -> xr.Dataset:
    vcf_to_zarr_sequential(input, output, region, chunk_length)
    ds: xr.Dataset = xr.open_zarr(output)  # type: ignore[no-untyped-call]
    return ds


def vcf_to_zarr_parallel(
    input: PathType,
    output: PathType,
    regions: Sequence[str],
    chunk_length: int = 10_000,
) -> None:
    """Convert specified regions of a VCF to zarr files, then concat, rechunk, write to zarr"""

    datasets = []
    for i, region in enumerate(regions):
        part = f"part-{i}.zarr"
        ds = dask.delayed(vcf_to_dataset)(
            input, output=part, region=region, chunk_length=chunk_length
        )
        datasets.append(ds)
    datasets = dask.compute(*datasets)

    # https://github.com/dask/dask/issues/5105
    dask.config.set({"optimization.fuse.ave-width": 50})

    # TODO: can we extend dask computation graph to cover these two operations too?
    ds = xr.concat(datasets, dim="variants", data_vars="minimal")  # type: ignore[no-untyped-call, no-redef]
    ds: xr.Dataset = ds.chunk({"variants": chunk_length})  # type: ignore

    # https://github.com/pydata/xarray/issues/3476
    # TODO: still need to fix xarray/conventions.py:188 warning
    del ds["variant_allele"].encoding["filters"]
    del ds["variant_id"].encoding["filters"]

    delayed = ds.to_zarr(output, mode="w", compute=False)
    # delayed.visualize()
    delayed.compute()


def vcf_to_zarr(
    input: PathType,
    output: PathType,
    regions: Union[None, str, Sequence[str]] = None,
    chunk_length: int = 10_000,
) -> None:
    if regions is None or isinstance(regions, str):
        vcf_to_zarr_sequential(input, output, regions, chunk_length)

    else:
        vcf_to_zarr_parallel(input, output, regions, chunk_length)


def count_variants(path: PathType, region: Optional[str] = None) -> int:
    """Count the number of variants in a VCF file."""
    with open_vcf(path) as vcf:
        if region is not None:
            vcf = vcf(region)
        count = 0
        for variant in region_filter(vcf, region):
            count = count + 1
        return count
