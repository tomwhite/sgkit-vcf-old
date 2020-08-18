import numpy as np
import xarray as xr
from numpy.testing import assert_array_equal

from sgkit_vcf import partition_into_regions, vcf_to_zarr


def test_vcf_to_zarr__small_vcf(shared_datadir):
    path = shared_datadir / "sample.vcf.gz"
    output = "vcf.zarr"

    vcf_to_zarr(path, output, chunk_length=5, chunk_width=2)
    ds = xr.open_zarr(output)  # type: ignore[no-untyped-call]

    assert ds.attrs["contigs"] == ["19", "20", "X"]
    assert_array_equal(ds["variant_contig"], [0, 0, 1, 1, 1, 1, 1, 1, 2])
    assert_array_equal(
        ds["variant_position"],
        [111, 112, 14370, 17330, 1110696, 1230237, 1234567, 1235237, 10],
    )
    assert_array_equal(
        ds["variant_allele"],
        [
            ["A", "C", "", ""],
            ["A", "G", "", ""],
            ["G", "A", "", ""],
            ["T", "A", "", ""],
            ["A", "G", "T", ""],
            ["T", "", "", ""],
            ["G", "GA", "GAC", ""],
            ["T", "", "", ""],
            ["AC", "A", "ATG", "C"],
        ],
    )
    assert_array_equal(
        ds["variant_id"],
        [".", ".", "rs6054257", ".", "rs6040355", ".", "microsat1", ".", "rsTest"],
    )
    assert_array_equal(
        ds["variant_id_mask"],
        [True, True, False, True, False, True, False, True, False],
    )

    assert_array_equal(ds["sample_id"], ["NA00001", "NA00002", "NA00003"])

    call_genotype = np.array(
        [
            [[0, 0], [0, 0], [0, 1]],
            [[0, 0], [0, 0], [0, 1]],
            [[0, 0], [1, 0], [1, 1]],
            [[0, 0], [0, 1], [0, 0]],
            [[1, 2], [2, 1], [2, 2]],
            [[0, 0], [0, 0], [0, 0]],
            [[0, 1], [0, 2], [-1, -1]],
            [[0, 0], [0, 0], [-1, -1]],
            [[0, -1], [0, 1], [0, 2]],
        ],
        dtype="i1",
    )
    call_genotype_phased = np.array(
        [
            [True, True, False],
            [True, True, False],
            [True, True, False],
            [True, True, False],
            [True, True, False],
            [True, True, False],
            [False, False, False],
            [False, True, False],
            [True, False, True],
        ],
        dtype=bool,
    )
    assert_array_equal(ds["call_genotype"], call_genotype)
    assert_array_equal(ds["call_genotype_mask"], call_genotype < 0)
    assert_array_equal(ds["call_genotype_phased"], call_genotype_phased)


def test_vcf_to_zarr__large_vcf(shared_datadir):
    path = path = shared_datadir / "CEUTrio.20.21.gatk3.4.g.vcf.bgz"
    output = "vcf.zarr"

    vcf_to_zarr(path, output, chunk_length=5_000)
    ds = xr.open_zarr(output)  # type: ignore[no-untyped-call]

    assert ds["sample_id"].shape == (1,)
    assert ds["call_genotype"].shape == (19910, 1, 2)
    assert ds["call_genotype_mask"].shape == (19910, 1, 2)
    assert ds["call_genotype_phased"].shape == (19910, 1)
    assert ds["variant_allele"].shape == (19910, 4)
    assert ds["variant_contig"].shape == (19910,)
    assert ds["variant_id"].shape == (19910,)
    assert ds["variant_id_mask"].shape == (19910,)
    assert ds["variant_position"].shape == (19910,)


def test_vcf_to_zarr_parallel(shared_datadir):
    path = path = shared_datadir / "CEUTrio.20.21.gatk3.4.g.vcf.bgz"
    output = "vcf_concat.zarr"
    regions = ["20", "21"]

    vcf_to_zarr(path, output, regions=regions, chunk_length=5_000)
    ds = xr.open_zarr(output)  # type: ignore[no-untyped-call]

    assert ds["sample_id"].shape == (1,)
    assert ds["call_genotype"].shape == (19910, 1, 2)
    assert ds["call_genotype_mask"].shape == (19910, 1, 2)
    assert ds["call_genotype_phased"].shape == (19910, 1)
    assert ds["variant_allele"].shape == (19910, 4)
    assert ds["variant_contig"].shape == (19910,)
    assert ds["variant_id"].shape == (19910,)
    assert ds["variant_id_mask"].shape == (19910,)
    assert ds["variant_position"].shape == (19910,)


def test_vcf_to_zarr_parallel_partitioned(shared_datadir):
    path = path = (
        shared_datadir / "1000G.phase3.broad.withGenotypes.chr20.10100000.vcf.gz"
    )
    output = "vcf_concat.zarr"

    regions = partition_into_regions(path, num_parts=4)

    vcf_to_zarr(path, output, regions=regions, chunk_length=1_000, chunk_width=1_000)
    ds = xr.open_zarr(output)  # type: ignore[no-untyped-call]

    assert ds["sample_id"].shape == (2535,)
    assert ds["variant_id"].shape == (1406,)
