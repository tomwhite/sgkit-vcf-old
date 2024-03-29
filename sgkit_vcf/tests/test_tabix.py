from pathlib import Path

import pytest
from cyvcf2 import VCF

from sgkit_vcf.tabix import (
    get_csi_path,
    get_tabix_path,
    partition_into_regions,
    read_csi,
    read_tabix,
)
from sgkit_vcf.vcf_reader import count_variants


@pytest.mark.parametrize(
    "vcf_file", ["CEUTrio.20.21.gatk3.4.g.vcf.bgz",],
)
def test_record_counts_tbi(shared_datadir, vcf_file):
    # Check record counts in tabix with actual count of VCF
    vcf_path = shared_datadir / vcf_file
    tabix_path = get_tabix_path(vcf_path)
    tabix = read_tabix(tabix_path)

    for i, contig in enumerate(tabix.sequence_names):
        assert tabix.record_counts[i] == count_variants(vcf_path, contig)


@pytest.mark.parametrize(
    "vcf_file", ["CEUTrio.20.21.gatk3.4.csi.g.vcf.bgz",],
)
def test_record_counts_csi(shared_datadir, vcf_file):
    # Check record counts in csi with actual count of VCF
    vcf_path = shared_datadir / vcf_file
    csi_path = get_csi_path(vcf_path)
    csi = read_csi(csi_path)

    for i, contig in enumerate(VCF(vcf_path).seqnames):
        assert csi.record_counts[i] == count_variants(vcf_path, contig)


@pytest.mark.parametrize(
    "vcf_file",
    [
        "CEUTrio.20.21.gatk3.4.g.bcf",
        "CEUTrio.20.21.gatk3.4.g.vcf.bgz",
        "NA12878.prod.chr20snippet.g.vcf.gz",
    ],
)
def test_partition_into_regions__num_parts(shared_datadir, vcf_file):
    vcf_path = shared_datadir / vcf_file

    regions = partition_into_regions(vcf_path, num_parts=4)

    assert regions is not None
    part_variant_counts = [count_variants(vcf_path, region) for region in regions]
    total_variants = count_variants(vcf_path)

    assert sum(part_variant_counts) == total_variants


def test_partition_into_regions__num_parts_large(shared_datadir):
    vcf_path = shared_datadir / "CEUTrio.20.21.gatk3.4.g.vcf.bgz"

    regions = partition_into_regions(vcf_path, num_parts=100)
    assert regions is not None
    assert len(regions) == 18

    part_variant_counts = [count_variants(vcf_path, region) for region in regions]
    total_variants = count_variants(vcf_path)

    assert sum(part_variant_counts) == total_variants


def test_partition_into_regions__target_part_size(shared_datadir):
    vcf_path = shared_datadir / "CEUTrio.20.21.gatk3.4.g.vcf.bgz"

    regions = partition_into_regions(vcf_path, target_part_size=100_000)
    assert regions is not None
    assert len(regions) == 5

    part_variant_counts = [count_variants(vcf_path, region) for region in regions]
    total_variants = count_variants(vcf_path)

    assert sum(part_variant_counts) == total_variants


def test_partition_into_regions__invalid_arguments(shared_datadir):
    vcf_path = shared_datadir / "CEUTrio.20.21.gatk3.4.g.vcf.bgz"

    with pytest.raises(
        ValueError, match=r"One of num_parts or target_part_size must be specified"
    ):
        partition_into_regions(vcf_path)

    with pytest.raises(
        ValueError, match=r"Only one of num_parts or target_part_size may be specified"
    ):
        partition_into_regions(vcf_path, num_parts=4, target_part_size=100_000)

    with pytest.raises(ValueError, match=r"num_parts must be positive"):
        partition_into_regions(vcf_path, num_parts=0)

    with pytest.raises(ValueError, match=r"target_part_size must be positive"):
        partition_into_regions(vcf_path, target_part_size=0)


def test_partition_into_regions__one_part(shared_datadir):
    vcf_path = shared_datadir / "CEUTrio.20.21.gatk3.4.g.vcf.bgz"
    assert partition_into_regions(vcf_path, num_parts=1) is None


def test_partition_into_regions__gnomad(shared_datadir):
    vcf_path = Path("/Users/tom/Downloads/gnomad.exomes.r2.1.1.sites.22.vcf.bgz")

    regions = partition_into_regions(vcf_path, num_parts=10)
    assert regions == [
        "22:1-20103168",
        "22:20103169-22740992",
        "22:22740993-25198592",
        "22:25198593-30703616",
        "22:30703617-34045952",
        "22:34045953-38354944",
        "22:38354945-41533440",
        "22:41533441-44335104",
        "22:44335105-50528256",
        "22:50528257-",
    ]
