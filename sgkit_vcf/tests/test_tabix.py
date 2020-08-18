import pytest

from sgkit_vcf.tabix import get_tabix_path, partition_into_regions, read_tabix
from sgkit_vcf.vcf_reader import count_variants


@pytest.mark.parametrize(
    "vcf_file",
    [
        "sample.vcf.gz",
        "CEUTrio.20.21.gatk3.4.g.vcf.bgz",
        "NA12878.prod.chr20snippet.g.vcf.gz",
    ],
)
def test_record_counts(shared_datadir, vcf_file):
    # Check record counts in tabix with actual count of VCF
    vcf_path = shared_datadir / vcf_file
    tabix_path = get_tabix_path(vcf_path)
    tabix = read_tabix(tabix_path)

    for i, contig in enumerate(tabix.sequence_names):
        assert tabix.record_counts[i] == count_variants(vcf_path, contig)


@pytest.mark.parametrize(
    "vcf_file",
    ["CEUTrio.20.21.gatk3.4.g.vcf.bgz", "NA12878.prod.chr20snippet.g.vcf.gz"],
)
def test_partition_into_regions(shared_datadir, vcf_file):
    vcf_path = shared_datadir / vcf_file

    regions = partition_into_regions(vcf_path, num_parts=4)

    part_variant_counts = [count_variants(vcf_path, region) for region in regions]
    total_variants = count_variants(vcf_path)

    assert sum(part_variant_counts) == total_variants


def test_partition_into_regions__target_part_size(shared_datadir):
    vcf_path = shared_datadir / "CEUTrio.20.21.gatk3.4.g.vcf.bgz"

    regions = partition_into_regions(vcf_path, target_part_size=100_000)
    assert len(regions) == 4

    part_variant_counts = [count_variants(vcf_path, region) for region in regions]
    total_variants = count_variants(vcf_path)

    assert sum(part_variant_counts) == total_variants


def test_partition_into_regions__missing_argument(shared_datadir):
    vcf_path = shared_datadir / "CEUTrio.20.21.gatk3.4.g.vcf.bgz"
    with pytest.raises(
        ValueError, match=r"One of num_parts or target_part_size must be specified"
    ):
        partition_into_regions(vcf_path)


def test_partition_into_regions__incompatible_arguments(shared_datadir):
    vcf_path = shared_datadir / "CEUTrio.20.21.gatk3.4.g.vcf.bgz"
    with pytest.raises(
        ValueError, match=r"Only one of num_parts or target_part_size may be specified"
    ):
        partition_into_regions(vcf_path, num_parts=4, target_part_size=100_000)
