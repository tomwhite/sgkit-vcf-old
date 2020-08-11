import numpy as np
import xarray as xr
from cyvcf2 import VCF

from sgkit.api import DIM_VARIANT, create_genotype_call_dataset
from sgkit.typing import PathType
from sgkit.utils import encode_array

DEFAULT_ALT_NUMBER = 3  # see vcf_read.py in scikit_allel


def read_vcf(path: PathType) -> xr.Dataset:

    chunk_size = 9
    alt_number = DEFAULT_ALT_NUMBER

    sample_id = np.array(VCF(path).samples, dtype=str)
    n_sample = len(sample_id)
    n_ploidy = 2

    variant_ids = []
    variant_contigs = []
    variant_position = np.zeros(chunk_size, dtype="i4")
    variant_alleles = []

    call_genotype = np.zeros((chunk_size, n_sample, n_ploidy), dtype="i1")
    call_genotype_phased = np.zeros((chunk_size, n_sample), dtype=bool)

    i = 0
    for variant in VCF(path):
        variant_ids.append(variant.ID if variant.ID is not None else ".")
        variant_contigs.append(variant.CHROM)
        variant_position[i] = variant.POS

        alleles = [variant.REF] + variant.ALT
        if len(alleles) > alt_number + 1:
            alleles = alleles[: alt_number + 1]
        elif len(alleles) < alt_number + 1:
            alleles = alleles + ([""] * (alt_number + 1 - len(alleles)))
        variant_alleles.append(alleles)

        gt = variant.genotype.array()
        call_genotype[i] = gt[..., 0:-1]
        call_genotype_phased[i] = gt[..., -1]

        i = i + 1

    variant_id = np.array(variant_ids, dtype=str)
    variant_id_mask = variant_id == "."
    variant_contig, variant_contig_names = encode_array(variant_contigs)
    variant_contig = variant_contig.astype("int16")
    variant_contig_names = list(variant_contig_names)
    variant_alleles = np.array(variant_alleles, dtype="O")  # TODO: or "S"

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

    return ds
