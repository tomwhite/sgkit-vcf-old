import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from cyvcf2 import VCF

from sgkit.typing import PathType
from sgkit_vcf.utils import at_eof, ceildiv, get_file_length, read_bytes

TABIX_LINEAR_INDEX_INTERVAL_SIZE = 1 << 14  # 16kb interval size


@dataclass
class Header:
    n_ref: int
    format: int
    col_seq: int
    col_beg: int
    col_end: int
    meta: int
    skip: int
    l_nm: int


@dataclass
class Chunk:
    cnk_beg: int
    cnk_end: int


@dataclass
class Bin:
    bin: int
    chunks: Sequence[Chunk]


@dataclass
class TabixIndex:
    header: Header
    sequence_names: Sequence[str]
    bins: Sequence[Sequence[Bin]]
    linear_indexes: Sequence[Sequence[int]]
    record_counts: Sequence[int]

    def offsets(self) -> Any:
        # Combine the linear indexes into one stacked array
        linear_indexes = self.linear_indexes
        linear_index = np.hstack([np.array(li) for li in linear_indexes])

        # Create file offsets for each element in the linear index
        file_offsets = np.array([get_file_offset(vfp) for vfp in linear_index])

        # Calculate corresponding contigs and positions or each element in the linear index
        contig_indexes = np.hstack(
            [np.full(len(li), i) for (i, li) in enumerate(linear_indexes)]
        )
        # positions are 1-based and inclusive
        positions = np.hstack(
            [
                np.arange(len(li)) * TABIX_LINEAR_INDEX_INTERVAL_SIZE + 1
                for li in linear_indexes
            ]
        )
        assert len(file_offsets) == len(contig_indexes)
        assert len(file_offsets) == len(positions)

        return file_offsets, contig_indexes, positions


def read_tabix(file: PathType) -> TabixIndex:
    """Parse a tabix file into a queryable datastructure"""
    with gzip.open(file) as f:
        (magic,) = read_bytes(f, "4s")
        assert magic == b"TBI\x01"

        header = Header(*read_bytes(f, "<8i"))

        sequence_names = []
        bins = []
        linear_indexes = []
        record_counts = []

        if header.l_nm > 0:
            (names,) = read_bytes(f, f"<{header.l_nm}s")
            # Convert \0-terminated names to strings
            sequence_names = [str(name, "utf-8") for name in names.split(b"\x00")[:-1]]

            for _ in range(header.n_ref):
                (n_bin,) = read_bytes(f, "<i")
                seq_bins = []
                record_count = -1
                for _ in range(n_bin):
                    bin, n_chunk = read_bytes(f, "<Ii")
                    chunks = []
                    for _ in range(n_chunk):
                        chunk = Chunk(*read_bytes(f, "<QQ"))
                        chunks.append(chunk)
                    seq_bins.append(Bin(bin, chunks))

                    if bin == 37450:  # pseudo-bin, see section 5.2 of BAM spec
                        assert len(chunks) == 2
                        n_mapped, n_unmapped = chunks[1].cnk_beg, chunks[1].cnk_end
                        record_count = n_mapped + n_unmapped
                (n_intv,) = read_bytes(f, "<i")
                linear_index = []
                for _ in range(n_intv):
                    (ioff,) = read_bytes(f, "<Q")
                    linear_index.append(ioff)
                bins.append(seq_bins)
                linear_indexes.append(linear_index)
                record_counts.append(record_count)

        (n_no_coor,) = read_bytes(f, "<Q", (0,))

        assert at_eof(f)

        return TabixIndex(header, sequence_names, bins, linear_indexes, record_counts)


@dataclass
class CSIBin:
    bin: int
    loffset: int
    chunks: Sequence[Chunk]


@dataclass
class CSIIndex:
    min_shift: int
    depth: int
    aux: Any  # TODO
    bins: Sequence[Sequence[CSIBin]]
    record_counts: Sequence[int]

    def offsets(self) -> Any:
        pseudo_bin = bin_limit(self.min_shift, self.depth) + 1

        file_offsets = []
        contig_indexes = []
        positions = []
        for contig_index, bins in enumerate(self.bins):
            # bins may be in any order within a contig, so sort by loffset
            for bin in sorted(bins, key=lambda b: b.loffset):
                if bin.bin == pseudo_bin:
                    continue  # skip pseudo bins
                file_offset = get_file_offset(bin.loffset)
                position = get_first_locus_in_bin(self, bin.bin)
                file_offsets.append(file_offset)
                contig_indexes.append(contig_index)
                positions.append(position)

        return np.array(file_offsets), np.array(contig_indexes), np.array(positions)


def bin_limit(min_shift: int, depth: int) -> int:
    """Defined in CSI spec"""
    return ((1 << (depth + 1) * 3) - 1) // 7


def get_first_bin_in_level(level: int) -> int:
    return ((1 << level * 3) - 1) // 7


def get_level_size(level: int) -> int:
    return 1 << level * 3


def get_level_for_bin(csi: CSIIndex, bin: int) -> int:
    for i in range(csi.depth, -1, -1):
        if bin >= get_first_bin_in_level(i):
            return i
    raise ValueError


def get_first_locus_in_bin(csi: CSIIndex, bin: int) -> int:
    level = get_level_for_bin(csi, bin)
    first_bin_on_level = get_first_bin_in_level(level)
    level_size = get_level_size(level)
    max_span = 1 << (csi.min_shift + 3 * csi.depth)
    return (bin - first_bin_on_level) * (max_span // level_size) + 1


def read_csi(file: PathType) -> CSIIndex:
    """Parse a CSI file into a queryable datastructure"""
    with gzip.open(file) as f:
        (magic,) = read_bytes(f, "4s")
        assert magic == b"CSI\x01"

        min_shift, depth, l_aux = read_bytes(f, "<3i")
        (aux,) = read_bytes(f, f"{l_aux}s", ("",))
        (n_ref,) = read_bytes(f, "<i")

        pseudo_bin = bin_limit(min_shift, depth) + 1

        bins = []
        record_counts = []

        if n_ref > 0:
            for _ in range(n_ref):
                (n_bin,) = read_bytes(f, "<i")
                seq_bins = []
                record_count = -1
                for _ in range(n_bin):
                    bin, loffset, n_chunk = read_bytes(f, "<IQi")
                    chunks = []
                    for _ in range(n_chunk):
                        chunk = Chunk(*read_bytes(f, "<QQ"))
                        chunks.append(chunk)
                    seq_bins.append(CSIBin(bin, loffset, chunks))

                    if bin == pseudo_bin:
                        assert len(chunks) == 2
                        n_mapped, n_unmapped = chunks[1].cnk_beg, chunks[1].cnk_end
                        record_count = n_mapped + n_unmapped
                bins.append(seq_bins)
                record_counts.append(record_count)

        (n_no_coor,) = read_bytes(f, "<Q", (0,))

        assert at_eof(f)

        return CSIIndex(min_shift, depth, aux, bins, record_counts)


def get_file_offset(vfp: int) -> int:
    """Convert a block compressed virtual file pointer to a file offset."""
    address_mask = 0xFFFFFFFFFFFF
    return vfp >> 16 & address_mask


def region_string(contig: str, start: int, end: Optional[int] = None) -> str:
    if end is not None:
        return f"{contig}:{start}-{end}"
    else:
        return f"{contig}:{start}-"


def get_tabix_path(vcf_path: PathType) -> Optional[Path]:
    tbi_path = Path(vcf_path).parent / (Path(vcf_path).name + ".tbi")
    if tbi_path.exists():
        return tbi_path
    else:
        return None


def get_csi_path(vcf_path: PathType) -> Optional[Path]:
    csi_path = Path(vcf_path).parent / (Path(vcf_path).name + ".csi")
    if csi_path.exists():
        return csi_path
    else:
        return None


def read_index(index_path: Path) -> Any:
    if index_path.suffix == ".tbi":
        return read_tabix(index_path)
    elif index_path.suffix == ".csi":
        return read_csi(index_path)
    else:
        return None


def get_sequence_names(vcf_path: Path, index: Any) -> Any:
    try:
        return index.sequence_names
    except AttributeError:
        return VCF(vcf_path).seqnames


def partition_into_regions(
    vcf_path: PathType,
    *,
    index_path: Optional[PathType] = None,
    num_parts: Optional[int] = None,
    target_part_size: Optional[int] = None,
) -> Optional[Sequence[str]]:
    """
    Calculate genomic region strings to partition a VCF file into roughly equal parts.
    """
    if num_parts is None and target_part_size is None:
        raise ValueError("One of num_parts or target_part_size must be specified")

    if num_parts is not None and target_part_size is not None:
        raise ValueError("Only one of num_parts or target_part_size may be specified")

    if num_parts is not None and num_parts < 1:
        raise ValueError("num_parts must be positive")

    if target_part_size is not None and target_part_size < 1:
        raise ValueError("target_part_size must be positive")

    if index_path is None:
        index_path = get_tabix_path(vcf_path)
        if index_path is None:
            index_path = get_csi_path(vcf_path)
            if index_path is None:
                raise ValueError("Cannot find .tbi or .csi file.")

    # Calculate the desired part file boundaries
    file_length = get_file_length(vcf_path)
    if num_parts is not None:
        target_part_size = file_length // num_parts
    elif target_part_size is not None:
        num_parts = ceildiv(file_length, target_part_size)
    if num_parts == 1:
        return None
    part_lengths = np.array([i * target_part_size for i in range(num_parts)])  # type: ignore

    # Get the file offsets from .tbi/.csi
    index = read_index(index_path)
    sequence_names = get_sequence_names(vcf_path, index)
    file_offsets, region_contig_indexes, region_positions = index.offsets()

    # Search the file offsets to find which indexes the part lengths fall at
    ind = np.searchsorted(file_offsets, part_lengths)

    # Drop any parts that are greater than the file offsets (these will be covered by a region with no end)
    ind = np.delete(ind, ind >= len(file_offsets))

    # Drop any duplicates
    ind = np.unique(ind)

    # Calculate region contig and start for each index
    region_contigs = region_contig_indexes[ind]
    region_starts = region_positions[ind]

    # Build region query strings
    regions = []
    for i in range(len(region_starts)):
        contig = sequence_names[region_contigs[i]]
        start = region_starts[i]

        if i == len(region_starts) - 1:  # final region
            regions.append(region_string(contig, start))
        else:
            next_contig = sequence_names[region_contigs[i + 1]]
            next_start = region_starts[i + 1]
            end = next_start - 1  # subtract one since positions are inclusive
            if next_contig == contig:  # contig doesn't change
                regions.append(region_string(contig, start, end))
            else:  # contig changes, so need two regions (or possibly more if any sequences were skipped)
                regions.append(region_string(contig, start))
                for ri in range(region_contigs[i] + 1, region_contigs[i + 1]):
                    regions.append(sequence_names[ri])  # pragma: no cover
                regions.append(region_string(next_contig, 1, end))
    # Add any sequences at the end that were not skipped
    for ri in range(region_contigs[-1] + 1, len(sequence_names)):
        regions.append(sequence_names[ri])  # pragma: no cover

    return regions
