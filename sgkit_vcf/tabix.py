import gzip
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Optional, Sequence, Union

import numpy as np

from sgkit.typing import PathType

TABIX_LINEAR_INDEX_INTERVAL_SIZE = 1 << 14  # 16kb interval size


def _read_bytes(f: IO[Any], fmt: str, nodata: Any = None) -> Union[Any, Sequence[Any]]:
    data = f.read(struct.calcsize(fmt))
    if not data:  # pragma: no cover
        return nodata
    return struct.Struct(fmt).unpack(data)


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
    bins: Sequence[Bin]
    linear_indexes: Sequence[Sequence[int]]
    record_counts: Sequence[int]


def read_tabix(file: PathType) -> TabixIndex:
    """Parse a tabix file into a queryable datastructure"""
    with gzip.open(file) as f:
        (magic,) = _read_bytes(f, "4s")
        assert magic == b"TBI\x01"

        header = Header(*_read_bytes(f, "<8i"))

        sequence_names = []
        linear_indexes = []
        record_counts = []

        if header.l_nm > 0:
            (names,) = _read_bytes(f, f"<{header.l_nm}s")
            # Convert \0-terminated names to strings
            sequence_names = [str(name, "utf-8") for name in names.split(b"\x00")[:-1]]

            for _ in range(header.n_ref):
                (n_bin,) = _read_bytes(f, "<i")
                bins = []
                record_count = -1
                for _ in range(n_bin):
                    bin, n_chunk = _read_bytes(f, "<Ii")
                    chunks = []
                    for _ in range(n_chunk):
                        chunk = Chunk(*_read_bytes(f, "<QQ"))
                        chunks.append(chunk)
                    bins.append(Bin(bin, chunks))

                    if bin == 37450:  # pseudo-bin, see section 5.2 of BAM spec
                        assert len(chunks) == 2
                        n_mapped, n_unmapped = chunks[1].cnk_beg, chunks[1].cnk_end
                        record_count = n_mapped + n_unmapped
                (n_intv,) = _read_bytes(f, "<i")
                linear_index = []
                for _ in range(n_intv):
                    (ioff,) = _read_bytes(f, "<Q")
                    linear_index.append(ioff)
                linear_indexes.append(linear_index)
                record_counts.append(record_count)

        (n_no_coor,) = _read_bytes(f, "<Q", (0,))

        # Check we have consumed all of file
        end = f.tell()

        f.seek(0, 2)
        uncompressed_file_length = f.tell()

        assert end == uncompressed_file_length

        return TabixIndex(header, sequence_names, bins, linear_indexes, record_counts)


def get_file_length(path: Path) -> int:
    """Get the length of a file in bytes."""
    return path.stat().st_size


def get_file_offset(vfp: int) -> int:
    """Convert a block compressed virtual file pointer to a file offset."""
    address_mask = 0xFFFFFFFFFFFF
    return vfp >> 16 & address_mask


def region_string(contig: str, start: int, end: Optional[int] = None) -> str:
    if end is not None:
        return f"{contig}:{start}-{end}"
    else:
        return f"{contig}:{start}-"


def get_tabix_path(vcf_path: Path) -> Path:
    return Path(vcf_path).parent / (Path(vcf_path).name + ".tbi")


def partition_into_regions(
    vcf_path: PathType,
    *,
    tabix_path: Optional[PathType] = None,
    num_parts: Optional[int] = None,
    target_part_size: Optional[int] = None,
) -> Sequence[str]:
    """
    Calculate genomic region strings to partition a VCF file into roughly equal parts.
    """
    if num_parts is None and target_part_size is None:
        raise ValueError("One of num_parts or target_part_size must be specified")

    if num_parts is not None and target_part_size is not None:
        raise ValueError("Only one of num_parts or target_part_size may be specified")

    if tabix_path is None:
        tabix_path = get_tabix_path(vcf_path)

    # Calculate the desired part file boundaries
    file_length = get_file_length(vcf_path)
    if num_parts is not None:
        target_part_size = file_length // num_parts
    elif target_part_size is not None:
        num_parts = file_length // target_part_size
    part_lengths = np.array([i * target_part_size for i in range(num_parts)])  # type: ignore

    # Get the contig names and their linear indexes from tabix
    tabix = read_tabix(tabix_path)

    # Combine the linear indexes and calculate their sizes
    linear_index = np.array([i for li in tabix.linear_indexes for i in li])
    linear_index_sizes = np.array([len(li) for li in tabix.linear_indexes])
    linear_index_cum_sizes = np.cumsum(linear_index_sizes)
    assert sum(linear_index_sizes) == len(linear_index)

    # Create file offsets for each position in the combined linear index
    file_offsets = np.array([get_file_offset(vfp) for vfp in linear_index])

    # Search the file offsets to find which indexes the part lengths fall at
    ind = np.searchsorted(file_offsets, part_lengths)

    # Calculate region contig and start for each index
    region_contig_indexes = np.searchsorted(linear_index_cum_sizes, ind)
    region_contigs = [tabix.sequence_names[i] for i in region_contig_indexes]

    linear_index_cum_sizes_0 = np.insert(linear_index_cum_sizes, 0, 0)
    per_contig_linear_index_ind = ind - linear_index_cum_sizes_0[region_contig_indexes]

    region_starts = (
        per_contig_linear_index_ind * TABIX_LINEAR_INDEX_INTERVAL_SIZE + 1
    )  # positions are 1-based and inclusive

    # Build region query strings
    regions = []
    for i in range(len(region_starts)):
        contig = region_contigs[i]
        start = region_starts[i]

        if i == len(region_starts) - 1:  # final region
            regions.append(region_string(contig, start))
        else:
            next_contig = region_contigs[i + 1]
            next_start = region_starts[i + 1]
            end = next_start - 1  # subtract one since positions are inclusive
            if next_contig == contig:  # contig doesn't change
                regions.append(region_string(contig, start, end))
            else:  # contig changes, so need two regions
                regions.append(region_string(contig, start))
                regions.append(region_string(next_contig, 1, end))

    return regions
