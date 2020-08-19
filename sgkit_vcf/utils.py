import itertools
import struct
from pathlib import Path
from typing import IO, Any, Iterator, Optional, Sequence, TypeVar

T = TypeVar("T")


def ceildiv(a: int, b: int) -> int:
    """Safe integer ceil function"""
    return -(-a // b)


# https://dev.to/orenovadia/solution-chunked-iterator-python-riddle-3ple
def chunks(iterator: Iterator[T], n: int) -> Iterator[Iterator[T]]:
    """
    Convert an iterator into an iterator of iterators, where the inner iterators
    each return `n` items, except the last, which may return fewer.
    """

    for first in iterator:  # take one item out (exits loop if `iterator` is empty)
        rest_of_chunk = itertools.islice(iterator, 0, n - 1)
        yield itertools.chain([first], rest_of_chunk)  # concatenate the first item back


def get_file_length(path: Path) -> int:
    """Get the length of a file in bytes."""
    return path.stat().st_size


def get_file_offset(vfp: int) -> int:
    """Convert a block compressed virtual file pointer to a file offset."""
    address_mask = 0xFFFFFFFFFFFF
    return vfp >> 16 & address_mask


def read_bytes(
    f: IO[Any], fmt: str, nodata: Optional[Sequence[Any]] = None
) -> Sequence[Any]:
    """Read bytes using a `struct` format string"""
    data = f.read(struct.calcsize(fmt))
    if not data:
        return nodata or ()
    return struct.Struct(fmt).unpack(data)


def at_eof(f: IO[Any]) -> bool:
    """Check if a file stream is at the end of the file."""
    pos = f.tell()
    try:
        f.seek(0, 2)  # seek to end of file
        eof = f.tell()
        return pos == eof
    finally:
        f.seek(pos)
