"""An Outlook message (.msg) built in memory, the way Outlook saves one when a mail is dragged out.

A .msg is a Compound File Binary container (an "OLE" file, [MS-CFB]) whose root storage holds
the message's MAPI properties ([MS-OXMSG]): a `__properties_version1.0` stream listing them and
one `__substg1.0_<id><type>` stream per text property. `outlook_message(subject, body,
sender_name, sender_address)` writes such a file with those four and nothing else, no recipients
and no attachments. Streams under 4096 bytes live in the mini stream, as the format asks.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

SECTOR = 512
MINI_SECTOR = 64
MINI_CUTOFF = 4096
FREE, END_OF_CHAIN, FAT_SECTOR, NO_STREAM = 0xFFFFFFFF, 0xFFFFFFFE, 0xFFFFFFFD, 0xFFFFFFFF
STREAM, ROOT = 2, 5
# {00020D0B-0000-0000-C000-000000000046}, the class of an Outlook message's root
MESSAGE_CLASS_ID = bytes.fromhex("0b0d020000000000c000000000000046")
UNICODE_STRING = 0x001F
PROPERTY_READABLE_WRITABLE = 0x6
TEXT_PROPERTIES = {
    "subject": 0x0037,
    "body": 0x1000,
    "sender_name": 0x0C1A,
    "sender_address": 0x0C1F,
    "message_class": 0x001A,
}


@dataclass
class _Entry:
    name: str
    kind: int
    data: bytes = b""
    start: int = END_OF_CHAIN
    right: int = NO_STREAM
    child: int = NO_STREAM


def _chain(table: list[int], first: int, count: int) -> None:
    """Link `count` sectors from `first` into one chain of `table`."""
    for index in range(first, first + count):
        table[index] = index + 1 if index + 1 < first + count else END_OF_CHAIN


def _sectors(data: bytes, size: int) -> int:
    return -(-len(data) // size)


def _padded(data: bytes, size: int) -> bytes:
    return data + b"\0" * (_sectors(data, size) * size - len(data))


def _directory_entry(entry: _Entry) -> bytes:
    name = (entry.name + "\0").encode("utf-16-le")
    class_id = MESSAGE_CLASS_ID if entry.kind == ROOT else b"\0" * 16
    return (
        name.ljust(64, b"\0")
        + struct.pack("<HBB", len(name), entry.kind, 1)  # every node black
        + struct.pack("<III", NO_STREAM, entry.right, entry.child)
        + class_id
        + b"\0" * 20  # state bits, created and modified times
        + struct.pack("<IQ", entry.start, len(entry.data))
    )


def _compound_file(streams: dict[str, bytes]) -> bytes:
    root = _Entry("Root Entry", ROOT)
    # siblings as a chain of right links, in the order the format sorts names
    names = sorted(streams, key=lambda name: (len(name), name.upper()))
    entries = [root, *(_Entry(name, STREAM, streams[name]) for name in names)]
    root.child = 1 if names else NO_STREAM
    for index in range(1, len(entries) - 1):
        entries[index].right = index + 1

    mini_stream, mini_fat = b"", []
    for entry in entries[1:]:
        if len(entry.data) < MINI_CUTOFF:
            entry.start = len(mini_stream) // MINI_SECTOR
            count = _sectors(entry.data, MINI_SECTOR)
            mini_fat.extend([FREE] * count)
            _chain(mini_fat, entry.start, count)
            mini_stream += _padded(entry.data, MINI_SECTOR)
    root.data = mini_stream

    # regular sectors: large streams, the mini stream, the mini FAT, the directory, the FAT
    blocks: list[tuple[_Entry | str, bytes]] = [
        (entry, entry.data) for entry in entries[1:] if len(entry.data) >= MINI_CUTOFF
    ]
    blocks.append((root, mini_stream))
    mini_fat_bytes = struct.pack(f"<{len(mini_fat)}I", *mini_fat) if mini_fat else b""
    blocks.append(("minifat", mini_fat_bytes))
    directory_count = _sectors(b"\0" * 128 * len(entries), SECTOR)
    data_sectors = sum(_sectors(data, SECTOR) for _, data in blocks) + directory_count
    fat_count = 1
    while fat_count * (SECTOR // 4) < data_sectors + fat_count:
        fat_count += 1
    fat = [FREE] * (fat_count * (SECTOR // 4))

    body, next_sector, mini_fat_start = b"", 0, END_OF_CHAIN
    for owner, data in blocks:
        count = _sectors(data, SECTOR)
        if count:
            _chain(fat, next_sector, count)
            if isinstance(owner, _Entry):
                owner.start = next_sector
            else:
                mini_fat_start = next_sector
        body += _padded(data, SECTOR)
        next_sector += count
    directory_start = next_sector
    _chain(fat, directory_start, directory_count)
    directory = b"".join(_directory_entry(entry) for entry in entries)
    body += directory.ljust(directory_count * SECTOR, b"\0")
    next_sector += directory_count
    fat_start = next_sector
    for index in range(fat_start, fat_start + fat_count):
        fat[index] = FAT_SECTOR
    body += struct.pack(f"<{len(fat)}I", *fat)

    difat = [fat_start + index for index in range(fat_count)]
    difat += [FREE] * (109 - len(difat))
    header = (
        bytes.fromhex("d0cf11e0a1b11ae1")
        + b"\0" * 16
        + struct.pack("<HHHHH", 0x003E, 0x0003, 0xFFFE, 9, 6)
        + b"\0" * 6
        + struct.pack("<IIII", 0, fat_count, directory_start, 0)
        + struct.pack(
            "<IIIII", MINI_CUTOFF, mini_fat_start, _sectors(mini_fat_bytes, SECTOR), END_OF_CHAIN, 0
        )
        + struct.pack("<109I", *difat)
    )
    return header + body


def _properties_stream(texts: dict[int, bytes]) -> bytes:
    # a top-level message: 8 reserved bytes, recipient and attachment ids and counts, 8 reserved
    header = struct.pack("<8xIIII8x", 0, 0, 0, 0)
    entries = b"".join(
        # the size counts the terminating null the stream leaves out
        struct.pack(
            "<IIII",
            (property_id << 16) | UNICODE_STRING,
            PROPERTY_READABLE_WRITABLE,
            len(value) + 2,
            0,
        )
        for property_id, value in texts.items()
    )
    return header + entries


def outlook_message(subject: str, body: str, sender_name: str, sender_address: str) -> bytes:
    """The bytes of a .msg holding one plain-text mail."""
    values = {
        "subject": subject,
        "body": body,
        "sender_name": sender_name,
        "sender_address": sender_address,
        "message_class": "IPM.Note",
    }
    texts = {TEXT_PROPERTIES[name]: value.encode("utf-16-le") for name, value in values.items()}
    streams = {
        f"__substg1.0_{property_id:04X}{UNICODE_STRING:04X}": value
        for property_id, value in texts.items()
    }
    streams["__properties_version1.0"] = _properties_stream(texts)
    return _compound_file(streams)
