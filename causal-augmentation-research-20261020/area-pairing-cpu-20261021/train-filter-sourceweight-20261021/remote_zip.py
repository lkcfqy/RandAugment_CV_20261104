"""Bounded, read-only access to an openly licensed Oxford Pet ZIP mirror.

No full archive, accounts, third-party dependencies, or execution of remote code.
"""
import datetime
import hashlib
import json
import pathlib
import re
import struct
import urllib.parse
import urllib.request
import zlib

ROOT = pathlib.Path(__file__).resolve().parent
REPO = 'antokun/The-Oxford-IIIT-Pet-Dataset-With-Annotations'
ARCHIVE = 'The Oxford-IIIT Pet Dataset With Annotations.zip'
SIZE = 815141550


def save_new(name, data):
    with (ROOT / name).open('xb') as f:
        f.write(data)


def save_json(name, value):
    save_new(name, (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode())


def parse_directory(directory):
    offset, entries = 0, {}
    while offset < len(directory):
        assert directory[offset:offset + 4] == b'PK\x01\x02'
        fields = struct.unpack_from('<6H3I5H2I', directory, offset + 4)
        made, needed, flags, method, tm, dt, crc, compressed, raw, nl, el, cl, disk_start, internal, external, local = fields
        name = directory[offset + 46:offset + 46 + nl].decode('utf8' if flags & 0x800 else 'cp437')
        extra = directory[offset + 46 + nl:offset + 46 + nl + el]
        ep = 0
        while ep + 4 <= len(extra):
            tag, size = struct.unpack_from('<HH', extra, ep)
            value = extra[ep + 4:ep + 4 + size]
            assert len(value) == size
            if tag == 1:  # ZIP64 sizes may be used even by a sub-4-GiB archive.
                vp = 0
                if raw == 0xffffffff:
                    raw = struct.unpack_from('<Q', value, vp)[0]; vp += 8
                if compressed == 0xffffffff:
                    compressed = struct.unpack_from('<Q', value, vp)[0]; vp += 8
                if local == 0xffffffff:
                    local = struct.unpack_from('<Q', value, vp)[0]; vp += 8
                if disk_start == 0xffff:
                    disk_start = struct.unpack_from('<I', value, vp)[0]
            ep += 4 + size
        assert name not in entries and disk_start == 0 and not flags & 1
        assert raw != 0xffffffff and compressed != 0xffffffff and 0 <= local < SIZE
        entries[name] = dict(compressed_bytes=compressed, bytes=raw, crc32=crc, method=method, local_offset=local)
        offset += 46 + nl + el + cl
    assert offset == len(directory)
    return entries


class RemoteZip:
    def __init__(self, revision, entries=None):
        self.url = 'https://huggingface.co/datasets/' + REPO + '/resolve/' + revision + '/' + urllib.parse.quote(ARCHIVE)
        self.revision = revision
        self.entries = entries or {}
        self.receipts = []

    def part(self, start, length):
        assert 0 <= start < SIZE and 0 < length <= 4 * 1024 * 1024 and start + length <= SIZE
        request = urllib.request.Request(self.url, headers={
            'User-Agent': 'OxfordPet-bounded-read/1.0', 'Range': 'bytes=%d-%d' % (start, start + length - 1)})
        with urllib.request.urlopen(request, timeout=30) as response:
            # Refuse a server that ignores Range; do not consume a full archive.
            assert response.status == 206, 'Range unsupported: status %s; body not read' % response.status
            expected = 'bytes %d-%d/%d' % (start, start + length - 1, SIZE)
            assert response.headers.get('Content-Range') == expected, 'Unexpected Content-Range'
            data = response.read(length + 1)
            assert len(data) == length, 'Range body size mismatch'
        self.receipts.append(dict(start=start, bytes=length, status=206, sha256=hashlib.sha256(data).hexdigest()))
        return data

    def index(self):
        tail = self.part(SIZE - 65557, 65557)
        pos = tail.rfind(b'PK\x05\x06')
        assert pos >= 0
        disk, cd_disk, disk_n, n, cd_size, cd_offset, comment = struct.unpack_from('<HHHHIIH', tail, pos + 4)
        assert disk == cd_disk == 0 and disk_n == n and n < 65535
        assert pos + 22 + comment == len(tail)
        assert 0 < cd_size <= 4 * 1024 * 1024
        directory = self.part(cd_offset, cd_size)
        save_new('zip-tail-metadata.bin', tail)
        entries = parse_directory(directory)
        assert len(entries) == n
        self.entries = entries
        save_json('zip-index.json', dict(revision=self.revision, archive_bytes=SIZE, central_directory_bytes=cd_size,
                                         central_directory_sha256=hashlib.sha256(directory).hexdigest(), entries=entries))
        return entries

    def member(self, name, limit):
        e = self.entries[name]
        assert 0 <= e['bytes'] <= limit and e['compressed_bytes'] <= limit
        header = self.part(e['local_offset'], 30)
        assert header[:4] == b'PK\x03\x04'
        nl, el = struct.unpack_from('<HH', header, 26)
        assert nl + el <= 8192
        local_name = self.part(e['local_offset'] + 30, nl + el)
        payload = self.part(e['local_offset'] + 30 + nl + el, e['compressed_bytes'])
        if e['method'] == 0:
            data = payload
        else:
            assert e['method'] == 8, 'Only stored/deflated members supported'
            dec = zlib.decompressobj(-15)
            data = dec.decompress(payload, limit + 1)
            assert dec.eof and not dec.unconsumed_tail
        assert len(data) == e['bytes'] and zlib.crc32(data) & 0xffffffff == e['crc32']
        return data

    def member_fast(self, name, limit):
        """One bounded range including at most 512 bytes of header slack."""
        e = self.entries[name]
        assert 0 < e['bytes'] <= limit and 0 < e['compressed_bytes'] <= limit
        length = 30 + len(name.encode('utf8')) + 512 + e['compressed_bytes']
        block = self.part(e['local_offset'], length)
        assert block[:4] == b'PK\x03\x04'
        nl, el = struct.unpack_from('<HH', block, 26)
        assert el <= 512 and nl == len(name.encode('utf8'))
        assert block[30:30 + nl].decode('utf8') == name
        payload = block[30 + nl + el:30 + nl + el + e['compressed_bytes']]
        if e['method'] == 0:
            data = payload
        else:
            assert e['method'] == 8
            decoder = zlib.decompressobj(-15)
            data = decoder.decompress(payload, limit + 1)
            assert decoder.eof and not decoder.unconsumed_tail
        assert len(data) == e['bytes'] and zlib.crc32(data) & 0xffffffff == e['crc32']
        return data


def main():
    api = 'https://huggingface.co/api/datasets/' + REPO + '?blobs=true'
    with urllib.request.urlopen(urllib.request.Request(api, headers={'User-Agent': 'OxfordPet-bounded-read/1.0'}), timeout=25) as response:
        b = response.read(1048577)
        assert len(b) <= 1048576 and response.status == 200
    meta = json.loads(b)
    assert not meta['private'] and not meta['gated']
    archive = next(x for x in meta['siblings'] if x['rfilename'] == ARCHIVE)
    assert archive['size'] == SIZE
    save_new('range-mirror-repository-metadata.json', b)
    reader = RemoteZip(meta['sha'])
    entries = reader.index()
    relevant = [n for n in entries if n.endswith(('list.txt', 'trainval.txt', 'test.txt', 'README'))]
    for i, name in enumerate(relevant):
        if entries[name]['bytes'] < 1024 * 1024:
            save_new('archive-text-%d.txt' % i, reader.member(name, 1024 * 1024))
    receipt = dict(status='bounded_ZIP_metadata_read_completed', repo=REPO, revision=meta['sha'], archive_bytes=SIZE,
                   all_bytes_read=sum(x['bytes'] for x in reader.receipts), range_requests=reader.receipts,
                   entry_count=len(entries), text_entries=relevant,
                   image_entries=sum(n.lower().endswith('.jpg') for n in entries),
                   trimap_entries=sum('/trimaps/' in n and n.lower().endswith('.png') for n in entries),
                   image_payloads_downloaded=0, at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    save_json('range-metadata-receipt.json', receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k != 'range_requests'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
