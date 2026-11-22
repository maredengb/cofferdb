use super::*;

// Each directory row stores an eight-byte data offset, a four-byte encoded record
// length and a four-byte CRC. Fixed width lets a binary-search midpoint locate its
// row without decoding any preceding keys.
const INDEX_ENTRY_BYTES: u64 = 16;
const PAGE_BYTES: usize = 2048;
const CACHE_PAGES: usize = 16;

// A probe can hold the record being checked and the key from the last successful
// lookup simultaneously. Reserve room for both at the maximum key size even if no
// cache page fits.
fn query_workspace(key: u128) -> u128 {
    (MIN_WORKSPACE as u128)
        .max(2 * key + ENTRY_HEADER_BYTES as u128 + std::mem::size_of::<Entry>() as u128)
}

pub(super) fn preferred_workspace(key: u128) -> u128 {
    // Extra headroom pays for the backing buffers and Page records of a bounded
    // cache. The smaller required grant remains usable because a probe can read
    // directly from either file.
    required_workspace(key).max(
        2 * key
            + ENTRY_HEADER_BYTES as u128
            + std::mem::size_of::<Entry>() as u128
            + CACHE_PAGES as u128 * (PAGE_BYTES + std::mem::size_of::<Page>()) as u128,
    )
}

pub(super) fn required_workspace(key: u128) -> u128 {
    // Directory construction reads a buffered run and writes a buffered index at
    // the same time. Query buffers take over after construction, so the workspace
    // need only cover the larger of these phases.
    query_workspace(key).max(2 * IO_BYTES as u128 + key + std::mem::size_of::<Entry>() as u128)
}

pub(crate) struct ExternalKeyLookup {
    // Paired run and directory files. An empty sorter produces None so even the
    // first lookup can return a miss without opening a file.
    files: Option<LookupFiles>,
    // Largest key length in bytes from the sorter. It bounds both a checked on-disk
    // entry and the memory reserved to probe it.
    max_key: usize,
    // Reusable buffer for one complete data entry, including its fixed-width
    // header. The entry CRC is checked before comparing the key.
    buffer: Vec<u8>,
    // Optional last successful hit, retaining its verified key, source line and
    // value so repeated probes can bypass both files.
    last: Option<Entry>,
    // Cached aligned pages from both files, ordered from least to most recently
    // used. Direct positional reads remain possible with zero pages.
    pages: Vec<Page>,
    // Workspace handed over by the sorter for the probe buffer, saved key and
    // optional page cache.
    _work: WorkspaceReservation,
}

struct Page {
    // Selects the fixed-width directory file when true and the variable-length data
    // file when false. Equal offsets in the two files denote different pages.
    index: bool,
    // Aligned byte offset of the page's first byte in the selected file.
    offset: u64,
    // Actual data read for that page. The last page of either file can be shorter
    // than PAGE_BYTES.
    bytes: Vec<u8>,
}

struct LookupFiles {
    // Sorted variable-width run containing full keys, source positions and values.
    data: TemporaryFile,
    // Keeps the data run and its disk charge alive after ownership moves out of the
    // sort cursor.
    _data_temporary: TempFileGuard,
    // Fixed-width directory file mapping entry ordinals to data offsets, encoded
    // lengths and record CRCs.
    index: TemporaryFile,
    // Retains ownership of the directory file's temporary name and disk reservation
    // for the lookup's lifetime.
    _index_temporary: TempFileGuard,
    // Number of sorted keys. It bounds binary search and determines the expected
    // directory length.
    count: u64,
    // Original run length in bytes, used to bound each directory-provided range
    // before reading from data.
    data_bytes: u64,
}

impl ExternalKeyLookup {
    pub(super) fn new(
        run: Option<Run>,
        directory: &Path,
        disk: Arc<TemporaryDiskBudget>,
        max_key: usize,
        work: WorkspaceReservation,
    ) -> Result<Self> {
        let mut lookup = Self {
            files: None,
            max_key,
            buffer: Vec::new(),
            last: None,
            pages: Vec::new(),
            _work: work,
        };
        // An empty run set needs neither file. An empty key set answers misses
        // without attempting to read or generate an index.
        let Some(run) = run else { return Ok(lookup) };
        let count = run.count;
        let data_bytes = run.bytes;
        // The cursor checks the source file's declared length on open. Reading it
        // through exhaustion below also checks the remaining extent and running
        // CRC before a lookup can be returned.
        let mut cursor = RunCursor::new(run, max_key)?;
        let (file, mut temporary) =
            create_budgeted_sort_file(directory, "import-lookup", disk, lookup._work.context())?;
        temporary.reserve_disk(
            // Reserve the complete fixed-size directory up front. Its data file
            // already belongs to the run, and the two disk reservations stay
            // separate even though the resulting lookup owns both.
            count
                .checked_mul(INDEX_ENTRY_BYTES)
                .ok_or_else(invalid_run)?,
        )?;
        let mut output = BufWriter::with_capacity(IO_BYTES, file);
        // Keep variable-length keys in the sorted run. A fixed-width directory
        // lets binary search seek by entry number; its CRC binds each probe to
        // the full data record before the key can influence the next seek.
        let mut offset = 0u64;
        while let Some(entry) = cursor.next()? {
            // Regenerate the canonical run header from the decoded entry. This
            // produces the same bytes stored at offset, allowing a later point
            // probe to check one record without rescanning the entire run.
            let header = entry.header()?;
            let length =
                // The directory length covers header and key together. Its
                // four-byte field must therefore fit even when the key length
                // alone was representable in a run header.
                u32::try_from(ENTRY_HEADER_BYTES + entry.key.len()).map_err(|_| invalid_run())?;
            let mut crc = Crc32c::default();
            crc.update(&header);
            crc.update(&entry.key);
            // The row is little-endian in offset, length, checksum order. Keep the
            // encoding here in step with get, which must validate the range before
            // using it to read data.
            output.write_all(&offset.to_le_bytes())?;
            output.write_all(&length.to_le_bytes())?;
            output.write_all(&crc.finish().to_le_bytes())?;
            offset = offset
                // Advance by the encoded record length including its header. A
                // wrapped byte offset would make later ordinal lookups seek into
                // unrelated records.
                .checked_add(u64::from(length))
                .ok_or_else(invalid_run)?;
        }
        // Complete the directory writes before installing either file. A build
        // error drops the local file owners, leaving no queryable lookup backed by
        // a partial directory.
        output.flush()?;
        lookup.files = Some(LookupFiles {
            data: cursor.input.into_inner(),
            _data_temporary: cursor._temporary,
            index: output.into_inner().map_err(|error| error.into_error())?,
            _index_temporary: temporary,
            count,
            data_bytes,
        });
        // Once both buffering adapters have surrendered their files, trim
        // construction credit to the preferred probe budget. Retain a smaller
        // grant as-is so a valid lookup can operate with less caching.
        let retained = lookup
            ._work
            .reserved_bytes()
            .min(preferred_workspace(max_key as u128) as usize);
        lookup._work.retain(retained);
        Ok(lookup)
    }

    pub(crate) fn get(&mut self, key: &[u8]) -> Result<Option<(u64, u64)>> {
        // The last hit already holds an owned key with its verified line and
        // value. Repeated references to the same key can return without consulting
        // either temporary file.
        if let Some(last) = &self.last {
            if last.key == key {
                return Ok(Some((last.line, last.value)));
            }
        }
        // An empty sorter has no files and released its workspace. Answer the miss
        // before subtracting the memory required by a nonempty lookup.
        let Some(files) = &mut self.files else {
            return Ok(None);
        };
        // Leave room for the decoded record, a separate saved key and its Entry.
        // Only the surplus lease is available for pages, and a zero-page lookup
        // can still read both files directly.
        let cache_bytes = self._work.reserved_bytes()
            - (2 * self.max_key + ENTRY_HEADER_BYTES + std::mem::size_of::<Entry>());
        let mut low = 0;
        let mut high = files.count;
        // Search the half-open interval of entry ordinals. Computing the midpoint
        // from the remaining width avoids overflowing when the directory is large.
        while low < high {
            let middle = low + (high - low) / 2;
            let mut index = [0; INDEX_ENTRY_BYTES as usize];
            // Every directory record has a fixed width, so its offset depends only
            // on the midpoint. The same page cache serves both files and the index
            // flag keeps their offsets distinct.
            read_cached(
                &mut self.pages,
                &mut files.index,
                true,
                files.count * INDEX_ENTRY_BYTES,
                &mut index,
                middle * INDEX_ENTRY_BYTES,
                cache_bytes,
            )?;
            // The directory entry comes from a separate temporary file. Bound its
            // data range and key length before using either value to allocate or
            // seek. The record checksum below then protects the comparison that
            // chooses the next half of the search.
            let offset = u64::from_le_bytes(index[..8].try_into().unwrap());
            let length = u32::from_le_bytes(index[8..12].try_into().unwrap()) as usize;
            let checksum = u32::from_le_bytes(index[12..].try_into().unwrap());
            // A complete record may consist solely of its header for an empty key.
            // Check the lower bound before subtracting it, and reject an
            // overflowing or out-of-run end offset.
            if length < ENTRY_HEADER_BYTES
                || length - ENTRY_HEADER_BYTES > self.max_key
                || offset
                    .checked_add(length as u64)
                    .filter(|end| *end <= files.data_bytes)
                    .is_none()
            {
                return Err(invalid_run());
            }
            if length > self.buffer.capacity() {
                // Drop the old probe allocation before acquiring a larger one.
                // Growing the existing vector could temporarily hold two record
                // buffers under a lease sized for only one.
                self.buffer = Vec::new();
                self.buffer
                    .try_reserve_exact(length)
                    .map_err(allocation_error)?;
            }
            // An exact reserve can still yield spare capacity. Fail if the actual
            // allocation exceeds the largest admitted record even though the
            // declared length passed the format checks.
            if self.buffer.capacity() > self.max_key + ENTRY_HEADER_BYTES {
                return Err(
                    std::io::Error::other("lookup buffer exceeds admitted capacity").into(),
                );
            }
            self.buffer.resize(length, 0);
            // Read the entire variable-width record into one buffer. Its data
            // extent comes from the run, whereas the location and expected
            // checksum came from the directory.
            read_cached(
                &mut self.pages,
                &mut files.data,
                false,
                files.data_bytes,
                &mut self.buffer,
                offset,
                cache_bytes,
            )?;
            let mut crc = Crc32c::default();
            crc.update(&self.buffer);
            // Check both the checksum and the key length encoded in the record
            // itself. A bad length is a malformed run even if the recorded bytes
            // happen to match the directory CRC.
            if crc.finish() != checksum
                || u32::from_le_bytes(self.buffer[..4].try_into().unwrap()) as usize
                    != length - ENTRY_HEADER_BYTES
            {
                return Err(invalid_run());
            }
            let stored_key = &self.buffer[ENTRY_HEADER_BYTES..];
            // Compare only the verified key bytes in the same order used to sort
            // the run. The line and payload do not contribute to ordering.
            match stored_key.cmp(key) {
                std::cmp::Ordering::Less => low = middle + 1,
                std::cmp::Ordering::Greater => high = middle,
                std::cmp::Ordering::Equal => {
                    let line = u64::from_le_bytes(self.buffer[4..12].try_into().unwrap());
                    let value = u64::from_le_bytes(self.buffer[12..20].try_into().unwrap());
                    // The verified line and value are already available. The saved
                    // key merely speeds up a repeated lookup; failing to cache it
                    // should not turn a valid hit into a resource error.
                    // Reuse the previous hit's key allocation when it fits. Give
                    // it up before reserving a larger key so two saved-hit buffers
                    // never overlap.
                    let mut saved = self.last.take().map(|entry| entry.key).unwrap_or_default();
                    if saved.capacity() < stored_key.len() {
                        saved = Vec::new();
                    }
                    saved.clear();
                    // A failed or oversized optional allocation leaves the
                    // last-hit slot empty. The verified lookup can still return
                    // its line and value immediately.
                    if saved.try_reserve_exact(stored_key.len()).is_err()
                        || saved.capacity() > self.max_key
                    {
                        return Ok(Some((line, value)));
                    }
                    // Copy the key out of the reusable record buffer before saving
                    // it. The next midpoint or query can overwrite that scratch
                    // space.
                    saved.extend_from_slice(stored_key);
                    self.last = Some(Entry {
                        key: saved,
                        line,
                        value,
                    });
                    return Ok(Some((line, value)));
                }
            }
        }
        // A clean miss leaves any prior successful hit in place. The empty
        // interval establishes absence without reading an extra directory entry.
        Ok(None)
    }
}

// Read from either file through a caller-sized optional page cache. The cache
// shares one byte allowance between index pages and run data pages.
fn read_cached(
    pages: &mut Vec<Page>,
    file: &mut File,
    index: bool,
    file_bytes: u64,
    mut output: &mut [u8],
    mut offset: u64,
    cache_bytes: usize,
) -> Result<()> {
    // Check the entire requested extent so cached and direct reads reject the same
    // overflow or out-of-file range before modifying the caller's buffer.
    if offset
        .checked_add(output.len() as u64)
        .filter(|end| *end <= file_bytes)
        .is_none()
    {
        return Err(invalid_run());
    }
    // Account for a full payload allocation and its metadata slot for each page.
    // The resulting bound applies to retained capacity even when the last page is
    // short.
    let page_limit = (cache_bytes / (PAGE_BYTES + std::mem::size_of::<Page>())).min(CACHE_PAGES);
    // Discard the old vector when its allocation exceeds the new grant. Merely
    // truncating the number of cached pages would keep oversized metadata
    // capacity.
    if pages.capacity() > page_limit {
        *pages = Vec::new();
    }
    // When a full cache slot does not fit, use the same bounded read directly. No
    // page needs to be allocated for correctness.
    if page_limit == 0 {
        return read_at(file, output, offset);
    }
    while !output.is_empty() {
        // Align down to the containing page. Key cache entries by both file kind
        // and byte offset because the index and data can share the same numeric
        // offsets.
        let start = offset / PAGE_BYTES as u64 * PAGE_BYTES as u64;
        let found = pages
            .iter()
            .position(|page| page.index == index && page.offset == start);
        // Take a matching page out of the list while using it. Putting it at the
        // end below makes it most recently used, leaving eviction at the front.
        let mut page = if let Some(position) = found {
            pages.remove(position)
        } else if pages.len() == page_limit {
            pages.remove(0)
        } else {
            // Verify metadata capacity after allocation as well as on entry. An
            // optional cache can abandon an allocator-rounded capacity and finish
            // this request directly.
            if pages.capacity() < page_limit
                && (pages.try_reserve_exact(page_limit - pages.len()).is_err()
                    || pages.capacity() > page_limit)
            {
                // Cache population is optional; direct reads preserve the
                // same checksums and lookup semantics without a cache entry.
                *pages = Vec::new();
                return read_at(file, output, offset);
            }
            Page {
                index,
                offset: start,
                bytes: Vec::new(),
            }
        };
        // On a miss, read the actual remaining file bytes for the last page. This
        // avoids requiring padding beyond the recorded extent.
        if found.is_none() {
            let length = (file_bytes - start).min(PAGE_BYTES as u64) as usize;
            if page.bytes.capacity() < length {
                page.bytes = Vec::new();
            }
            // Reuse an evicted page's buffer if it already fits. If allocating
            // another buffer fails or rounds above the page size, read directly
            // into the caller's remaining slice.
            if page.bytes.capacity() < length
                && (page
                    .bytes
                    .try_reserve_exact(length - page.bytes.len())
                    .is_err()
                    || page.bytes.capacity() > PAGE_BYTES)
            {
                return read_at(file, output, offset);
            }
            page.bytes.resize(length, 0);
            read_at(file, &mut page.bytes, start)?;
            // Publish page identity only after the exact fill succeeded. A failed
            // fill drops the detached page without installing stale data at the
            // new offset.
            page.index = index;
            page.offset = start;
        }
        // Copy just the requested portion within this page. Update both the
        // destination slice and absolute offset to complete reads that cross page
        // boundaries.
        let within = (offset - start) as usize;
        let count = output.len().min(page.bytes.len() - within);
        output[..count].copy_from_slice(&page.bytes[within..within + count]);
        offset += count as u64;
        output = &mut output[count..];
        pages.push(page);
    }
    Ok(())
}

// Read exactly at the requested byte position. Unix uses positional I/O, while
// other targets use a seek on the mutably borrowed file followed by the same
// exact-read check.
fn read_at(file: &mut File, bytes: &mut [u8], offset: u64) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        file.read_exact_at(bytes, offset).map_err(read_error)?;
    }
    #[cfg(not(unix))]
    {
        file.seek(std::io::SeekFrom::Start(offset))?;
        file.read_exact(bytes).map_err(read_error)?;
    }
    Ok(())
}

// An early EOF inside the already validated extent means an incomplete run. Keep
// other read failures as I/O errors so callers can distinguish operational failure
// from corrupt layout.
fn read_error(error: std::io::Error) -> Error {
    if error.kind() == std::io::ErrorKind::UnexpectedEof {
        invalid_run()
    } else {
        error.into()
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::{sorter, TestDirectory};
    use super::*;
    use std::collections::BTreeMap;
    use std::io::SeekFrom;

    #[test]
    fn optional_pages_do_not_raise_the_required_merge_workspace() {
        let directory = TestDirectory::new("lookup-optional-minimum");
        let disk = Arc::new(TemporaryDiskBudget::new(1024 * 1024));
        for extra in 0..100usize {
            let size = 19 * 1024 + extra;
            let mut small = vec![31; 3];
            let mut large = vec![31; size];
            large[size - 1] = 32;
            let mut second = large.clone();
            second[size - 1] = 33;
            let mut keys = sorter(&directory, disk.clone());
            keys.push_value(&[&large], 7, 11).unwrap();
            keys.flush_run().unwrap();
            keys.push_value(&[&second], 19, 23).unwrap();
            keys.flush_run().unwrap();
            keys.push_value(&[&small], 13, 17).unwrap();
            let mut lookup = keys.finish_lookup().unwrap();
            assert_eq!(lookup._work.reserved_bytes(), MIN_WORKSPACE);
            assert_eq!(lookup.get(&small).unwrap(), Some((13, 17)));
            assert_eq!(lookup.get(&large).unwrap(), Some((7, 11)));
            assert_eq!(lookup.get(&second).unwrap(), Some((19, 23)));
            let held = lookup.buffer.capacity()
                + lookup.last.as_ref().unwrap().key.capacity()
                + std::mem::size_of::<Entry>()
                + lookup.pages.capacity() * std::mem::size_of::<Page>()
                + lookup
                    .pages
                    .iter()
                    .map(|page| page.bytes.capacity())
                    .sum::<usize>();
            assert!(held <= lookup._work.reserved_bytes());
            let page_limit =
                (MIN_WORKSPACE - 2 * size - ENTRY_HEADER_BYTES - std::mem::size_of::<Entry>())
                    / (PAGE_BYTES + std::mem::size_of::<Page>());
            assert!(lookup.pages.capacity() <= page_limit);
            assert!(lookup.pages.len() < CACHE_PAGES);
            small.push(30);
            assert_eq!(lookup.get(&small).unwrap(), None);
            drop(lookup);
            directory.assert_clean(&disk);
        }
    }

    #[test]
    fn bounded_page_cache_handles_cross_page_reads_eviction_and_file_identity() {
        let directory = TestDirectory::new("lookup-pages");
        let disk = Arc::new(TemporaryDiskBudget::new(2 * 1024 * 1024));
        let mut sources = Vec::new();
        for index in 0..2 {
            let bytes = (0..PAGE_BYTES * 35 + 7)
                .map(|offset| (offset.wrapping_mul(31 + index * 6) ^ (offset >> 7)) as u8)
                .collect::<Vec<_>>();
            let (mut file, mut guard) =
                create_budgeted_temporary_file_in(&directory.0, "cache-test", "tmp", disk.clone())
                    .unwrap();
            guard.reserve_disk(bytes.len() as u64).unwrap();
            file.write_all(&bytes).unwrap();
            sources.push((file, guard, bytes));
        }
        let mut pages = Vec::new();
        for seed in 0..100 {
            for length in [
                0,
                1,
                PAGE_BYTES - 1,
                PAGE_BYTES,
                PAGE_BYTES + 1,
                PAGE_BYTES * 4,
            ] {
                for (index, (file, _, expected)) in sources.iter_mut().enumerate() {
                    let offset = seed * 997 % (expected.len() - length + 1);
                    let mut actual = vec![0; length];
                    read_cached(
                        &mut pages,
                        file,
                        index == 1,
                        expected.len() as u64,
                        &mut actual,
                        offset as u64,
                        64 * 1024,
                    )
                    .unwrap();
                    assert_eq!(actual, expected[offset..offset + length]);
                    assert!(pages.len() <= CACHE_PAGES);
                    assert!(pages.capacity() <= CACHE_PAGES);
                    assert!(pages.iter().all(|page| page.bytes.capacity() <= PAGE_BYTES));
                }
            }
        }
        let (file, _, expected) = &mut sources[0];
        for offset in [expected.len() as u64, u64::MAX] {
            assert_eq!(
                read_cached(
                    &mut pages,
                    file,
                    false,
                    expected.len() as u64,
                    &mut [0],
                    offset,
                    64 * 1024
                )
                .unwrap_err()
                .code(),
                ErrorCode::ChecksumMismatch
            );
        }
        drop(sources);
        directory.assert_clean(&disk);
    }

    #[test]
    fn shrinking_or_disabling_optional_cache_preserves_reads_and_integrity_checks() {
        let directory = TestDirectory::new("lookup-cache-fallback");
        let disk = Arc::new(TemporaryDiskBudget::new(128 * 1024));
        let (mut file, mut guard) =
            create_budgeted_temporary_file_in(&directory.0, "cache-fallback", "tmp", disk.clone())
                .unwrap();
        let bytes = (0..PAGE_BYTES * 20 + 9)
            .map(|n| (n * 31) as u8)
            .collect::<Vec<_>>();
        guard.reserve_disk(bytes.len() as u64).unwrap();
        file.write_all(&bytes).unwrap();
        let mut pages = Vec::new();
        for seed in 0..100 {
            for workspace in [65536, 0, 1, 8192, 65536] {
                let start = seed * 127 % (bytes.len() - PAGE_BYTES * 2);
                let mut output = vec![0; PAGE_BYTES * 2];
                read_cached(
                    &mut pages,
                    &mut file,
                    false,
                    bytes.len() as u64,
                    &mut output,
                    start as u64,
                    workspace,
                )
                .unwrap();
                assert_eq!(output, bytes[start..start + output.len()]);
                let held = pages.capacity() * std::mem::size_of::<Page>()
                    + pages
                        .iter()
                        .map(|page| page.bytes.capacity())
                        .sum::<usize>();
                assert!(held <= workspace);
            }
        }
        for workspace in [0, 65536] {
            let error = read_cached(
                &mut pages,
                &mut file,
                false,
                bytes.len() as u64,
                &mut [0],
                bytes.len() as u64,
                workspace,
            )
            .unwrap_err();
            assert_eq!(error.code(), ErrorCode::ChecksumMismatch);
        }
        drop((file, guard));
        directory.assert_clean(&disk);
    }

    #[test]
    fn disk_lookup_matches_100_independent_shuffled_maps_with_bounded_buffers() {
        let directory = TestDirectory::new("lookup-oracle");
        let disk = Arc::new(TemporaryDiskBudget::new(8 * 1024 * 1024));
        for seed in 0..100u64 {
            let mut state = seed + 1;
            let mut random = || {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state
            };
            let mut expected = BTreeMap::new();
            for number in 0..257u64 {
                let mut key = number.to_be_bytes().to_vec();
                key.resize(128 + (random() % 256) as usize, random() as u8);
                expected.insert(key, (number * 3 + 1, random()));
            }
            expected.insert(Vec::new(), (u64::MAX, 0));
            expected.insert(vec![255; 8192], (1, u64::MAX));
            let mut input = expected.iter().collect::<Vec<_>>();
            for index in (1..input.len()).rev() {
                input.swap(index, random() as usize % (index + 1));
            }
            let mut keys = sorter(&directory, disk.clone());
            for (key, (line, value)) in &input {
                keys.push_value(&[key], *line, *value).unwrap();
            }
            let mut lookup = keys.finish_lookup().unwrap();
            let files = lookup.files.as_ref().unwrap();
            assert_eq!(
                disk.used(),
                files.data_bytes + files.count * INDEX_ENTRY_BYTES
            );
            for (key, value) in input.into_iter().rev() {
                assert_eq!(lookup.get(key).unwrap(), Some(*value), "seed={seed}");
                assert_eq!(lookup.get(key).unwrap(), Some(*value));
                assert!(lookup.buffer.capacity() <= lookup.max_key + ENTRY_HEADER_BYTES);
                assert!(lookup.last.as_ref().unwrap().key.capacity() <= lookup.max_key);
                assert!(lookup.pages.len() <= CACHE_PAGES);
                assert!(
                    lookup
                        .pages
                        .iter()
                        .map(|page| page.bytes.capacity())
                        .sum::<usize>()
                        <= CACHE_PAGES * PAGE_BYTES
                );
            }
            for missing in [b"missing".as_slice(), &[255; 2], &[0; 8193]] {
                assert_eq!(lookup.get(missing).unwrap(), None);
            }
            drop(lookup);
            directory.assert_clean(&disk);
        }
    }

    #[test]
    fn lookup_construction_enforces_exact_disk_quota_and_empty_lookup_needs_no_file() {
        let directory = TestDirectory::new("lookup-quota");
        for limit in [22, 23, 38, 39] {
            let disk = Arc::new(TemporaryDiskBudget::new(limit));
            let mut keys = sorter(&directory, disk.clone());
            keys.push_value(&[b"key"], 1, 5).unwrap();
            match keys.finish_lookup() {
                Ok(mut lookup) => {
                    assert_eq!(limit, 39);
                    assert_eq!(disk.used(), 39);
                    assert_eq!(lookup.get(b"key").unwrap(), Some((1, 5)));
                }
                Err(error) => {
                    assert!(limit < 39);
                    assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                }
            }
            directory.assert_clean(&disk);
        }
        let disk = Arc::new(TemporaryDiskBudget::new(0));
        let mut empty = sorter(&directory, disk.clone()).finish_lookup().unwrap();
        assert_eq!(empty.get(b"anything").unwrap(), None);
        assert!(empty.files.is_none());
        directory.assert_clean(&disk);
    }

    #[test]
    fn lookup_index_offsets_lengths_and_record_values_are_checked_before_use() {
        let directory = TestDirectory::new("lookup-corruption");
        let disk = Arc::new(TemporaryDiskBudget::new(1024));
        for damage in 0..7 {
            let mut keys = sorter(&directory, disk.clone());
            for (line, key) in [b"a", b"b", b"c"].into_iter().enumerate() {
                keys.push_value(&[key], line as u64 + 1, line as u64 + 10)
                    .unwrap();
            }
            let mut lookup = keys.finish_lookup().unwrap();
            let files = lookup.files.as_mut().unwrap();
            match damage {
                0 => {
                    files.index.rewind().unwrap();
                    files.index.write_all(&u64::MAX.to_le_bytes()).unwrap();
                }
                1 => {
                    files.index.seek(SeekFrom::Start(8)).unwrap();
                    files.index.write_all(&u32::MAX.to_le_bytes()).unwrap();
                }
                2 => {
                    files.index.seek(SeekFrom::Start(12)).unwrap();
                    files.index.write_all(&[0; 4]).unwrap();
                }
                3 => files.index.set_len(0).unwrap(),
                4 => files.data.set_len(20).unwrap(),
                5 => {
                    files.data.seek(SeekFrom::Start(12)).unwrap();
                    files.data.write_all(&u64::MAX.to_le_bytes()).unwrap();
                }
                _ => {
                    files
                        .data
                        .seek(SeekFrom::Start(ENTRY_HEADER_BYTES as u64))
                        .unwrap();
                    files.data.write_all(b"z").unwrap();
                }
            }
            assert_eq!(
                lookup.get(b"a").unwrap_err().code(),
                ErrorCode::ChecksumMismatch,
                "damage={damage}"
            );
            assert!(lookup.buffer.capacity() <= lookup.max_key + ENTRY_HEADER_BYTES);
            drop(lookup);
            directory.assert_clean(&disk);
        }
    }
}
