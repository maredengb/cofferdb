// Legacy records store the 64-bit FNV-1a checksum of their payload. Use the same
// streaming accumulator for complete slices and incremental reads so the result is
// independent of how a record was buffered.
pub fn checksum64(bytes: &[u8]) -> u64 {
    let mut hash = Checksum64::default();
    hash.update(bytes);
    hash.finish()
}

pub(crate) struct Checksum64(u64);

impl Default for Checksum64 {
    fn default() -> Self {
        // The FNV offset basis is also the checksum for empty input. Changing the
        // seed would invalidate stored checksums even when every payload byte
        // remained identical.
        Self(0xcbf29ce484222325)
    }
}

impl Checksum64 {
    pub(crate) fn update(&mut self, bytes: &[u8]) {
        for byte in bytes {
            // Mix each byte before multiplying by the FNV prime. Overflow is the
            // intended modulo-2^64 arithmetic in both debug and optimized builds.
            self.0 ^= *byte as u64;
            self.0 = self.0.wrapping_mul(0x100000001b3);
        }
    }

    pub(crate) fn finish(self) -> u64 {
        self.0
    }
}

// Some legacy headers store only the low 32 bits of this same FNV-1a result. Their
// format requires this truncation even though newer record formats also use a
// different 32-bit checksum, CRC-32C.
pub fn checksum32(bytes: &[u8]) -> u32 {
    (checksum64(bytes) & 0xffff_ffff) as u32
}

const CRC32C_TABLES: [[u32; 256]; 8] = {
    // Generate the slicing tables at compile time. The first table describes one
    // reflected byte step, and each later slice advances that contribution by an
    // additional trailing byte.
    let mut tables = [[0; 256]; 8];
    let mut i = 0;
    while i < 256 {
        let mut crc = i as u32;
        let mut bit = 0;
        while bit < 8 {
            // 0x82f63b78 is the reflected Castagnoli polynomial. The low bit
            // determines the feedback on each right shift, matching the byte order
            // used by the software update below.
            crc = (crc >> 1) ^ if crc & 1 == 1 { 0x82f63b78 } else { 0 };
            bit += 1;
        }
        tables[0][i] = crc;
        i += 1;
    }
    let mut slice = 1;
    while slice < 8 {
        i = 0;
        while i < 256 {
            let previous = tables[slice - 1][i];
            // Take one more zero-byte step from the preceding table. Slice seven
            // can then account for a byte followed by seven more bytes without
            // walking the intervening bytes at runtime.
            tables[slice][i] = (previous >> 8) ^ tables[0][(previous & 0xff) as usize];
            i += 1;
        }
        slice += 1;
    }
    tables
};

// Treat the register as a bit vector over GF(2). A set bit selects one transformed
// basis vector; XOR combines the selected vectors without carries.
const fn crc_matrix_times(matrix: &[u32; 32], mut value: u32) -> u32 {
    let mut result = 0;
    // Stop once all remaining input bits are zero. The all-zero vector therefore
    // maps to zero without reading any matrix columns, as required for a linear
    // CRC transformation.
    let mut bit = 0;
    while value != 0 {
        if value & 1 != 0 {
            result ^= matrix[bit];
        }
        value >>= 1;
        bit += 1;
    }
    result
}

// Each matrix advances a CRC by 2^n zero bytes, without initialization/finalization.
const CRC32C_BYTE_SHIFTS: [[u32; 32]; 64] = {
    let mut powers = [[0; 32]; 64];
    let mut bit = 0;
    // Apply a one-byte zero shift to each of the 32 bit basis values. These
    // columns form the base matrix from which every longer byte shift is derived.
    while bit < 32 {
        let value = 1u32 << bit;
        powers[0][bit] = (value >> 8) ^ CRC32C_TABLES[0][(value & 0xff) as usize];
        bit += 1;
    }
    let mut power = 1;
    while power < 64 {
        bit = 0;
        // Compose the previous matrix with itself one column at a time. Squaring
        // doubles the number of zero bytes advanced, which lets append_crc choose
        // powers by bits of a u64 length.
        while bit < 32 {
            powers[power][bit] = crc_matrix_times(&powers[power - 1], powers[power - 1][bit]);
            bit += 1;
        }
        power += 1;
    }
    powers
};

#[derive(Clone, Copy)]
pub(crate) struct Crc32c(u32);

// Keep the running CRC in its pre-finalization representation. Incremental updates
// and architecture-specific instructions consume that same register value.
impl Default for Crc32c {
    fn default() -> Self {
        // A fresh hasher uses this seed regardless of which backend processes its
        // first bytes.
        Self(u32::MAX)
    }
}

impl Crc32c {
    // Hardware dispatch is per target and feature availability. Each path
    // continues the existing CRC register and returns after consuming this slice,
    // so mixed chunk sizes have the same result as one update.
    pub(crate) fn update(&mut self, bytes: &[u8]) {
        #[cfg(target_arch = "aarch64")]
        if std::arch::is_aarch64_feature_detected!("crc") && {
            #[cfg(test)]
            {
                !test_backend::forced()
            }
            #[cfg(not(test))]
            {
                true
            }
        } {
            #[cfg(test)]
            test_backend::record_hardware();
            // SAFETY: The runtime probe establishes the function's CRC instruction requirement.
            self.0 = unsafe { crc32c_aarch64(self.0, bytes) };
            return;
        }
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("sse4.2") && {
            #[cfg(test)]
            {
                !test_backend::forced()
            }
            #[cfg(not(test))]
            {
                true
            }
        } {
            #[cfg(test)]
            test_backend::record_hardware();
            // SAFETY: The runtime probe establishes the function's SSE4.2 requirement.
            self.0 = unsafe { crc32c_x86_64(self.0, bytes) };
            return;
        }
        #[cfg(test)]
        test_backend::record_software();
        self.update_software(bytes);
    }

    fn update_software(&mut self, bytes: &[u8]) {
        // Eight independent table lookups avoid the byte-by-byte dependency chain.
        let mut blocks = bytes.chunks_exact(8);
        for block in &mut blocks {
            // Fold the old register into the lowest four input bytes in
            // little-endian order. Its first byte then has the longest distance to
            // advance and uses the highest-numbered table.
            let low = self.0 ^ u32::from_le_bytes([block[0], block[1], block[2], block[3]]);
            self.0 = CRC32C_TABLES[7][(low & 0xff) as usize]
                ^ CRC32C_TABLES[6][((low >> 8) & 0xff) as usize]
                ^ CRC32C_TABLES[5][((low >> 16) & 0xff) as usize]
                ^ CRC32C_TABLES[4][(low >> 24) as usize]
                ^ CRC32C_TABLES[3][block[4] as usize]
                ^ CRC32C_TABLES[2][block[5] as usize]
                ^ CRC32C_TABLES[1][block[6] as usize]
                ^ CRC32C_TABLES[0][block[7] as usize];
        }
        // Handle just the unconsumed tail with the one-byte recurrence. A short
        // input or the end of an incrementally supplied slice needs no padding.
        for &byte in blocks.remainder() {
            self.0 = (self.0 >> 8) ^ CRC32C_TABLES[0][((self.0 as u8) ^ byte) as usize];
        }
    }

    // Complement only when exposing the checksum. Repeated update calls must carry
    // the uncomplemented register between slices.
    pub(crate) fn finish(self) -> u32 {
        !self.0
    }

    // Checkpoint writers already know a verified extent's length and checksum.
    // Appending that pair preserves the extent's place after the preceding bytes
    // when constructing the record checksum.
    /// Append bytes represented by their finalized CRC and exact byte length.
    pub(crate) fn append_crc(&mut self, suffix_crc: u32, mut suffix_len: u64) {
        // An empty suffix contributes no bytes and cannot change the running
        // state. The caller is responsible for pairing every nonempty suffix CRC
        // with its exact encoded byte length.
        if suffix_len == 0 {
            return;
        }
        // Work on the finalized prefix CRC. Each table is the GF(2) transformation
        // for appending a power-of-two number of zero bytes. Applying the tables
        // selected by the bits of the suffix length shifts the prefix contribution
        // to its place in the concatenation.
        let mut crc = self.finish();
        let mut bit = 0;
        // Consume the length one bit at a time. The table has one power for each
        // bit of a u64, so even a suffix too large to address in memory can be
        // combined without scanning it.
        while suffix_len != 0 {
            if suffix_len & 1 != 0 {
                crc = crc_matrix_times(&CRC32C_BYTE_SHIFTS[bit], crc);
            }
            suffix_len >>= 1;
            bit += 1;
        }
        // XOR in the finalized checksum of the suffix, then complement once to
        // restore the running representation expected by update(). No prefix or
        // suffix bytes need to be read again.
        self.0 = !(crc ^ suffix_crc);
    }
}

#[cfg(test)]
pub(crate) mod test_backend {
    use std::cell::Cell;
    use std::marker::PhantomData;
    use std::rc::Rc;

    thread_local! {
        static SOFTWARE_ONLY: Cell<bool> = const { Cell::new(false) };
        static CALLS: Cell<(usize, usize)> = const { Cell::new((0, 0)) };
    }

    // The guard must be dropped on the thread whose dispatch it changes.
    pub(crate) struct SoftwareOnly {
        previous: bool,
        _same_thread: PhantomData<Rc<()>>,
    }

    impl SoftwareOnly {
        pub(crate) fn new() -> Self {
            Self {
                previous: SOFTWARE_ONLY.replace(true),
                _same_thread: PhantomData,
            }
        }
    }

    impl Drop for SoftwareOnly {
        fn drop(&mut self) {
            SOFTWARE_ONLY.set(self.previous);
        }
    }

    pub(crate) fn forced() -> bool {
        SOFTWARE_ONLY.get()
    }
    pub(crate) fn counts() -> (usize, usize) {
        CALLS.get()
    }
    pub(super) fn record_hardware() {
        let (software, hardware) = CALLS.get();
        CALLS.set((software, hardware + 1));
    }
    pub(super) fn record_software() {
        let (software, hardware) = CALLS.get();
        CALLS.set((software + 1, hardware));
    }

    #[test]
    fn software_guard_restores_nested_unwind_and_thread_state() {
        assert!(!forced());
        let outer = SoftwareOnly::new();
        assert!(forced());
        {
            let _inner = SoftwareOnly::new();
            assert!(forced());
        }
        assert!(forced());
        assert!(std::panic::catch_unwind(|| {
            let _inner = SoftwareOnly::new();
            panic!("exercise guard unwinding");
        })
        .is_err());
        assert!(forced());
        std::thread::spawn(|| {
            assert!(!forced());
            let _other = SoftwareOnly::new();
            assert!(forced());
        })
        .join()
        .unwrap();
        assert!(forced());
        drop(outer);
        assert!(!forced());
    }
}

/// # Safety
///
/// Dispatch here only on CPUs that support the `crc` target feature.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "crc")]
unsafe fn crc32c_aarch64(mut state: u32, bytes: &[u8]) -> u32 {
    use std::arch::aarch64::{__crc32cb, __crc32cd};
    // Read full groups from the byte slice rather than casting its address to a
    // word pointer. Little-endian decoding presents the bytes in stream order to
    // the Castagnoli instruction without imposing alignment.
    let mut blocks = bytes.chunks_exact(8);
    for block in &mut blocks {
        state = __crc32cd(state, u64::from_le_bytes(block.try_into().unwrap()));
    }
    // Feed the trailing bytes individually. Padding a short group to eight bytes
    // would change the checksum and break equivalence across update boundaries.
    for &byte in blocks.remainder() {
        state = __crc32cb(state, byte);
    }
    state
}

/// # Safety
///
/// Dispatch here only on CPUs that support the `sse4.2` target feature.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2")]
unsafe fn crc32c_x86_64(mut state: u32, bytes: &[u8]) -> u32 {
    use std::arch::x86_64::{_mm_crc32_u64, _mm_crc32_u8};
    // Every word exposed by chunks_exact contains eight bytes, even if the caller
    // passed an unaligned subslice. Converting the bytes to an integer avoids a
    // native word load through an aligned pointer.
    let mut blocks = bytes.chunks_exact(8);
    for block in &mut blocks {
        // The wide intrinsic takes and returns a 64-bit operand, but the running
        // CRC is 32 bits. Carry only that accumulator into the next group, without
        // applying the final complement here.
        state = _mm_crc32_u64(
            u64::from(state),
            u64::from_le_bytes(block.try_into().unwrap()),
        ) as u32;
    }
    // The tail continues from the previous instruction's running state. Empty
    // input returns that state unchanged, as does the software path.
    for &byte in blocks.remainder() {
        state = _mm_crc32_u8(state, byte);
    }
    state
}

// This convenience function uses the same initial state and final complement as
// incremental storage checks. Splitting an input across several update calls must
// yield the same result.
pub(crate) fn crc32c(bytes: &[u8]) -> u32 {
    let mut crc = Crc32c::default();
    crc.update(bytes);
    crc.finish()
}

#[cfg(test)]
mod crc_tests {
    use super::*;

    #[test]
    fn castagnoli_known_answers_and_incremental_updates() {
        assert_eq!(crc32c(b""), 0);
        assert_eq!(crc32c(b"123456789"), 0xe3069283);
        assert_eq!(crc32c(&[0; 32]), 0x8a9136aa);
        let mut crc = Crc32c::default();
        crc.update(b"1234");
        crc.update(b"56789");
        assert_eq!(crc.finish(), 0xe3069283);
        assert_ne!(crc32c(b"ab"), crc32c(b"ba"));
    }

    #[test]
    fn combined_crc_matches_independent_bitwise_reference_and_continued_writes() {
        let reference = |bytes: &[u8]| {
            let mut state = u32::MAX;
            for byte in bytes {
                state ^= u32::from(*byte);
                for _ in 0..8 {
                    state = (state >> 1) ^ if state & 1 != 0 { 0x82f63b78 } else { 0 };
                }
            }
            !state
        };
        for seed in 0..100 {
            let bytes = (0..(seed * 733) % 65537)
                .map(|index| (index * 31 + seed * 17) as u8)
                .collect::<Vec<_>>();
            let expected = reference(&bytes);
            for width in [1, 7, 64, 4093, 65536] {
                let mut crc = Crc32c::default();
                for part in bytes.chunks(width) {
                    crc.append_crc(reference(part), part.len() as u64);
                    crc.append_crc(0, 0);
                }
                assert_eq!(crc.finish(), expected, "seed {seed}, width {width}");
                crc.update(b"tail");
                assert_eq!(
                    crc.finish(),
                    reference(&[bytes.as_slice(), b"tail"].concat())
                );
            }
        }
    }

    #[test]
    fn combined_crc_preserves_associativity_across_all_u64_length_bits() {
        // An independent bit-at-a-time operator checks powers beyond allocatable inputs.
        let mut one_bit = [0; 32];
        one_bit[0] = 0x82f63b78;
        for (bit, entry) in one_bit.iter_mut().enumerate().skip(1) {
            *entry = 1 << (bit - 1);
        }
        for _ in 0..3 {
            one_bit = std::array::from_fn(|bit| crc_matrix_times(&one_bit, one_bit[bit]));
        }
        assert_eq!(one_bit, CRC32C_BYTE_SHIFTS[0]);
        let mut repeated = Crc32c::default();
        repeated.update(b"x");
        for (bit, matrix) in CRC32C_BYTE_SHIFTS.iter().enumerate() {
            assert_eq!(*matrix, one_bit);
            let len = 1u64 << bit;
            for seed in 0..100u32 {
                let prefix = seed.wrapping_mul(0x9e3779b9);
                let mut combined = Crc32c(!prefix);
                combined.append_crc(repeated.finish(), len);
                assert_eq!(
                    combined.finish(),
                    crc_matrix_times(&one_bit, prefix) ^ repeated.finish()
                );
            }
            if bit < 63 {
                let before = repeated;
                repeated.append_crc(before.finish(), len);
                let mut prefix = Crc32c::default();
                prefix.update(b"prefix");
                let mut split = prefix;
                split.append_crc(before.finish(), len);
                split.append_crc(before.finish(), len);
                prefix.append_crc(repeated.finish(), len * 2);
                assert_eq!(prefix.finish(), split.finish());
            }
            one_bit = std::array::from_fn(|bit| crc_matrix_times(&one_bit, one_bit[bit]));
        }
        let mut maximum = Crc32c::default();
        let mut length = 0u64;
        let mut block = Crc32c::default();
        block.update(b"x");
        for bit in 0..64 {
            maximum.append_crc(block.finish(), 1u64 << bit);
            length |= 1u64 << bit;
            if bit < 63 {
                block.append_crc(block.finish(), 1u64 << bit);
            }
        }
        assert_eq!(length, u64::MAX);
        let mut from_empty = Crc32c::default();
        from_empty.append_crc(maximum.finish(), length);
        assert_eq!(from_empty.finish(), maximum.finish());
    }

    #[test]
    fn sliced_crc_matches_independent_bitwise_reference_for_all_alignments_and_tails() {
        fn reference(bytes: &[u8]) -> u32 {
            let mut crc = u32::MAX;
            for byte in bytes {
                crc ^= u32::from(*byte);
                for _ in 0..8 {
                    crc = (crc >> 1) ^ if crc & 1 == 1 { 0x82f63b78 } else { 0 };
                }
            }
            !crc
        }
        for seed in 0..100u32 {
            let length = (seed as usize * 131) % 8193;
            let mut state = seed + 1;
            let bytes = (0..length + 8)
                .map(|_| {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    (state >> 24) as u8
                })
                .collect::<Vec<_>>();
            for alignment in 0..8 {
                let bytes = &bytes[alignment..alignment + length];
                let expected = reference(bytes);
                assert_eq!(crc32c(bytes), expected);
                for width in [1, 7, 8, 9, 63, 64, 65, 1024] {
                    let mut crc = Crc32c::default();
                    let mut portable = Crc32c::default();
                    for part in bytes.chunks(width) {
                        crc.update(part);
                        portable.update_software(part);
                    }
                    assert_eq!(portable.finish(), expected);
                    assert_eq!(
                        crc.finish(),
                        expected,
                        "seed={seed} alignment={alignment} width={width}"
                    );
                }
            }
        }
    }

    #[test]
    fn crc_backends_preserve_arbitrary_states_and_mixed_incremental_updates() {
        for seed in 0..100_u32 {
            let data = (0..4096 + seed as usize)
                .map(|index| (index as u32 * 137 + seed * 31) as u8)
                .collect::<Vec<_>>();
            for initial in [0, u32::MAX, seed.wrapping_mul(0x9e3779b9)] {
                let mut expected = initial;
                for &byte in &data {
                    expected ^= u32::from(byte);
                    for _ in 0..8 {
                        expected = (expected >> 1) ^ if expected & 1 != 0 { 0x82f63b78 } else { 0 };
                    }
                }
                let mut mixed = Crc32c(initial);
                for (index, part) in data.chunks(1 + seed as usize).enumerate() {
                    if index % 2 == 0 {
                        mixed.update(part);
                    } else {
                        mixed.update_software(part);
                    }
                    let before = mixed.0;
                    mixed.update(&[]);
                    mixed.update_software(&[]);
                    assert_eq!(mixed.0, before);
                }
                assert_eq!(mixed.finish(), !expected, "seed={seed}, initial={initial}");
            }
        }
    }
}

#[cfg(test)]
pub fn sha256_hex(bytes: &[u8]) -> String {
    digest_hex(sha256(bytes))
}

pub(crate) fn digest_hex(digest: [u8; 32]) -> String {
    // Two lowercase hex digits encode each digest byte, including leading zeroes.
    // Reserve the complete representation so this owned result can be rendered in
    // place.
    let mut out = String::with_capacity(64);
    append_digest_hex(&mut out, digest);
    out
}

pub(crate) fn append_digest_hex(out: &mut String, digest: [u8; 32]) {
    // String accepts each formatted character. The fallible formatter entry point
    // below is also used by bounded writers, where the same conversion can fail.
    write_digest_hex(out, digest).expect("writing to String is infallible");
}

// Render into the caller's formatter so a bounded output can append the digest
// without building an intermediate String. A formatter error can occur after
// earlier digits have been written.
pub(crate) fn write_digest_hex(
    out: &mut (impl std::fmt::Write + ?Sized),
    digest: [u8; 32],
) -> std::fmt::Result {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    for byte in digest {
        out.write_char(HEX[usize::from(byte >> 4)] as char)?;
        out.write_char(HEX[usize::from(byte & 15)] as char)?;
    }
    Ok(())
}

pub fn sha256(bytes: &[u8]) -> [u8; 32] {
    let mut hash = Sha256::default();
    hash.update(bytes);
    hash.finish()
}

pub(crate) fn sha256_reader(reader: impl std::io::Read) -> std::io::Result<String> {
    // The convenience reader owns a fixed scratch buffer. Snapshot verification
    // can call the buffer-taking variant to reuse storage it already admitted.
    sha256_reader_with_buffer(reader, &mut [0; 64 * 1024])
}

pub(crate) fn sha256_reader_with_buffer(
    mut reader: impl std::io::Read,
    buffer: &mut [u8],
) -> std::io::Result<String> {
    // A zero-length read normally signals EOF. An empty scratch buffer could
    // produce the same result while data remains, so require at least one byte.
    assert!(!buffer.is_empty(), "hash buffer must be nonempty");
    let mut hash = Sha256::default();
    loop {
        // Retry interrupted reads without updating the hash. On any other read
        // error return the error instead of reporting a digest for an incomplete
        // stream.
        let count = match reader.read(buffer) {
            Err(err) if err.kind() == std::io::ErrorKind::Interrupted => continue,
            result => result?,
        };
        // Stop on EOF, including a genuinely empty stream. Finalization still
        // emits SHA-256 for an empty message when no update ran.
        if count == 0 {
            break;
        }
        // Only the count returned by Read is fresh input. Bytes beyond it may
        // still be left over from an earlier full buffer.
        hash.update(&buffer[..count]);
    }
    Ok(digest_hex(hash.finish()))
}

pub(crate) struct Sha256Reader<R> {
    // Input stream whose bytes feed the digest as they are read. EOF contributes no
    // additional data.
    source: R,
    // Incremental SHA-256 state for bytes already returned by the source. The
    // digest can therefore name the same input that was parsed.
    hash: Sha256,
}

impl<R> Sha256Reader<R> {
    pub(crate) fn new(source: R) -> Self {
        Self {
            source,
            hash: Sha256::default(),
        }
    }

    // Finalizing this wrapper does not drain its source. A caller identifying an
    // entire input must have read it through EOF before taking the digest.
    pub(crate) fn finish(self) -> String {
        digest_hex(self.hash.finish())
    }
}

impl<R: std::io::Read> std::io::Read for Sha256Reader<R> {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        // Hash only successful reads, in exactly the slices made available to the
        // consumer. A parser and its digest can then share a single pass over the
        // stream.
        let count = self.source.read(buffer)?;
        self.hash.update(&buffer[..count]);
        Ok(count)
    }
}

pub(crate) struct Sha256Writer<W> {
    // Output sink receiving the bytes to be hashed, including any short write
    // accepted by the sink.
    destination: W,
    // Incremental digest of the bytes the destination accepted. An unwritten suffix
    // must not enter the digest twice during a retry.
    hash: Sha256,
}

impl<W: std::io::Write> Sha256Writer<W> {
    pub(crate) fn new(destination: W) -> Self {
        Self {
            destination,
            hash: Sha256::default(),
        }
    }

    pub(crate) fn finish(mut self) -> std::io::Result<String> {
        // Flush the destination before releasing its digest. A sink error at this
        // stage must still reach the caller even though earlier writes were
        // accepted.
        self.destination.flush()?;
        Ok(digest_hex(self.hash.finish()))
    }
}

impl<W: std::io::Write> std::io::Write for Sha256Writer<W> {
    // Hash just the byte count accepted by the destination. A short write leaves
    // the rest for the caller to retry, so the digest still corresponds to the
    // emitted stream.
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let count = self.destination.write(bytes)?;
        self.hash.update(&bytes[..count]);
        Ok(count)
    }

    // Flushing affects the destination only. The digest already includes bytes
    // accepted into any lower-level buffer.
    fn flush(&mut self) -> std::io::Result<()> {
        self.destination.flush()
    }
}

pub(crate) struct Sha256 {
    // Eight running compression words after all complete input blocks processed so
    // far.
    state: [u32; 8],
    // Storage for a partial compression block carried between updates and for the
    // final padding.
    buffer: [u8; 64],
    // Number of input bytes currently buffered in the partial block. Only this
    // prefix belongs to the message.
    used: usize,
    // Total input bytes accepted before padding. Finalization encodes the message
    // length in bits.
    byte_count: u64,
}

impl Default for Sha256 {
    // Start with the SHA-256 initial chaining words. A new stream has no pending
    // input and a length of zero, independent of any previous writer.
    fn default() -> Self {
        Self {
            state: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
                0x5be0cd19,
            ],
            buffer: [0; 64],
            used: 0,
            byte_count: 0,
        }
    }
}

impl Sha256 {
    // Accept arbitrary fragment boundaries. Each byte contributes once to the
    // total length, whether compressed directly or staged across multiple updates.
    pub(crate) fn update(&mut self, mut bytes: &[u8]) {
        // Use a wrapping byte count so conversion to the standard 64-bit
        // bit-length field also has defined modulo arithmetic.
        self.byte_count = self.byte_count.wrapping_add(bytes.len() as u64);
        // Complete any pending block first. A short fragment can only fill part of
        // that block and must leave the loaded state waiting for the next update.
        if self.used != 0 {
            let count = bytes.len().min(64 - self.used);
            self.buffer[self.used..self.used + count].copy_from_slice(&bytes[..count]);
            self.used += count;
            bytes = &bytes[count..];
            // Return before processing direct blocks whenever the carry buffer is
            // still incomplete. Its active prefix spans the previous update and
            // this fragment.
            if self.used != 64 {
                return;
            }
            // Reset the tail after compression so subsequent full blocks can
            // borrow the caller's slice directly.
            sha256_compress(&mut self.state, &self.buffer);
            self.used = 0;
        }
        // Process complete blocks without copying them through the carry buffer.
        // Keep only the exact suffix for the next call.
        let mut blocks = bytes.chunks_exact(64);
        for block in &mut blocks {
            sha256_compress(&mut self.state, block);
        }
        // Copy the uncompressed suffix into the front of the carry block. Bytes
        // beyond used can be stale and will be overwritten by later input or
        // padding.
        let tail = blocks.remainder();
        self.buffer[..tail.len()].copy_from_slice(tail);
        self.used = tail.len();
    }

    // Consume the running state to append padding exactly once. The stored byte
    // count describes the original message and excludes these extra blocks.
    pub(crate) fn finish(mut self) -> [u8; 32] {
        // Append the high bit immediately after the last input byte and zero every
        // remaining byte before inserting the length.
        self.buffer[self.used] = 0x80;
        self.buffer[self.used + 1..].fill(0);
        // The last eight bytes of the final block hold the original bit length. If
        // the padding marker has reached that field, compress it in this block and
        // write the length into a second, initially zero, block.
        if self.used >= 56 {
            sha256_compress(&mut self.state, &self.buffer);
            self.buffer.fill(0);
        }
        // Encode the bit count in big-endian form at the fixed trailing offset.
        // The digest likewise serializes each final word in big-endian order.
        self.buffer[56..].copy_from_slice(&self.byte_count.wrapping_mul(8).to_be_bytes());
        sha256_compress(&mut self.state, &self.buffer);
        // The digest representation is 32 bytes. Serializing the words explicitly
        // keeps the output the same on hosts with either endianness.
        let mut out = [0; 32];
        for (i, word) in self.state.iter().enumerate() {
            out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
        }
        out
    }
}

// Compress one complete 64-byte block into the eight chaining words. Callers
// supply blocks from chunks_exact, a filled carry buffer or final padding.
fn sha256_compress(h: &mut [u32; 8], chunk: &[u8]) {
    // These are the SHA-256 round constants in schedule order. Preserve the
    // positions so the compression function stays compatible with standard
    // digests.
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];

    // Build the message schedule from the block's first sixteen big-endian words.
    // The remaining words depend on earlier schedule entries, not on input bytes
    // outside this block.
    let mut w = [0u32; 64];
    for (i, word) in w.iter_mut().take(16).enumerate() {
        let j = i * 4;
        *word = u32::from_be_bytes([chunk[j], chunk[j + 1], chunk[j + 2], chunk[j + 3]]);
    }
    // Extend the schedule in ascending order using the small-sigma rotations and
    // shifts. Combining words with wrapping addition implements arithmetic modulo
    // 2^32.
    for i in 16..64 {
        let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
        let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16]
            .wrapping_add(s0)
            .wrapping_add(w[i - 7])
            .wrapping_add(s1);
    }

    // Initialize the eight round registers from the previous chaining state.
    // Retain the original state so it can be added back after the last round.
    let mut a = h[0];
    let mut b = h[1];
    let mut c = h[2];
    let mut d = h[3];
    let mut e = h[4];
    let mut f = h[5];
    let mut g = h[6];
    let mut hh = h[7];

    for i in 0..64 {
        // The first round term combines the eighth register, the upper-case sigma
        // of e, choose(e, f, g), the round constant and the current message word.
        let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        // For each bit, choose takes f where e has a one and g where e has a zero.
        // Using bitwise operators performs all 32 selections together.
        let ch = (e & f) ^ ((!e) & g);
        let temp1 = hh
            .wrapping_add(s1)
            .wrapping_add(ch)
            .wrapping_add(K[i])
            .wrapping_add(w[i]);
        // Use the other upper-case sigma function on a. Its sum with majority
        // contributes to the next a register, while e receives the first term and
        // the old d.
        let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        // For each bit, majority is set when at least two of a, b, and c have that
        // bit. It contributes to the new a word through the second round term.
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let temp2 = s0.wrapping_add(maj);
        // Move the old registers in descending order so the assignments cannot
        // overwrite a value that still needs to be carried to the next register.
        hh = g;
        g = f;
        f = e;
        e = d.wrapping_add(temp1);
        d = c;
        c = b;
        b = a;
        a = temp1.wrapping_add(temp2);
    }

    // Feed the completed registers back into the previous chaining state. The
    // result becomes the starting state for the next block or the final digest.
    h[0] = h[0].wrapping_add(a);
    h[1] = h[1].wrapping_add(b);
    h[2] = h[2].wrapping_add(c);
    h[3] = h[3].wrapping_add(d);
    h[4] = h[4].wrapping_add(e);
    h[5] = h[5].wrapping_add(f);
    h[6] = h[6].wrapping_add(g);
    h[7] = h[7].wrapping_add(hh);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufWriter, Read, Write};

    #[test]
    fn incremental_legacy_checksum_matches_flat_reference_across_chunk_boundaries() {
        for seed in 0..100usize {
            let bytes = (0..seed * 257)
                .map(|index| (index * 31 + seed) as u8)
                .collect::<Vec<_>>();
            let expected = bytes.iter().fold(0xcbf29ce484222325u64, |hash, byte| {
                (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
            });
            assert_eq!(checksum64(&bytes), expected);
            for width in [1, 2, 7, 64, 8192] {
                let mut actual = Checksum64::default();
                for chunk in bytes.chunks(width) {
                    actual.update(chunk);
                }
                assert_eq!(actual.finish(), expected);
            }
        }
    }

    #[test]
    fn hexadecimal_output_matches_formatter_without_growing_preallocated_storage() {
        for seed in 0..100_u64 {
            let mut state = seed + 1;
            let digest = std::array::from_fn(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                (state >> 32) as u8
            });
            let expected = digest
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>();
            let mut output = String::with_capacity(64);
            let pointer = output.as_ptr();
            append_digest_hex(&mut output, digest);
            assert_eq!(output.as_ptr(), pointer);
            assert_eq!(output.capacity(), 64);
            assert_eq!(output, expected);
            assert_eq!(digest_hex(digest), expected);
        }
    }

    #[test]
    fn hashing_writer_counts_only_accepted_bytes_across_short_writes_and_buffer_boundaries() {
        struct ShortWriter<'a> {
            bytes: &'a mut Vec<u8>,
            maximum: usize,
            interrupt: bool,
        }
        impl Write for ShortWriter<'_> {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                if std::mem::take(&mut self.interrupt) {
                    return Err(std::io::ErrorKind::Interrupted.into());
                }
                let count = bytes.len().min(self.maximum);
                self.bytes.extend_from_slice(&bytes[..count]);
                Ok(count)
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        for seed in 0..100 {
            let bytes = (0..(seed * 733) % 10001)
                .map(|index| (index * 31 + seed * 17) as u8)
                .collect::<Vec<_>>();
            for (capacity, maximum) in [(1, 1), (7, 13), (8192, 4096)] {
                let mut written = Vec::new();
                let sink = ShortWriter {
                    bytes: &mut written,
                    maximum,
                    interrupt: true,
                };
                let mut output = BufWriter::with_capacity(capacity, Sha256Writer::new(sink));
                for part in bytes.chunks(73) {
                    output.write_all(part).unwrap();
                    output.write_all(&[]).unwrap();
                }
                let hash = output
                    .into_inner()
                    .map_err(|error| error.into_error())
                    .unwrap()
                    .finish()
                    .unwrap();
                assert_eq!(written, bytes, "seed {seed}");
                assert_eq!(hash, sha256_hex(&bytes), "seed {seed}");
            }
        }
        let mut output = Sha256Writer::new(std::io::sink());
        for _ in 0..1000 {
            output.write_all(&[b'a'; 1000]).unwrap();
        }
        assert_eq!(
            output.finish().unwrap(),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
        );
    }

    #[test]
    fn hashing_writer_preserves_write_and_flush_failures_without_hashing_rejected_bytes() {
        struct FailingWriter {
            remaining: usize,
            zero: bool,
            flush_error: bool,
        }
        impl Write for FailingWriter {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                if self.remaining == 0 {
                    return if self.zero {
                        Ok(0)
                    } else {
                        Err(std::io::ErrorKind::PermissionDenied.into())
                    };
                }
                let count = bytes.len().min(self.remaining).min(7);
                self.remaining -= count;
                Ok(count)
            }
            fn flush(&mut self) -> std::io::Result<()> {
                if self.flush_error {
                    Err(std::io::ErrorKind::Other.into())
                } else {
                    Ok(())
                }
            }
        }
        let bytes = [0xab; 101];
        let mut buffered = BufWriter::with_capacity(
            8192,
            Sha256Writer::new(FailingWriter {
                remaining: 13,
                zero: false,
                flush_error: false,
            }),
        );
        buffered.write_all(&bytes).unwrap();
        let failure = buffered.into_inner().err().unwrap();
        assert_eq!(failure.error().kind(), std::io::ErrorKind::PermissionDenied);
        let (writer, pending) = failure.into_inner().into_parts();
        assert_eq!(pending.unwrap(), bytes[13..]);
        assert_eq!(writer.finish().unwrap(), sha256_hex(&bytes[..13]));
        for accepted in 0..100 {
            for zero in [false, true] {
                let mut output = Sha256Writer::new(FailingWriter {
                    remaining: accepted,
                    zero,
                    flush_error: false,
                });
                let error = output.write_all(&bytes).unwrap_err();
                assert_eq!(
                    error.kind(),
                    if zero {
                        std::io::ErrorKind::WriteZero
                    } else {
                        std::io::ErrorKind::PermissionDenied
                    }
                );
                assert_eq!(output.finish().unwrap(), sha256_hex(&bytes[..accepted]));
            }
        }
        let mut output = Sha256Writer::new(FailingWriter {
            remaining: bytes.len(),
            zero: false,
            flush_error: true,
        });
        output.write_all(&bytes).unwrap();
        assert_eq!(
            output.flush().unwrap_err().kind(),
            std::io::ErrorKind::Other
        );
        assert_eq!(
            output.finish().unwrap_err().kind(),
            std::io::ErrorKind::Other
        );
    }

    #[test]
    fn sha256_matches_known_vectors() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn streaming_sha256_matches_known_million_byte_vector() {
        let mut hash = Sha256::default();
        for _ in 0..1000 {
            hash.update(&[b'a'; 1000]);
        }
        assert_eq!(
            digest_hex(hash.finish()),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
        );
    }

    #[test]
    fn streaming_sha256_handles_padding_boundaries_and_arbitrary_fragments() {
        for len in [0, 1, 55, 56, 63, 64, 65, 119, 120, 127, 128, 129, 4097] {
            let bytes = (0..len).map(|i| (i * 31) as u8).collect::<Vec<_>>();
            let expected = sha256(&bytes);
            for size in [1, 3, 17, 64, 73, 1024] {
                let mut hash = Sha256::default();
                for chunk in bytes.chunks(size) {
                    hash.update(chunk);
                    hash.update(&[]);
                }
                assert_eq!(hash.finish(), expected, "len={len}, fragment={size}");
            }
            assert_eq!(
                sha256_reader(std::io::Cursor::new(&bytes)).unwrap(),
                digest_hex(expected)
            );
            for size in [1, 7, 64, 1024] {
                let mut reader = Sha256Reader::new(std::io::Cursor::new(&bytes));
                let mut buffer = vec![0; size];
                while reader.read(&mut buffer).unwrap() != 0 {}
                assert_eq!(reader.finish(), digest_hex(expected));
            }
        }
    }
}
