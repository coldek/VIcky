//! Which title memory the title wrote since a host copy took it (decision
//! 0007, "Copies of title memory"). The runtime reserves title GPU memory
//! with write watching (`victory-hle` xmem), so the kernel keeps a dirty bit
//! per 4 KiB page; asking for a range's written pages also clears them. Host
//! copies of the same memory can overlap (nested buffers, textures in
//! buffers), so what one asks is kept here for the others: a journal of
//! each page's last write, numbered by the asking.
//!
//! Physical-page views (xmem's sections) cannot be write-watched. There a
//! copy leaves each 64 KiB block it took read-only, and our vectored handler
//! takes the first write to it: it notes the block as written and gives the
//! write access back. The runtime tells us when it maps or unmaps views
//! ([`title_memory_remapped`]), which resets their protection. Other
//! mappings reported by the runtime are guarded and dirtied together. Blocks found
//! wholly uncommitted are skipped while they remain so; one query can
//! confirm a whole region, including ordinary reservations between views.
//!
//! shortcut: writes into physical pages through another view (xmem's
//! `write_physical`, DirectStorage's page destinations) are not seen at
//! the guarded addresses; they go to the texture pool, whose textures are
//! hashed, not guarded. A system call writing into a guarded block fails
//! (none is known to).

use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;
use std::sync::Mutex;

/// The unit tracked and copied: 64 KiB, the console's page (the kernel's
/// 4 KiB pages fold into it). Finer units made the checks of the world's
/// big buffers cost more than the copies they saved.
pub(crate) const PAGE: usize = 64 << 10;
/// The kernel's page, the unit of its answers.
const KERNEL_PAGE: usize = 4096;

struct Journal {
    /// Numbers the askings and mapping changes; 0 is "never".
    asked: u64,
    /// Page number → the asking that last found it written.
    written: BTreeMap<usize, u64>,
    remapped: Remaps,
    /// Room for the kernel's answer.
    found: Vec<usize>,
    /// Page ranges asked ahead for a submission's refreshes ([`ask_ahead`]):
    /// `writes` answers for a range inside one from the journal.
    ahead: Vec<(usize, usize)>,
    /// The ahead asking stands for a whole submission ([`ask_once`]): no
    /// other ahead asking is made, and `ahead` stays, until
    /// [`ask_once_end`].
    kept: bool,
}

static JOURNAL: Mutex<Journal> = Mutex::new(Journal {
    asked: 0,
    written: BTreeMap::new(),
    remapped: Remaps(BTreeMap::new()),
    found: Vec::new(),
    ahead: Vec::new(),
    kept: false,
});

/// Disjoint remapped page ranges and their generations, keyed by start page.
/// Even an uncommitted reservation can still answer `GetWriteWatch` without
/// reporting a write. Keep its mapping change independently of the old data's
/// write stamps so every reader can see it.
struct Remaps(BTreeMap<usize, (usize, u64)>);

const MAX_REMAP_RANGES: usize = 8 * 1024;

impl Remaps {
    fn record(&mut self, range: Range<usize>, now: u64) {
        let start = range.start;
        let end = range.end;
        if let Some((to, stamp)) = self.0.range_mut(..start).next_back().map(|(_, r)| r)
            && *to > start
        {
            let (old_end, old_stamp) = (*to, *stamp);
            *to = start;
            if old_end > end {
                self.0.insert(end, (old_end, old_stamp));
            }
        }
        while let Some((&from, &(to, stamp))) = self.0.range(range.clone()).next() {
            self.0.remove(&from);
            if to > end {
                self.0.insert(end, (to, stamp));
            }
        }
        self.0.insert(start, (end, now));
        if self.0.len() > MAX_REMAP_RANGES {
            self.compact();
        }
    }

    fn compact(&mut self) {
        // Pair adjacent older ranges so the newest quarter retain their
        // boundaries. Use the later stamp over each pair and its gap so
        // older readers can never miss a change. Removing at least a
        // quarter of the entries amortizes this pass over later remaps.
        let mut stamps: Vec<_> = self.0.values().map(|&(_, stamp)| stamp).collect();
        let cutoff = stamps.len() * 3 / 4;
        let newest = *stamps.select_nth_unstable(cutoff).1;
        let mut previous: Option<(usize, u64)> = None;
        let mut merged = Vec::new();
        self.0.retain(|&from, &mut (to, stamp)| {
            if stamp > newest {
                previous = None;
                return true;
            }
            if let Some((start, earlier)) = previous.take() {
                merged.push((start, (to, earlier.max(stamp))));
                return false;
            }
            previous = Some((from, stamp));
            true
        });
        self.0.extend(merged);
    }

    fn within(&self, first: usize, last: usize) -> impl Iterator<Item = (Range<usize>, u64)> + '_ {
        self.0
            .range(..first)
            .next_back()
            .filter(|&(_, &(end, _))| end > first)
            .into_iter()
            .chain(self.0.range(first..last))
            .map(move |(&from, &(to, stamp))| (from.max(first)..to.min(last), stamp))
    }
}

/// The write and mapping changes known after an asking. Remaps stay as
/// ranges even when a large reservation changes in a single call.
pub(crate) struct Changes {
    pub(crate) now: u64,
    written: Vec<(usize, u64)>,
    remapped: Vec<(Range<usize>, u64)>,
}

impl Journal {
    fn changes(&self, first: usize, last: usize) -> Changes {
        Changes {
            now: self.asked,
            written: self
                .written
                .range(first..last)
                .map(|(&p, &w)| (p, w))
                .collect(),
            remapped: self.remapped.within(first, last).collect(),
        }
    }
}

impl Changes {
    pub(crate) fn changed_since(&self, at: u64) -> bool {
        self.written.iter().any(|&(_, stamp)| stamp > at)
            || self.remapped.iter().any(|&(_, stamp)| stamp > at)
    }

    pub(crate) fn remapped_since(&self, page: usize, at: u64) -> bool {
        let i = self.remapped.partition_point(|(range, _)| range.end <= page);
        self.remapped
            .get(i)
            .is_some_and(|(range, stamp)| range.contains(&page) && *stamp > at)
    }

    /// Pages in address order, with the later of their write and remap stamps.
    pub(crate) fn pages(&self) -> impl Iterator<Item = (usize, u64)> + '_ {
        let mut written = self.written.iter().copied().peekable();
        let mut remapped = self
            .remapped
            .iter()
            .flat_map(|(range, stamp)| range.clone().map(move |page| (page, *stamp)))
            .peekable();
        std::iter::from_fn(move || match (written.peek().copied(), remapped.peek().copied()) {
            (Some((page, w)), Some((p, r))) if page == p => {
                written.next();
                remapped.next();
                Some((page, w.max(r)))
            }
            (Some((page, _)), Some((p, _))) if page < p => written.next(),
            (_, Some(_)) => remapped.next(),
            _ => written.next(),
        })
    }
}

/// Ranges of one submission closer than this are asked about together:
/// walking the pages between them costs less than another kernel call (a
/// call that resets written pages costs 30-70 microseconds whatever the
/// range).
const AHEAD_GAP: usize = (8 << 20) / PAGE;
/// The most one asking covers (its answer needs 8 bytes per 4 KiB page).
const AHEAD_MAX: usize = (1 << 30) / PAGE;

/// Asks the kernel about many ranges `(start, end)` at once, before the
/// [`writes`] of each: ranges close together are one asking, and `writes`
/// then answers from the journal until [`ask_ahead_done`]. The world's
/// submissions refresh thousands of host copies (6,664 askings a frame,
/// 456 ms, in run 14). Ranges that are not write-watched, or lie alone, are
/// left to `writes`.
pub(crate) fn ask_ahead(ranges: &[(usize, usize)]) {
    let mut pieces: Vec<(usize, usize)> = ranges
        .iter()
        .filter(|r| r.0 < r.1)
        .map(|&(start, end)| (start / PAGE, end.div_ceil(PAGE)))
        .collect();
    if pieces.len() < 2 {
        return;
    }
    pieces.sort_unstable();
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    let j = &mut *j;
    // The submission's one asking stands ([`ask_once`]): what it does not
    // hold is left to `writes`.
    if j.kept {
        return;
    }
    j.asked += 1;
    j.ahead.clear();
    let mut i = 0;
    while i < pieces.len() {
        let first = pieces[i].0;
        let mut last = pieces[i].1;
        let mut k = i + 1;
        while k < pieces.len()
            && pieces[k].0 <= last + AHEAD_GAP
            && last.max(pieces[k].1) - first <= AHEAD_MAX
        {
            last = last.max(pieces[k].1);
            k += 1;
        }
        ask_run(j, &pieces[i..k]);
        i = k;
    }
}

/// Since the last frame line: the kernel askings that took a millisecond
/// or more, their microseconds, and the longest asking's. An asking walks
/// page tables, some ten microseconds for most ranges; in run 75 the 2,644
/// askings of a frame took 219 ms in the lines where the pass over the
/// heap's views ran long, 25-50 ms in the others. These numbers say
/// whether a few askings wait for something or all of them are slow.
#[cfg(feature = "diagnostics")]
static SLOW: [std::sync::atomic::AtomicU64; 3] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 3];

#[cfg(feature = "diagnostics")]
fn slow(us: u64) {
    use std::sync::atomic::Ordering::Relaxed;
    if us >= 1000 {
        SLOW[0].fetch_add(1, Relaxed);
        SLOW[1].fetch_add(us, Relaxed);
    }
    SLOW[2].fetch_max(us, Relaxed);
}

/// [`SLOW`], taken.
#[cfg(feature = "diagnostics")]
pub(crate) fn slow_asks() -> [u64; 3] {
    [0, 1, 2].map(|i| SLOW[i].swap(0, std::sync::atomic::Ordering::Relaxed))
}

/// The submission's refreshes are done: `writes` asks the kernel again.
pub(crate) fn ask_ahead_done() {
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    if !j.kept {
        j.ahead.clear();
    }
}

/// Whether a submission asks the kernel once, before its first refresh,
/// about every range its lists refresh (`VICTORY_ASK_ONCE=1`; off by
/// default).
///
/// A submission's refreshes are done in groups, between the title's own
/// buffer copies (each copy flushes the refreshes recorded before it), and
/// every group asked the kernel again: 2,644 askings a frame, 70 a
/// submission. With the switch the journal answers all of them from one
/// asking made when the submission starts.
///
/// What changes: memory the title's other threads write while the
/// submission is being prepared is seen by the next submission's asking,
/// not by this one's later groups. What this submission's lists read was
/// written before they were submitted, so before the asking. Our own
/// writes into title memory while preparing are told to the journal
/// ([`wrote`]) or end the asking ([`ask_once_end`]).
pub(crate) fn once() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // On by default since run 81 (with card buffers: 375-450 ms a frame
    // against 573-647, the picture unchanged over four minutes);
    // `VICTORY_ASK_ONCE=0` asks per range, as before.
    *ON.get_or_init(|| std::env::var_os("VICTORY_ASK_ONCE").is_none_or(|v| v != "0"))
}

/// With [`once`]: asks about `ranges` (everything a submission's lists
/// refresh) as [`ask_ahead`] does, and lets that asking stand until
/// [`ask_once_end`].
pub(crate) fn ask_once(ranges: &[(usize, usize)]) {
    if !once() {
        return;
    }
    ask_once_end();
    ask_ahead(ranges);
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    let j = &mut *j;
    // Another reader or remap may have advanced the journal since
    // `ask_ahead` released the lock. The remaining asks get a new stamp.
    j.asked += 1;
    // `ask_ahead` leaves a range that lies alone to `writes`, which asks
    // once per flush: with one asking for the submission that was every
    // such range asked again at each of its 70 flushes (run 76: 2,300
    // askings a frame with the switch on, as without it). They are asked
    // here, once; what is not write-watched stays with `writes`.
    for &(start, end) in ranges {
        if start >= end {
            continue;
        }
        let (first, last) = (start / PAGE, end.div_ceil(PAGE));
        if !j.ahead.iter().any(|&(a, b)| a <= first && last <= b) && ask_kernel(j, first, last) {
            j.ahead.push((first, last));
        }
    }
    j.kept = !j.ahead.is_empty();
}

/// The submission's one asking stands no longer: its refreshes are done,
/// or title memory was written in a way the journal was not told.
pub(crate) fn ask_once_end() {
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    if j.kept {
        j.kept = false;
        j.ahead.clear();
    }
}

/// We wrote title memory `[at, at + len)` while a submission's one asking
/// stands (the title's buffer copies, done at submission): its pages count
/// as written after that asking, so the refreshes that follow copy them.
pub(crate) fn wrote(at: usize, len: usize) {
    if len == 0 {
        return;
    }
    let end = at.saturating_add(len);
    open(&mut guards(), at, end);
    write_ahead(at, end);
}

fn write_ahead(at: usize, end: usize) {
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    if j.ahead.is_empty() || at >= end {
        return;
    }
    j.asked += 1;
    let now = j.asked;
    let (first, last) = (at / PAGE, end.div_ceil(PAGE));
    let Journal { ahead, written, .. } = &mut *j;
    for &(a, b) in ahead.iter() {
        for page in a.max(first)..b.min(last) {
            written.insert(page, now);
        }
    }
}

/// One kernel asking for `pieces` (page ranges in order); by halves if it
/// fails, as a run across two reservations or into memory that is not
/// write-watched does. A piece alone is left to [`writes`].
fn ask_run(j: &mut Journal, pieces: &[(usize, usize)]) {
    if pieces.len() < 2 {
        return;
    }
    let first = pieces[0].0;
    let last = pieces.iter().map(|p| p.1).max().unwrap_or(first);
    if ask_kernel(j, first, last) {
        j.ahead.push((first, last));
        return;
    }
    let (a, b) = pieces.split_at(pieces.len() / 2);
    ask_run(j, a);
    ask_run(j, b);
}

/// Asks the kernel for the written pages of pages `[first, last)` and notes
/// them in the journal under the current asking; false (and nothing changed)
/// if the range is not write-watched memory of one reservation.
fn ask_kernel(j: &mut Journal, first: usize, last: usize) -> bool {
    use windows_sys::Win32::System::Memory::GetWriteWatch;
    const WRITE_WATCH_FLAG_RESET: u32 = 1;
    let mut count = (last - first) * PAGE / KERNEL_PAGE;
    j.found.resize(count.max(j.found.len()), 0);
    let mut granularity = 0;
    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    // SAFETY: the answer buffer holds `count` entries.
    let failed = unsafe {
        GetWriteWatch(
            WRITE_WATCH_FLAG_RESET,
            (first * PAGE) as *const _,
            (last - first) * PAGE,
            j.found.as_mut_ptr().cast(),
            &mut count,
            &mut granularity,
        )
    } != 0;
    if failed {
        return false;
    }
    #[cfg(feature = "diagnostics")]
    {
        use crate::copystat::{C, add};
        add(C::Watches, 1);
        add(C::WatchUs, started.elapsed().as_micros() as u64);
        slow(started.elapsed().as_micros() as u64);
    }
    let now = j.asked;
    for k in 0..count {
        let page = j.found[k] / PAGE;
        j.written.insert(page, now);
    }
    true
}

/// What [`writes`] would say of `[start, end)` from the journal alone, if
/// the range lies in one asked ahead ([`ask_ahead`], [`ask_once`]): the
/// write and mapping changes known to the current asking. `None` if the
/// kernel would have to be asked.
pub(crate) fn asked(start: usize, end: usize) -> Option<Changes> {
    if start >= end {
        return None;
    }
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    let j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    j.ahead
        .iter()
        .any(|&(a, b)| a <= first && last <= b)
        .then(|| j.changes(first, last))
}

/// The changes of `[start, end)` (whole pages) known after this asking.
/// `None` if the memory is neither write-watched nor, with `guarded`, view
/// memory that can be guarded.
pub(crate) fn writes(start: usize, end: usize, guarded: bool) -> Option<Changes> {
    use windows_sys::Win32::System::Memory::GetWriteWatch;
    const WRITE_WATCH_FLAG_RESET: u32 = 1;
    if start >= end {
        return None;
    }
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    let j = &mut *j;
    // Asked ahead with the submission's other ranges: the journal has it.
    if j.ahead.iter().any(|&(a, b)| a <= first && last <= b) {
        return Some(j.changes(first, last));
    }
    let mut count = (last - first) * PAGE / KERNEL_PAGE;
    j.found.resize(count.max(j.found.len()), 0);
    let mut granularity = 0;
    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    // SAFETY: the answer buffer holds `count` entries.
    let failed = unsafe {
        GetWriteWatch(
            WRITE_WATCH_FLAG_RESET,
            (first * PAGE) as *const _,
            (last - first) * PAGE,
            j.found.as_mut_ptr().cast(),
            &mut count,
            &mut granularity,
        )
    } != 0;
    let guarded = if failed {
        count = 0;
        if !guarded {
            return None;
        }
        #[cfg(feature = "diagnostics")]
        let guarding = std::time::Instant::now();
        let written = guard(first * PAGE, last * PAGE);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(
            crate::copystat::C::GuardUs,
            guarding.elapsed().as_micros() as u64,
        );
        written?
    } else {
        Vec::new()
    };
    #[cfg(feature = "diagnostics")]
    {
        use crate::copystat::{C, add};
        add(C::Watches, 1);
        add(C::WatchUs, started.elapsed().as_micros() as u64);
        slow(started.elapsed().as_micros() as u64);
    }
    j.asked += 1;
    let now = j.asked;
    let pages = j.found[..count].iter().map(|&at| at / PAGE).chain(guarded);
    for page in pages.collect::<Vec<_>>() {
        j.written.insert(page, now);
    }
    Some(j.changes(first, last))
}

/// Guarded blocks of view memory.
const BLOCK: usize = PAGE;

struct Guards {
    /// Blocks we made read-only.
    protected: BTreeSet<usize>,
    /// Of those, the ones not written since a copy took them.
    clean: BTreeSet<usize>,
    /// Blocks already copied as zeros. While they remain wholly uncommitted,
    /// the copy can skip hundreds of MB of failing kernel reads per world frame.
    absent: BTreeSet<usize>,
    /// Each page points to the other mappings of its physical page.
    aliases: BTreeMap<usize, BTreeSet<usize>>,
    /// Kernel writes in progress, as page ranges.
    busy: Vec<(usize, usize)>,
    #[cfg(feature = "diagnostics")]
    written: BTreeSet<usize>,
    #[cfg(feature = "diagnostics")]
    written_before: BTreeSet<usize>,
}

static GUARDS: Mutex<Guards> = Mutex::new(Guards {
    protected: BTreeSet::new(),
    clean: BTreeSet::new(),
    absent: BTreeSet::new(),
    aliases: BTreeMap::new(),
    busy: Vec::new(),
    #[cfg(feature = "diagnostics")]
    written: BTreeSet::new(),
    #[cfg(feature = "diagnostics")]
    written_before: BTreeSet::new(),
});

#[cfg(not(feature = "diagnostics"))]
fn guards() -> std::sync::MutexGuard<'static, Guards> {
    GUARDS.lock().unwrap_or_else(|e| e.into_inner())
}

/// The guard lock, timed while held by the line that took it
/// (diagnostics: [`held_times`]).
#[cfg(feature = "diagnostics")]
#[track_caller]
fn guards() -> Held {
    let line = std::panic::Location::caller().line();
    let guard = GUARDS.lock().unwrap_or_else(|e| e.into_inner());
    Held(Some(guard), std::time::Instant::now(), line)
}

#[cfg(feature = "diagnostics")]
struct Held(
    Option<std::sync::MutexGuard<'static, Guards>>,
    std::time::Instant,
    u32,
);

#[cfg(feature = "diagnostics")]
impl std::ops::Deref for Held {
    type Target = Guards;
    fn deref(&self) -> &Guards {
        self.0.as_ref().expect("held until dropped")
    }
}

#[cfg(feature = "diagnostics")]
impl std::ops::DerefMut for Held {
    fn deref_mut(&mut self) -> &mut Guards {
        self.0.as_mut().expect("held until dropped")
    }
}

/// Per line taking the guard lock: the line, microseconds held in all, the
/// longest hold (no allocation: the fault handler takes the lock).
#[cfg(feature = "diagnostics")]
static HELD: [[std::sync::atomic::AtomicU64; 3]; 16] =
    [const { [const { std::sync::atomic::AtomicU64::new(0) }; 3] }; 16];

#[cfg(feature = "diagnostics")]
impl Drop for Held {
    fn drop(&mut self) {
        use std::sync::atomic::Ordering::Relaxed;
        let us = self.1.elapsed().as_micros() as u64;
        drop(self.0.take());
        let line = u64::from(self.2);
        if let Some(slot) = HELD.iter().find(|s| {
            s[0].load(Relaxed) == line || s[0].compare_exchange(0, line, Relaxed, Relaxed).is_ok()
        }) {
            slot[1].fetch_add(us, Relaxed);
            slot[2].fetch_max(us, Relaxed);
        }
    }
}

/// Since the last call, per line of this file taking the guard lock: (line,
/// milliseconds held, longest hold in milliseconds), most held first.
#[cfg(feature = "diagnostics")]
pub(crate) fn held_times() -> Vec<(u64, f64, f64)> {
    use std::sync::atomic::Ordering::Relaxed;
    let mut out: Vec<(u64, f64, f64)> = HELD
        .iter()
        .filter(|s| s[0].load(Relaxed) != 0)
        .map(|s| {
            (
                s[0].load(Relaxed),
                s[1].swap(0, Relaxed) as f64 / 1e3,
                s[2].swap(0, Relaxed) as f64 / 1e3,
            )
        })
        .collect();
    out.sort_by(|a, b| b.1.total_cmp(&a.1));
    out
}

/// For view memory: the pages of `[start, end)`'s blocks written since
/// they were last guarded (all of a block's pages), guarding them again
/// before the caller copies them. `None` if the memory is not a view.
fn guard(start: usize, end: usize) -> Option<Vec<usize>> {
    // A range whose first block the guards hold already (read-only by our
    // hand, or noted absent) is a view: no question to Windows. The
    // question walks the page tables to the end of the pages that share the
    // first one's state, and a guarded view is one long run of read-only
    // pages: 0.6 ms for the title's 174 MB heap, 16 times a frame, and most
    // of the 68 ms a frame of "guards" (run 58).
    // `VICTORY_GUARD_ASK=1` asks every time, as before.
    static ASK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let known =
        !*ASK.get_or_init(|| std::env::var_os("VICTORY_GUARD_ASK").is_some_and(|v| v == "1")) && {
            let g = guards();
            let first = start / BLOCK;
            g.protected.contains(&first) || g.absent.contains(&first)
        };
    if !known && !is_view(start, end) {
        return None;
    }
    install();
    let mut g = guards();
    let mut written = Vec::new();
    let mut run: Option<usize> = None;
    let (first, last) = (start / BLOCK, end.div_ceil(BLOCK));
    // The range's clean blocks, in order: read off the set in one pass,
    // not looked up one block at a time. A frame asks about some 10 GB of
    // view memory, 150,000 blocks, nearly all clean: the lookups were most
    // of the 60-90 ms a frame of "guards" (run 54). What the loop adds to
    // the set meanwhile is the block it stands at or ones before it.
    let clean: Vec<usize> = g.clean.range(first..last).copied().collect();
    if clean.len() == last - first && g.absent.range(first..last).next().is_none() {
        // Nothing written, mapped or new since the range was last taken.
        return Some(written);
    }
    let mut next_clean = 0;
    let mut uncommitted_until = first;
    for block in first..=last {
        while next_clean < clean.len() && clean[next_clean] < block {
            next_clean += 1;
        }
        let is_clean = next_clean < clean.len() && clean[next_clean] == block;
        let mut open = block < last && !is_clean;
        if block < last && g.absent.contains(&block) {
            // Ordinary commits need no runtime mapping notification. Reuse
            // this asking's query for the following blocks in the same region.
            if block >= uncommitted_until {
                uncommitted_until = uncommitted_end(block);
            }
            open = block >= uncommitted_until;
            if open {
                g.absent.remove(&block);
                g.clean.remove(&block);
            }
        }
        if open {
            written.extend(block * BLOCK / PAGE..(block + 1) * BLOCK / PAGE);
            if !busy(&g, block) {
                if g.protected.contains(&block) {
                    // Another mapping armed this block before it was asked.
                    g.clean.insert(block);
                } else {
                    run.get_or_insert(block);
                    continue;
                }
            }
        }
        if let Some(from) = run.take() {
            protect(&mut g, from, block);
        }
    }
    // Arm every mapping before allowing the requested page to become
    // clean. A failed guard (a partially committed page, for example)
    // leaves the requesting page dirty so its next reader retries.
    let aliases: Vec<_> = written
        .iter()
        .flat_map(|&p| {
            g.aliases
                .get(&p)
                .into_iter()
                .flatten()
                .map(move |&a| (p, a))
        })
        .collect();
    if !aliases.is_empty() {
        #[cfg(feature = "diagnostics")]
        let _timed = Timed(std::time::Instant::now(), crate::copystat::C::AliasUs);
        for (page, other) in aliases {
            if !guard_alias(&mut g, other) {
                g.clean.remove(&page);
            }
        }
    }
    Some(written)
}

/// Whether `[start, end)` is view memory. The title's views lie in
/// placeholders, mapped and unmapped block by block as it streams, so a
/// range can begin in a block that is not mapped right now: the first
/// pages of the range that are there decide. A range with nothing there
/// counts as a view (its blocks are noted absent, and looked at again when
/// the runtime maps them).
///
/// Looking at the first address only took such a range for "not a view":
/// its copy was marked not watched for good and copied whole at every
/// refresh, around its unmapped pages (run 54: 300 such ranges, 570 MB
/// copied a frame for them, 255 copies a frame reading around holes).
/// `VICTORY_GUARD_HOLES=0` looks at the first address only, as before.
fn is_view(start: usize, end: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_MAPPED, MEMORY_BASIC_INFORMATION, VirtualQuery,
    };
    static HOLES: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let holes =
        *HOLES.get_or_init(|| std::env::var_os("VICTORY_GUARD_HOLES").is_none_or(|v| v != "0"));
    let mut at = start;
    loop {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
        if n == 0 {
            return false;
        }
        if info.Type == MEM_MAPPED {
            return true;
        }
        // Committed memory that is no view (the title's heaps, ours), or
        // the answer as it was before.
        if info.State == MEM_COMMIT || !holes {
            return false;
        }
        // A hole (a placeholder, or free): on to what follows it.
        let next = (info.BaseAddress as usize).saturating_add(info.RegionSize);
        if next <= at || next >= end {
            return true;
        }
        at = next;
    }
}

/// Makes blocks `[from, to)` read-only and notes them guarded and clean;
/// block by block if they span views, leaving out any that fail, and
/// noting those not committed (unmapped) absent.
fn protect(g: &mut Guards, from: usize, to: usize) {
    use windows_sys::Win32::System::Memory::{PAGE_READONLY, VirtualProtect};
    #[cfg(feature = "diagnostics")]
    let _timed = Timed(std::time::Instant::now(), crate::copystat::C::ProtectUs);
    let set = |from: usize, to: usize| {
        #[cfg(feature = "diagnostics")]
        {
            crate::copystat::add(crate::copystat::C::ProtectCalls, 1);
            crate::copystat::add(crate::copystat::C::ProtectPages, (to - from) as u64);
        }
        let mut old = 0;
        // SAFETY: title view memory; only its protection changes.
        unsafe {
            VirtualProtect(
                (from * BLOCK) as *const _,
                (to - from) * BLOCK,
                PAGE_READONLY,
                &mut old,
            ) != 0
        }
    };
    let note = |g: &mut Guards, from: usize, to: usize| {
        g.protected.extend(from..to);
        g.clean.extend(from..to);
    };
    if set(from, to) {
        note(g, from, to);
        return;
    }
    for block in from..to {
        if set(block, block + 1) {
            note(g, block, block + 1);
        } else if block < uncommitted_end(block) {
            g.absent.insert(block);
            g.clean.insert(block);
        }
    }
}

fn busy(g: &Guards, block: usize) -> bool {
    g.busy.iter().any(|&(from, to)| (from..to).contains(&block))
}

/// Arm a secondary mapping without consuming its own pending changes.
fn guard_alias(g: &mut Guards, block: usize) -> bool {
    if busy(g, block) {
        return false;
    }
    if !g.protected.contains(&block) {
        let clean = g.clean.contains(&block);
        protect(g, block, block + 1);
        if !clean {
            g.clean.remove(&block);
        }
    }
    g.protected.contains(&block) || g.absent.contains(&block)
}

/// The end (exclusive) of the whole uncommitted blocks starting at `block`.
fn uncommitted_end(block: usize) -> usize {
    use windows_sys::Win32::System::Memory::{
        MEM_FREE, MEM_RESERVE, MEMORY_BASIC_INFORMATION, VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery((block * BLOCK) as *const _, &mut info, size_of_val(&info)) };
    if n != 0 && matches!(info.State, MEM_RESERVE | MEM_FREE) {
        (info.BaseAddress as usize).saturating_add(info.RegionSize) / BLOCK
    } else {
        block
    }
}

/// Gives write access back to the guarded blocks and marks their aliases
/// dirty. Only the addresses being written need write access.
fn open(g: &mut Guards, start: usize, end: usize) -> bool {
    use windows_sys::Win32::System::Memory::{PAGE_READWRITE, VirtualProtect};
    if start >= end {
        return false;
    }
    let range = start / BLOCK..end.div_ceil(BLOCK);
    let aliases: Vec<_> = g
        .aliases
        .range(range.clone())
        .flat_map(|(_, others)| others.iter().copied())
        .collect();
    for block in aliases {
        if g.clean.remove(&block) {
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::AliasMarks, 1);
        }
        #[cfg(feature = "diagnostics")]
        g.written.insert(block);
    }
    let clean: Vec<_> = g.clean.range(range.clone()).copied().collect();
    for block in clean {
        g.clean.remove(&block);
        #[cfg(feature = "diagnostics")]
        g.written.insert(block);
    }
    let blocks: Vec<_> = g.protected.range(range).copied().collect();
    for &block in &blocks {
        g.protected.remove(&block);
        let mut old = 0;
        // SAFETY: a block we made read-only.
        unsafe { VirtualProtect((block * BLOCK) as *const _, BLOCK, PAGE_READWRITE, &mut old) };
    }
    !blocks.is_empty()
}

/// A kernel write keeps its destination writable until it completes.
#[must_use = "the write must take place while this guard lives"]
pub struct Writing {
    at: usize,
    end: usize,
}

impl Drop for Writing {
    fn drop(&mut self) {
        if self.at >= self.end {
            return;
        }
        let range = (self.at / BLOCK, self.end.div_ceil(BLOCK));
        {
            let mut g = guards();
            if let Some(i) = g.busy.iter().position(|&r| r == range) {
                g.busy.swap_remove(i);
            }
            open(&mut g, self.at, self.end);
        }
        write_ahead(self.at, self.end);
    }
}

/// Opens title memory for a kernel write, which cannot use the fault handler.
pub fn writing(at: usize, len: usize) -> Writing {
    let end = at.saturating_add(len);
    if at < end {
        {
            let mut g = guards();
            g.busy.push((at / BLOCK, end.div_ceil(BLOCK)));
            open(&mut g, at, end);
        }
        crate::descriptor::host::remapped(at, len);
    }
    Writing { at, end }
}

/// Held shared by a read of title memory in place, and alone by the runtime
/// while it changes what title memory shows.
///
/// The runtime does not change a mapping in one step. To unmap one page of
/// a view it unmaps the whole view and maps what stays again
/// (`xmem.rs` `unmap`), and a view it maps is a reservation until the next
/// call commits it (`map`, then `commit`). For that moment pages the title
/// never gave up are "reserved, never committed", and a thread reading
/// them in place takes an access violation: run 71 in `Layout::detile`,
/// run 74 in the hash before it, both under `Staging::refresh_now`, which
/// had asked whether the texture was all mapped and been told yes. Asking
/// cannot help; not changing the mapping while such a read is under way
/// does.
static MAPPING: std::sync::RwLock<()> = std::sync::RwLock::new(());

/// What the runtime holds while it changes what title memory shows
/// ([`title_memory_remapping`]).
pub struct Remapping(#[allow(dead_code)] std::sync::RwLockWriteGuard<'static, ()>);

/// The runtime is about to map, unmap or move physical pages of title
/// memory: until what it returns is dropped, no read of title memory in
/// place is under way and none begins ([`title_memory_reading`]). Taken
/// around the whole change, the unmapping and the mapping again.
pub fn title_memory_remapping() -> Remapping {
    Remapping(MAPPING.write().unwrap_or_else(|e| e.into_inner()))
}

/// What a read of title memory in place holds ([`title_memory_reading`]).
pub struct Reading(#[allow(dead_code)] std::sync::RwLockReadGuard<'static, ()>);

/// For reading title memory in place: until what it returns is dropped,
/// the runtime changes no mapping, so memory found mapped stays mapped.
/// Not to be taken twice by one thread, nor held across anything that
/// maps title memory.
pub fn title_memory_reading() -> Reading {
    Reading(MAPPING.read().unwrap_or_else(|e| e.into_inner()))
}

/// [`title_memory_reading`] if it can be had at once: `None` while the
/// runtime changes a mapping (or waits to), for a caller that has another
/// way to read and would rather not wait.
pub fn title_memory_reading_now() -> Option<Reading> {
    match MAPPING.try_read() {
        Ok(guard) => Some(Reading(guard)),
        Err(std::sync::TryLockError::Poisoned(e)) => Some(Reading(e.into_inner())),
        Err(std::sync::TryLockError::WouldBlock) => None,
    }
}

/// The runtime changed the mapping of `[at, at + len)`. Discard guard
/// state for the old mapping and publish the change to readers.
pub fn title_memory_remapped(at: usize, len: usize) {
    unmap_title_memory(at, len, || {});
}

/// Retires in-place views before the supplied unmap and publishes the
/// resulting mapping change afterward. The runtime also holds
/// [`title_memory_remapping`] across the full mapping change.
pub fn unmap_title_memory(at: usize, len: usize, unmap: impl FnOnce()) {
    if len == 0 {
        unmap();
        return;
    }
    crate::inplace::views_gone(at, len);
    crate::inplace::title_memory_freeing(at, len);
    unmap();
    let range = at / BLOCK..at.saturating_add(len).div_ceil(BLOCK);
    // A query racing the unmap can see the old generation. Publish the new
    // generation only after the unmap and discard the old mapping's write
    // stamps and cached asks under both locks, in the order used by writes.
    let mut journal = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    let mut g = guards();
    let stale_writes: Vec<_> = journal
        .written
        .range(range.clone())
        .map(|(&p, _)| p)
        .collect();
    for page in stale_writes {
        journal.written.remove(&page);
    }
    journal.asked += 1;
    let now = journal.asked;
    journal.remapped.record(range.clone(), now);
    journal
        .ahead
        .retain(|&(from, to)| to <= range.start || range.end <= from);
    let absent: Vec<usize> = g.absent.range(range.clone()).copied().collect();
    for block in absent {
        g.absent.remove(&block);
    }
    let blocks: Vec<_> = g.protected.range(range.clone()).copied().collect();
    for block in blocks {
        g.protected.remove(&block);
    }
    let clean: Vec<_> = g.clean.range(range.clone()).copied().collect();
    for block in clean {
        g.clean.remove(&block);
    }
    let stale: Vec<_> = g.aliases.range(range).map(|(&p, _)| p).collect();
    for page in stale {
        for other in g.aliases.remove(&page).unwrap_or_default() {
            if let Some(links) = g.aliases.get_mut(&other) {
                links.remove(&page);
                if links.is_empty() {
                    g.aliases.remove(&other);
                }
            }
        }
    }
    drop(g);
    drop(journal);
    // A placed texture there arrives, goes or changes with no descriptor
    // written.
    crate::descriptor::host::remapped(at, len);
}

/// Reports all known virtual mappings of the same physical page.
pub fn title_memory_aliased(at: usize, others: &[usize]) {
    let mut g = guards();
    let mut pages = BTreeSet::from([at / BLOCK]);
    for other in others.iter().map(|&p| p / BLOCK) {
        pages.insert(other);
        pages.extend(g.aliases.get(&other).into_iter().flatten());
    }
    if pages.len() < 2 {
        return;
    }
    for &page in &pages {
        g.aliases
            .insert(page, pages.iter().copied().filter(|&p| p != page).collect());
    }
    if pages.iter().any(|page| g.protected.contains(page)) {
        let mut armed = true;
        for &page in &pages {
            armed &= guard_alias(&mut g, page);
        }
        if !armed {
            for page in pages {
                g.clean.remove(&page);
            }
        }
    }
}

/// Registers the handler that takes writes to guarded blocks, once, ahead
/// of the crash reporter's.
fn install() {
    use windows_sys::Win32::Foundation::{EXCEPTION_ACCESS_VIOLATION, NTSTATUS};
    use windows_sys::Win32::System::Diagnostics::Debug::{
        AddVectoredExceptionHandler, EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH,
        EXCEPTION_POINTERS,
    };
    unsafe extern "system" fn on_write(info: *mut EXCEPTION_POINTERS) -> i32 {
        // SAFETY: the system passes the exception being dispatched.
        let r = unsafe { &*(*info).ExceptionRecord };
        const WRITE: usize = 1;
        if r.ExceptionCode != EXCEPTION_ACCESS_VIOLATION as NTSTATUS
            || r.NumberParameters < 2
            || r.ExceptionInformation[0] != WRITE
        {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        let at = r.ExceptionInformation[1];
        #[cfg(feature = "diagnostics")]
        let _timed = Timed(std::time::Instant::now(), crate::copystat::C::HandlerUs);
        // Held across both checks: another thread may have opened the block
        // since this one faulted, and a copy may have guarded it again since
        // (then it is ours again, not read-only for some other reason).
        let mut g = guards();
        if open(&mut g, at, at + 1) {
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::GuardFaults, 1);
            return EXCEPTION_CONTINUE_EXECUTION;
        }
        if writable(at) {
            EXCEPTION_CONTINUE_EXECUTION
        } else {
            EXCEPTION_CONTINUE_SEARCH
        }
    }
    static ONCE: std::sync::Once = std::sync::Once::new();
    // SAFETY: the handler only touches our guard state and protections.
    ONCE.call_once(|| unsafe {
        AddVectoredExceptionHandler(1, Some(on_write));
    });
}

/// Whether `at` is committed read-write memory now.
fn writable(at: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_READWRITE, VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    n != 0 && info.State == MEM_COMMIT && info.Protect == PAGE_READWRITE
}

/// Adds the time since it was made to a counter when dropped (diagnostics).
#[cfg(feature = "diagnostics")]
pub(crate) struct Timed(pub(crate) std::time::Instant, pub(crate) crate::copystat::C);

#[cfg(feature = "diagnostics")]
impl Drop for Timed {
    fn drop(&mut self) {
        let c = std::mem::replace(&mut self.1, crate::copystat::C::Submissions);
        crate::copystat::add(c, self.0.elapsed().as_micros() as u64);
    }
}

/// A cursor into the shared write journal, used e.g. by descriptor heaps.
pub(crate) struct Tracked {
    start: usize,
    end: usize,
    views: bool,
    copied: Mutex<Vec<u64>>,
}

pub(crate) fn track(start: usize, len: usize, views: bool) -> Option<Tracked> {
    let end = start.checked_add(len)?;
    if start >= end || writes(start, end, views).is_none() {
        return None;
    }
    Some(Tracked {
        start,
        end,
        views,
        copied: Mutex::new(vec![0; end.div_ceil(PAGE) - start / PAGE]),
    })
}

impl Tracked {
    /// Ranges changed since this tracker took them, in address order.
    pub(crate) fn take(&self, from: usize, to: usize) -> Vec<(usize, usize)> {
        let (from, to) = (from.max(self.start), to.min(self.end));
        if from >= to {
            return Vec::new();
        }
        let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
        let Some(changes) = writes(from, to, self.views) else {
            return vec![(from, to)];
        };
        let mut written = changes.pages().peekable();
        let mut ranges: Vec<(usize, usize)> = Vec::new();
        for p in from / PAGE..to.div_ceil(PAGE) {
            let last_write = written
                .next_if(|&(page, _)| page == p)
                .map_or(0, |(_, w)| w);
            let stamp = &mut copied[p - self.start / PAGE];
            if *stamp != 0 && *stamp >= last_write {
                continue;
            }
            let (a, b) = (
                (p * PAGE).max(from),
                (p.saturating_mul(PAGE).saturating_add(PAGE)).min(to),
            );
            if a <= (p * PAGE).max(self.start) && b >= (p * PAGE).saturating_add(PAGE).min(self.end)
            {
                *stamp = changes.now;
            }
            match ranges.last_mut() {
                Some((_, end)) if *end == a => *end = b,
                _ => ranges.push((a, b)),
            }
        }
        ranges
    }
}

#[cfg(feature = "diagnostics")]
pub(crate) fn frame_done() {
    let mut g = guards();
    let written = std::mem::take(&mut g.written);
    let again = written.intersection(&g.written_before).count();
    crate::copystat::add(crate::copystat::C::WrittenPages, written.len() as u64);
    crate::copystat::add(crate::copystat::C::WrittenAgain, again as u64);
    g.written_before = written;
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
    };
    pub(crate) const MEM_WRITE_WATCH: u32 = 0x20_0000;

    /// A write-watched allocation of `pages` pages (as the runtime's pools),
    /// on a 64 KiB boundary (the allocation granularity).
    pub(crate) fn pool(pages: usize) -> usize {
        // SAFETY: a fresh allocation, freed by the test.
        let p = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                pages * PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert!(p != 0 && p.is_multiple_of(PAGE));
        p
    }

    pub(crate) fn free(p: usize) {
        // SAFETY: an allocation of ours.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
    }

    #[test]
    fn written_pages_are_reported_to_every_asker() {
        // SAFETY: a fresh write-watched allocation, freed at the end.
        let p = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                4 * PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(p, 0);
        let (start, end) = (p, p + 4 * PAGE);
        title_memory_remapped(start, end - start);
        let before = writes(start, end, false).expect("watched");
        assert!(before.written.is_empty());
        // SAFETY: committed pages of ours.
        unsafe { ((p + PAGE + 8) as *mut u8).write(1) };
        let after = writes(start, end, false).expect("watched");
        assert_eq!(after.written, [(p / PAGE + 1, after.now)]);
        // Another asker after the kernel's bit was cleared still sees it.
        let again = writes(p + PAGE, p + 2 * PAGE, false).expect("watched");
        assert!(again.now > after.now);
        assert_eq!(
            again.pages().collect::<Vec<_>>(),
            [(p / PAGE + 1, after.now)]
        );
        let mut plain = vec![0u8; PAGE];
        assert!(
            writes(
                plain.as_mut_ptr() as usize,
                plain.as_ptr() as usize + PAGE,
                true
            )
            .is_none()
        );
        title_memory_remapped(start, end - start);
        // SAFETY: our allocation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
    }

    #[test]
    fn remapped_write_watch_pages_stay_changed_for_each_tracker() {
        use windows_sys::Win32::System::Memory::MEM_DECOMMIT;

        let p = pool(3);
        let (middle, end) = (p + PAGE, p + 3 * PAGE);
        // SAFETY: materialize this test's committed pages before tracking.
        unsafe { (p as *mut u8).write_bytes(0, 3 * PAGE) };
        let first = track(p, 3 * PAGE, false).expect("watched");
        let second = track(p, 3 * PAGE, false).expect("watched");
        for tracker in [&first, &second] {
            assert_eq!(tracker.take(p, end), [(p, end)]);
            assert!(tracker.take(p, end).is_empty());
        }
        // SAFETY: the test owns this committed page and its reservation.
        unsafe { ((middle + 8) as *mut u8).write_volatile(1) };
        assert_eq!(
            first.take(middle + 8, middle + 9),
            [(middle + 8, middle + 9)]
        );
        {
            let _changing = title_memory_remapping();
            unmap_title_memory(middle, PAGE, || {
                // SAFETY: the test still owns the reservation.
                assert_ne!(
                    unsafe { VirtualFree(middle as *mut _, PAGE, MEM_DECOMMIT) },
                    0
                );
            });
        }
        for tracker in [&first, &second] {
            assert!(tracker.take(p, middle).is_empty());
            assert!(tracker.take(middle + PAGE, end).is_empty());
            assert_eq!(tracker.take(p, end), [(middle, middle + PAGE)]);
            assert!(tracker.take(p, end).is_empty());
        }
        title_memory_remapped(p, end - p);
        free(p);
    }

    #[test]
    fn remap_history_preserves_overlaps_and_bounds_older_ranges() {
        let mut remaps = Remaps(BTreeMap::new());
        remaps.record(10..40, 1);
        remaps.record(20..30, 2);
        remaps.record(25..35, 3);
        assert_eq!(
            remaps.within(15, 38).collect::<Vec<_>>(),
            [(15..20, 1), (20..25, 2), (25..35, 3), (35..38, 1)]
        );

        let start = 50;
        for i in 0..MAX_REMAP_RANGES {
            let page = start + i * 2;
            remaps.record(page..page + 1, i as u64 + 4);
        }
        assert!(remaps.0.len() <= MAX_REMAP_RANGES);
        for i in 0..MAX_REMAP_RANGES {
            let page = start + i * 2;
            let stamp = i as u64 + 4;
            assert!(remaps.within(page, page + 1).any(|(_, at)| at >= stamp));
        }
        let last = start + (MAX_REMAP_RANGES - 1) * 2;
        assert_eq!(
            remaps.within(last - 1, last + 2).collect::<Vec<_>>(),
            [(last..last + 1, MAX_REMAP_RANGES as u64 + 3)]
        );
    }

    /// A kernel read into a guarded view block fails; opened first, it
    /// lands and the block counts as written.
    #[test]
    fn kernel_reads_into_guarded_blocks_after_opening() {
        use std::io::{Read, Seek, Write};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, UnmapViewOfFile,
        };
        // SAFETY: a pagefile section of two blocks, mapped read-write.
        let view = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 2 * BLOCK).Value as usize
        };
        assert_ne!(view, 0);
        assert!(guard(view, view + 2 * BLOCK).is_some()); // now read-only
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(b"from the disk").unwrap();
        // `ReadFile` inside: the kernel writes the view.
        let read = |file: &mut std::fs::File| {
            file.rewind().unwrap();
            // SAFETY: 13 bytes of our view, nothing else refers to them.
            let into = unsafe { std::slice::from_raw_parts_mut((view + 100) as *mut u8, 13) };
            file.read(into).is_ok_and(|n| n == 13)
        };
        assert!(
            !read(&mut file),
            "the kernel does not fault into our handler"
        );
        let writing = writing(view + 100, 13);
        assert_eq!(guard(view, view + 2 * BLOCK), Some(vec![view / PAGE]));
        assert!(read(&mut file));
        drop(writing);
        // SAFETY: the bytes just read.
        let landed = unsafe { std::slice::from_raw_parts((view + 100) as *const u8, 13) };
        assert_eq!(landed, b"from the disk");
        assert_eq!(
            guard(view, view + 2 * BLOCK),
            Some((view / PAGE..view / PAGE + 1).collect())
        );
        // SAFETY: our view.
        unsafe {
            UnmapViewOfFile(
                windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: view as *mut _,
                },
            )
        };
        title_memory_remapped(view, 2 * BLOCK);
    }

    #[test]
    fn writes_in_partly_committed_view_blocks_are_reported() {
        use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile,
            SEC_RESERVE, UnmapViewOfFile,
        };

        const SIZE: usize = 3 * BLOCK;
        struct View(MEMORY_MAPPED_VIEW_ADDRESS);
        impl Drop for View {
            fn drop(&mut self) {
                title_memory_remapped(self.0.Value as usize, SIZE);
                // SAFETY: our view is still mapped.
                unsafe { UnmapViewOfFile(self.0) };
            }
        }
        // SAFETY: a new reserved pagefile section. The view holds its own
        // reference after we close the handle, including on assertion failure.
        let view = unsafe {
            let section = CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE | SEC_RESERVE,
                0,
                SIZE as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let view = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, SIZE);
            CloseHandle(section);
            assert!(!view.Value.is_null());
            View(view)
        };
        let start = view.0.Value as usize;
        let commit = |at: usize, len| {
            // SAFETY: these pages belong to our reserved view.
            let committed = unsafe { VirtualAlloc(at as _, len, MEM_COMMIT, PAGE_READWRITE) };
            assert_eq!(committed as usize, at);
        };
        let early = start + BLOCK + 3 * KERNEL_PAGE;
        let late = start + 2 * BLOCK + 3 * KERNEL_PAGE;
        commit(start, BLOCK);
        commit(early, KERNEL_PAGE);
        let refresh = || guard(start, start + SIZE).unwrap();
        assert_eq!(refresh(), [start / PAGE, early / PAGE, late / PAGE]);
        // The last block is still entirely reserved.
        assert!(!refresh().contains(&(late / PAGE)));

        commit(late, KERNEL_PAGE); // Ordinary commit, without a runtime mapping notification.
        for at in [early, late] {
            assert!(writable(at));
            // SAFETY: each address is inside its committed read-write page.
            unsafe { (at as *mut u8).write_volatile(1) };
        }
        assert_eq!(refresh(), [early / PAGE, late / PAGE]);
    }

    /// A range of view memory whose first block is not mapped is a view
    /// all the same: its blocks are reported once, then the ones written
    /// or mapped since.
    #[test]
    fn a_range_that_begins_in_an_unmapped_block_is_guarded() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER,
            MEM_RESERVE_PLACEHOLDER, MapViewOfFile3, PAGE_NOACCESS, UnmapViewOfFile2,
            VirtualAlloc2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        // SAFETY: a fresh placeholder of three blocks and a pagefile
        // section of three; the middle block is mapped first.
        let (p, section) = unsafe {
            let p = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                3 * BLOCK,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_ne!(p, 0);
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (3 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            (p, section)
        };
        let map = |block: usize| {
            // SAFETY: splits the placeholder at the block and maps the
            // section's block there.
            let view = unsafe {
                VirtualFree(
                    (p + block * BLOCK) as *mut _,
                    BLOCK,
                    MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
                );
                MapViewOfFile3(
                    section,
                    GetCurrentProcess(),
                    (p + block * BLOCK) as *const _,
                    (block * BLOCK) as u64,
                    BLOCK,
                    MEM_REPLACE_PLACEHOLDER,
                    PAGE_READWRITE,
                    std::ptr::null_mut(),
                    0,
                )
            };
            assert_eq!(view.Value as usize, p + block * BLOCK);
            view
        };
        let pages = |block: usize| (p + block * BLOCK) / PAGE..(p + (block + 1) * BLOCK) / PAGE;
        let all: Vec<usize> = (0..3).flat_map(pages).collect();
        // Nothing mapped at all: a view with every block absent.
        assert!(is_view(p, p + 3 * BLOCK));
        let middle = map(1);
        title_memory_remapped(p + BLOCK, BLOCK);
        // The range begins in the placeholder, its middle block is a view.
        assert!(is_view(p, p + 3 * BLOCK));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(all));
        assert_eq!(
            guard(p, p + 3 * BLOCK),
            Some(Vec::new()),
            "nothing written since"
        );
        // SAFETY: inside the mapped middle block.
        unsafe { ((p + BLOCK + 8) as *mut u8).write_volatile(1) };
        assert_eq!(guard(p, p + 3 * BLOCK), Some(pages(1).collect()));
        let first = map(0);
        title_memory_remapped(p, BLOCK);
        assert_eq!(guard(p, p + 3 * BLOCK), Some(pages(0).collect()));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(Vec::new()));
        // Memory of ours that is no view stays none.
        let mut local = [0u8; 16];
        let at = local.as_mut_ptr() as usize;
        assert!(!is_view(at, at + local.len()));
        // SAFETY: our views and placeholder.
        unsafe {
            UnmapViewOfFile2(GetCurrentProcess(), first, MEM_PRESERVE_PLACEHOLDER);
            UnmapViewOfFile2(GetCurrentProcess(), middle, MEM_PRESERVE_PLACEHOLDER);
            for block in 0..3 {
                VirtualFree((p + block * BLOCK) as *mut _, 0, MEM_RELEASE);
            }
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
        title_memory_remapped(p, 3 * BLOCK);
    }

    /// Blocks of a placeholder between views are reported once, then not
    /// again (they cannot be guarded) until the runtime maps them.
    #[test]
    fn unmapped_blocks_are_reported_once_until_mapped() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER,
            MEM_RESERVE_PLACEHOLDER, MapViewOfFile3, PAGE_NOACCESS, UnmapViewOfFile2,
            VirtualAlloc2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        // SAFETY: a fresh placeholder of three blocks and a pagefile
        // section of two; block 0 shows the section's first block, as the
        // runtime maps physical pages.
        let (p, section) = unsafe {
            let p = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                3 * BLOCK,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_ne!(p, 0);
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            (p, section)
        };
        let map = |block: usize| {
            // SAFETY: splits the placeholder at the block and maps the
            // section's block there.
            let view = unsafe {
                VirtualFree(
                    (p + block * BLOCK) as *mut _,
                    BLOCK,
                    MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
                );
                MapViewOfFile3(
                    section,
                    GetCurrentProcess(),
                    (p + block * BLOCK) as *const _,
                    (block * BLOCK) as u64,
                    BLOCK,
                    MEM_REPLACE_PLACEHOLDER,
                    PAGE_READWRITE,
                    std::ptr::null_mut(),
                    0,
                )
            };
            assert_eq!(view.Value as usize, p + block * BLOCK);
            view
        };
        let first = map(0);
        let pages = |block: usize| (p + block * BLOCK) / PAGE..(p + (block + 1) * BLOCK) / PAGE;
        let all: Vec<usize> = (0..3).flat_map(pages).collect();
        assert_eq!(guard(p, p + 3 * BLOCK), Some(all));
        assert_eq!(
            guard(p, p + 3 * BLOCK),
            Some(Vec::new()),
            "nothing written since"
        );
        let second = map(1);
        title_memory_remapped(p + BLOCK, BLOCK);
        assert_eq!(guard(p, p + 3 * BLOCK), Some(pages(1).collect()));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(Vec::new()));
        // SAFETY: our views and placeholder.
        unsafe {
            UnmapViewOfFile2(GetCurrentProcess(), first, MEM_PRESERVE_PLACEHOLDER);
            UnmapViewOfFile2(GetCurrentProcess(), second, MEM_PRESERVE_PLACEHOLDER);
            for block in 0..3 {
                VirtualFree((p + block * BLOCK) as *mut _, 0, MEM_RELEASE);
            }
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
        title_memory_remapped(p, 3 * BLOCK);
    }

    /// `pages` pages of a fresh pagefile section mapped twice (as the
    /// title's pool view and tiled heap show the same physical pages),
    /// reported as the runtime does when it maps them.
    pub(crate) fn twin_views(pages: usize) -> (usize, usize) {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile,
        };
        let len = pages * PAGE;
        // SAFETY: a pagefile section mapped read-write twice; the test
        // keeps both views.
        let (a, b) = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                len as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let views = (
                MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, len).Value as usize,
                MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, len).Value as usize,
            );
            windows_sys::Win32::Foundation::CloseHandle(section);
            views
        };
        assert!(a != 0 && b != 0);
        for k in (0..len).step_by(PAGE) {
            title_memory_aliased(b + k, &[a + k]);
        }
        (a, b)
    }

    /// A write through another mapping of the same pages marks the tracker
    /// (the guard there faults, the runtime says where else it is mapped);
    /// so does a write the runtime reports at the other mapping.
    #[test]
    fn writes_through_another_mapping_mark_the_tracker() {
        let (a, b) = twin_views(2);
        let t = track(a, 2 * PAGE, true).expect("views are tracked");
        assert_eq!(t.take(a, a + 2 * PAGE), [(a, a + 2 * PAGE)]);
        // SAFETY: the second view of the pages, guarded by the take.
        unsafe { ((b + PAGE + 4) as *mut u32).write_volatile(7) };
        assert_eq!(t.take(a, a + 2 * PAGE), [(a + PAGE, a + 2 * PAGE)]);
        // SAFETY: the tracked view, which shows the write.
        assert_eq!(unsafe { ((a + PAGE + 4) as *const u32).read_volatile() }, 7);
        drop(writing(b, 8));
        assert_eq!(t.take(a, a + 2 * PAGE), [(a, a + PAGE)]);
        assert!(t.take(a, a + 2 * PAGE).is_empty());
        for p in [a, b] {
            // SAFETY: each address is a mapped view from twin_views.
            unsafe {
                windows_sys::Win32::System::Memory::UnmapViewOfFile(
                    windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS {
                        Value: p as *mut _,
                    },
                )
            };
            title_memory_remapped(p, 2 * PAGE);
        }
    }

    /// A write or mapping that runs past the end of the address space (a
    /// title's `fread` with a huge count) covers up to the end; the sum used
    /// to overflow. Up there no test has guarded blocks.
    #[test]
    fn ranges_past_the_address_space_end_do_not_overflow() {
        let top = usize::MAX - 16;
        drop(writing(top, PAGE));
        title_memory_remapped(top, usize::MAX);
    }
}

#[cfg(test)]
mod mapping_tests {
    use super::{title_memory_reading, title_memory_remapping};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, mpsc};
    use std::time::Duration;

    /// A change of mapping waits for a read in place that is under way, and
    /// a read in place waits for a change of mapping.
    #[test]
    fn reads_in_place_and_remappings_wait_for_each_other() {
        // A reader holds; the change does not begin until it lets go.
        let changed = Arc::new(AtomicBool::new(false));
        let reading = title_memory_reading();
        let (began, wait) = mpsc::channel();
        let change = {
            let changed = changed.clone();
            std::thread::spawn(move || {
                began.send(()).unwrap();
                let _changing = title_memory_remapping();
                changed.store(true, Ordering::SeqCst);
            })
        };
        wait.recv().unwrap();
        std::thread::sleep(Duration::from_millis(100));
        assert!(
            !changed.load(Ordering::SeqCst),
            "the mapping changed under a read in place"
        );
        drop(reading);
        change.join().unwrap();
        assert!(changed.load(Ordering::SeqCst));
        // A change holds; the read does not begin until it is done.
        let read = Arc::new(AtomicBool::new(false));
        let changing = title_memory_remapping();
        let (began, wait) = mpsc::channel();
        let reader = {
            let read = read.clone();
            std::thread::spawn(move || {
                began.send(()).unwrap();
                let _reading = title_memory_reading();
                read.store(true, Ordering::SeqCst);
            })
        };
        wait.recv().unwrap();
        std::thread::sleep(Duration::from_millis(100));
        assert!(
            !read.load(Ordering::SeqCst),
            "a read in place began during a change of mapping"
        );
        drop(changing);
        reader.join().unwrap();
        assert!(read.load(Ordering::SeqCst));
    }
}
