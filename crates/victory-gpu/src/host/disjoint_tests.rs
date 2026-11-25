//! Adapted from aycow's VIctory PR 1818, under the project GPL license.

use super::Batch;

/// Two requested ranges leave a dirty interior page available to its
/// own reader. The change journal must not consume it with the first two.
#[test]
fn disjoint_refreshes_leave_the_dirty_gap_for_its_reader() {
    use crate::watch::PAGE;

    let title = crate::watch::tests::pool(3);
    let len = 3 * PAGE;
    // SAFETY: the test owns this writable allocation.
    unsafe { (title as *mut u8).write_bytes(0, len) };
    crate::shadow::register(title, title as u64, len as u64);
    let Some(copy) = crate::shadow::find(title as u64, len as u64) else {
        crate::shadow::unregister(title, title as u64);
        crate::watch::tests::free(title);
        assert!(std::env::var_os("VICTORY_TEST_SHADOW_DISJOINT_REQUIRE_GPU").is_none());
        return;
    };
    let mut warm = Batch::default();
    warm.add(&copy, title, len);
    let warmed = warm.flush();
    for (page, marker) in [11u8, 22, 33].into_iter().enumerate() {
        // SAFETY: each address belongs to the test's guarded allocation.
        unsafe { ((title + page * PAGE) as *mut u8).write_volatile(marker) };
    }

    let mut batch = Batch::default();
    batch.add(&copy, title + 2 * PAGE, PAGE);
    batch.add(&copy, title, PAGE);
    batch.add(&copy, title + PAGE / 2, PAGE / 2);
    let first_bytes = batch.flush();
    let markers = [0, 1, 2].map(|page| copy.read_host((title + page * PAGE) as u64, 1)[0]);
    // SAFETY: the test still owns the title allocation.
    unsafe { ((title + 2 * PAGE) as *mut u8).write_volatile(44) };
    batch.add(&copy, title + 2 * PAGE, PAGE);
    let repeated_bytes = batch.flush();
    batch.add(&copy, title + PAGE, PAGE);
    let late_bytes = batch.flush();
    let late_marker = copy.read_host((title + PAGE) as u64, 1)[0];
    batch.title_written();
    batch.add(&copy, title + 2 * PAGE, PAGE);
    let next_phase_bytes = batch.flush();
    let next_phase_marker = copy.read_host((title + 2 * PAGE) as u64, 1)[0];

    // The temporary read batches hold copies of the backing while live.
    drop((warm, batch));
    crate::shadow::unregister(title, title as u64);
    drop(copy);
    crate::watch::tests::free(title);

    eprintln!(
        "disjoint refresh copied {first_bytes} then {late_bytes} bytes, markers {markers:?} then {late_marker}"
    );
    assert_eq!(warmed, len);
    assert_eq!(first_bytes, 2 * PAGE);
    assert_eq!(markers, [11, 0, 33]);
    assert_eq!(repeated_bytes, 0);
    assert_eq!(late_bytes, PAGE);
    assert_eq!(late_marker, 22);
    assert_eq!(next_phase_bytes, PAGE);
    assert_eq!(next_phase_marker, 44);
}

#[test]
fn disjoint_refreshes_see_alias_writes_on_repeated_submissions() {
    use crate::watch::{PAGE, tests::twin_views, title_memory_remapped};
    use windows_sys::Win32::System::Memory::{MEMORY_MAPPED_VIEW_ADDRESS, UnmapViewOfFile};

    if crate::host::host().is_none() {
        assert!(std::env::var_os("VICTORY_TEST_SHADOW_DISJOINT_REQUIRE_GPU").is_none());
        return;
    }
    let len = 3 * PAGE;
    let (title, alias) = twin_views(3);
    crate::shadow::register(title, title as u64, len as u64);
    crate::shadow::register(alias, alias as u64, len as u64);
    let copy = crate::shadow::find(title as u64, len as u64).unwrap();
    let other_copy = crate::shadow::find(alias as u64, len as u64).unwrap();
    assert!(!std::sync::Arc::ptr_eq(&copy, &other_copy));
    assert_eq!(copy.update(title, len), len);
    assert_eq!(other_copy.update(alias, len), len);

    let mut previous_gap = 0;
    for value in [11u32, 21, 31] {
        // SAFETY: both addresses are in our alias view, outside the gap.
        unsafe {
            (alias as *mut u32).write_volatile(value);
            ((alias + 2 * PAGE) as *mut u32).write_volatile(value + 2);
        }
        let mut batch = Batch::default();
        batch.add(&copy, title, PAGE);
        batch.add(&copy, title + 2 * PAGE, PAGE);
        let gate = std::sync::Barrier::new(2);
        std::thread::scope(|scope| {
            let writer = scope.spawn(|| {
                gate.wait();
                // SAFETY: the gap is outside both active refresh ranges.
                unsafe { ((alias + PAGE) as *mut u32).write_volatile(value + 1) };
            });
            gate.wait();
            let bytes = batch.flush();
            writer.join().unwrap();
            assert_eq!(bytes, 2 * PAGE);
        });
        let markers = |s: &crate::shadow::Shadow, base| {
            [0, 1, 2].map(|page| s.read_host((base + page * PAGE) as u64, 1)[0])
        };
        assert_eq!(markers(&copy, title), [value, previous_gap, value + 2]);
        assert_eq!(
            other_copy.update(alias, len),
            len,
            "the other backing sees its own dirt"
        );
        assert_eq!(markers(&other_copy, alias), [value, value + 1, value + 2]);
        batch.add(&copy, title + PAGE, PAGE);
        assert_eq!(batch.flush(), PAGE);
        assert_eq!(markers(&copy, title), [value, value + 1, value + 2]);
        previous_gap = value + 1;
    }

    crate::shadow::unregister(title, title as u64);
    crate::shadow::unregister(alias, alias as u64);
    drop((copy, other_copy));
    for address in [title, alias] {
        // SAFETY: both views were made by this test and are no longer read.
        assert_ne!(
            unsafe {
                UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: address as *mut _,
                })
            },
            0
        );
        title_memory_remapped(address, len);
    }
}

#[test]
fn disjoint_refreshes_handle_mapping_invalidation_between_submissions() {
    use crate::watch::{
        PAGE,
        tests::{free, pool},
        title_memory_remapped,
    };
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_DECOMMIT, PAGE_READWRITE, VirtualAlloc, VirtualFree,
    };

    if crate::host::host().is_none() {
        assert!(std::env::var_os("VICTORY_TEST_SHADOW_DISJOINT_REQUIRE_GPU").is_none());
        return;
    }
    let title = pool(3);
    let middle = title + PAGE;
    let len = 3 * PAGE;
    // Materialize the outer pages before registering them with write-watch.
    // Otherwise the initial read can register demand-zero faults as writes.
    unsafe { (title as *mut u8).write_bytes(0, len) };
    crate::shadow::register(title, title as u64, len as u64);
    let copy = crate::shadow::find(title as u64, len as u64).unwrap();
    // SAFETY: the committed middle page is owned by the test.
    unsafe { (middle as *mut u8).write_volatile(12) };
    assert_eq!(copy.update(title, len), len);
    assert_eq!(copy.read_host(middle as u64, 1)[0], 12);
    // SAFETY: the middle page is committed inside our reservation.
    assert_ne!(
        unsafe { VirtualFree(middle as *mut _, PAGE, MEM_DECOMMIT) },
        0
    );
    title_memory_remapped(middle, PAGE);

    let mut batch = Batch::default();
    batch.add(&copy, title, PAGE);
    batch.add(&copy, title + 2 * PAGE, PAGE);
    assert_eq!(batch.flush(), 0);
    assert_eq!(copy.read_host(middle as u64, 1)[0], 12);
    batch.add(&copy, middle, PAGE);
    assert_eq!(batch.flush(), PAGE);
    assert_eq!(copy.read_host(middle as u64, 1)[0], 0);
    drop(batch);

    // SAFETY: recommit only the missing region of our live reservation.
    assert_eq!(
        unsafe { VirtualAlloc(middle as *mut _, PAGE, MEM_COMMIT, PAGE_READWRITE) } as usize,
        middle
    );
    title_memory_remapped(middle, PAGE);
    // SAFETY: the recommitted page is writable and stays live below.
    unsafe { (middle as *mut u8).write_volatile(34) };
    let mut batch = Batch::default();
    batch.add(&copy, middle, PAGE);
    assert_eq!(batch.flush(), PAGE);
    assert_eq!(copy.read_host(middle as u64, 1)[0], 34);

    drop(batch);
    crate::shadow::unregister(title, title as u64);
    drop(copy);
    free(title);
}

#[test]
fn disjoint_partial_refreshes_update_declared_write_snapshots() {
    use crate::watch::{
        PAGE,
        tests::{free, pool},
    };

    let Some(host) = crate::host::host() else {
        assert!(std::env::var_os("VICTORY_TEST_SHADOW_DISJOINT_REQUIRE_GPU").is_none());
        return;
    };
    let title = pool(1);
    crate::shadow::register(title, title as u64, PAGE as u64);
    let copy = crate::shadow::find(title as u64, PAGE as u64).unwrap();
    assert_eq!(copy.update(title, PAGE), PAGE);
    let busy = title + 32;
    // The submission is still preparing. An earlier list declared this
    // range writable, but no GPU work in this batch has run yet.
    copy.writing(busy, 4);
    for (offset, value) in [(4, 11u8), (16, 22), (32, 33), (33, 66), (40, 44)] {
        // SAFETY: every marker belongs to the test's live title page.
        unsafe { ((title + offset) as *mut u8).write_volatile(value) };
    }
    let mut batch = Batch::default();
    batch.add(&copy, title + 4, 4);
    batch.add(&copy, busy, 12);
    assert_eq!(
        batch.flush(),
        16,
        "both requested ranges refresh without the gap"
    );
    let markers = || {
        [4, 16, 32, 40].map(|offset| copy.read_host((title + offset) as u64, 1)[0].to_le_bytes()[0])
    };
    assert_eq!(markers(), [11, 0, 33, 44]);
    assert_eq!(
        copy.read_host(busy as u64, 1)[0].to_le_bytes(),
        [33, 66, 0, 0]
    );

    let (upload, data) = host.upload_buffer(4).unwrap();
    let writer = host.new_list().unwrap();
    // SAFETY: both buffers live through the synchronous submission.
    // The GPU changes only the first byte relative to the refreshed
    // snapshot. The title also changes the second byte after refresh.
    unsafe {
        data.cast::<u32>().write(u32::from_le_bytes([55, 66, 0, 0]));
        writer
            .list
            .CopyBufferRegion(&copy.buffer, 32, &upload, 0, 4);
        ((busy + 1) as *mut u8).write_volatile(77);
    }
    writer.close();
    host.execute(&[&writer]);
    // This backend merges only the one byte the GPU changed.
    assert_eq!(copy.copy_back_with(busy, 4, None), 1);
    // SAFETY: the copy-back completed and the test owns these bytes.
    let delivered =
        [busy, busy + 1].map(|address| unsafe { (address as *const u8).read_volatile() });
    assert_eq!(delivered, [55, 77]);

    batch.add(&copy, title + 16, 4);
    assert_eq!(batch.flush(), 4, "the unobserved partial page stays dirty");
    assert_eq!(markers(), [11, 22, 55, 44]);
    drop(batch);
    crate::shadow::unregister(title, title as u64);
    drop(copy);
    free(title);
}
