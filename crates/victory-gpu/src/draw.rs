//! Draws on the host GPU (decision 0007, step 3). A list keeps the state
//! the title sets for its draws (pipeline, descriptor heaps, root
//! arguments, vertex and index buffers, targets); each draw then brings the
//! host list up to date: the host pipeline and our root signature, the
//! user data the translated shaders read, the host copies of the buffers
//! and descriptors they read, and the states of the textures involved.
//! State the host keeps across draws (viewports, scissors, blend factor,
//! stencil reference, topology, targets) is set on the host list as the
//! title sets it.

use std::sync::atomic::{AtomicU64, Ordering};

use victory_shader::resource::RootIndexedView;
use victory_shader::{ResourceType, RootType, UsedResource};

use crate::descriptor::View;
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;
use windows::core::Interface;

use crate::host::HostList;
use crate::object::This;
use crate::pipeline::host::HostPipeline;

/// A root argument as the title set it.
#[derive(Debug, Clone, Default)]
enum Root {
    #[default]
    None,
    /// A descriptor table's first descriptor (an address in a heap).
    Table(u64),
    Constants(Vec<u32>),
    /// A root view's GPU address.
    View(u64),
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct VertexView {
    address: u64,
    size: u32,
    stride: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct IndexView {
    address: u64,
    size: u32,
    format: u32,
}

/// Host states of a depth target, and of a texture shaders read.
const DEPTH_WRITE: u32 = 0x10;
const DEPTH_READ: u32 = 0x20;
const SHADER_RESOURCE: u32 = 0x40 | 0x80;

thread_local! {
    /// While a draw is prepared (`State::bind`, then its targets' states):
    /// the depth target the draw only tests (0: none, or one it writes),
    /// and whether a view its shaders read is of that same texture.
    static TESTED_DEPTH: std::cell::Cell<(usize, bool)> =
        const { std::cell::Cell::new((0, false)) };
    /// While a stage's tables are walked (`State::tables`): whether that
    /// stage's shader can store to buffers (`Translated::stores`).
    static STORES: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
}

/// Whether a buffer range a shader names through an unordered access view
/// is given to shaders as written (snapshot before the submission, compare
/// and write back after) only if that shader's code can store to buffers.
/// The title binds buffers its shaders only read through such views, and
/// every range declared costs a snapshot and a compare of its bytes: 3.97
/// GB a frame in 31 submissions, 0.49 s of snapshots and 0.28 s of
/// comparing (run 28). On by default; `VICTORY_WRITE_SKIP=0` declares
/// every such range, as before.
fn write_skip() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WRITE_SKIP").is_none_or(|v| v != "0"))
}

/// A run of indirect records clamped in one pass ([`windows`]): the host
/// copy they came from (its number), their kind and stride, the offset of
/// the first in that copy, how many, and the list's scratch buffer that
/// holds them clamped, record `i` at `i * step`.
#[derive(Clone)]
struct Window {
    copy: u64,
    kind: u32,
    stride: u32,
    /// How far apart the records are in the copy, and in the scratch
    /// buffer: what the title walks them by ([`windows`]).
    step: u32,
    base: u64,
    records: u32,
    scratch: ID3D12Resource,
}

/// The most records clamped in one pass ([`windows`]).
const WINDOW_MOST: u64 = 64;

/// Whether indirect draws that read record after record of one argument
/// buffer are clamped a window at a time (`VICTORY_INDIRECT_WINDOW=1`; off
/// by default).
///
/// The title issues its mesh draws one record per call, each from the next
/// slot of the same argument buffer: 3,460 a frame in gameplay, and before
/// each our clamp ran as a compute dispatch of its own between two waits,
/// so the card went from drawing to compute and back for every record. In
/// the timing run of the lit scene (run 79) a shadow draw of next to
/// nothing cost 14-16 microseconds, 4,126 records a frame.
///
/// With the switch, when a draw asks for a record a short step after the
/// one drawn before, the clamp takes that record and the ones that step
/// apart after it (16, then twice as many each time, [`WINDOW_MOST`] at
/// most, never past the buffer) into
/// the list's scratch buffer in one pass, and the draws that follow read
/// their record there with no pass of ours, until one of these: a compute
/// dispatch, a direct draw, an indirect call with a count buffer or more
/// than one record, a draw whose shaders can store to buffers, a record
/// outside the window, another host list.
///
/// The step is the one the title is seen to take, not the stride its
/// command signature names: it draws one record a call, for which the
/// stride means nothing, and Grand Theft Auto VI names 12 bytes (the three
/// words of a mesh dispatch) for records it keeps 32 bytes apart (2,173 of
/// a frame's 3,349 mesh records in run 87's listed frame; 283 are 16
/// apart, which ends a window). Walked by the named stride the window
/// never opened: run 87, with the switch, still clamped 5,068 records a
/// frame one at a time.
///
/// What it costs the guard against garbage records: nothing against a
/// hang (every record is bounded as before and charged to the list's
/// budget as before), but the budget is charged when a window is clamped,
/// so also for records of the window's tail that are never drawn. Those
/// hold garbage in this title (slots of all ones: run 93 spent lists'
/// budgets on them and emptied the draws after), so in a window a record
/// that asks for more than can run is left empty and charges nothing
/// (`crate::indirect::bound_ahead`); a tail record that looks real is
/// charged what it asks, a few thousand of 2^27. And arguments the title's CPU side changes are seen as
/// before, at the next submission.
fn windows() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_INDIRECT_WINDOW").is_some_and(|v| v == "1"))
}

/// What one title list has set, and what its host list was last given.
#[derive(Default)]
pub(crate) struct State {
    pipeline: usize,
    /// View heap, sampler heap.
    heaps: [usize; 2],
    graphics: Vec<Root>,
    compute: Vec<Root>,
    vertex: [VertexView; 32],
    index: IndexView,
    /// Bound render target and depth resources, for their host states.
    targets: Vec<usize>,
    depth: usize,
    /// The host state the bound depth view needs ([`depth_state`]).
    depth_state: u32,
    /// The title's primitive topology (the host's may differ).
    topology: u32,
    /// What the title set that goes straight onto the host list, kept for
    /// the host list that continues this one after a cut ([`State::replay`]):
    /// whether a topology was set, the render target and depth handles, the
    /// viewports, the scissors, the blend factor and the stencil reference.
    topology_set: bool,
    target_handles: Vec<u64>,
    depth_handle: u64,
    targets_set: bool,
    viewports: Vec<D3D12_VIEWPORT>,
    scissors: Vec<RECT>,
    blend: Option<[f32; 4]>,
    stencil: Option<u32>,
    /// The indirect records clamped last, if later draws may still read
    /// them ([`windows`]).
    window: Option<Window>,
    /// What the host list has: pipeline, root signature, heaps.
    host_pipeline: usize,
    /// Graphics, compute.
    host_roots: [usize; 2],
    host_heaps: [usize; 2],
    /// Tables made afresh in this list: (table, slot count, contents) → first
    /// dynamic slot.
    fresh_tables: std::collections::HashMap<(u64, u32, u64), u32>,
    /// The dynamic slots the list holds: title heap, first, count; given
    /// back when the list has run, is reset or released.
    dynamic: Vec<(usize, u32, u32)>,
    /// Root SRVs bound as one-view tables in this list: (title heap, title
    /// address; 0 for none) → dynamic slot, and the host copy the slot
    /// shows: later draws of the list bring that one up to date, whatever
    /// copy the address would find by then (`shadow.rs` `tails`).
    root_srv_slots: std::collections::HashMap<(usize, u64), RootSlot>,
    /// Root UAVs bound as one-view tables in this list, likewise.
    root_uav_slots: std::collections::HashMap<(usize, u64), RootSlot>,
    /// The entries the bind being prepared added to the three maps above
    /// ([`State::bind`]): a rejected bind takes them out again.
    bind_made: Vec<BindMade>,
    /// The list ran and its dynamic slots were given back.
    tables_gone: bool,
}

/// An entry a bind added to one of [`State`]'s per-list slot maps.
enum BindMade {
    Table((u64, u32, u64)),
    RootSrv((usize, u64)),
    RootUav((usize, u64)),
}

impl Drop for State {
    fn drop(&mut self) {
        for &(heap, first, n) in &self.dynamic {
            release_slots(heap, first, n);
        }
    }
}

/// The key of our own view heap (`crate::memory::heap`) among the title's.
const MEMORY_HEAP: usize = crate::memory::OURS;

/// Dynamic slots taken for root views' one-view tables
/// ([`State::root_srv_table`], [`State::root_uav`]) since the last call.
pub(crate) fn take_root_view_slots() -> u64 {
    ROOT_VIEW_SLOTS.swap(0, Ordering::Relaxed)
}
static ROOT_VIEW_SLOTS: AtomicU64 = AtomicU64::new(0);

/// Gives back dynamic slots a list held in view heap `heap` (a title
/// heap's address, or ours).
fn release_slots(heap: usize, first: u32, n: u32) {
    match (heap, crate::memory::heap()) {
        (MEMORY_HEAP, Some(ours)) => ours.release(first, n),
        (MEMORY_HEAP, None) => {}
        _ => crate::descriptor::release_dynamic(heap, first, n),
    }
}

impl State {
    /// The list's submission has run (submissions are waited for): the
    /// dynamic slots its tables were made in are given back, not kept
    /// until the title resets the list. A frame of the opening cutscene
    /// asks for 12,500 tables; held until the resets they ran the 131,072
    /// slots out, and up to 4,900 draws a frame went without their tables.
    /// shortcut: a list the title runs a second time without a reset then
    /// reads slots others may have taken (logged once).
    pub(crate) fn tables_done(&mut self) {
        if self.tables_gone && self.dynamic.is_empty() {
            return;
        }
        if self.tables_gone {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| tracing::warn!("a list ran again after its descriptor tables were given back"));
        }
        for (heap, first, n) in self.dynamic.drain(..) {
            release_slots(heap, first, n);
        }
        self.fresh_tables.clear();
        self.root_srv_slots.clear();
        self.root_uav_slots.clear();
        self.tables_gone = true;
    }
}

/// An experiment on the title's scissor rectangles (`VICTORY_SCISSOR_TEST=full`):
/// they are all widened to the whole target. Grand Theft Auto VI's room at
/// the start of the prologue lost its walls in five runs of seven, and in
/// those runs the player's body and the world outside were cut by one
/// rectangle around the window (run 85, frame 2046): one run with this on
/// says whether the walls are drawn and cut away by such a rectangle, or
/// not drawn at all. Off by default: a rectangle the title sets is part of
/// its picture.
fn scissor_test() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_SCISSOR_TEST").as_deref() == Ok("full"))
}

/// An experiment on mesh draws (`VICTORY_MESH_TEST`): 1 for `skip` (none
/// is issued), 2 for `scissor` (each runs its shaders but rasterises one
/// pixel), 0 otherwise. It tells a card hung by a mesh shader from one hung
/// by what the shader's triangles cover.
fn mesh_test() -> u8 {
    static ON: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *ON.get_or_init(|| match std::env::var("VICTORY_MESH_TEST").as_deref() {
        Ok("skip") => 1,
        Ok("scissor") => 2,
        _ => 0,
    })
}

/// `VICTORY_MESH_TEST=keep:<hex digits>[:<hex digits>]`: only the mesh
/// draws whose mesh shader's hash ends in one of the first digits (and
/// whose digit before it is one of the second, if given) are issued: a
/// search for the shaders that stop the card.
fn mesh_kept(pipeline: usize) -> bool {
    static KEEP: std::sync::OnceLock<Option<(String, String)>> = std::sync::OnceLock::new();
    let Some((last, before)) = KEEP.get_or_init(|| {
        let value = std::env::var("VICTORY_MESH_TEST").ok()?;
        let mut parts = value.strip_prefix("keep:")?.split(':');
        Some((parts.next()?.to_owned(), parts.next().unwrap_or("").to_owned()))
    }) else {
        return true;
    };
    let Some(p) = crate::pipeline::host_pipeline(pipeline as This) else {
        return true;
    };
    let Some(at) = p.name.find("mesh ") else {
        return true;
    };
    let hash = p.name[at + 5..].as_bytes();
    if hash.len() < 16 {
        return true;
    }
    last.as_bytes().contains(&hash[15]) && (before.is_empty() || before.as_bytes().contains(&hash[14]))
}

/// Draws recorded on the host GPU since the last frame line.
static DRAWN: AtomicU64 = AtomicU64::new(0);

/// Draws run on the host so far.
pub(crate) fn drawn() -> u64 {
    DRAWN.load(Ordering::Relaxed)
}

/// The draws traced since the last present (diagnostics builds, with
/// debug logging for this module): logged when the frame comes out black.
#[cfg(feature = "diagnostics")]
static FRAME: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

/// The draws traced since the last call.
#[cfg(feature = "diagnostics")]
pub(crate) fn take_frame() -> Vec<String> {
    std::mem::take(&mut *FRAME.lock().unwrap_or_else(|e| e.into_inner()))
}

/// Draws, mesh dispatches and dispatches recorded but not run on the host
/// (no host pipeline) since the last call.
pub(crate) fn take_dropped() -> u64 {
    DROPPED.swap(0, Ordering::Relaxed)
}
/// A root SRV's dynamic slot in a list, and the host copy it shows.
type RootSlot = (u32, Option<std::sync::Arc<crate::shadow::Shadow>>);

/// Buffer views of the heap whose slot showed another host buffer than
/// the copy the pass brought up to date for them, since the start
/// (diagnostics; for the memory line). It stays 0 while the pass and the
/// slots agree; with `VICTORY_VIEW_EXTENT=0` it counts the views that were
/// read stale.
#[cfg(feature = "diagnostics")]
pub(crate) static HEAP_VIEWS_ELSEWHERE: AtomicU64 = AtomicU64::new(0);
/// Passes over the view heap's buffer views for shaders that pick them by
/// index ([`State::heap_buffers`]), and the microseconds they took.
pub(crate) static HEAP_BUFFER_PASSES: AtomicU64 = AtomicU64::new(0);
pub(crate) static HEAP_BUFFER_MICROS: AtomicU64 = AtomicU64::new(0);

/// Recording-time texture work within a heap pass, in microseconds.
#[cfg(feature = "diagnostics")]
#[derive(Default)]
struct HeapTextureTiming {
    remap_us: u64,
    queue_us: u64,
    scan_us: u64,
}

/// Textures brought up to date for shaders that pick them by index
/// ([`State::heap_textures`]), the bytes of their memory, and the slots
/// still waiting.
pub(crate) static HEAP_TEXTURES: AtomicU64 = AtomicU64::new(0);
pub(crate) static HEAP_TEXTURE_BYTES: AtomicU64 = AtomicU64::new(0);
pub(crate) static HEAP_TEXTURES_WAITING: AtomicU64 = AtomicU64::new(0);
/// Passes that made no texture because the card had no room
/// (`crate::resource::room`), for the memory line.
pub(crate) static HEAP_TEXTURES_FULL: AtomicU64 = AtomicU64::new(0);
/// Times a placed texture's slot was passed over because its memory is not
/// mapped.
pub(crate) static HEAP_TEXTURES_AWAY: AtomicU64 = AtomicU64::new(0);
/// Slots the pass found up to date already: their texture is uploaded by
/// the draws that read it through a table (it has a staging buffer).
pub(crate) static HEAP_TEXTURES_TABLED: AtomicU64 = AtomicU64::new(0);
/// The queue of textures picked by index at the last pass
/// (`TextureQueue::census`): slots in it, due, waiting for their memory,
/// uploaded in part, and the frames the oldest due one has waited.
pub(crate) static HEAP_TEXTURES_QUEUE: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];
/// Since the start: looks at a slot uploaded in part that found no more of
/// its memory mapped; uploads of textures not all mapped; and of those,
/// the ones whose first page was not there (asked by the first page alone,
/// as before, they counted as not there at all).
pub(crate) static HEAP_TEXTURES_PARTLY: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
static DROPPED: AtomicU64 = AtomicU64::new(0);

/// Marks the last traced draw as not run on the host, counts it by why
/// (the reason [`not_run`] gave, if its pipeline has a host one) and notes
/// it in the list (diagnostics).
#[cfg(feature = "diagnostics")]
fn dropped(list: &HostList, pipeline: usize, kind: &str) {
    let why = NOT_RUN.with(|w| w.replace(""));
    if why.is_empty() || crate::pipeline::host_pipeline(pipeline as This).is_none() {
        crate::pipeline::note_drop(pipeline as This);
    } else {
        crate::pipeline::note_why(format!("{kind}: {why}"));
    }
    list.note_op(|| format!("DROPPED {kind} ({why}) {}", crate::pipeline::shaders_of(pipeline as This)));
    if let Some(last) = FRAME.lock().unwrap_or_else(|e| e.into_inner()).last_mut() {
        last.push_str(" DROPPED");
    }
}

#[cfg(feature = "diagnostics")]
thread_local! {
    /// Why the draw or dispatch this thread last prepared cannot run.
    static NOT_RUN: std::cell::Cell<&'static str> = const { std::cell::Cell::new("") };
}

/// Notes why the draw or dispatch being prepared cannot run (diagnostics).
fn not_run(why: &'static str) {
    #[cfg(feature = "diagnostics")]
    NOT_RUN.with(|w| w.set(why));
    #[cfg(not(feature = "diagnostics"))]
    let _ = why;
}

/// Dispatches run on the host so far.
static DISPATCHED: AtomicU64 = AtomicU64::new(0);

pub(crate) fn dispatched() -> u64 {
    DISPATCHED.load(Ordering::Relaxed)
}

/// The host address of a root view: its buffer's host copy, or for a
/// constant buffer not 256-byte aligned there (the console needs 16) a
/// copy in upload memory.
/// shortcut: a root view has no size; constant buffers are taken as 64 KiB
/// (cut at their buffer's end), other views as all that is left of their
/// buffer. (They were taken as 1 MiB: 1,452 of the 2,021 mesh vertex
/// streams of a cutscene frame reach further, and what lay past the first
/// megabyte was never brought up to date - meshes without bodies, exploded
/// triangles. `VICTORY_ROOT_VIEW_MB` sets a limit again.)
fn root_view(list: &HostList, address: u64, srv: bool) -> Option<u64> {
    static LIMIT: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    let limit = *LIMIT.get_or_init(|| {
        std::env::var("VICTORY_ROOT_VIEW_MB")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .map_or(u64::MAX, |mb| mb << 20)
    });
    let shadow = crate::shadow::find(address, 1)?;
    let len = shadow
        .left(address)
        .min(if srv { limit } else { 1 << 16 });
    let gpu = shadow.gpu(address);
    if srv || gpu % 256 == 0 {
        shadow.refresh(list, address, len);
        return Some(gpu);
    }
    let (gpu, to) = list.scratch(len)?;
    list.defer_fill(to, address as *const u8, len as usize);
    Some(gpu)
}

macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

fn set(roots: &mut Vec<Root>, slot: u32, root: Root) {
    let slot = slot as usize;
    if roots.len() <= slot {
        roots.resize(slot + 1, Root::None);
    }
    roots[slot] = root;
}

impl State {
    pub(crate) fn reset(&mut self, pipeline: This) {
        *self = State::default();
        self.pipeline = pipeline as usize;
    }

    /// The title's topology; console-only ones become host lists (see
    /// [`crate::primitive`]).
    pub(crate) fn set_topology(&mut self, list: &HostList, topology: u32) {
        self.topology = topology;
        self.topology_set = true;
        let host = match topology {
            RECTANGLES => 4,
            TRIANGLE_FAN | LINE_LOOP | QUADS | QUAD_STRIP => {
                once!(topology, "console topology drawn as a triangle list");
                4
            }
            t => t,
        };
        // SAFETY: an open list.
        unsafe {
            list.list
                .IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY(host as i32))
        };
    }

    pub(crate) fn set_pipeline(&mut self, pipeline: This) {
        self.pipeline = pipeline as usize;
    }

    /// Others recorded into the host list (`crate::external`): its heaps,
    /// root signatures and pipeline are set again before the next draw or
    /// dispatch.
    pub(crate) fn host_state_lost(&mut self) {
        // No title heaps (0) differ from what the host list has, too.
        (self.host_pipeline, self.host_roots, self.host_heaps) = (0, [0; 2], [usize::MAX; 2]);
    }

    pub(crate) fn set_heaps(&mut self, heaps: &[This]) {
        self.heaps = [0; 2];
        for &h in heaps {
            if !h.is_null() {
                let samplers = crate::descriptor::heap(h).samplers();
                self.heaps[usize::from(samplers)] = h as usize;
            }
        }
    }

    fn roots(&mut self, compute: bool) -> &mut Vec<Root> {
        if compute {
            &mut self.compute
        } else {
            &mut self.graphics
        }
    }

    pub(crate) fn set_table(&mut self, compute: bool, slot: u32, handle: u64) {
        set(self.roots(compute), slot, Root::Table(handle));
    }

    pub(crate) fn set_view(&mut self, compute: bool, slot: u32, address: u64) {
        set(self.roots(compute), slot, Root::View(address));
    }

    pub(crate) fn set_constants(&mut self, compute: bool, slot: u32, values: &[u32], offset: u32) {
        let roots = self.roots(compute);
        let s = slot as usize;
        if !matches!(roots.get(s), Some(Root::Constants(_))) {
            set(roots, slot, Root::Constants(Vec::new()));
        }
        if let Root::Constants(c) = &mut roots[s] {
            let end = offset as usize + values.len();
            if c.len() < end {
                c.resize(end, 0);
            }
            c[offset as usize..end].copy_from_slice(values);
        }
    }

    pub(crate) fn set_vertex_buffers(&mut self, start: u32, views: &[VertexView]) {
        for (i, v) in views.iter().enumerate() {
            if let Some(slot) = self.vertex.get_mut(start as usize + i) {
                *slot = *v;
            }
        }
    }

    pub(crate) fn clear_vertex_buffers(&mut self, start: u32, count: u32) {
        for slot in self.vertex.iter_mut().skip(start as usize).take(count as usize) {
            *slot = VertexView::default();
        }
    }

    pub(crate) fn set_index_buffer(&mut self, view: Option<&IndexView>) {
        self.index = view.copied().unwrap_or_default();
    }

    /// Render targets and depth, set on the host list at once.
    pub(crate) fn set_targets(&mut self, list: &HostList, handles: &[u64], depth: u64) {
        let Some(host) = crate::host::host() else {
            return;
        };
        self.target_handles.clear();
        self.target_handles.extend_from_slice(handles);
        self.depth_handle = depth;
        self.targets_set = true;
        self.targets.clear();
        let mut rtvs = Vec::new();
        for &h in handles {
            let Some(record) = crate::descriptor::rtv_record(h) else {
                once!(
                    handle = format_args!("{h:#x}"),
                    "render target without a recorded view"
                );
                continue;
            };
            let Some(texture) = crate::resource::host_resource(record.resource) else {
                // A typed null view (or a target without a host copy) still
                // takes its slot: left out, the next live target would move
                // into it and the pixel shader's outputs reach the wrong
                // targets.
                // shortcut: a view without a description or a format has no
                // null view to stand for it and is left out, as before.
                let null = record
                    .desc()
                    .filter(|d| d.Format.0 != 0 && d.ViewDimension.0 != 0);
                if let Some(desc) = null {
                    rtvs.push(host.rtv(None, Some(&desc)));
                }
                continue;
            };
            self.targets.push(record.resource as usize);
            #[cfg(feature = "diagnostics")]
            crate::capture::note_target(list, record.resource);
            rtvs.push(host.rtv(Some(texture), record.desc().as_ref()));
        }
        self.depth = 0;
        self.depth_state = DEPTH_WRITE;
        let dsv = crate::descriptor::dsv_record(depth).and_then(|record| {
            let texture = crate::resource::host_resource(record.resource)?;
            self.depth = record.resource as usize;
            self.depth_state = depth_state(texture, record.depth_desc().as_ref());
            if crate::resource::depth_and_unordered(record.resource) {
                crate::resource::DEPTH_UNORDERED[1].fetch_add(1, Ordering::Relaxed);
            }
            #[cfg(feature = "diagnostics")]
            crate::capture::note_target(list, record.resource);
            Some(host.dsv(texture, record.depth_desc().as_ref()))
        });
        // SAFETY: an open list; the views were just made.
        unsafe {
            list.list.OMSetRenderTargets(
                rtvs.len() as u32,
                (!rtvs.is_empty()).then_some(rtvs.as_ptr()),
                false,
                dsv.as_ref().map(std::ptr::from_ref),
            )
        };
    }

    /// Viewports, scissors, the blend factor and the stencil reference: set
    /// on the host list at once, and kept for [`State::replay`].
    pub(crate) fn set_viewports(&mut self, list: &HostList, viewports: &[D3D12_VIEWPORT]) {
        self.viewports.clear();
        self.viewports.extend_from_slice(viewports);
        crate::draw::viewports(list, viewports);
    }

    pub(crate) fn set_scissors(&mut self, list: &HostList, rects: &[RECT]) {
        self.scissors.clear();
        self.scissors.extend_from_slice(rects);
        if scissor_test() {
            // The experiment: every rectangle the title sets is widened to
            // the largest target there is, so nothing is cut by one. What
            // the listing shows (`op_text`) stays the title's own.
            let full = RECT {
                left: 0,
                top: 0,
                right: 16384,
                bottom: 16384,
            };
            crate::draw::scissors(list, &vec![full; rects.len()]);
            return;
        }
        crate::draw::scissors(list, rects);
    }

    pub(crate) fn set_blend_factor(&mut self, list: &HostList, factor: Option<&[f32; 4]>) {
        self.blend = factor.copied();
        crate::draw::blend_factor(list, factor);
    }

    pub(crate) fn set_stencil_ref(&mut self, list: &HostList, reference: u32) {
        self.stencil = Some(reference);
        crate::draw::stencil_ref(list, reference);
    }

    /// `list` is a new host list that continues the title's list (it was
    /// cut at a wait for a memory value, `crate::list::cut`): a new host
    /// list has nothing set, so what the title set so far is set again.
    /// The pipeline, the root signatures and the heaps are set by the next
    /// draw or dispatch (`bind`, which is told here that the host list has
    /// none); root arguments, vertex and index buffers are set by every
    /// draw anyway. The dynamic descriptor slots and the tables made in
    /// them stay: the parts before the cut have not run yet, and they are
    /// given back when the title's list is reset.
    /// shortcut: the render target views are made again from the handles'
    /// descriptors as they are now, not as they were when the title set
    /// the targets (a descriptor rewritten in between names another view).
    pub(crate) fn replay(&mut self, list: &HostList) {
        self.host_pipeline = 0;
        self.host_roots = [0; 2];
        self.host_heaps = [0; 2];
        if self.topology_set {
            self.set_topology(list, self.topology);
        }
        if self.targets_set {
            let handles = std::mem::take(&mut self.target_handles);
            self.set_targets(list, &handles, self.depth_handle);
        }
        if !self.viewports.is_empty() {
            crate::draw::viewports(list, &self.viewports);
        }
        if !self.scissors.is_empty() {
            let rects = std::mem::take(&mut self.scissors);
            self.set_scissors(list, &rects);
        }
        if let Some(factor) = self.blend {
            crate::draw::blend_factor(list, Some(&factor));
        }
        if let Some(reference) = self.stencil {
            crate::draw::stencil_ref(list, reference);
        }
    }

    /// The user-data value of a root argument, as translated shaders read
    /// it: a table's byte offset in its heap, a constant's value, a root
    /// view's address (title memory).
    fn user_value(&self, compute: bool, slot: u8, root: RootType, dword: u32) -> u32 {
        match (self.roots_ref(compute).get(usize::from(slot)), root) {
            (Some(Root::View(address)), RootType::Descriptor) if dword < 2 => {
                (address >> (32 * dword)) as u32
            }
            (Some(Root::Table(handle)), RootType::Table) => match self.heap_of(*handle) {
                Some(heap) => (handle - heap.start) as u32,
                None => {
                    once!(
                        handle = format_args!("{handle:#x}"),
                        "descriptor table outside the bound heaps"
                    );
                    0
                }
            },
            (Some(Root::Constants(c)), RootType::Constants) => {
                c.get(dword as usize).copied().unwrap_or(0)
            }
            // The host's own ([`Self::host_word`]): no root argument.
            (_, RootType::Other(_)) => 0,
            _ => {
                once!(
                    slot,
                    ?root,
                    "a shader reads a root argument the title did not set"
                );
                0
            }
        }
    }

    /// A user-data dword the host computes for a translated shader
    /// (`victory_shader::emit::USER_MEMORY_BASE`, `USER_VIEW_ADDRESS`).
    fn host_word(
        &self,
        compute: bool,
        kind: u32,
        slot: u8,
        dword: u32,
        memory: Option<&crate::memory::Ready>,
    ) -> u32 {
        use victory_shader::emit::{USER_MEMORY_BASE, USER_VIEW_ADDRESS, USER_VIEW_BIAS};
        if kind == USER_MEMORY_BASE {
            return memory.map_or(0, |m| m.base);
        }
        if kind == USER_VIEW_BIAS {
            // The view begins on 16 bytes of the copy the address finds
            // (`root_uav`, `root_srv_table`).
            // shortcut: found again here, not taken from the list's slot.
            return match self.roots_ref(compute).get(usize::from(slot)) {
                Some(&Root::View(address)) => crate::shadow::find(address, 1).map_or(0, |s| {
                    // SAFETY: a live buffer.
                    let base = unsafe { s.buffer.GetGPUVirtualAddress() };
                    ((s.gpu(address) - base) & 15) as u32
                }),
                _ => 0,
            };
        }
        if kind != USER_VIEW_ADDRESS && kind != USER_VIEW_ADDRESS + 1 {
            once!(kind, "a shader reads a host value this build does not know");
            return 0;
        }
        // The title address of the buffer view at that dword of the table.
        let address = match self.roots_ref(compute).get(usize::from(slot)) {
            Some(&Root::Table(table)) => crate::descriptor::view(table + u64::from(dword) * 4)
                .and_then(|view| crate::descriptor::host::buffer_range(&view))
                .map_or(0, |range| range.0),
            _ => 0,
        };
        if address == 0 {
            once!(slot, dword, "a shader takes the address of a buffer view that is not there: it reads 0");
        }
        if kind == USER_VIEW_ADDRESS {
            address as u32
        } else {
            (address >> 32) as u32
        }
    }

    /// For a shader that reads title memory by address
    /// ([`crate::memory`]): its directory, fault notes and view slots,
    /// with the pages of the root views the list has set made ready.
    /// With no view heap of the title's bound, the copies' views go into
    /// ours (`bind` binds it for such a shader).
    fn memory_ready(&self, list: &HostList, compute: bool, writes: bool) -> Option<crate::memory::Ready> {
        let (key, heap) = match self.heaps[0] {
            0 => (MEMORY_HEAP, crate::memory::heap()?),
            h => (h, crate::descriptor::heap(h as This).host()?),
        };
        let (mut addresses, mut constants) = (Vec::new(), Vec::new());
        for root in self.roots_ref(compute) {
            match root {
                Root::View(address) => addresses.push(*address),
                Root::Constants(values) => constants.extend_from_slice(values),
                _ => {}
            }
        }
        crate::memory::ready(list, key, heap, &addresses, &constants, writes)
    }

    /// The bound heap holding `handle`.
    fn heap_of(&self, handle: u64) -> Option<&'static crate::descriptor::Heap> {
        self.heaps
            .iter()
            .filter(|&&h| h != 0)
            .map(|&h| crate::descriptor::heap(h as This))
            .find(|h| h.index(handle).is_some())
    }

    /// Brings the host list up to date for a draw; false if the draw cannot
    /// run on the host.
    fn prepare(&mut self, list: &HostList, indexed: bool) -> bool {
        self.prepare_for(list, indexed, false)
    }

    /// As [`State::prepare`], for draws (`mesh` false) or mesh dispatches.
    fn prepare_for(&mut self, list: &HostList, indexed: bool, mesh: bool) -> bool {
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            return false;
        }
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return false;
        };
        if p.compute || p.mesh != mesh {
            once!(mesh, "a draw or mesh dispatch with another kind of pipeline");
            not_run("another kind of pipeline");
            return false;
        }
        if p.rects && self.topology != RECTANGLES {
            // Its geometry stage would turn each triangle into a rectangle.
            once!("a rectangle pipeline drawn with another topology");
        }
        if !self.bind(list, p, false) {
            return false;
        }
        let l = &list.list;
        // Vertex buffers the input layout reads.
        let mut views = [D3D12_VERTEX_BUFFER_VIEW::default(); 32];
        let mut count = 0;
        for (slot, (view, &v)) in views.iter_mut().zip(&self.vertex).enumerate() {
            if p.slots & 1 << slot == 0 {
                continue;
            }
            count = slot + 1;
            if v.address == 0 || v.size == 0 {
                continue;
            }
            let Some(shadow) = crate::shadow::find_any(v.address, u64::from(v.size)) else {
                continue;
            };
            shadow.refresh(list, v.address, u64::from(v.size));
            *view = D3D12_VERTEX_BUFFER_VIEW {
                BufferLocation: shadow.gpu(v.address),
                SizeInBytes: v.size,
                StrideInBytes: v.stride,
            };
        }
        // SAFETY: an open list; the views name live host buffers.
        unsafe {
            if count > 0 {
                l.IASetVertexBuffers(0, Some(&views[..count]));
            }
        }
        if indexed {
            let v = self.index;
            let Some(shadow) = crate::shadow::find_any(v.address, u64::from(v.size)) else {
                not_run("index buffer outside title buffers");
                return false;
            };
            shadow.refresh(list, v.address, u64::from(v.size));
            let view = D3D12_INDEX_BUFFER_VIEW {
                BufferLocation: shadow.gpu(v.address),
                SizeInBytes: v.size,
                Format: DXGI_FORMAT(v.format as i32),
            };
            // SAFETY: an open list; the view names a live host buffer.
            unsafe { l.IASetIndexBuffer(Some(&view)) };
        }
        const RENDER_TARGET: u32 = 0x4;
        for &t in &self.targets {
            crate::list::ensure_state(list, t as This, RENDER_TARGET);
        }
        if self.depth != 0 {
            // A depth texture shaders also write: what they wrote, first.
            list.twin_use(self.depth as This, false, true);
            // A view that writes neither plane keeps the texture readable:
            // the depth read state, together with the shader read states
            // if this draw's shaders read the same texture (the write state
            // put it out of the state those views need, and a move to it
            // and back for every such draw).
            let state = if self.depth_state == DEPTH_READ {
                let (tested, read) = TESTED_DEPTH.get();
                if read && tested == self.depth {
                    DEPTH_READ | SHADER_RESOURCE
                } else {
                    DEPTH_READ
                }
            } else {
                DEPTH_WRITE
            };
            crate::list::ensure_state(list, self.depth as This, state);
        }
        // A depth copy of ours (`crate::twin`) ran a compute pipeline.
        if list.take_rebind() {
            // SAFETY: an open list and a live pipeline.
            unsafe { l.SetPipelineState(&p.pso) };
            (self.host_pipeline, self.host_roots[1]) = (p.pso.as_raw() as usize, 0);
        }
        true
    }

    /// A descriptor a shader reads through a table: its host slot made if
    /// it changed, constant data copied, textures made readable.
    /// What draws and dispatches share: heaps, our root signature, the
    /// pipeline, and per stage the user data, root views and the
    /// descriptors its tables hold.
    /// False if the draw or dispatch must not run: a root SRV it reads
    /// could not be given a view.
    pub(crate) fn bind(&mut self, list: &HostList, p: &HostPipeline, compute: bool) -> bool {
        let reserved = self.dynamic.len();
        self.bind_made.clear();
        let bound = self.bind_attempt(list, p, compute);
        if !bound {
            self.release_rejected(reserved);
        }
        self.bind_made.clear();
        bound
    }

    /// Release the descriptors reserved for an operation that will never be
    /// recorded. Earlier draws can still reference reservations before the mark.
    fn release_rejected(&mut self, reserved: usize) {
        for made in self.bind_made.drain(..) {
            match made {
                BindMade::Table(key) => {
                    self.fresh_tables.remove(&key);
                }
                BindMade::RootSrv(key) => {
                    self.root_srv_slots.remove(&key);
                }
                BindMade::RootUav(key) => {
                    self.root_uav_slots.remove(&key);
                }
            }
        }
        for (heap, first, n) in self.dynamic.drain(reserved..) {
            release_slots(heap, first, n);
        }
    }

    /// [`State::bind`] before a rejection's slots are given back.
    fn bind_attempt(&mut self, list: &HostList, p: &HostPipeline, compute: bool) -> bool {
        let l = &list.list;
        // A depth copy of ours (`crate::twin`) since the last one ran a
        // compute pipeline.
        if list.take_rebind() {
            (self.host_pipeline, self.host_roots[1]) = (0, 0);
        }
        // Heaps first: a root signature with directly indexed heaps must be
        // set after them. Title memory's views need a view heap: ours where
        // the title binds none (the BVH builder's dispatches may not).
        let memory_heap = self.heaps[0] == 0 && p.stages.iter().any(|s| s.translated().host.memory);
        self.bind_heaps(list, memory_heap);
        // The depth target a draw only tests, for the views of it that the
        // draw's shaders read (`readable`).
        let tested = if !compute && self.depth_state == DEPTH_READ {
            self.depth
        } else {
            0
        };
        TESTED_DEPTH.set((tested, false));
        let root = p.root.as_raw() as usize;
        if self.host_roots[usize::from(compute)] != root {
            // SAFETY: an open list and a live root signature.
            unsafe {
                if compute {
                    l.SetComputeRootSignature(&p.root);
                } else {
                    l.SetGraphicsRootSignature(&p.root);
                }
            }
            self.host_roots[usize::from(compute)] = root;
        }
        // By host object: a title pipeline released and another made at its
        // address is another host pipeline (the list keeps the old one).
        if self.host_pipeline != p.pso.as_raw() as usize {
            // SAFETY: an open list and a live pipeline.
            unsafe { l.SetPipelineState(&p.pso) };
            self.host_pipeline = p.pso.as_raw() as usize;
            list.used_pipeline(&p.name);
            // The title may release the pipeline before it submits the list.
            list.keep(&p.pso);
            list.keep(&p.root);
        }
        #[cfg(feature = "diagnostics")]
        crate::capture::binding(&p.name);
        // Shaders taking views from the heap itself read (or write) views no
        // table names: everything the bound heap's views reach is brought up
        // to date at submission (`crate::bindless`).
        if p.stages.iter().any(|s| s.translated().host.heap) {
            let writes = p.stages.iter().any(|s| s.translated().host.heap_writes);
            if let Some(host) = (self.heaps[0] != 0)
                .then(|| crate::descriptor::heap(self.heaps[0] as This).host())
                .flatten()
            {
                list.defer_heap(&host.viewed);
            }
            crate::bindless::note_draw(writes);
        }
        for stage in &p.stages {
            let t = stage.translated();
            STORES.set(t.stores);
            let made = self.tables(list, &t.used, &t.host.root_indexed_views, compute);
            STORES.set(true);
            // A depth copy of ours made readable a texture the tables hold
            // (`crate::twin`), with a compute pipeline: this one's again
            // (root arguments are set below).
            if list.take_rebind() {
                // SAFETY: an open list, a live root signature and pipeline.
                unsafe {
                    if compute {
                        l.SetComputeRootSignature(&p.root);
                    }
                    l.SetPipelineState(&p.pso);
                }
                self.host_pipeline = p.pso.as_raw() as usize;
                self.host_roots[1] = if compute { root } else { 0 };
            }
            if made.iter().any(|m| m.1 == u32::MAX) {
                static DROPPED_TABLES: AtomicU64 = AtomicU64::new(0);
                let n = DROPPED_TABLES.fetch_add(1, Ordering::Relaxed) + 1;
                if n.is_power_of_two() {
                    tracing::warn!(
                        dropped = n,
                        "a draw or dispatch dropped: no dynamic slots for its tables"
                    );
                }
                not_run("no dynamic slots for its tables");
                return false;
            }
            // What the shader stores through root views and raw addresses is
            // copied back into title memory afterwards (not the structure
            // builder's under `VICTORY_BVH_QUARANTINE=1`).
            let quarantined = crate::pipeline::builder_quarantined(self.pipeline as This);
            let writes = (t.stores || !write_skip()) && !quarantined;
            // Title memory by address: what the shader finds it by.
            let memory = if t.host.memory {
                // Our heap holds no view of the title's: a shader that also
                // picks views from the heap would index past it.
                if memory_heap
                    && (t.host.heap || t.host.root_srvs.contains(&victory_shader::emit::SLOT_STRIDES))
                {
                    once!("a shader that picks views from a view heap, with none bound: the draw is dropped");
                    not_run("heap views without a view heap");
                    return false;
                }
                let Some(ready) = self.memory_ready(list, compute, t.host.memory_writes && !quarantined) else {
                    once!("a shader that reads title memory by address without a view heap to hold its views: the draw is dropped");
                    not_run("title memory without a view heap");
                    return false;
                };
                crate::memory::note_use(&p.name);
                Some(ready)
            } else {
                None
            };
            if let Some((param, dwords)) = stage.user_data {
                let mut values = vec![0u32; dwords as usize];
                for u in &t.host.user_data {
                    if let RootType::Other(kind) = u.root {
                        values[u.index as usize] =
                            self.host_word(compute, kind, u.slot, u.dword, memory.as_ref());
                        continue;
                    }
                    let fresh = made
                        .iter()
                        .find(|m| m.0 == u.slot && u.root == RootType::Table);
                    values[u.index as usize] = match fresh {
                        Some(&(_, offset)) => offset,
                        None => self.user_value(compute, u.slot, u.root, u.dword),
                    };
                }
                // SAFETY: an open list; `values` covers the parameter.
                unsafe {
                    if compute {
                        l.SetComputeRoot32BitConstants(param, dwords, values.as_ptr().cast(), 0);
                    } else {
                        l.SetGraphicsRoot32BitConstants(param, dwords, values.as_ptr().cast(), 0);
                    }
                }
            }
            for (views, srv) in [(&stage.root_cbvs, false), (&stage.root_srvs, true)] {
                for &(slot, param) in views {
                    if srv && slot == victory_shader::emit::SLOT_MEMORY {
                        // Ours: the directory of title memory by address. A
                        // root view left unbound is undefined: no draw.
                        let Some(m) = &memory else {
                            once!("a shader reads the title-memory directory without `HostShader::memory`: dropped");
                            not_run("title-memory directory without its metadata");
                            return false;
                        };
                        // SAFETY: an open list; the directory lives as long
                        // as the process and the list holds it.
                        unsafe {
                            if compute {
                                l.SetComputeRootShaderResourceView(param, m.directory);
                            } else {
                                l.SetGraphicsRootShaderResourceView(param, m.directory);
                            }
                        }
                        continue;
                    }
                    if srv && slot == victory_shader::emit::SLOT_STRIDES {
                        // Ours, not the title's: the shader picks buffers
                        // from the view heap by an index it computes.
                        // Without the table the shader would read an unbound
                        // root view: no draw.
                        let Some(gpu) = self.heap_buffers(list) else {
                            once!("a shader picks buffers from a view heap that is not set: dropped");
                            not_run("view strides without a view heap");
                            return false;
                        };
                        // SAFETY: an open list; the table lives as long as
                        // the heap the list holds.
                        unsafe {
                            if compute {
                                l.SetComputeRootShaderResourceView(param, gpu);
                            } else {
                                l.SetGraphicsRootShaderResourceView(param, gpu);
                            }
                        }
                        continue;
                    }
                    let tabled = srv && crate::pipeline::host::bounded_root_srvs();
                    let Some(&Root::View(address)) = self.roots_ref(compute).get(usize::from(slot))
                    else {
                        once!(slot, "a shader reads a root view the title did not set");
                        // A table left unset is undefined: a null view.
                        if tabled && !self.root_srv_table(list, None, param, compute, false) {
                            return false;
                        }
                        continue;
                    };
                    // The shader adds the bytes a raw view begins early by.
                    let biased = srv
                        && t.host.user_data.iter().any(|u| {
                            u.slot == slot
                                && u.root == RootType::Other(victory_shader::emit::USER_VIEW_BIAS)
                        });
                    if tabled {
                        if !self.root_srv_table(list, Some(address), param, compute, biased) {
                            return false;
                        }
                        continue;
                    }
                    let Some(mut gpu) = root_view(list, address, srv) else {
                        continue;
                    };
                    if biased {
                        gpu -= u64::from(self.host_word(
                            compute,
                            victory_shader::emit::USER_VIEW_BIAS,
                            slot,
                            0,
                            None,
                        ));
                    }
                    // SAFETY: an open list; the address is in live memory.
                    unsafe {
                        match (compute, srv) {
                            (false, false) => l.SetGraphicsRootConstantBufferView(param, gpu),
                            (false, true) => l.SetGraphicsRootShaderResourceView(param, gpu),
                            (true, false) => l.SetComputeRootConstantBufferView(param, gpu),
                            (true, true) => l.SetComputeRootShaderResourceView(param, gpu),
                        }
                    }
                }
            }
            for &(slot, param) in &stage.root_uavs {
                if slot == victory_shader::emit::SLOT_FAULTS {
                    // Ours: where the list's shaders note the pages they
                    // found empty (`crate::memory`).
                    let Some(m) = &memory else {
                        once!("a shader notes title-memory faults without `HostShader::memory`: dropped");
                        not_run("title-memory faults without its metadata");
                        return false;
                    };
                    // SAFETY: an open list; the list holds the buffer.
                    unsafe {
                        if compute {
                            l.SetComputeRootUnorderedAccessView(param, m.faults);
                        } else {
                            l.SetGraphicsRootUnorderedAccessView(param, m.faults);
                        }
                    }
                    continue;
                }
                let address = match self.roots_ref(compute).get(usize::from(slot)) {
                    Some(&Root::View(address)) => Some(address),
                    _ => {
                        once!(slot, "a shader writes through a root view the title did not set");
                        None
                    }
                };
                if !self.root_uav(list, address, param, compute, writes) {
                    return false;
                }
            }
        }
        // Depth textures it writes through unordered access: their colour
        // twins hold what they lack from here on.
        list.twins_written();
        true
    }

    /// Binds a root UAV: a raw unordered access view of the title buffer's
    /// host copy from `address` on, as a one-view table (a null view
    /// without an address, without a host copy shaders can write, or at an
    /// address not on 16 bytes), or as a root descriptor with
    /// `VICTORY_ROOT_SRV_DIRECT=1` (the dispatch is dropped then if there
    /// is nothing to bind). With `writes`, what the shader stored is copied
    /// back into title memory when the list has run. False if the draw
    /// must be dropped.
    /// shortcut: a root view has no size: the view, the refresh and the
    /// copy back cover all that is left of the title's buffer
    /// (`VICTORY_ROOT_UAV_MB` sets a limit for all three).
    fn root_uav(
        &mut self,
        list: &HostList,
        address: Option<u64>,
        param: u32,
        compute: bool,
        writes: bool,
    ) -> bool {
        const UNORDERED: i32 = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0;
        static LIMIT: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
        let limit = *LIMIT.get_or_init(|| {
            std::env::var("VICTORY_ROOT_UAV_MB")
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
                .map_or(u64::MAX, |mb| mb << 20)
        });
        let find = |a: u64| {
            let s = crate::shadow::find(a, 1)?;
            // SAFETY: a live buffer.
            let flags = unsafe { s.buffer.GetDesc() }.Flags;
            if flags.0 & UNORDERED == 0 {
                once!("a root UAV of a buffer whose host copy shaders cannot write");
                return None;
            }
            Some(s)
        };
        let ready = |a: u64, s: &std::sync::Arc<crate::shadow::Shadow>| {
            let len = s.left(a).min(limit);
            s.refresh(list, a, len);
            if writes {
                s.gpu_writes(list, a, len);
            }
        };
        if !crate::pipeline::host::bounded_root_srvs() {
            let Some((a, s)) = address.and_then(|a| Some((a, find(a)?))) else {
                once!("a root UAV with nothing to bind: the draw is dropped");
                return false;
            };
            ready(a, &s);
            // From the 16 bytes below: the shader adds the rest.
            // SAFETY: a live buffer.
            let base = unsafe { s.buffer.GetGPUVirtualAddress() };
            let gpu = base + ((s.gpu(a) - base) & !15);
            // SAFETY: an open list; the address is in a live buffer.
            unsafe {
                if compute {
                    list.list.SetComputeRootUnorderedAccessView(param, gpu);
                } else {
                    list.list.SetGraphicsRootUnorderedAccessView(param, gpu);
                }
            }
            return true;
        }
        let heap = match self.heaps[0] {
            0 => None,
            h => Some(crate::descriptor::heap(h as This)),
        };
        let Some((heap, host)) = heap.and_then(|h| Some((h, h.host()?))) else {
            once!("a root UAV without a bound view heap: the draw is dropped");
            return false;
        };
        let key = (std::ptr::from_ref(heap) as usize, address.unwrap_or(0));
        let (index, shadow) = match self.root_uav_slots.get(&key) {
            Some((index, shadow)) => (*index, address.zip(shadow.clone())),
            None => {
                let shadow = address.and_then(|a| find(a).map(|s| (a, s)));
                let Some(index) = host.dynamic(1) else {
                    return false;
                };
                ROOT_VIEW_SLOTS.fetch_add(1, Ordering::Relaxed);
                self.dynamic.push((key.0, index, 1));
                self.bind_made.push(BindMade::RootUav(key));
                if let Some((a, s)) = &shadow {
                    // SAFETY: a live buffer.
                    let base = unsafe { s.buffer.GetGPUVirtualAddress() };
                    let offset = s.gpu(*a) - base;
                    // Raw views start on 16 bytes: from the 16 bytes
                    // below, and the shader adds the rest to its offsets
                    // (`victory_shader::emit::USER_VIEW_BIAS`). (An address
                    // not on 16 bytes got a null view at first: the
                    // builder's counters are at such addresses.)
                    let bias = offset & 15;
                    host.raw_uav_at(index, &s.buffer, offset - bias, s.left(*a).min(limit) + bias);
                }
                if shadow.is_none() {
                    host.null_raw_uav_at(index);
                }
                self.root_uav_slots
                    .insert(key, (index, shadow.as_ref().map(|(_, s)| s.clone())));
                (index, shadow)
            }
        };
        if let Some((a, s)) = &shadow {
            ready(*a, s);
        }
        let handle = host.gpu_handle(index);
        // SAFETY: an open list; the slot is in the bound view heap.
        unsafe {
            if compute {
                list.list.SetComputeRootDescriptorTable(param, handle);
            } else {
                list.list.SetGraphicsRootDescriptorTable(param, handle);
            }
        }
        #[cfg(feature = "diagnostics")]
        crate::capture::binding("");
        true
    }

    /// Gives the host list the title's heaps if it has others.
    fn bind_heaps(&mut self, list: &HostList, memory_heap: bool) {
        let want = if memory_heap {
            [MEMORY_HEAP, self.heaps[1]]
        } else {
            self.heaps
        };
        if self.host_heaps != want {
            let mut heaps: Vec<Option<ID3D12DescriptorHeap>> = self
                .heaps
                .iter()
                .filter(|&&h| h != 0)
                .filter_map(|&h| crate::descriptor::heap(h as This).host())
                .map(|h| Some(h.heap.clone()))
                .collect();
            if memory_heap && let Some(h) = crate::memory::heap() {
                heaps.insert(0, Some(h.heap.clone()));
            }
            heaps.iter().flatten().for_each(|h| list.keep(h));
            // SAFETY: an open list and shader-visible heaps.
            unsafe { list.list.SetDescriptorHeaps(&heaps) };
            self.host_heaps = want;
            self.host_roots = [0; 2];
        }
    }

    /// `ClearUnorderedAccessViewUint` or `...Float` (`float`): the view the
    /// title made at `cpu` (its CPU handle; the GPU one names the same view
    /// in the bound heap) cleared to `values` on the host, through a
    /// dynamic slot of the bound view heap that the list holds until reset.
    pub(crate) fn clear_unordered(
        &mut self,
        list: &HostList,
        cpu: u64,
        values: [u32; 4],
        float: bool,
        rects: &[RECT],
    ) {
        let Some(view) = crate::descriptor::view(cpu) else {
            once!("an unordered access clear of a slot without a view");
            return;
        };
        let heap = (self.heaps[0] != 0).then(|| crate::descriptor::heap(self.heaps[0] as This));
        let (Some(heap), Some(host)) = (heap, crate::host::host()) else {
            once!("an unordered access clear without a bound view heap");
            return;
        };
        let Some(visible) = heap.host() else { return };
        let Some(slot) = visible.dynamic(1) else {
            return;
        };
        self.dynamic
            .push((std::ptr::from_ref(heap) as usize, slot, 1));
        visible.make_at(slot, &view);
        let at = host.cpu_view();
        let Some(resource) = crate::descriptor::host::make_view(&view, at) else {
            once!("an unordered access clear of a view the host cannot make");
            return;
        };
        // What the clear writes goes back into title memory (a buffer).
        readable(list, ResourceType::Uav, &view);
        self.bind_heaps(list, false);
        list.keep(&resource);
        let gpu = visible.gpu_handle(slot);
        // SAFETY: an open list; both slots view the live resource.
        unsafe {
            if float {
                let values = values.map(f32::from_bits);
                list.list
                    .ClearUnorderedAccessViewFloat(gpu, at, &resource, &values, rects);
            } else {
                list.list
                    .ClearUnorderedAccessViewUint(gpu, at, &resource, &values, rects);
            }
        }
    }

    /// Binds a root SRV as a one-view table
    /// ([`crate::pipeline::host::bounded_root_srvs`]): a raw view of the
    /// title buffer's host copy from `address` to the copy's end, or a null
    /// view (no address, no host copy, or an address not on 16 bytes). One
    /// dynamic slot per address per list, given back with the list's
    /// others. False if no slot could be had or no view heap is bound.
    fn root_srv_table(
        &mut self,
        list: &HostList,
        address: Option<u64>,
        param: u32,
        compute: bool,
        biased: bool,
    ) -> bool {
        let heap = match self.heaps[0] {
            0 => None,
            h => Some(crate::descriptor::heap(h as This)),
        };
        let Some((heap, host)) = heap.and_then(|h| Some((h, h.host()?))) else {
            once!("a root SRV without a bound view heap: the draw is dropped");
            return false;
        };
        // A view that begins early is another view than one that does not
        // (title addresses stay below bit 63).
        let key = (
            std::ptr::from_ref(heap) as usize,
            address.unwrap_or(0) | u64::from(biased) << 63,
        );
        let (index, shadow) = match self.root_srv_slots.get(&key) {
            Some((index, shadow)) => (*index, address.zip(shadow.clone())),
            None => {
                let shadow = address.and_then(|a| crate::shadow::find(a, 1).map(|s| (a, s)));
                let Some(index) = host.dynamic(1) else {
                    return false;
                };
                ROOT_VIEW_SLOTS.fetch_add(1, Ordering::Relaxed);
                self.dynamic.push((key.0, index, 1));
                self.bind_made.push(BindMade::RootSrv(key));
                self.root_srv_slots
                    .insert(key, (index, shadow.as_ref().map(|(_, s)| s.clone())));
                match &shadow {
                    Some((a, s)) => {
                        // SAFETY: a live buffer.
                        let base = unsafe { s.buffer.GetGPUVirtualAddress() };
                        let offset = s.gpu(*a) - base;
                        if biased {
                            // From the 16 bytes below; the shader adds the
                            // rest (`victory_shader::emit::USER_VIEW_BIAS`).
                            let bias = offset & 15;
                            host.raw_at(index, &s.buffer, offset - bias, s.left(*a) + bias);
                        } else if offset.is_multiple_of(16) {
                            host.raw_at(index, &s.buffer, offset, s.left(*a));
                        } else {
                            // Raw views start on 16 bytes; one from the
                            // aligned start would shift every read.
                            once!(
                                address = format_args!("{a:#x}"),
                                "a root SRV not on 16 bytes reads zeros"
                            );
                            host.null_raw_at(index);
                        }
                    }
                    None => host.null_raw_at(index),
                }
                (index, shadow)
            }
        };
        // As for root descriptors (`root_view`): all that is left of the
        // buffer is brought up to date. The first 1 MiB only left what lay
        // past it stale (most mesh vertex streams reach further).
        if let Some((a, s)) = &shadow {
            s.refresh(list, *a, s.left(*a));
        }
        let handle = host.gpu_handle(index);
        // SAFETY: an open list; the slot is in the bound view heap.
        unsafe {
            if compute {
                list.list.SetComputeRootDescriptorTable(param, handle);
            } else {
                list.list.SetGraphicsRootDescriptorTable(param, handle);
            }
        }
        true
    }

    /// For a shader that picks buffers from the view heap by an index it
    /// computes (SM 6.6 dynamic resources): the GPU address of the heap's
    /// table of element strides, with every buffer view of the heap given
    /// its host slot and its memory brought up to date when `list` is
    /// submitted. Tables tell us which views a draw reads; nothing tells
    /// us which of the heap's a computed index names, so all of them are
    /// made ready, once per list.
    /// shortcut: the buffers are read, not written: what such a shader
    /// writes through a picked unordered access view stays in the host
    /// copy. A list refreshes them where it first runs such a shader;
    /// title writes recorded later in the same list are not seen.
    fn heap_buffers(&self, list: &HostList) -> Option<u64> {
        let heap = self
            .heaps
            .iter()
            .filter(|&&h| h != 0)
            .map(|&h| crate::descriptor::heap(h as This))
            .find(|h| !h.samplers())?;
        let host = heap.host()?;
        let gpu = host.strides_gpu()?;
        // One key per list, as textures have: no title resource is there.
        const ONCE: usize = usize::MAX - 0x5e;
        if !list.first_use(ONCE) {
            return Some(gpu);
        }
        // The list may outlive the title's heap object, which owns the table.
        if let Some(buffer) = host.strides_buffer() {
            list.keep(buffer);
        }
        // The pass itself once per frame: per list it ran up to 51 times a
        // frame over 81,000 views and 4,100 host copies, 18 s of every
        // frame (2026-11-24, run 11).
        // shortcut: what the title writes into such buffers between lists
        // of one frame is seen a frame late.
        static FRAME: AtomicU64 = AtomicU64::new(u64::MAX);
        let frame = crate::frame::current();
        if FRAME.swap(frame, Ordering::Relaxed) == frame {
            return Some(gpu);
        }
        let started = std::time::Instant::now();
        #[cfg(feature = "diagnostics")]
        let profile = crate::bindless::heap_pass_timing();
        #[cfg(feature = "diagnostics")]
        let tick = || if profile { std::time::Instant::now() } else { started };
        // Per host heap: the host copy and range each slot's buffer view
        // reads, kept from pass to pass. Only the slots written since the
        // last pass are looked at again (83,000 slots a pass otherwise, 0.1 s
        // at best); all of them every 16 passes, because a view of a
        // released resource goes stale without a write, and a slot written
        // before its memory had a host copy is only made then.
        type Ranges = std::collections::HashMap<u32, (std::sync::Arc<crate::shadow::Shadow>, u64, u64)>;
        static RANGES: std::sync::Mutex<Option<std::collections::HashMap<usize, (u64, Ranges)>>> =
            std::sync::Mutex::new(None);
        let mut kept = RANGES.lock().unwrap_or_else(|e| e.into_inner());
        #[cfg(feature = "diagnostics")]
        let ranges_locked = tick();
        let (passes, ranges) = kept
            .get_or_insert_with(Default::default)
            .entry(std::ptr::from_ref(host) as usize)
            .or_insert_with(|| (0, Ranges::new()));
        let changed = host.buffer_changes();
        let full_scan = *passes % 16 == 0;
        let slots = if full_scan {
            ranges.clear();
            host.buffer_slots()
        } else {
            changed
        };
        *passes += 1;
        // Whether the device was there before the pass: a view that removes
        // it is named below.
        static GONE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        let mut alive = !GONE.load(Ordering::Relaxed) && crate::descriptor::host::device_removed().is_none();
        for (n, &index) in slots.iter().enumerate() {
            ranges.remove(&index);
            let Some(handle) = heap.handle(index) else {
                continue;
            };
            let Some(view) = host.mirror(index, handle, true) else {
                continue;
            };
            // Every 64 views while the device lives (a cheap query).
            if alive
                && n % 64 == 63
                && let Some(why) = crate::descriptor::host::device_removed()
            {
                alive = false;
                GONE.store(true, Ordering::Relaxed);
                let last: Vec<String> = slots[n - 63..=n]
                    .iter()
                    .filter_map(|&i| crate::descriptor::view(heap.handle(i)?).map(|v| format!("{i}: {v:?}")))
                    .map(|s| s.chars().take(200).collect())
                    .collect();
                tracing::error!(why, slot = index, "the host device went while buffer views of the heap were made; one of: {last:#?}");
                crate::host::log_removal_context();
            }
            // The copy the slot shows (`buffer_copy`), not the one the
            // view's range alone would find.
            let Some((shadow, address, len)) = crate::descriptor::host::buffer_copy(&view) else {
                continue;
            };
            #[cfg(feature = "diagnostics")]
            if host.shows(index, &shadow.buffer) == Some(false) {
                HEAP_VIEWS_ELSEWHERE.fetch_add(1, Ordering::Relaxed);
            }
            ranges.insert(index, (shadow, address, address + len));
        }
        #[cfg(feature = "diagnostics")]
        let slots_cached = tick();
        let views = ranges.len() as u32;
        // Per host copy, the span its picked views cover.
        let mut spans: std::collections::HashMap<usize, (std::sync::Arc<crate::shadow::Shadow>, u64, u64)> =
            std::collections::HashMap::new();
        for (shadow, from, to) in ranges.values() {
            let key = std::sync::Arc::as_ptr(shadow) as usize;
            let span = spans.entry(key).or_insert_with(|| (shadow.clone(), *from, *to));
            span.1 = span.1.min(*from);
            span.2 = span.2.max(*to);
        }
        drop(kept);
        let bytes: u64 = spans.values().map(|s| s.2 - s.1).sum();
        // Memory that is not write-watched is copied whole by every refresh.
        // It was once refreshed only every 120 frames (unmapped pages read
        // one by one cost 14 GB a frame in run 11, before they were
        // remembered): streamed geometry then drew from stale buffers, and
        // the garbage ended runs (the card stopped answering 14-90 frames
        // into the opening cutscene; with every frame it ran on). The pass
        // costs some 30 ms a frame this way.
        static WHOLE: std::sync::Mutex<Option<std::collections::HashMap<usize, u64>>> = std::sync::Mutex::new(None);
        #[cfg(feature = "diagnostics")]
        let spans_folded = tick();
        let mut whole_guard = WHOLE.lock().unwrap_or_else(|e| e.into_inner());
        #[cfg(feature = "diagnostics")]
        let whole_locked = tick();
        let whole = whole_guard.get_or_insert_with(Default::default);
        for (shadow, from, to) in spans.values() {
            if shadow.unwatched() {
                // By the copy's number: a copy made where one let go was
                // (`shadow::let_go_idle`) has never been refreshed.
                let key = shadow.id() as usize;
                // `VICTORY_UNWATCHED_FRAMES` sets the interval (1: every frame).
                static EVERY: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
                let every = *EVERY.get_or_init(|| {
                    std::env::var("VICTORY_UNWATCHED_FRAMES")
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(1)
                });
                match whole.get(&key) {
                    Some(&last) if frame < last + every => continue,
                    _ => {
                        whole.insert(key, frame);
                    }
                }
            }
            shadow.refresh(list, *from, to - from);
        }
        #[cfg(feature = "diagnostics")]
        let refresh_queued = tick();
        #[cfg(feature = "diagnostics")]
        let mut texture_timing = profile.then(HeapTextureTiming::default);
        #[cfg(feature = "diagnostics")]
        self.heap_textures(list, heap, host, texture_timing.as_mut());
        #[cfg(not(feature = "diagnostics"))]
        self.heap_textures(list, heap, host);
        let finished = std::time::Instant::now();
        let elapsed = finished - started;
        HEAP_BUFFER_PASSES.fetch_add(1, Ordering::Relaxed);
        HEAP_BUFFER_MICROS.fetch_add(elapsed.as_micros() as u64, Ordering::Relaxed);
        static LOGGED: AtomicU64 = AtomicU64::new(0);
        // The first pass and every growth of the set by half.
        let seen = LOGGED.load(Ordering::Relaxed);
        if u64::from(views) > seen + seen / 2 {
            LOGGED.store(u64::from(views), Ordering::Relaxed);
            tracing::info!(
                slots = slots.len(),
                views,
                copies = spans.len(),
                mb = bytes >> 20,
                ms = elapsed.as_millis(),
                "buffer views of the heap made ready for shaders that pick them by index"
            );
        }
        drop(whole_guard);
        #[cfg(feature = "diagnostics")]
        if let Some(texture) = texture_timing {
            // The three texture timings are included in `textures_us`.
            tracing::info!(
                frame,
                full_scan,
                slots = slots.len(),
                views,
                copies = spans.len(),
                total_us = elapsed.as_micros() as u64,
                ranges_lock_us = (ranges_locked - started).as_micros() as u64,
                cache_slots_us = (slots_cached - ranges_locked).as_micros() as u64,
                fold_spans_us = (spans_folded - slots_cached).as_micros() as u64,
                whole_lock_us = (whole_locked - spans_folded).as_micros() as u64,
                defer_refresh_us = (refresh_queued - whole_locked).as_micros() as u64,
                textures_us = (finished - refresh_queued).as_micros() as u64,
                textures_remap_us = texture.remap_us,
                textures_queue_us = texture.queue_us,
                textures_scan_us = texture.scan_us,
                "heap pass recording"
            );
        }
        Some(gpu)
    }

    /// The texture side of [`State::heap_buffers`]: textures the title fills
    /// itself (streamed pages, layout-library copies) whose views sit in the
    /// view heap are created (placed textures have no host texture until
    /// something reads them) and uploaded, as a table read would do, for
    /// shaders that pick them by a computed index. Within a budget of bytes
    /// per frame and in all; the rest waits for later lists and frames.
    /// The most recently written slots first (`TextureQueue::due`); slots
    /// draws read through tables are not in the queue.
    /// shortcut: uploaded once, and again only when the title rewrites the
    /// slot. A placed texture not all mapped yet is uploaded with what is
    /// there and looked at again (1, 2, 4, ... frames later, at most about
    /// a second): uploaded again when more of it is mapped. Pages mapped
    /// in place of others, and contents changed with no descriptor write,
    /// are missed (hook the page-mapping events for those).
    /// `VICTORY_HEAP_TEXTURES=0` turns it off.
    fn heap_textures(
        &self,
        list: &HostList,
        heap: &crate::descriptor::Heap,
        host: &crate::descriptor::host::HostHeap,
        #[cfg(feature = "diagnostics")] mut timing: Option<&mut HeapTextureTiming>,
    ) {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if !*ON.get_or_init(|| std::env::var_os("VICTORY_HEAP_TEXTURES").is_none_or(|v| v != "0")) {
            return;
        }
        /// Bytes of texture memory staged per frame.
        const PER_FRAME: u64 = 192 << 20;
        static FRAME: AtomicU64 = AtomicU64::new(u64::MAX);
        static SPENT: AtomicU64 = AtomicU64::new(0);
        let frame = crate::frame::current();
        if FRAME.swap(frame, Ordering::Relaxed) != frame {
            SPENT.store(0, Ordering::Relaxed);
        }
        // No more of them are made while the card has no room
        // (`resource::room`): by what is alive now, not by what was ever
        // staged (4 GB in all, after which later textures stayed empty for
        // the run). The ones nothing names any more are let go, which
        // makes room again: so the slots are still looked at (the pass
        // returned here, and a slot whose texture had been streamed out
        // kept its host copy for as long as the card was full).
        let room = crate::resource::room();
        if !room {
            HEAP_TEXTURES_FULL.fetch_add(1, Ordering::Relaxed);
            once!("textures picked by index: the card has no room for more; they wait");
        }
        let mut refused = 0;
        // Slots whose texture's memory the runtime mapped, unmapped or
        // wrote since the last pass are due again.
        #[cfg(feature = "diagnostics")]
        let remap_started = timing.as_ref().map(|_| std::time::Instant::now());
        host.texture_slots_remapped(frame);
        #[cfg(feature = "diagnostics")]
        let queue_started = timing.as_ref().map(|_| std::time::Instant::now());
        let (due, census) = host.texture_slots_due(frame, 4096);
        #[cfg(feature = "diagnostics")]
        if let (Some(timing), Some(remap_started), Some(queue_started)) =
            (timing.as_mut(), remap_started, queue_started)
        {
            timing.remap_us = (queue_started - remap_started).as_micros() as u64;
            timing.queue_us = queue_started.elapsed().as_micros() as u64;
        }
        HEAP_TEXTURES_WAITING.store(census[0], Ordering::Relaxed);
        for (count, now) in HEAP_TEXTURES_QUEUE.iter().zip(census) {
            count.store(now, Ordering::Relaxed);
        }
        // The frames in about four seconds, between 16 and 256: how long a
        // slot with nothing to upload waits at most for its next look. In
        // frames alone it was 120: four minutes at the opening cutscene's
        // half a frame per second, while the title streams by the clock.
        // (About a second before the runtime told of the memory it maps:
        // the looks are now only for what that does not tell.)
        static PASSED: AtomicU64 = AtomicU64::new(0);
        let now = crate::host::now_ms();
        let took = now.saturating_sub(PASSED.swap(now, Ordering::Relaxed)).max(1);
        let most = (4000 / took).clamp(16, 256);
        // What a table entry of a texture a shader samples says.
        let sampled = UsedResource {
            kind: ResourceType::Srv,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 8,
            user_data: 0,
            slot: 0,
        };
        // Also within a time per pass: the pass runs on a thread that
        // records the frame, and making and staging 192 MB of textures took
        // 1.6-2.2 s of each of the world's first twenty frames (run 20).
        // The rest waits for the next frames, as it does past the bytes.
        const SPAN: std::time::Duration = std::time::Duration::from_millis(250);
        let began = std::time::Instant::now();
        for (index, write) in due {
            if SPENT.load(Ordering::Relaxed) >= PER_FRAME || began.elapsed() > SPAN {
                break;
            }
            // A placed texture whose memory is not mapped (not streamed in
            // yet, or streamed out again: the heap keeps tens of thousands
            // of such views) gets no host texture; looked at again soon (a
            // title writes a texture's view before its pages are mapped,
            // and the pass runs early in a frame). One of which some is
            // mapped (a texture streaming in: its smaller mips lie at the
            // start of its memory and arrive first) is uploaded with what
            // is there, and again when more is. (It was uploaded again
            // every 120 frames, changed or not; and asked by its first
            // page alone, one whose first page is the one missing counted
            // as not there.)
            let mut partly = None;
            if let Some(View::Placed(false, resource, desc)) = heap.handle(index).and_then(crate::descriptor::view) {
                let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
                let len = crate::resource::placed_bytes(&resource).unwrap_or(1);
                let mapped = crate::texture::mapped_bytes(address as *const u8, len as usize);
                if mapped == 0 {
                    HEAP_TEXTURES_AWAY.fetch_add(1, Ordering::Relaxed);
                    // Streamed out: what the slot showed is let go.
                    host.texture_slot_gone(index);
                    host.texture_slot_later(index, write, frame, most);
                    continue;
                }
                if mapped < len {
                    if mapped <= host.texture_slot_mapped(index) {
                        HEAP_TEXTURES_PARTLY[0].fetch_add(1, Ordering::Relaxed);
                        host.texture_slot_later(index, write, frame, most);
                        continue;
                    }
                    if !crate::texture::committed(address as *const u8, 1) {
                        HEAP_TEXTURES_PARTLY[2].fetch_add(1, Ordering::Relaxed);
                    }
                    partly = Some(mapped);
                }
            }
            // A texture object the host renders to or writes holds the
            // host's own contents: nothing to bring up to date, and this
            // pass is no use of it (its host copy is made when something
            // uses it, and its slots then: `descriptor::make_waiting`).
            if let Some(View::Shader(object, _)) = heap.handle(index).and_then(crate::descriptor::view)
                && crate::resource::buffer_bytes(object as This).is_none()
                && !crate::resource::title_fills(object as This)
            {
                host.texture_slot_done(index, write, None);
                continue;
            }
            // From here on a host texture is made: not without room. The
            // slot stays due.
            if !room {
                refused += 1;
                if refused > 256 {
                    break;
                }
                continue;
            }
            let view = heap.handle(index).and_then(|handle| host.mirror(index, handle, true));
            let placed = view.as_ref().and_then(crate::descriptor::host::placed_texture);
            let resource = match placed.as_ref().or(view.as_ref()) {
                Some(&View::Shader(resource, _)) => resource as This,
                _ => {
                    host.texture_slot_done(index, write, None);
                    continue;
                }
            };
            let title_fills = crate::resource::host_resource(resource).is_some_and(crate::texture::title_written);
            if !title_fills {
                // a render target, a depth or unordered access texture: the host's own
                host.texture_slot_done(index, write, None);
                continue;
            }
            // A texture that has a staging buffer now is one draws read
            // through a table: they upload it, and check it at every
            // submission that samples it.
            let tabled = crate::texture::is_staged(resource);
            if let Some(view) = &view {
                readable(list, sampled.kind, view);
            }
            // A texture view the host refuses removes the device: name it.
            static GONE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            if !GONE.load(Ordering::Relaxed)
                && let Some(why) = crate::descriptor::host::device_removed()
            {
                GONE.store(true, Ordering::Relaxed);
                let text: String = format!("{view:?}").chars().take(300).collect();
                tracing::error!(why, slot = index, "the host device went while textures picked by index were made ready; last: {text}");
                crate::host::log_removal_context();
                break;
            }
            let (bytes, whole) = crate::texture::staged(resource).unwrap_or((0, true));
            // Not all of it there: looked at again next frame, and from
            // then on as a slot with nothing new (a placed texture: until
            // more of it is mapped; a texture object at an address the
            // title mapped: again in 120 frames, as before).
            let again = match partly {
                Some(mapped) => Some((frame + 1, mapped)),
                None => (!whole).then_some((frame + 120, 0)),
            };
            if again.is_some() {
                HEAP_TEXTURES_PARTLY[1].fetch_add(1, Ordering::Relaxed);
            }
            if tabled {
                // Nothing was staged here, so nothing is counted or spent,
                // and the staging buffer stays: taking it away made the
                // next draw that reads the texture through a table stage,
                // detile and upload all of it once more.
                HEAP_TEXTURES_TABLED.fetch_add(1, Ordering::Relaxed);
                host.texture_slot_done(index, write, again);
                continue;
            }
            // Uploaded: the list holds the staging buffer until it has run;
            // a table read, or the next look at a texture not all mapped
            // yet, makes it anew. (Kept for those until now: every partly
            // streamed texture of the heap held a staging buffer of its full
            // size for good.)
            crate::texture::unstage(resource);
            SPENT.fetch_add(bytes, Ordering::Relaxed);
            HEAP_TEXTURE_BYTES.fetch_add(bytes, Ordering::Relaxed);
            HEAP_TEXTURES.fetch_add(1, Ordering::Relaxed);
            host.texture_slot_done(index, write, again);
        }
        #[cfg(feature = "diagnostics")]
        if let Some(timing) = timing {
            timing.scan_us = began.elapsed().as_micros() as u64;
        }
    }

    /// For the frame listing, after a dropped dispatch's line: its group
    /// counts and the compute root arguments it was recorded with (root
    /// constants as hex dwords, root views with the title buffer they
    /// point into). A program that does not translate has no bindings to
    /// list; these say which memory it would have read and written (the
    /// ray-tracing structure builder passes its buffers this way).
    #[cfg(feature = "diagnostics")]
    fn list_dropped_roots(&self, groups: [u32; 3]) {
        crate::capture::op(0, || {
            use std::fmt::Write;
            let [x, y, z] = groups;
            let mut text = format!("  its roots ({x}x{y}x{z}):");
            for (slot, root) in self.compute.iter().enumerate() {
                match root {
                    Root::None => {}
                    Root::Table(a) => {
                        let _ = write!(text, " {slot}=table {a:#x}");
                    }
                    Root::Constants(v) => {
                        let _ = write!(text, " {slot}=constants {:x?}", &v[..v.len().min(48)]);
                    }
                    Root::View(a) => {
                        let _ = write!(text, " {slot}=view {a:#x}");
                        match crate::shadow::extent(*a) {
                            Some((start, len)) => {
                                let _ = write!(text, " in buffer {start:#x}+{len:#x}");
                            }
                            None => text.push_str(" in no buffer"),
                        }
                    }
                }
            }
            text
        });
    }

    /// Runs a dispatch on the host.
    pub(crate) fn dispatch(&mut self, list: &HostList, x: u32, y: u32, z: u32) {
        // It may write the arguments a window holds clamped.
        self.window = None;
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            dropped(list, self.pipeline, "dispatch");
            return;
        }
        // The ray-tracing structure builder's stages run all or none.
        if crate::pipeline::is_builder(self.pipeline as This) && !crate::pipeline::builder_runs() {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            {
                crate::pipeline::note_why("a stage of the ray-tracing structure builder, which is off".to_owned());
                self.list_dropped_roots([x, y, z]);
            }
            return;
        }
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            once!(
                pipeline = format_args!("{:#x}", self.pipeline),
                "a dispatch without a host pipeline dropped"
            );
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            {
                dropped(list, self.pipeline, "dispatch");
                self.list_dropped_roots([x, y, z]);
            }
            return;
        };
        if !p.compute {
            once!("a dispatch with a graphics pipeline");
            return;
        }
        if !self.bind(list, p, true) {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            dropped(list, self.pipeline, "dispatch");
            // A stage of the structure builder that cannot be bound is a
            // dropped stage: the others stop too.
            if crate::pipeline::is_builder(self.pipeline as This) {
                crate::pipeline::builder_off("one of its dispatches could not be bound");
            }
            return;
        }
        #[cfg(feature = "diagnostics")]
        list.note_op(|| self.op_text(format!("dispatch {x}x{y}x{z}"), true));
        #[cfg(feature = "diagnostics")]
        self.probe_small(list, true);
        // SAFETY: an open list with everything set.
        unsafe { list.list.Dispatch(x, y, z) };
        #[cfg(feature = "diagnostics")]
        self.probe_written(list);
        DISPATCHED.fetch_add(1, Ordering::Relaxed);
    }

    /// What a draw or dispatch runs, for the device-removed report
    /// (diagnostics): the pipeline, then per stage the user data it gets
    /// (with the view at that heap index, if one is), the host addresses
    /// of its root views and the views its tables name.
    #[cfg(feature = "diagnostics")]
    fn op_text(&self, kind: String, compute: bool) -> String {
        use std::fmt::Write;
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return kind;
        };
        let mut text = format!("{kind} {}", p.name);
        // What it draws into (a dispatch writes through its views below).
        if !compute {
            let described = |t: usize| {
                crate::resource::host_made(t as This).map(|r| {
                    // SAFETY: a live resource.
                    let d = unsafe { r.GetDesc() };
                    format!("{t:#x} {}x{} f{}", d.Width, d.Height, d.Format.0)
                })
            };
            let targets: Vec<String> = self.targets.iter().filter_map(|&t| described(t)).collect();
            let _ = write!(text, " targets {targets:?} depth {:?} state [{}]", described(self.depth), p.state);
            // The title's scissor and viewport at this draw (the first of
            // each): Grand Theft Auto VI draws what it sees through a
            // portal (a window, from inside a room) under the portal's
            // rectangle, and the listing did not say which draws those are.
            if let Some(r) = self.scissors.first() {
                let _ = write!(text, " scissor [{} {} {} {}]", r.left, r.top, r.right, r.bottom);
            }
            if let Some(v) = self.viewports.first() {
                let _ = write!(
                    text,
                    " viewport [{} {} {} {} depth {}..{}]",
                    v.TopLeftX, v.TopLeftY, v.Width, v.Height, v.MinDepth, v.MaxDepth
                );
            }
        }
        let views = (self.heaps[0] != 0)
            .then(|| crate::descriptor::heap(self.heaps[0] as This).host())
            .flatten();
        for stage in &p.stages {
            let t = stage.translated();
            let _ = write!(text, " | {}:", stage.shader.hash);
            for u in &t.host.user_data {
                let v = match u.root {
                    RootType::Other(kind) => self.host_word(compute, kind, u.slot, u.dword, None),
                    _ => self.user_value(compute, u.slot, u.root, u.dword),
                };
                let _ = write!(text, " u{}={v:#x}", u.index);
                let index = if u.root == RootType::Table { v / 32 } else { v };
                if let Some(slot) = views.and_then(|h| h.describe_slot(index)) {
                    let _ = write!(text, " [heap {index}: {slot}]");
                    // A slot the host never made: what the title's memory
                    // holds there (our record, zeros, or words it wrote itself).
                    if slot == "null" && self.heaps[0] != 0 {
                        let at = crate::descriptor::heap(self.heaps[0] as This).start + u64::from(index) * 32;
                        let mut words = [0u8; 32];
                        if crate::host::read_into(at as *const u8, &mut words) {
                            let w: Vec<u32> =
                                words.chunks(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                            let _ = write!(text, " (title words {w:x?}");
                            if let Some(view) = crate::descriptor::view(at) {
                                let _ = write!(text, ": {}", describe(Some(&view)));
                            }
                            let _ = write!(text, ")");
                        }
                    }
                }
            }
            for &(slot, _) in stage.root_cbvs.iter().chain(&stage.root_srvs).chain(&stage.root_uavs) {
                if let Some(Root::View(address)) = self.roots_ref(compute).get(usize::from(slot)) {
                    let host = crate::shadow::find(*address, 1)
                        .map(|s| (s.gpu(*address), s.left(*address)));
                    // Its first dwords (a vertex stream header, say).
                    let mut head = [0u8; 16];
                    let head = crate::host::read_into(*address as *const u8, &mut head)
                        .then(|| head.chunks(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect::<Vec<_>>());
                    let _ = write!(text, " root{slot}={address:#x} host {host:x?} head {head:x?}");
                }
            }
        }
        let _ = write!(text, " tables {:?}", self.reads(compute));
        if !compute {
            let mem = |r: usize| crate::resource::texture_memory(r as This).map_or(0, |(p, _)| p as u64);
            let targets: Vec<String> = self.targets.iter().map(|&t| format!("{t:#x} mem {:#x}", mem(t))).collect();
            let _ = write!(text, " targets {targets:?} depth {:#x} mem {:#x}", self.depth, mem(self.depth));
        }
        text
    }

    /// A mesh draw under `VICTORY_MESH_TEST=scissor` rasterises one pixel:
    /// the scissors before it (`on`), the title's after it.
    fn mesh_scissor(&self, list: &HostList, on: bool) {
        if mesh_test() != 2 {
            return;
        }
        if on {
            let one = RECT { left: 0, top: 0, right: 1, bottom: 1 };
            crate::draw::scissors(list, &vec![one; self.scissors.len().max(1)]);
        } else if !self.scissors.is_empty() {
            crate::draw::scissors(list, &self.scissors);
        }
    }

    /// Runs a mesh dispatch of `x` groups on the host (the title's mesh
    /// programs are one-dimensional).
    pub(crate) fn dispatch_mesh(&mut self, list: &HostList, x: u32) {
        self.window = None;
        #[cfg(feature = "diagnostics")]
        self.trace("mesh", x, 0);
        let ran = mesh_test() != 1
            && mesh_kept(self.pipeline)
            && self.prepare_for(list, false, true)
            && match list.list.cast::<ID3D12GraphicsCommandList6>() {
                Ok(l6) => {
                    #[cfg(feature = "diagnostics")]
                    list.note_op(|| self.op_text(format!("mesh {x}"), false));
                    self.mesh_scissor(list, true);
                    // SAFETY: an open list with everything set.
                    unsafe { l6.DispatchMesh(x, 1, 1) };
                    self.mesh_scissor(list, false);
                    DRAWN.fetch_add(1, Ordering::Relaxed);
                    true
                }
                Err(_) => false,
            };
        if !ran {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            dropped(list, self.pipeline, "mesh");
        }
    }

    fn roots_ref(&self, compute: bool) -> &Vec<Root> {
        if compute {
            &self.compute
        } else {
            &self.graphics
        }
    }

    fn table_entry(&self, list: &HostList, u: &UsedResource, compute: bool) {
        let Some(Root::Table(table)) = self.roots_ref(compute).get(usize::from(u.slot)) else {
            return;
        };
        let handle = table + u64::from(u.table_offset_dwords) * 4;
        let Some(heap) = self.heap_of(handle) else {
            return;
        };
        let (Some(host), Some(index)) = (heap.host(), heap.index(handle)) else {
            return;
        };
        if let Some(view) = host.mirror(index, handle, true) {
            readable(list, u.kind, &view);
            // Up to date by this draw: not the texture pass's to make.
            if crate::descriptor::host::is_texture_view(&view) {
                host.texture_slot_tabled(index, true);
            }
        }
    }

    /// Mirrors tables in place where possible. A table that crosses the host
    /// identity window or has a root-selected entry is materialized as one
    /// contiguous range. The returned offsets name those ranges for this list.
    pub(crate) fn tables(
        &mut self,
        list: &HostList,
        used: &[UsedResource],
        indexed: &[RootIndexedView],
        compute: bool,
    ) -> Vec<(u8, u32)> {
        let tables = |u: &&UsedResource| u.root == RootType::Table;
        let mut fresh: Vec<u8> = indexed.iter().map(|entry| entry.table_slot).collect();
        // Tables reaching past a host heap smaller than the title's (the
        // host's limit, 1,000,000 views on NVIDIA) are made afresh too: the
        // title's slots there have no host slot of their own.
        for u in used.iter().filter(tables) {
            if fresh.contains(&u.slot) {
                continue;
            }
            let Some(&Root::Table(table)) = self.roots_ref(compute).get(usize::from(u.slot)) else {
                continue;
            };
            let end = table + u64::from(u.table_offset_dwords) * 4;
            let Some(heap) = self.heap_of(table) else {
                continue;
            };
            let (Some(host), Some(index)) = (heap.host(), heap.index(end)) else {
                continue;
            };
            if index >= host.count() {
                fresh.push(u.slot);
            }
        }
        fresh.sort_unstable();
        fresh.dedup();
        for u in used.iter().filter(tables) {
            if !fresh.contains(&u.slot) {
                self.table_entry(list, u, compute);
            }
        }
        let mut made = Vec::new();
        for slot in fresh {
            let Some(&Root::Table(table)) = self.roots_ref(compute).get(usize::from(slot)) else {
                continue;
            };
            let entries: Vec<&UsedResource> = used
                .iter()
                .filter(tables)
                .filter(|u| u.slot == slot)
                .collect();
            let Some(heap) = self.heap_of(table) else {
                continue;
            };
            if heap.samplers() || (table - heap.start) % 32 != 0 {
                continue;
            }
            let Some(host) = heap.host() else {
                continue;
            };
            let mut selected: Vec<u32> = indexed
                .iter()
                .filter(|index| index.table_slot == slot)
                .map(|index| {
                    let constant = self.user_value(
                        compute,
                        index.constant_slot,
                        RootType::Constants,
                        index.constant_dword,
                    );
                    // The shader adds i32 byte offsets and shifts by 5.
                    index.byte_offset.wrapping_add(constant << index.shift) / 32
                })
                .collect();
            selected.sort_unstable();
            selected.dedup();
            let n = entries
                .iter()
                .map(|u| u.table_offset_dwords / 8)
                .chain(selected.last().copied())
                .max()
                .unwrap_or(0)
                + 1;
            let last = table.checked_add(u64::from(n - 1) * 32);
            if n > crate::descriptor::host::DYNAMIC
                || last.and_then(|handle| heap.index(handle)).is_none()
            {
                once!(slot, n, "fresh descriptor table exceeds the available span");
                made.push((slot, u32::MAX));
                continue;
            }
            // The table and what its entries hold (record words).
            let key = {
                use std::hash::{Hash, Hasher};
                let mut h = std::collections::hash_map::DefaultHasher::new();
                for k in 0..n {
                    let at = table + u64::from(k) * 32;
                    // SAFETY: a descriptor slot in title memory.
                    unsafe { (at as *const [u32; 3]).read_unaligned() }.hash(&mut h);
                }
                (table, n, h.finish())
            };
            if let Some(&first) = self.fresh_tables.get(&key) {
                for u in &entries {
                    let handle = table + u64::from(u.table_offset_dwords) * 4;
                    if let Some(view) = crate::descriptor::view(handle) {
                        readable(list, u.kind, &view);
                    }
                }
                for k in selected {
                    if entries.iter().any(|u| u.table_offset_dwords / 8 == k) {
                        continue;
                    }
                    let handle = table + u64::from(k) * 32;
                    if let Some(view) = crate::descriptor::view(handle)
                        && let Some(kind) = view_kind(&view)
                    {
                        readable(list, kind, &view);
                    }
                }
                made.push((slot, first * 32));
                continue;
            }
            let Some(first) = host.dynamic(n) else {
                // No room: the draw must not run with the title's own
                // offsets (slots never made for it); `bind` drops it and
                // gives back the tables made before this one.
                made.push((slot, u32::MAX));
                break;
            };
            self.fresh_tables.insert(key, first);
            self.bind_made.push(BindMade::Table(key));
            self.dynamic
                .push((std::ptr::from_ref(heap) as usize, first, n));
            // The title's own slots of this table: what the draw reads of
            // them it brings up to date here, in the dynamic slots.
            let title_first = heap.index(table).filter(|first| first + n <= host.count());
            // The whole range: a shader may index past the entries its
            // used-resource list names (descriptor arrays).
            for k in 0..n {
                let index = first + k;
                let used = entries.iter().find(|u| u.table_offset_dwords / 8 == k);
                match crate::descriptor::view(table + u64::from(k) * 32) {
                    Some(view) => {
                        host.make_at(index, &view);
                        let kind = used.map(|u| u.kind).or_else(|| {
                            selected.binary_search(&k).ok().and_then(|_| view_kind(&view))
                        });
                        if let Some(kind) = kind {
                            readable(list, kind, &view);
                        }
                        // The title's slot is a table's: out of the queue
                        // of textures picked by index if the draw read it
                        // (its texture is up to date now), and made in
                        // place as the texture pass would have made it,
                        // for a shader that picks it by index all the same.
                        if let Some(title) = title_first
                            && crate::descriptor::host::is_texture_view(&view)
                        {
                            host.texture_slot_tabled(title + k, kind.is_some());
                            if kind.is_some() {
                                host.mirror(title + k, table + u64::from(k) * 32, true);
                            }
                        }
                    }
                    None => {
                        host.make_null(index);
                        if used.is_some() {
                            once!(slot, "a table entry without a view");
                        }
                    }
                }
            }
            made.push((slot, first * 32));
        }
        made
    }

    /// The first draws, at debug level: what they bind and read
    /// (diagnostics builds).
    #[cfg(feature = "diagnostics")]
    fn trace(&self, kind: &str, count: u32, first: u32) {
        static TRACED: AtomicU64 = AtomicU64::new(0);
        if !tracing::enabled!(tracing::Level::DEBUG) {
            return;
        }
        let listed = crate::capture::listed(crate::queue::presented() + 1);
        let logged = listed || TRACED.fetch_add(1, Ordering::Relaxed) < 64;
        let targets: Vec<String> = self
            .targets
            .iter()
            .filter_map(|&t| crate::resource::host_made(t as This))
            .map(|t| {
                // SAFETY: a live resource.
                let d = unsafe { t.GetDesc() };
                format!("{}x{} f{}", d.Width, d.Height, d.Format.0)
            })
            .collect();
        let v = self.vertex[0];
        let line = format!(
            "{kind} {count} from {first} topology {} pipeline {:#x} targets {targets:?} vb {:#x}+{} stride {} reads {:?}",
            self.topology,
            self.pipeline,
            v.address,
            v.size,
            v.stride,
            self.reads(false)
        );
        if logged {
            tracing::debug!("draw {line}");
        }
        let mut frame = FRAME.lock().unwrap_or_else(|e| e.into_inner());
        if frame.len() < 256 {
            frame.push(line);
        }
    }

    /// In a listed frame: each small buffer (64 bytes at most) the draw's
    /// tables name, as title memory now and as its host copy when the list
    /// is submitted (before the GPU runs it) and after it has run.
    #[cfg(feature = "diagnostics")]
    fn probe_small(&self, list: &HostList, compute: bool) {
        if !crate::capture::listed(crate::queue::presented() + 1) {
            return;
        }
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else { return };
        for stage in &p.stages {
            for u in &stage.translated().used {
                let (RootType::Table, Some(Root::Table(table))) =
                    (u.root, self.roots_ref(compute).get(usize::from(u.slot)))
                else {
                    continue;
                };
                let Some(view) = crate::descriptor::view(table + u64::from(u.table_offset_dwords) * 4) else {
                    continue;
                };
                let range = match view {
                    View::Constant(address, size) => Some((address, u64::from(size))),
                    ref v => crate::descriptor::host::buffer_range(v),
                };
                let Some((at, len)) = range.filter(|r| r.1 <= 64) else {
                    continue;
                };
                let Some(shadow) = crate::shadow::find(at, len) else { continue };
                let who = format!("{} {:?} s{}+{} {at:#x}+{len}", stage.shader.hash, u.kind, u.slot, u.table_offset_dwords);
                let (s, w) = (shadow.clone(), who.clone());
                list.defer_host(move || tracing::info!(target: "victory_gpu::trace", "small {w} at submission: {}", s.peek(at, len)));
                list.defer_after(move || tracing::info!(target: "victory_gpu::trace", "small {who} after the list: {}", shadow.peek(at, len)));
            }
        }
    }

    /// In a listed frame, right after a dispatch: each 1D or 3D texture it
    /// writes through a table, read back in the list itself (what the
    /// dispatch left there, before anything later in the frame).
    #[cfg(feature = "diagnostics")]
    fn probe_written(&self, list: &HostList) {
        if !crate::capture::listed(crate::queue::presented() + 1) {
            return;
        }
        let (Some(p), Some(host)) = (crate::pipeline::host_pipeline(self.pipeline as This), crate::host::host()) else {
            return;
        };
        for stage in &p.stages {
            for u in stage.translated().used.iter().filter(|u| u.kind == ResourceType::Uav) {
                let (RootType::Table, Some(Root::Table(table))) = (u.root, self.compute.get(usize::from(u.slot))) else {
                    continue;
                };
                let Some(View::Unordered(r, ..)) = crate::descriptor::view(table + u64::from(u.table_offset_dwords) * 4) else {
                    continue;
                };
                let Some(t) = crate::resource::host_resource(r as This) else { continue };
                // SAFETY: a live resource.
                let d = unsafe { t.GetDesc() };
                if d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE1D && d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE3D {
                    continue;
                }
                const COPY_SOURCE: u32 = 0x800;
                const UNORDERED: u32 = 0x8;
                // All UAV writes done before the copy.
                let all = D3D12_RESOURCE_BARRIER {
                    Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                    Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                    Anonymous: D3D12_RESOURCE_BARRIER_0 {
                        UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER { pResource: std::mem::ManuallyDrop::new(None) }),
                    },
                };
                // SAFETY: an open list.
                unsafe { list.list.ResourceBarrier(&[all]) };
                crate::list::ensure_state(list, r as This, COPY_SOURCE);
                let read = crate::capture::read_back(&host.device, &list.list, t);
                crate::list::ensure_state(list, r as This, UNORDERED);
                let Some((buffer, f)) = read else { continue };
                list.keep(&buffer);
                struct Readback(ID3D12Resource);
                // SAFETY: D3D12 objects are free-threaded.
                unsafe impl Send for Readback {}
                // SAFETY: as above.
                unsafe impl Sync for Readback {}
                let buffer = Readback(buffer);
                let who = format!("{} s{}+{} {r:#x} {}x{}x{} f{}", stage.shader.hash, u.slot, u.table_offset_dwords, d.Width, d.Height, d.DepthOrArraySize, d.Format.0);
                list.defer_after(move || {
                    let mut data = std::ptr::null_mut();
                    // SAFETY: the list has run; the mapping spans the first slice.
                    unsafe {
                        if buffer.0.Map(0, None, Some(&mut data)).is_err() {
                            return;
                        }
                        let n = (f.Footprint.RowPitch * f.Footprint.Height) as usize / 4;
                        let w = std::slice::from_raw_parts(data.cast::<u32>(), n);
                        let set = w.iter().filter(|&&x| x != 0).count();
                        let first: Vec<f32> = w.iter().take(8).map(|&x| f32::from_bits(x)).collect();
                        tracing::info!(target: "victory_gpu::trace", "written {who} right after the dispatch: {set} of {n} dwords set, first {first:?}");
                        buffer.0.Unmap(0, None);
                    }
                });
            }
        }
    }

    /// What a draw's shaders read, one line per used resource (diagnostics
    /// builds).
    #[cfg(feature = "diagnostics")]
    fn reads(&self, compute: bool) -> Vec<String> {
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return vec!["no host pipeline".into()];
        };
        let mut out = Vec::new();
        for stage in &p.stages {
            let t = stage.translated();
            for u in &t.used {
                let what = match (u.root, self.roots_ref(compute).get(usize::from(u.slot))) {
                    (RootType::Table, Some(Root::Table(table))) => {
                        let handle = table + u64::from(u.table_offset_dwords) * 4;
                        let view = crate::descriptor::view(handle);
                        let mut text = describe(view.as_ref());
                        // The title's view words, and whether the host slot
                        // the shader reads was made from a view.
                        if let Some(View::Shader(_, Some(w)) | View::Unordered(_, _, Some(w))) = &view {
                            text.push_str(&format!(" words {w:?}"));
                        }
                        let slot = (self.heaps[0] != 0)
                            .then(|| crate::descriptor::heap(self.heaps[0] as This))
                            .and_then(|h| {
                                let i = handle.checked_sub(h.start)? / 32;
                                h.host()?.describe_slot(u32::try_from(i).ok()?)
                            });
                        let current: String = view.as_ref().map(|v| format!("{v:?}")).unwrap_or_default().chars().take(120).collect();
                        text.push_str(match slot.as_deref() {
                            Some("null") => " slot null",
                            Some(s) if s.starts_with(&current) => " slot made",
                            Some(_) => " slot stale",
                            None => " slot unknown",
                        });
                        text
                    }
                    (root, _) => format!("{root:?}"),
                };
                out.push(format!(
                    "{} {:?} s{}+{}: {what}",
                    stage.shader.hash, u.kind, u.slot, u.table_offset_dwords
                ));
            }
        }
        out
    }

    pub(crate) fn draw(
        &mut self,
        list: &HostList,
        vertices: u32,
        instances: u32,
        first: u32,
        first_instance: u32,
    ) {
        self.window = None;
        #[cfg(feature = "diagnostics")]
        self.trace("draw", vertices, first);
        let ready = self.prepare(list, false);
        // A pipeline with the rectangle geometry stage draws the title's
        // vertices as they are; any other gets its rectangles expanded.
        let rects = self.topology == RECTANGLES && !self.completes_rects();
        let ran = ready
            && (!rects || self.rectangles(list, vertices, first) || {
                not_run("rectangle vertices outside their buffer");
                false
            });
        if ran {
            // Rectangles were expanded into their own vertex buffers.
            let (count, from) = if rects {
                (vertices / 3 * 6, 0)
            } else {
                (vertices, first)
            };
            #[cfg(feature = "diagnostics")]
            list.note_op(|| self.op_text(format!("draw {count}x{instances}"), false));
            #[cfg(feature = "diagnostics")]
            self.probe_small(list, false);
            // SAFETY: an open list with everything set.
            unsafe {
                list.list
                    .DrawInstanced(count, instances, from, first_instance)
            };
            DRAWN.fetch_add(1, Ordering::Relaxed);
        }
        if !ran {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            dropped(list, self.pipeline, "draw");
        }
    }

    /// Whether the bound pipeline completes rectangles itself, in the
    /// geometry stage rectangle pipelines get (`HostPipeline::rects`).
    fn completes_rects(&self) -> bool {
        crate::pipeline::host_pipeline(self.pipeline as This).is_some_and(|p| p.rects)
    }

    /// Binds vertex buffers holding the rectangles of vertices `first..first
    /// + count` as triangles, made when the list is submitted.
    fn rectangles(&self, list: &HostList, count: u32, first: u32) -> bool {
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return false;
        };
        let rects = u64::from(count / 3);
        for slot in 0..32 {
            let elements: Vec<(u32, u32)> = p
                .inputs
                .iter()
                .filter(|e| e.slot == slot && !e.per_instance)
                .map(|e| (e.offset, e.format))
                .collect();
            if elements.is_empty() {
                continue;
            }
            let v = self.vertex[slot as usize];
            let stride = u64::from(v.stride);
            let from = v.address + u64::from(first) * stride;
            if v.address == 0
                || stride == 0
                || u64::from(first + count) * stride > u64::from(v.size)
                || crate::shadow::find(from, rects * 3 * stride).is_none()
            {
                once!(slot, "rectangle vertices outside their buffer");
                return false;
            }
            let Some((gpu, to)) = list.scratch(rects * 6 * stride) else {
                return false;
            };
            let (from, to, stride) = (from as usize, to as usize, stride as usize);
            let n = rects as usize;
            list.defer_host(move || {
                // The vertices are read in place, at submission: no change
                // of mapping under the read (see
                // `crate::watch::title_memory_reading`).
                let _mapped = crate::watch::title_memory_reading();
                // SAFETY: the vertices lie in a title buffer (checked when
                // recorded) and the scratch memory is the list's.
                let (src, dst) = unsafe {
                    (
                        std::slice::from_raw_parts(from as *const u8, n * 3 * stride),
                        std::slice::from_raw_parts_mut(to as *mut u8, n * 6 * stride),
                    )
                };
                crate::primitive::expand_rects(src, dst, stride, &elements);
            });
            let view = D3D12_VERTEX_BUFFER_VIEW {
                BufferLocation: gpu,
                SizeInBytes: (rects * 6 * stride as u64) as u32,
                StrideInBytes: v.stride,
            };
            // SAFETY: an open list; the view names live scratch memory.
            unsafe { list.list.IASetVertexBuffers(slot, Some(&[view])) };
        }
        true
    }

    pub(crate) fn draw_indexed(
        &mut self,
        list: &HostList,
        indices: u32,
        instances: u32,
        first: u32,
        base: i32,
        first_instance: u32,
    ) {
        self.window = None;
        #[cfg(feature = "diagnostics")]
        self.trace("indexed", indices, first);
        // Indexed rectangles are drawn by the rectangle geometry stage
        // alone (nothing expands them on the CPU).
        let lost = self.topology == RECTANGLES && !self.completes_rects();
        if lost {
            once!("indexed rectangles not drawn");
            not_run("indexed rectangles");
        }
        let ran = !lost && self.prepare(list, true);
        if ran {
            #[cfg(feature = "diagnostics")]
            list.note_op(|| self.op_text(format!("indexed {indices}x{instances}"), false));
            // SAFETY: an open list with everything set.
            unsafe {
                list.list
                    .DrawIndexedInstanced(indices, instances, first, base, first_instance)
            };
            DRAWN.fetch_add(1, Ordering::Relaxed);
        }
        if !ran {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            dropped(list, self.pipeline, "indexed");
        }
    }
}

impl State {
    /// Indirect execution of a one-argument signature of `kind` (PC
    /// numbering: 0 draw, 1 indexed draw, 2 dispatch, 10 mesh dispatch) with
    /// records `stride` bytes apart at title address `arguments`, and the
    /// record count at `count` if the title gives one: the GPU reads them
    /// from the buffer's host copy, where earlier dispatches may have
    /// written them. False if it cannot run on the host.
    pub(crate) fn execute_indirect(
        &mut self,
        list: &HostList,
        (kind, stride, max): (u32, u32, u32),
        arguments: u64,
        count: Option<u64>,
    ) -> bool {
        // Skip before the argument clamp can mutate the indirect buffer.
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            crate::pipeline::note_drop(self.pipeline as This);
            list.note_op(|| format!("DROPPED indirect kind {kind} (skipped) {}", crate::pipeline::shaders_of(self.pipeline as This)));
            return false;
        }
        if crate::pipeline::is_builder(self.pipeline as This) && !crate::pipeline::builder_runs() {
            return false;
        }
        const DRAW: u32 = 0;
        const INDEXED: u32 = 1;
        const DISPATCH: u32 = 2;
        const MESH: u32 = 10;
        #[cfg(feature = "diagnostics")]
        let pipeline = self.pipeline as This;
        #[cfg(feature = "diagnostics")]
        let why = |what: &str| {
            crate::pipeline::note_why(format!("indirect kind {kind}: {what}"));
            list.note_op(|| format!("DROPPED indirect kind {kind} max {max} ({what}) {}", crate::pipeline::shaders_of(pipeline)));
        };
        #[cfg(not(feature = "diagnostics"))]
        let why = |_: &str| {};
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            #[cfg(feature = "diagnostics")]
            crate::pipeline::note_drop(self.pipeline as This);
            why("no host pipeline");
            return false;
        };
        // Indirect rectangles (Grand Theft Auto VI's tiled lighting, 256
        // indirect draws a frame, makes their corners from the vertex
        // number) are completed by the pipeline's geometry stage: the
        // arguments stay the title's, three vertices a rectangle. They
        // cannot be expanded on the CPU, which does not know their count.
        // shortcut: in a pipeline without that stage, a rectangle with no
        // per-vertex inputs is drawn as its first triangle (three vertices,
        // half of it), and one with vertex buffers is not drawn.
        let made_rects = !p.rects
            && kind == DRAW
            && self.topology == RECTANGLES
            && !p.inputs.iter().any(|e| !e.per_instance);
        if made_rects {
            once!("indirect rectangles without vertex inputs drawn as their first triangle");
        }
        let matches_kind = match kind {
            DRAW | INDEXED => {
                (self.topology != RECTANGLES || made_rects || p.rects) && !p.compute && !p.mesh
            }
            MESH => !p.compute && p.mesh,
            DISPATCH => p.compute,
            _ => false,
        };
        if !matches_kind {
            why(&format!(
                "pipeline mismatch (compute {}, mesh {}, topology {})",
                p.compute, p.mesh, self.topology
            ));
            return false;
        }
        let len = u64::from(stride) * u64::from(max.max(1));
        let Some(args) = crate::shadow::find_any(arguments, len) else {
            why("arguments outside title buffers");
            return false;
        };
        let counter = match count {
            Some(at) => match crate::shadow::find_any(at, 4) {
                Some(s) => Some((s, at)),
                None => {
                    why("count outside title buffers");
                    return false;
                }
            },
            None => None,
        };
        let Some(signature) = matches!(kind, DRAW | INDEXED | DISPATCH | MESH)
            .then(|| host_signature(kind, stride))
            .flatten()
        else {
            why("no host signature");
            return false;
        };
        if kind == MESH && (mesh_test() == 1 || !mesh_kept(self.pipeline)) {
            why("mesh draws are switched off (VICTORY_MESH_TEST=skip)");
            return false;
        }
        // A window of records clamped in one pass (`windows`): for single
        // records without a count buffer, drawn by shaders that store to no
        // buffer. Anything else ends the window: it uses the scratch buffer
        // or may write the arguments.
        const WINDOW_USE: usize = usize::MAX - 0x77;
        let offset = args.offset(arguments);
        let windowed = windows()
            && kind != DISPATCH
            && max == 1
            && count.is_none()
            && !p.stages.iter().any(|stage| stage.translated().stores);
        // Another host list (a new one after a cut, or this one after its
        // reset) has another scratch buffer, or this one's with old bytes.
        if !windowed || list.first_use(WINDOW_USE) {
            self.window = None;
        }
        let same = |w: &Window| w.copy == args.id() && w.kind == kind && w.stride == stride;
        let hit = self
            .window
            .as_ref()
            .filter(|w| same(w) && offset >= w.base)
            .filter(|w| (offset - w.base) % u64::from(w.step) == 0 && (offset - w.base) / u64::from(w.step) < u64::from(w.records))
            .map(|w| (w.scratch.clone(), offset - w.base));
        let (bounded, at) = match hit {
            Some(hit) => hit,
            None => {
                // A record a short step after the last window's last one
                // (the same step as that window's, if it had one): the
                // title walks the buffer, so the ones that step apart
                // after it are clamped with it.
                let (step, records) = match &self.window {
                    Some(w) if windowed && same(w) => {
                        let last = w.base + u64::from(w.records - 1) * u64::from(w.step);
                        let step = offset.saturating_sub(last);
                        let walks = step >= u64::from(stride)
                            && step <= 256
                            && step % 4 == 0
                            && (w.records == 1 || step == u64::from(w.step));
                        if walks {
                            let fit = (args.left(arguments) / step).min(WINDOW_MOST) as u32;
                            (step as u32, (if w.records == 1 { 16 } else { w.records * 2 }).min(fit).max(1))
                        } else {
                            (stride, max)
                        }
                    }
                    _ => (stride, max),
                };
                args.refresh(list, arguments, len.max(u64::from(step) * u64::from(records)));
                if let Some((s, at)) = &counter {
                    s.refresh(list, *at, 4);
                }
                // Bounded first: the clamp binds a compute pipeline of its
                // own and copies clamped active records into a scratch
                // buffer at offset 0.
                let count_arg = counter.as_ref().map(|(s, at)| (&s.buffer, s.offset(*at)));
                // More than the record asked for: a window clamped ahead,
                // whose unfilled slots must not spend the list's budget.
                let bounded = if windowed && records > 1 {
                    crate::indirect::bound_ahead(list, (&args.buffer, offset), (kind, step, records))
                } else {
                    crate::indirect::bound(list, (&args.buffer, offset), count_arg, (kind, step, records))
                };
                let Some(bounded) = bounded else {
                    self.window = None;
                    why("not bounded (no clamp, or arguments not 4-byte aligned)");
                    return false;
                };
                self.host_pipeline = 0;
                self.host_roots[1] = 0;
                self.window = windowed.then(|| Window {
                    copy: args.id(),
                    kind,
                    stride,
                    step,
                    base: offset,
                    records,
                    scratch: bounded.clone(),
                });
                (bounded, 0)
            }
        };
        let ran = match kind {
            DRAW | INDEXED => self.prepare(list, kind == INDEXED),
            MESH => self.prepare_for(list, false, true),
            DISPATCH => self.bind(list, p, true),
            _ => false,
        };
        if !ran {
            crate::indirect::done(list, &bounded);
            #[cfg(feature = "diagnostics")]
            match crate::pipeline::host_pipeline(self.pipeline as This) {
                None => {
                    crate::pipeline::note_drop(self.pipeline as This);
                    list.note_op(|| format!("DROPPED indirect kind {kind} (no host pipeline) {}", crate::pipeline::shaders_of(pipeline)));
                }
                Some(p) => why(&format!(
                    "prepare failed (compute {}, mesh {}): {}",
                    p.compute,
                    p.mesh,
                    NOT_RUN.with(|w| w.replace(""))
                )),
            }
            #[cfg(not(feature = "diagnostics"))]
            why("prepare failed");
            return false;
        }
        #[cfg(feature = "diagnostics")]
        list.note_op(|| {
            self.op_text(format!("indirect kind {kind} max {max} at {arguments:#x}"), kind == DISPATCH)
        });
        if kind == MESH {
            self.mesh_scissor(list, true);
        }
        // SAFETY: an open list with everything set; the scratch buffer and
        // host copies are kept alive by the list, and the signature is cached.
        unsafe {
            list.list.ExecuteIndirect(
                &signature,
                max,
                &bounded,
                at,
                counter.as_ref().map(|(s, _)| &s.buffer),
                counter.as_ref().map_or(0, |(s, at)| s.offset(*at)),
            )
        };
        if kind == MESH {
            self.mesh_scissor(list, false);
        }
        // A listed frame: what the records held (after the clamp, in the
        // scratch buffer the execution read).
        #[cfg(feature = "diagnostics")]
        if crate::capture::dump_on() {
            let count = counter.as_ref().map(|(s, at)| (&s.buffer, s.offset(*at)));
            let name = crate::pipeline::shaders_of(pipeline);
            crate::capture::read_indirect(list, (&bounded, at), count, (kind, stride, max), name);
        }
        crate::indirect::done(list, &bounded);
        if kind == DISPATCH {
            DISPATCHED.fetch_add(1, Ordering::Relaxed);
        } else {
            DRAWN.fetch_add(1, Ordering::Relaxed);
        }
        true
    }
}

/// The host command signature for one argument of `kind` with records
/// `stride` bytes apart (no root arguments, so no root signature), made
/// once.
fn host_signature(kind: u32, stride: u32) -> Option<ID3D12CommandSignature> {
    static MADE: std::sync::Mutex<Vec<((u32, u32), ID3D12CommandSignature)>> =
        std::sync::Mutex::new(Vec::new());
    let mut made = MADE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((_, s)) = made.iter().find(|(k, _)| *k == (kind, stride)) {
        return Some(s.clone());
    }
    let argument = D3D12_INDIRECT_ARGUMENT_DESC {
        Type: D3D12_INDIRECT_ARGUMENT_TYPE(kind as i32),
        ..Default::default()
    };
    let desc = D3D12_COMMAND_SIGNATURE_DESC {
        ByteStride: stride,
        NumArgumentDescs: 1,
        pArgumentDescs: &argument,
        NodeMask: 0,
    };
    let mut signature: Option<ID3D12CommandSignature> = None;
    // SAFETY: a description of one argument; no root arguments.
    let result = unsafe {
        crate::host::host()?.device.CreateCommandSignature(
            &desc,
            None::<&ID3D12RootSignature>,
            &mut signature,
        )
    };
    if let Err(e) = result {
        once!(error = %e, kind, stride, "host command signature refused");
        return None;
    }
    let signature = signature?;
    made.push(((kind, stride), signature.clone()));
    Some(signature)
}

/// Gets what a shader reads or writes through a view ready: buffer data
/// copied, textures moved to the state they need.
fn view_kind(view: &View) -> Option<ResourceType> {
    match view {
        View::NullConstant | View::Constant(..) => Some(ResourceType::Cbv),
        View::Shader(..) | View::Placed(false, ..) => Some(ResourceType::Srv),
        View::Unordered(..) | View::Placed(true, ..) => Some(ResourceType::Uav),
        _ => None,
    }
}

/// The host state a depth view needs of its texture: the read state if the
/// view writes no plane the format has (the title's read-only flags), else
/// the write state.
/// shortcut: whole textures have one state, so a view that writes one of
/// two planes counts as writing.
fn depth_state(texture: &ID3D12Resource, desc: Option<&D3D12_DEPTH_STENCIL_VIEW_DESC>) -> u32 {
    // D16_UNORM and D32_FLOAT: depth formats without a stencil plane.
    const ONE_PLANE: [i32; 2] = [55, 40];
    let Some(desc) = desc else {
        return DEPTH_WRITE;
    };
    let format = if desc.Format.0 == 0 {
        // SAFETY: a live texture; a view without a format has its own.
        unsafe { texture.GetDesc() }.Format.0
    } else {
        desc.Format.0
    };
    let mut needed = D3D12_DSV_FLAG_READ_ONLY_DEPTH.0;
    if !ONE_PLANE.contains(&format) {
        needed |= D3D12_DSV_FLAG_READ_ONLY_STENCIL.0;
    }
    if desc.Flags.0 & needed == needed {
        DEPTH_READ
    } else {
        DEPTH_WRITE
    }
}

fn readable(list: &HostList, kind: ResourceType, view: &View) {
    const UNORDERED_ACCESS: u32 = 0x8;
    let placed = crate::descriptor::host::placed_texture(view);
    let view = placed.as_ref().unwrap_or(view);
    // Buffer views read title memory: their range is copied at submission;
    // what shaders write through unordered access goes back to it after.
    if let Some((shadow, address, len)) = crate::descriptor::host::buffer_copy(view) {
        shadow.refresh(list, address, len);
        if kind == ResourceType::Uav {
            // A shader that cannot store to buffers leaves the range as
            // the title has it: nothing to find afterwards.
            if STORES.get() || !write_skip() {
                shadow.gpu_writes(list, address, len);
            } else {
                #[cfg(feature = "diagnostics")]
                {
                    crate::copystat::add(crate::copystat::C::GpuWritesLeftOut, 1);
                    crate::copystat::add(crate::copystat::C::GpuWritesLeftOutBytes, len);
                }
            }
        }
    }
    match (view, kind) {
        (View::Shader(resource, _), ResourceType::Srv) => {
            // A view the title placed, in another format, in the memory of
            // a texture the host writes: that texture's texels, copied on
            // the host (title memory does not have them).
            let aliased = crate::resource::alias_source(*resource as This)
                .is_some_and(|source| crate::texture::alias_copy(list, source, *resource as This));
            if !aliased
                && let Some(t) = crate::resource::host_resource(*resource as This)
                && crate::texture::title_written(t)
            {
                crate::texture::upload(list, *resource as This, t);
            }
            // A depth texture shaders also write: what they wrote, first.
            list.twin_use(*resource as This, false, false);
            // The depth target the draw only tests is read in both ways at
            // once: both read states, which its preparation asks for too.
            let (tested, _) = TESTED_DEPTH.get();
            let state = if tested != 0 && tested == *resource {
                TESTED_DEPTH.set((tested, true));
                DEPTH_READ | SHADER_RESOURCE
            } else {
                SHADER_RESOURCE
            };
            crate::list::ensure_state(list, *resource as This, state);
            #[cfg(feature = "diagnostics")]
            if let Some(t) = crate::resource::host_resource(*resource as This) {
                // SAFETY: a live resource.
                let d = unsafe { t.GetDesc() };
                if d.Flags.0 & 7 == 0 {
                    tracing::debug!(
                        resource = format_args!("{resource:#x}"),
                        ?d,
                        "a shader reads a texture the title fills"
                    );
                }
            }
        }
        (View::Unordered(resource, _, _), ResourceType::Uav) => {
            #[cfg(feature = "diagnostics")]
            crate::capture::note_target(list, *resource as This);
            // Unordered access to a depth texture is its colour twin's.
            if !list.twin_use(*resource as This, true, true) {
                crate::list::ensure_state(list, *resource as This, UNORDERED_ACCESS);
            }
        }
        _ => {}
    }
}

/// A view in a few words (diagnostics builds).
#[cfg(feature = "diagnostics")]
fn describe(view: Option<&View>) -> String {
    let texture = |r: This| {
        crate::resource::host_made(r).map(|t| {
            // SAFETY: a live resource.
            let d = unsafe { t.GetDesc() };
            // Its console memory: another view of it (a placed one) is
            // another host texture.
            let at = crate::resource::texture_memory(r).map_or(0, |(p, _)| p as u64);
            format!("{}x{} f{} flags {:#x} mem {at:#x} of {r:?}", d.Width, d.Height, d.Format.0, d.Flags.0)
        })
    };
    // Small buffers (exposure, counters): their title memory now, as floats.
    let small = |range: Option<(u64, u64)>| match range {
        Some((at, len)) if len <= 64 && crate::texture::committed(at as *const u8, len as usize) => {
            // SAFETY: `len` committed bytes of title memory.
            let b = unsafe { std::slice::from_raw_parts(at as *const u8, len as usize) };
            let f: Vec<f32> = b.as_chunks::<4>().0.iter().map(|&c| f32::from_le_bytes(c)).collect();
            format!(" = {f:?}")
        }
        _ => String::new(),
    };
    match view {
        None => "no view".into(),
        Some(v @ View::Shader(r, _)) => texture(*r as This).unwrap_or_else(|| {
            let range = crate::descriptor::host::buffer_range(v);
            format!("buffer {range:?}{}", small(range))
        }),
        // A buffer has no host texture: its title range, as for a shader view.
        Some(v @ View::Unordered(r, ..)) => match texture(*r as This) {
            Some(t) => format!("uav texture {t}"),
            None => {
                let range = crate::descriptor::host::buffer_range(v);
                format!("uav resource {r:#x} buffer {range:?}{}", small(range))
            }
        },
        Some(View::Placed(u, r, d)) => format!(
            "placed uav={u} dim {} {}x{} f{} layout {:#x} at {:#x}",
            r[0],
            u64::from(r[4]) | u64::from(r[5]) << 32,
            r[6],
            r[8],
            r[11],
            u64::from(d[10]) | u64::from(d[11]) << 32
        ),
        // The first four values, as a colour rectangle's constants are, read
        // through the kernel (asking `VirtualQuery` first cost milliseconds
        // per draw in the title's pools, with every draw noted).
        Some(View::Constant(address, size)) if *size >= 16 => {
            let mut bytes = [0u8; 16];
            if crate::host::read_into(*address as *const u8, &mut bytes) {
                let v: [f32; 4] = std::array::from_fn(|i| f32::from_le_bytes([0, 1, 2, 3].map(|k| bytes[4 * i + k])));
                format!("Constant({address:#x}, {size}) {v:?}")
            } else {
                format!("Constant({address:#x}, {size})")
            }
        }
        Some(v) => format!("{v:?}").chars().take(60).collect(),
    }
}

/// The console's own primitive topologies.
const TRIANGLE_FAN: u32 = 6;
const RECTANGLES: u32 = 17;
const LINE_LOOP: u32 = 18;
const QUADS: u32 = 19;
const QUAD_STRIP: u32 = 20;

/// Fixed-function state, straight onto the host list.
pub(crate) fn viewports(list: &HostList, viewports: &[D3D12_VIEWPORT]) {
    // SAFETY: an open list.
    unsafe { list.list.RSSetViewports(viewports) };
}

pub(crate) fn scissors(list: &HostList, rects: &[RECT]) {
    // SAFETY: an open list.
    unsafe { list.list.RSSetScissorRects(rects) };
}

pub(crate) fn blend_factor(list: &HostList, factor: Option<&[f32; 4]>) {
    // SAFETY: an open list.
    unsafe { list.list.OMSetBlendFactor(factor) };
}

pub(crate) fn stencil_ref(list: &HostList, reference: u32) {
    #[cfg(feature = "diagnostics")]
    crate::capture::op(std::ptr::from_ref(list) as usize, || format!("STENCIL REF {reference:#x}"));
    // SAFETY: an open list.
    unsafe { list.list.OMSetStencilRef(reference) };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::{HRESULT, S_OK};
    use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
    use windows::Win32::Graphics::Direct3D::ID3DBlob;
    use windows::core::s;

    /// A compute shader with its root signature inside (`RS`).
    fn compile_cs(source: &str) -> (ID3D12RootSignature, ID3D12PipelineState) {
        let host = crate::host::host().unwrap();
        let (mut code, mut errors): (Option<ID3DBlob>, Option<ID3DBlob>) = (None, None);
        // SAFETY: the source and names outlive the call.
        unsafe {
            D3DCompile(
                source.as_ptr().cast(),
                source.len(),
                s!("root"),
                None,
                None,
                s!("main"),
                s!("cs_5_1"),
                0,
                0,
                &mut code,
                Some(&mut errors),
            )
            .unwrap();
        }
        let code = code.unwrap();
        // SAFETY: the blob's own bytes.
        let bytes = unsafe {
            std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize())
        };
        // SAFETY: a compiled shader with its root signature inside.
        unsafe {
            let root: ID3D12RootSignature = host.device.CreateRootSignature(0, bytes).unwrap();
            let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: bytes.as_ptr().cast(),
                    BytecodeLength: bytes.len(),
                },
                ..Default::default()
            };
            let pso = host.device.CreateComputePipelineState(&desc).unwrap();
            (root, pso)
        }
    }

    /// A title buffer of `bytes` in memory of its own.
    fn buffer(bytes: u64) -> This {
        let desc: [u32; 14] = [1, 0, 0, 0, bytes as u32, (bytes >> 32) as u32, 1, 1 | 1 << 16, 0, 1, 0, 1, 0, 0];
        let properties = [1u32, 0, 0, 0, 0];
        let mut out = std::ptr::null();
        let iid = crate::object::guid(0x696442be, 0xa72e, 0x4059, [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad]);
        let made: HRESULT = crate::resource::create_committed(
            std::ptr::null(),
            properties.as_ptr().cast(),
            0,
            desc.as_ptr().cast(),
            0,
            std::ptr::null(),
            &iid,
            &mut out,
        );
        assert_eq!(made, S_OK);
        out
    }


    /// Title memory of `bytes` (64 KiB aligned) for textures at an address.
    fn title_memory(bytes: usize) -> u64 {
        let layout = std::alloc::Layout::from_size_align(bytes, 1 << 16).unwrap();
        // SAFETY: a non-zero size; leaked, as title memory outlives a test.
        unsafe { std::alloc::alloc_zeroed(layout) as u64 }
    }

    /// A 2D resource description as words: size, format, layout, flags.
    fn texture_words(width: u32, height: u32, format: u32, layout: u32, flags: u32) -> [u32; 14] {
        [3, 0, 0, 0, width, 0, height, 1 | 1 << 16, format, 1, 0, layout, flags, 0]
    }

    /// A title texture (resource object) at `address`.
    fn texture_at_address(address: u64, words: &[u32; 14]) -> This {
        let iid = crate::object::guid(0x696442be, 0xa72e, 0x4059, [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad]);
        let mut out = std::ptr::null();
        let made = crate::resource::create_at_address(std::ptr::null(), address, words.as_ptr().cast(), 0, std::ptr::null(), &iid, &mut out);
        assert_eq!(made, S_OK);
        out
    }

    /// A placed shader view (device 0x230) of a 2D texture at `address`.
    fn placed_view(address: u64, words: [u32; 14]) -> View {
        let mut desc = [0u32; 16];
        desc[..6].copy_from_slice(&[words[8], 4, 0x1688, 0, 0, 1]);
        (desc[10], desc[11]) = (address as u32, (address >> 32) as u32);
        View::Placed(false, words, desc)
    }


    /// Slots of a shader-visible view heap of ours.
    struct Slots(ID3D12DescriptorHeap, u32);

    impl Slots {
        fn new(n: u32) -> Self {
            let host = crate::host::host().unwrap();
            let desc = D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: n,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            };
            // SAFETY: plain creation.
            let heap = unsafe { host.device.CreateDescriptorHeap(&desc) }.unwrap();
            // SAFETY: a plain query.
            let step = unsafe { host.device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) };
            Slots(heap, step)
        }

        fn make(&self, i: u32, view: &View) {
            // SAFETY: a live heap.
            let mut at = unsafe { self.0.GetCPUDescriptorHandleForHeapStart() };
            at.ptr += (i * self.1) as usize;
            assert!(crate::descriptor::host::make_view(view, at).is_some(), "slot {i} made");
        }

        fn gpu(&self, i: u32) -> D3D12_GPU_DESCRIPTOR_HANDLE {
            // SAFETY: a live shader-visible heap.
            let mut at = unsafe { self.0.GetGPUDescriptorHandleForHeapStart() };
            at.ptr += u64::from(i * self.1);
            at
        }
    }

    /// Records a dispatch of `pso` over `groups` with the table at slot
    /// `table` as parameter 0 and, if given, root UAV `out` as parameter 1.
    fn run(
        list: &HostList,
        slots: &Slots,
        (root, pso): &(ID3D12RootSignature, ID3D12PipelineState),
        table: u32,
        out: Option<&ID3D12Resource>,
        groups: (u32, u32),
    ) {
        // SAFETY: an open list and live objects.
        unsafe {
            list.list.SetDescriptorHeaps(&[Some(slots.0.clone())]);
            list.list.SetComputeRootSignature(root);
            list.list.SetPipelineState(pso);
            list.list.SetComputeRootDescriptorTable(0, slots.gpu(table));
            if let Some(out) = out {
                list.list.SetComputeRootUnorderedAccessView(1, out.GetGPUVirtualAddress());
            }
            list.list.Dispatch(groups.0, groups.1, 1);
        }
    }

    const READ: &str = r#"
#define RS "DescriptorTable(SRV(t0)), UAV(u1)"
Texture2D<float4> t : register(t0);
RWByteAddressBuffer o : register(u1);
[RootSignature(RS)]
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    o.Store4((id.y * 64 + id.x) * 16, asuint(t.Load(int3(id.xy, 0))));
}
"#;

    /// One texture memory written through an unordered view of a resource
    /// object in one format and read through a placed view in another: a
    /// castable pair shares one host texture (no copy), a pair of other
    /// shapes over the same bytes (block-compressed against plain) is two,
    /// bridged by a copy at the read.
    #[test]
    fn a_texture_written_in_one_format_reads_in_another_through_a_placed_view() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let slots = Slots::new(4);
        let read = compile_cs(READ);
        let (out, data) = host.shared_buffer(64 * 64 * 16).unwrap();
        // Format written and size, what the shader writes, format read and
        // size, the texel expected, whether one host texture.
        type Case = (u32, u32, &'static str, u32, u32, [f32; 4], bool);
        let cases: [Case; 2] = [
            // R32_UINT holding R9G9B9E5 (1, 0.5, 0.25): the sky's tables.
            (
                42,
                64,
                "RWTexture2D<uint> o : register(u0); void put(uint2 p) { o[p] = 256u | 128u << 9 | 64u << 18 | 16u << 27; }",
                67,
                64,
                [1.0, 0.5, 0.25, 1.0],
                true,
            ),
            // R32G32_UINT blocks read as BC1: solid red blocks.
            (
                17,
                16,
                "RWTexture2D<uint2> o : register(u0); void put(uint2 p) { o[p] = uint2(0xF800, 0); }",
                71,
                64,
                [1.0, 0.0, 0.0, 1.0],
                false,
            ),
        ];
        for (round, (written, size, put, format, read_size, want, shared)) in cases.into_iter().enumerate() {
            let address = title_memory(1 << 20);
            let object = texture_at_address(address, &texture_words(size, size, written, 0x105, 4));
            let view = placed_view(address, texture_words(read_size, read_size, format, 0x105, 0));
            let Some(View::Shader(placed, _)) = crate::descriptor::host::placed_texture(&view) else {
                panic!("round {round}: a placed texture");
            };
            let a = crate::resource::host_copy(object).unwrap();
            let b = crate::resource::host_copy(placed as This).unwrap();
            // A cross-family cast only shares storage if the device accepted
            // this format in the source texture's cast list. The copy path
            // must produce the same sampled values on legacy devices.
            let shared = shared && a.castable.contains(&format);
            assert_eq!(std::sync::Arc::ptr_eq(&a, &b), shared, "round {round}: one host texture");
            let unordered = View::Unordered(object as usize, 0, Some([written, 4, 0, 0, 0, 0, 0, 0, 0, 0]));
            slots.make(0, &unordered);
            slots.make(1, &view);
            let write = compile_cs(&format!(
                "#define RS \"DescriptorTable(UAV(u0))\"\n{put}\n[RootSignature(RS)]\n[numthreads(8, 8, 1)]\nvoid main(uint3 id : SV_DispatchThreadID) {{ put(id.xy); }}"
            ));
            let list = host.new_list().unwrap();
            let copies = crate::texture::ALIAS_COPIES.load(Ordering::Relaxed);
            readable(&list, ResourceType::Uav, &unordered);
            run(&list, &slots, &write, 0, None, (size / 8, size / 8));
            readable(&list, ResourceType::Srv, &view);
            run(&list, &slots, &read, 1, Some(&out), (read_size / 8, read_size / 8));
            list.close();
            host.execute(&[&list]);
            let copied = crate::texture::ALIAS_COPIES.load(Ordering::Relaxed) > copies;
            assert_eq!(copied, !shared, "round {round}: a copy only where not shared");
            let n = read_size as usize;
            for i in [0usize, 7, n * 9 + 33, n * n - 1] {
                let (x, y) = (i % n, i / n);
                // SAFETY: inside the mapped output.
                let got = unsafe { (data.add((y * 64 + x) * 16) as *const [f32; 4]).read() };
                assert_eq!(got, want, "round {round} texel {x},{y}");
            }
        }
    }

    /// A depth texture with stencil (the console's 16-bit depth, the
    /// host's 24-bit) written through unordered views of its planes, then
    /// read as depth and stencil: the 16-bit depth arrives in 24 bits as
    /// the host converts the same value, the stencil as written. Rendered
    /// as depth (a clear), then used through the unordered view: back in
    /// 16 bits.
    #[test]
    fn a_depth_texture_written_through_unordered_views_reads_as_depth() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let slots = Slots::new(4);
        let address = title_memory(1 << 22);
        let depth = texture_at_address(address, &texture_words(64, 64, 118, 0, 2));
        let copy = crate::resource::host_copy(depth).expect("a host depth texture");
        let planes = [
            View::Unordered(depth as usize, 0, Some([56, 4, 0, 0, 0, 0, 0, 0, 0, 0])),
            View::Unordered(depth as usize, 0, Some([62, 4, 0, 1, 0, 0, 0, 0, 0, 0])),
        ];
        slots.make(0, &planes[0]);
        slots.make(1, &planes[1]);
        let reads = [
            View::Shader(depth as usize, Some([119, 4, 0x1688, 0, 0, 1, 0, 0, 0, 0])),
            View::Shader(depth as usize, Some([120, 4, 0x1688, 0, 0, 1, 1, 0, 0, 0])),
        ];
        slots.make(2, &reads[0]);
        slots.make(3, &reads[1]);
        let write = compile_cs(
            r#"
#define RS "DescriptorTable(UAV(u0, numDescriptors=2))"
RWTexture2D<float> d : register(u0);
RWTexture2D<uint> s : register(u1);
[RootSignature(RS)]
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    d[id.xy] = (1 + id.x + 64 * id.y) * (1.0 / 65535.0);
    s[id.xy] = (id.x ^ id.y) & 0xff;
}
"#,
        );
        let read = compile_cs(
            r#"
#define RS "DescriptorTable(SRV(t0, numDescriptors=2)), UAV(u1)"
Texture2D<float> d : register(t0);
Texture2D<uint2> s : register(t1);
RWByteAddressBuffer o : register(u1);
[RootSignature(RS)]
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    o.Store2((id.y * 64 + id.x) * 8, uint2(asuint(d.Load(int3(id.xy, 0))), s.Load(int3(id.xy, 0)).g));
}
"#,
        );
        let (out, data) = host.shared_buffer(64 * 64 * 8).unwrap();
        // The depth texture's planes as bytes, read back after.
        // SAFETY: a live resource.
        let dd = unsafe { copy.texture.GetDesc() };
        let mut f = [D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); 2];
        let mut total = 0;
        // SAFETY: outputs for two subresources.
        unsafe { host.device.GetCopyableFootprints(&dd, 0, 2, 0, Some(f.as_mut_ptr()), None, None, Some(&mut total)) };
        let (raw, raw_data) = host.shared_buffer(total).unwrap();
        let list = host.new_list().unwrap();
        for v in &planes {
            readable(&list, ResourceType::Uav, v);
        }
        run(&list, &slots, &write, 0, None, (8, 8));
        for v in &reads {
            readable(&list, ResourceType::Srv, v);
        }
        assert!(list.take_rebind(), "the conversion ran in the list");
        run(&list, &slots, &read, 2, Some(&out), (8, 8));
        crate::list::ensure_state(&list, depth, 0x800);
        let at = |r: &ID3D12Resource, sub: u32| D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the copy does not release.
            pResource: unsafe { std::mem::transmute_copy(r) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: sub },
        };
        let to = |fp: D3D12_PLACED_SUBRESOURCE_FOOTPRINT| D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(&raw) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: fp },
        };
        // SAFETY: an open list, the texture in the copy source state.
        unsafe {
            list.list.CopyTextureRegion(&to(f[0]), 0, 0, 0, &at(&copy.texture, 0), None);
            list.list.CopyTextureRegion(&to(f[1]), 0, 0, 0, &at(&copy.texture, 1), None);
        }
        list.close();
        host.execute(&[&list]);
        let d24 = |u: u32| u * 256 + (u * 255 + 32767) / 65535;
        for (x, y) in [(0usize, 0usize), (5, 0), (63, 2), (17, 40), (63, 63)] {
            let k = (1 + x + 64 * y) as u32;
            // SAFETY: inside the mapped buffers.
            let [depth_bits, stencil] = unsafe { (data.add((y * 64 + x) * 8) as *const [u32; 2]).read() };
            let got = f32::from_bits(depth_bits);
            assert!((got - k as f32 / 65535.0).abs() < 1e-7, "depth at {x},{y}: {got}");
            assert_eq!(stencil, ((x ^ y) & 0xff) as u32, "stencil at {x},{y}");
            // SAFETY: inside the mapped buffer, by the footprints.
            let (word, byte) = unsafe {
                (
                    (raw_data.add(f[0].Offset as usize + y * f[0].Footprint.RowPitch as usize + x * 4) as *const u32).read(),
                    raw_data.add(f[1].Offset as usize + y * f[1].Footprint.RowPitch as usize + x).read(),
                )
            };
            assert_eq!(word & 0xff_ffff, d24(k), "24-bit depth at {x},{y}");
            assert_eq!(u32::from(byte), ((x ^ y) & 0xff) as u32);
        }
        // Rendered as depth (a clear of both planes), then used through
        // the unordered view: copied to the twin, converted back.
        list.reset();
        crate::list::ensure_state(&list, depth, 0x10);
        let dsv = D3D12_DEPTH_STENCIL_VIEW_DESC {
            Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_D24_UNORM_S8_UINT,
            ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2D,
            ..Default::default()
        };
        let handle = host.dsv(&copy.texture, Some(&dsv));
        // SAFETY: an open list, the texture in the depth write state.
        unsafe {
            list.list.ClearDepthStencilView(handle, D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, 0.5, 0x11, None)
        };
        // The next list's prelude must copy from the cleared depth texture
        // before publishing the final COPY_SOURCE states of the two planes.
        list.close();
        host.execute(&[&list]);
        list.reset();
        readable(&list, ResourceType::Uav, &planes[0]);
        let twin = copy.twin.get().and_then(Option::as_ref).expect("a twin");
        let mut g = [D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); 2];
        for (p, plane) in twin.planes.iter().enumerate() {
            // SAFETY: a live resource.
            let pd = unsafe { plane.texture.GetDesc() };
            // SAFETY: an output for one subresource.
            unsafe { host.device.GetCopyableFootprints(&pd, 0, 1, f[p].Offset, Some(&mut g[p]), None, None, None) };
            if let Some(b) = list.move_texture(plane, 0x800) {
                // SAFETY: an open list.
                unsafe { list.list.ResourceBarrier(&[b]) };
            }
        }
        // SAFETY: an open list, the twin's planes in the copy source state.
        unsafe {
            list.list.CopyTextureRegion(&to(g[0]), 0, 0, 0, &at(&twin.planes[0].texture, 0), None);
            list.list.CopyTextureRegion(&to(g[1]), 0, 0, 0, &at(&twin.planes[1].texture, 0), None);
        }
        list.close();
        host.execute(&[&list]);
        for (x, y) in [(0usize, 0usize), (63, 63)] {
            // SAFETY: inside the mapped buffer, by the footprints.
            let (half, byte) = unsafe {
                (
                    (raw_data.add(g[0].Offset as usize + y * g[0].Footprint.RowPitch as usize + x * 2) as *const u16).read(),
                    raw_data.add(g[1].Offset as usize + y * g[1].Footprint.RowPitch as usize + x).read(),
                )
            };
            assert_eq!(half, 32768, "16-bit depth at {x},{y}");
            assert_eq!(byte, 0x11, "stencil at {x},{y}");
        }
    }


    /// Unordered-access-only 1D and 3D title textures (the colour grading
    /// bakes: a 4096 R32_FLOAT curve, a 64^3 RGBA16F table) keep what a
    /// dispatch writes through their unordered views, whole (a 3D view of
    /// every depth slice, -1), and a shader view reads it.
    #[test]
    fn unordered_1d_and_3d_textures_keep_their_writes() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let slots = Slots::new(2);
        let (out, data) = host.shared_buffer(64 * 64 * 16).unwrap();
        // Description words (dimension, width, height, depth | mips << 16,
        // format), the unordered view's and shader view's words, the
        // shaders' bodies, groups, a texel to check and its value.
        type Case = ([u32; 5], [u32; 10], [u32; 10], &'static str, &'static str, (u32, u32), [f32; 4]);
        let cases: [Case; 2] = [
            (
                [2, 4096, 1, 1 | 1 << 16, 41],
                [41, 2, 0, 0, 0, 0, 0, 0, 0, 0],
                [41, 2, 0x1688, 0, 0, 1, 0, 0, 0, 0],
                "RWTexture1D<float> o : register(u0); void put(uint3 p) { if (p.y == 0 && p.z == 0) o[p.x] = p.x * 0.25; }",
                "Texture1D<float4> t : register(t0); float4 get(uint3 p) { return t.Load(int2(p.y * 64 + p.x, 0)); }",
                (512, 1),
                [(2 * 64 + 7) as f32 * 0.25, 0.0, 0.0, 1.0],
            ),
            (
                [4, 64, 64, 64 | 1 << 16, 10],
                [10, 8, 0, 0, u32::MAX, 0, 0, 0, 0, 0],
                [10, 8, 0x1688, 0, 0, 1, 0, 0, 0, 0],
                "RWTexture3D<float4> o : register(u0); void put(uint3 p) { o[p] = float4(p, 1) / 64; }",
                "Texture3D<float4> t : register(t0); float4 get(uint3 p) { return t.Load(int4(p.x, p.y, 40, 0)); }",
                (8, 8),
                [7.0 / 64.0, 2.0 / 64.0, 40.0 / 64.0, 1.0 / 64.0],
            ),
        ];
        for (round, (d, uav, srv, put, get, groups, want)) in cases.into_iter().enumerate() {
            let address = title_memory(1 << 22);
            let words = [d[0], 0, 0, 0, d[1], 0, d[2], d[3], d[4], 1, 0, 0, 4, 0];
            let texture = texture_at_address(address, &words);
            let unordered = View::Unordered(texture as usize, 0, Some(uav));
            let shader = View::Shader(texture as usize, Some(srv));
            slots.make(0, &unordered);
            slots.make(1, &shader);
            let write = compile_cs(&format!(
                "#define RS \"DescriptorTable(UAV(u0))\"\n{put}\n[RootSignature(RS)]\n[numthreads(8, 8, 1)]\nvoid main(uint3 id : SV_DispatchThreadID) {{ for (uint z = 0; z < 64; z++) put(uint3(id.xy, z)); }}"
            ));
            let read = compile_cs(&format!(
                "#define RS \"DescriptorTable(SRV(t0)), UAV(u1)\"\n{get}\nRWByteAddressBuffer o : register(u1);\n[RootSignature(RS)]\n[numthreads(8, 8, 1)]\nvoid main(uint3 id : SV_DispatchThreadID) {{ o.Store4((id.y * 64 + id.x) * 16, asuint(get(id))); }}"
            ));
            let list = host.new_list().unwrap();
            readable(&list, ResourceType::Uav, &unordered);
            run(&list, &slots, &write, 0, None, groups);
            readable(&list, ResourceType::Srv, &shader);
            run(&list, &slots, &read, 1, Some(&out), (8, 8));
            list.close();
            host.execute(&[&list]);
            // SAFETY: inside the mapped output: texel (7, 2) of the read.
            let got = unsafe { (data.add((2 * 64 + 7) * 16) as *const [f32; 4]).read() };
            assert_eq!(got, want, "round {round}");
        }
    }

    /// A structured view the title places off its buffer's elements (the
    /// culling constants at a ring position) is made over a copy of its own
    /// bytes; one on an element views the buffer's copy.
    #[test]
    fn structured_views_off_their_buffers_elements_get_their_own_copy() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let src = buffer(1 << 16);
        let (bytes, _) = crate::resource::buffer_bytes(src).unwrap();
        let start = bytes as u64;
        // SAFETY: a plain CPU-side heap of one view.
        let cpu: ID3D12DescriptorHeap = unsafe {
            host.device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: 1,
                ..Default::default()
            })
        }
        .unwrap();
        // SAFETY: a live heap.
        let at = unsafe { cpu.GetCPUDescriptorHandleForHeapStart() };
        // Four 456-byte elements placed at `address` (resource and view
        // descriptions as the title passes them to device 0x230).
        let view = |address: u64| {
            let resource: [u32; 14] = [1, 0, 0, 0, 4 * 456, 0, 1, 1 | 1 << 16, 0, 1, 0, 1, 0, 0];
            let mut desc = [0u32; 16];
            desc[..10].copy_from_slice(&[0, 1, 5768, 0, 0, 0, 4, 456, 0, 0]);
            (desc[10], desc[11]) = (address as u32, (address >> 32) as u32);
            View::Placed(false, resource, desc)
        };
        let width = |address: u64| {
            let made = crate::descriptor::host::make_view(&view(address), at).expect("a view");
            // SAFETY: a live buffer.
            unsafe { made.GetDesc() }.Width
        };
        assert_eq!(width(start + 456), 1 << 16, "on an element: the buffer's copy");
        assert_eq!(width(start + 100), (4 * 456u64).next_multiple_of(256), "off: its own bytes");
        let (copy, address, len) = crate::descriptor::host::view_copy(&view(start + 100)).unwrap();
        assert_eq!((copy.range(), address, len), ((start + 100, 4 * 456), start + 100, 4 * 456));
        drop(copy);
        // A slot made from the view holds its copy: after more than a
        // thousand other views' copies, draws still find (and refresh) the
        // copy the slot shows; rewritten, the slot lets it go.
        let null = std::ptr::null();
        let heap_desc: [u32; 8] = [0, 8, 1, 0, 0, 0, 0, 0];
        let mut heap = null;
        let iid = crate::object::guid(0x8efb471d, 0x616c, 0x4f49, [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51]);
        assert_eq!(crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &iid, &mut heap), S_OK);
        let slots = crate::descriptor::heap(heap).host().expect("a host view heap");
        slots.make_at(0, &view(start + 100));
        let shown = std::sync::Arc::downgrade(&crate::descriptor::host::view_copy(&view(start + 100)).unwrap().0);
        for k in 1..1200 {
            if (100 + 4 * k) % 456 != 0 {
                crate::descriptor::host::view_copy(&view(start + 100 + 4 * k)).unwrap();
            }
        }
        let found = crate::descriptor::host::view_copy(&view(start + 100)).unwrap().0;
        assert!(shown.upgrade().is_some_and(|s| std::sync::Arc::ptr_eq(&s, &found)), "the slot's copy");
        drop(found);
        slots.make_null(0);
        // Rewriting a descriptor retires its previous view through the next
        // submission so a previously recorded list can still use the resource.
        let list = host.new_list().unwrap();
        list.close();
        host.execute(&[&list]);
        assert!(shown.upgrade().is_none(), "released after the submission finishes");
    }

    /// With a heap map, a table whose entries lie below the identity window
    /// is read in place, and one whose entries reach past it (its title
    /// slots have no host slot at their own index) is made afresh in
    /// dynamic slots, which follow the map's slot.
    #[test]
    fn tables_past_the_heap_maps_window_are_made_afresh() {
        use crate::descriptor::host::{DYNAMIC, TEST_MAP};
        let Some(host) = crate::host::host() else {
            return;
        };
        let list = host.new_list().expect("host list");
        let null: This = std::ptr::null();
        let mut heap_obj: This = null;
        let desc = [0u32, 64, 1, 0, 0, 0, 0, 0];
        let iid = ID3D12DescriptorHeap::IID;
        assert_eq!(
            crate::descriptor::create_heap(
                null,
                desc.as_ptr().cast(),
                std::ptr::from_ref(&iid).cast(),
                &mut heap_obj,
            ),
            crate::object::S_OK
        );
        let start = crate::descriptor::heap(heap_obj).start;
        TEST_MAP.set(Some((16, 24)));
        let heap = crate::descriptor::heap(heap_obj).host().expect("a host heap");
        TEST_MAP.set(None);
        assert_eq!(heap.count(), 16);

        let mut state = State::default();
        state.set_heaps(&[heap_obj]);
        // Entry 1 of the table: one descriptor past its start.
        let used = [UsedResource {
            kind: ResourceType::Srv,
            root: RootType::Table,
            table_offset_dwords: 8,
            lower_bound: 0,
            stride: 0,
            dwords: 8,
            user_data: 0,
            slot: 0,
        }];
        // A table at 2: entry 1 is title slot 3, in the window.
        state.set_table(false, 0, start + 2 * 32);
        assert!(state.tables(&list, &used, &[], false).is_empty(), "read in place");
        // A table at 15: entry 1 is title slot 16, past it.
        state.set_table(false, 0, start + 15 * 32);
        let made = state.tables(&list, &used, &[], false);
        assert_eq!(made.len(), 1, "made afresh");
        let first = made[0].1 / 32;
        assert!((25..25 + DYNAMIC).contains(&first), "{first}: a dynamic slot");
        list.close();
        drop(state);
        let release = crate::object::slot::<extern "system" fn(This) -> u32>(heap_obj, 0x10);
        assert_eq!(release(heap_obj), 0);
    }
}
