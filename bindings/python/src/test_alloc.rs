//! Test-only observation immediately before an owned buffer is actually freed.

use cofferdb::binding::Input;
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::marker::PhantomData;

#[derive(Clone, Copy)]
struct Watch {
    pointer: *const u8,
    input: *mut Input,
    result: *const Cell<Option<bool>>,
}

thread_local! {
    static WATCH: Cell<Option<Watch>> = const { Cell::new(None) };
}

struct Allocator;
#[global_allocator]
static ALLOCATOR: Allocator = Allocator;

// SAFETY: allocations use System unchanged. The one-thread watch is disarmed
// before its callback; its guard exclusively borrows Input and keeps the result
// alive until observation ends. No pointer is retained past that guard.
unsafe impl GlobalAlloc for Allocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        unsafe { System.realloc(pointer, layout, size) }
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        let watch = WATCH
            .try_with(|slot| {
                if slot.get().is_some_and(|watch| watch.pointer == pointer) {
                    slot.take()
                } else {
                    None
                }
            })
            .ok()
            .flatten();
        if let Some(watch) = watch {
            let bytes = [b'q'; 512];
            // SAFETY: these bytes are ASCII; the guard's exclusive Input and
            // result references remain live during this synchronous deallocation.
            let text = unsafe { std::str::from_utf8_unchecked(&bytes) };
            let admitted = unsafe { (&mut *watch.input).text(text).is_ok() };
            unsafe {
                (&*watch.result).set(Some(admitted));
            }
        }
        unsafe { System.dealloc(pointer, layout) }
    }
}

pub(crate) struct Guard<'a>(PhantomData<(&'a mut Input, &'a Cell<Option<bool>>)>);
impl Drop for Guard<'_> {
    fn drop(&mut self) {
        let _ = WATCH.try_with(|watch| watch.set(None));
    }
}

pub(crate) fn watch<'a>(
    pointer: *const u8,
    input: &'a mut Input,
    result: &'a Cell<Option<bool>>,
) -> Guard<'a> {
    WATCH.with(|watch| {
        assert!(watch.get().is_none(), "allocation observations cannot nest");
        watch.set(Some(Watch {
            pointer,
            input,
            result,
        }));
    });
    Guard(PhantomData)
}
