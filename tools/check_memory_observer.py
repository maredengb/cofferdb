"""Control observer counters with a touched allocation, without loading CofferDB."""
import ctypes
import gc
import json
import platform
import sys
import tracemalloc

from memory_observer import ProcessMemoryObserver

assert 'cofferdb' not in sys.modules
tracemalloc.start()
observer = ProcessMemoryObserver()
libc = ctypes.CDLL(None)
libc.malloc.argtypes = [ctypes.c_size_t]
libc.malloc.restype = ctypes.c_void_p
libc.free.argtypes = [ctypes.c_void_p]
libc.free.restype = None
before = observer.observe()
size = 2 * 1024 * 1024
pointer = libc.malloc(size)
if not pointer:
    raise MemoryError('control allocation refused')
try:
    ctypes.memset(pointer, 1, size)
    during = observer.observe()
finally:
    libc.free(pointer)
gc.collect()
after = observer.observe()
assert all(value['rss']['current_bytes'] > 0 for value in [before, during, after])
assert all(value['python_heap']['enabled'] for value in [before, during, after])
assert all(value['native_allocator']['method'] != 'unavailable' for value in [before, during, after])
if platform.system() == 'Darwin':
    assert during['native_allocator']['size_in_use_bytes'] - before['native_allocator']['size_in_use_bytes'] >= size
    assert during['native_allocator']['size_in_use_bytes'] - after['native_allocator']['size_in_use_bytes'] >= size - 131072
else:
    assert during['native_allocator']['mmap_current_bytes'] >= before['native_allocator']['mmap_current_bytes'] + size
    assert during['native_allocator']['mmap_current_bytes'] >= after['native_allocator']['mmap_current_bytes'] + size
assert 'cofferdb' not in sys.modules
print(json.dumps({'kind': 'NO_ENGINE_OBSERVER_CONTROL', 'platform': platform.platform(),
                  'libc': platform.libc_ver(), 'explicit_touched_allocation_bytes': size,
                  'before': before, 'during': during, 'after_free_and_gc': after,
                  'does_not_measure': 'CofferDB performance, exclusive Rust bytes, lifecycle growth or platform package acceptance'}, sort_keys=True))
