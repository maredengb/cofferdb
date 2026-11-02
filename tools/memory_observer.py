"""Observe Python heap, process allocator totals and RSS for local benchmarks.

Allocator totals include Python and other native consumers, overlap with traced
Python memory and RSS, and are not exclusive Rust engine bytes. Snapshot peaks
are sampled lower bounds. No subtraction or addition attributes these totals to
an engine. Unsupported observations remain null with an explicit reason.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
import platform
import resource
import tracemalloc
import xml.etree.ElementTree as ET


class MallocStatistics(ctypes.Structure):
    _fields_ = [("blocks_in_use", ctypes.c_uint), ("size_in_use", ctypes.c_size_t),
                ("max_size_in_use", ctypes.c_size_t), ("size_allocated", ctypes.c_size_t)]


class TaskBasicInfo(ctypes.Structure):
    _fields_ = [("virtual_size", ctypes.c_uint64), ("resident_size", ctypes.c_uint64),
                ("resident_size_max", ctypes.c_uint64), ("user_seconds", ctypes.c_int32),
                ("user_microseconds", ctypes.c_int32), ("system_seconds", ctypes.c_int32),
                ("system_microseconds", ctypes.c_int32), ("policy", ctypes.c_int32),
                ("suspend_count", ctypes.c_int32)]


class ProcessMemoryObserver:
    def __init__(self):
        self.system = platform.system()
        self.native = ctypes.CDLL(None, use_errno=True)
        if self.system == "Darwin":
            self.native.malloc_zone_statistics.argtypes = [ctypes.c_void_p, ctypes.POINTER(MallocStatistics)]
            self.native.malloc_zone_statistics.restype = None
            self.native.task_info.argtypes = [ctypes.c_uint32, ctypes.c_int32, ctypes.c_void_p,
                                             ctypes.POINTER(ctypes.c_uint32)]
            self.native.task_info.restype = ctypes.c_int32
            self.self_task = ctypes.c_uint32.in_dll(self.native, "mach_task_self_").value
        elif self.system == "Linux":
            self.native.open_memstream.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_size_t)]
            self.native.open_memstream.restype = ctypes.c_void_p
            self.native.malloc_info.argtypes = [ctypes.c_int, ctypes.c_void_p]
            self.native.malloc_info.restype = ctypes.c_int
            self.native.fclose.argtypes = [ctypes.c_void_p]
            self.native.fclose.restype = ctypes.c_int
            self.native.free.argtypes = [ctypes.c_void_p]
            self.native.free.restype = None
        else:
            raise ValueError("benchmark observations support Darwin and glibc Linux")

    def native_totals(self):
        if self.system == "Darwin":
            value = MallocStatistics()
            self.native.malloc_zone_statistics(None, ctypes.byref(value))
            high_water = value.max_size_in_use
            return {"method": "darwin_malloc_zone_statistics_all_zones",
                    "size_in_use_bytes": value.size_in_use, "reserved_bytes": value.size_allocated,
                    "blocks_in_use": value.blocks_in_use,
                    "high_water_reported_bytes": high_water,
                    "process_lifetime_high_water_bytes": high_water if high_water >= value.size_in_use else None,
                    "high_water_status": "reported" if high_water >= value.size_in_use else "unavailable_counter_below_current_use"}
        buffer = ctypes.c_void_p()
        length = ctypes.c_size_t()
        stream = self.native.open_memstream(ctypes.byref(buffer), ctypes.byref(length))
        if not stream:
            raise OSError(ctypes.get_errno(), "open_memstream refused")
        try:
            try:
                status = self.native.malloc_info(0, stream)
            finally:
                closed = self.native.fclose(stream)
            if status or closed:
                raise OSError(ctypes.get_errno(), "malloc_info or stream close failed")
            if not buffer.value or length.value > 4 * 1024 * 1024:
                raise ValueError("malloc_info output is absent or exceeds the bounded observer buffer")
            data = ctypes.string_at(buffer, length.value)
        finally:
            self.native.free(buffer)
        root = ET.fromstring(data)
        def size(path):
            element = root.find(path)
            return None if element is None else int(element.attrib["size"])
        return {"method": "glibc_malloc_info_root_totals",
                "arena_system_current_bytes": size("system[@type='current']"),
                "arena_system_max_bytes": size("system[@type='max']"),
                "free_fast_chunks_bytes": size("total[@type='fast']"),
                "free_other_chunks_bytes": size("total[@type='rest']"),
                "mmap_current_bytes": size("total[@type='mmap']"),
                "size_in_use_bytes": None,
                "size_in_use_status": "not_derived_from_overlapping_allocator_counters"}

    def rss(self):
        if self.system == "Darwin":
            value = TaskBasicInfo()
            count = ctypes.c_uint32(ctypes.sizeof(value) // ctypes.sizeof(ctypes.c_uint32))
            status = self.native.task_info(self.self_task, 20, ctypes.byref(value), ctypes.byref(count))
            if status:
                raise OSError(status, "self task_info refused")
            return {"method": "mach_task_basic_info_self", "current_bytes": value.resident_size,
                    "process_lifetime_peak_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        fields = {}
        for line in Path("/proc/self/status").read_text().splitlines():
            key, _, value = line.partition(":")
            if key in {"VmRSS", "VmHWM"}:
                number, unit = value.split()
                if unit != "kB":
                    raise ValueError("unrecognized proc memory unit")
                fields[key] = int(number) * 1024
        return {"method": "proc_self_status", "current_bytes": fields["VmRSS"],
                "process_lifetime_peak_bytes": fields["VmHWM"]}

    def observe(self):
        traced = tracemalloc.is_tracing()
        current, peak = tracemalloc.get_traced_memory() if traced else (None, None)
        try:
            native = self.native_totals()
        except (OSError, ValueError, ET.ParseError) as error:
            native = {"method": "unavailable", "reason": type(error).__name__ + ": " + str(error)}
        directory = "/dev/fd" if self.system == "Darwin" else "/proc/self/fd"
        try:
            fd_count = len(os.listdir(directory))
        except OSError:
            fd_count = None
        return {"python_heap": {"method": "tracemalloc", "enabled": traced,
                                "current_bytes": current, "peak_since_last_reset_bytes": peak},
                "native_allocator": native, "rss": self.rss(),
                "open_fd_snapshot_count": fd_count}


if __name__ == "__main__":
    import argparse
    import json
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace-python',action='store_true',help='start Python tracing for this observer process')
    args=parser.parse_args()
    if args.trace_python:
        tracemalloc.start()
    print(json.dumps(ProcessMemoryObserver().observe(),sort_keys=True))
