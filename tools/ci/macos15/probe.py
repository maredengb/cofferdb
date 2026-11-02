"""Run only inside the exact selected runtime sandbox; no SDK assertions."""
import argparse
import errno
import json
import os
import platform
import socket
import subprocess
import sys
import sysconfig


def denial(action):
    try:
        action()
    except OSError as error:
        return {"denied": error.errno in (errno.EPERM, errno.EACCES), "errno": error.errno}
    return {"denied": False, "errno": None}


def probe(compiler, address, port, expected_minor):
    def execute():
        # A mere nonzero compiler exit is NOT a sandbox denial.
        # If an unexpected compiler launch succeeds, its actual output remains
        # in the enclosing supervisor's original streams, not a hidden PIPE.
        result = subprocess.run([compiler, '--version'], stdin=subprocess.DEVNULL, timeout=3)
        if result.returncode:
            raise RuntimeError('Compiler launched; nonzero exit is not denial proof')

    def read():
        with open(compiler, 'rb') as stream:
            stream.read(1)

    def connect():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as stream:
            stream.settimeout(3)
            stream.connect((address, port))

    facts = {"compiler_exec": denial(execute), "compiler_read": denial(read),
             "network_outbound": denial(connect)}
    runtime = {"system": platform.system(), "mac_version": platform.mac_ver()[0],
               "machine": platform.machine(), "implementation": sys.implementation.name,
               "minor": sys.version_info.minor, "major": sys.version_info.major,
               "free_threaded": bool(sysconfig.get_config_var('Py_GIL_DISABLED'))}
    good_runtime = (runtime['system'] == 'Darwin' and runtime['mac_version'].split('.')[0] == '15'
                    and runtime['machine'] == 'arm64' and runtime['implementation'] == 'cpython'
                    and runtime['major'] == 3 and runtime['minor'] == expected_minor and not runtime['free_threaded'])
    return {"status": 'MAC_SANDBOX_PROBES_PASSED' if good_runtime and all(
                value['denied'] for value in facts.values()) else 'HOLD_MAC_SANDBOX_PROBES',
            "runtime": runtime, "facts": facts, "product_acceptance": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler', required=True)
    parser.add_argument('--address', required=True)
    parser.add_argument('--port', required=True, type=int)
    parser.add_argument('--expected-minor', required=True, type=int, choices=[12, 13, 14])
    args = parser.parse_args()
    result = probe(args.compiler, args.address, args.port, args.expected_minor)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result['status'] == 'MAC_SANDBOX_PROBES_PASSED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
