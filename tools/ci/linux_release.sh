#!/usr/bin/env bash
# Run inside the pinned manylinux image on its native GitHub runner.
set -euo pipefail
umask 022

phase=${1:?build, accept or restore required}
machine=$(uname -m)
case "$RUNNER_ARCH:$machine" in
    ARM64:aarch64)
        rustup_sha=e3853c5a252fca15252d07cb23a1bdd9377a8c6f3efa01531109281ae47f841c
        maturin_url=https://files.pythonhosted.org/packages/b9/f9/f5ca9fe8cad70cac6f3b6008598cc708f8a74dd619baced99784a6253f23/maturin-1.10.2-py3-none-manylinux_2_17_aarch64.manylinux2014_aarch64.musllinux_1_1_aarch64.whl
        maturin_sha=a41ec70d99e27c05377be90f8e3c3def2a7bae4d0d9d5ea874aaf2d1da625d5c
        ;;
    X64:x86_64)
        rustup_sha=20a06e644b0d9bd2fbdbfd52d42540bdde820ea7df86e92e533c073da0cdd43c
        maturin_url=https://files.pythonhosted.org/packages/46/8d/753148c0d0472acd31a297f6d11c3263cd2668d38278ed29d523625f7290/maturin-1.10.2-py3-none-manylinux_2_12_x86_64.manylinux2010_x86_64.musllinux_1_1_x86_64.whl
        maturin_sha=efcd496a3202ffe0d0489df1f83d08b91399782fb2dd545d5a1e7bf6fd81af39
        ;;
    *) exit 1 ;;
esac
[[ "$(getconf GNU_LIBC_VERSION)" == "glibc 2.28" ]]
[[ "$SOURCE_SHA" =~ ^[0-9a-f]{40}$ ]]
abis=(cp312 cp313 cp314)
pythons=()
# Both pinned images supply these regular CPython builds.
versions=(3.12.16 3.13.18 3.14.10)
for abi in "${abis[@]}"; do
    python="/opt/python/$abi-$abi/bin/python"
    pythons+=("$python")
done
arguments=(--commit "$SOURCE_SHA" --arch "$RUNNER_ARCH" --python "${pythons[@]}" --versions "${versions[@]}")
cd /work
if [[ "$phase" == restore ]]; then
    exec "${pythons[0]}" -I -B /assertions/release_wheel.py restore /work "${arguments[@]}" --wheels /wheels --snapshot /snapshot
fi
git config --global --add safe.directory /src
driver=/src/tools/ci/release_wheel.py
if [[ "$phase" == accept ]]; then
    exec "${pythons[0]}" -I -B "$driver" accept /work "${arguments[@]}" --repo /src --built /build-input --fixture /fixture
fi
[[ "$phase" == build ]]
verify_download() {
    local url=$1 output=$2 checksum=$3
    curl --fail --silent --show-error --location --retry 3 --proto '=https' "$url" --output "$output"
    printf '%s  %s\n' "$checksum" "$output" | sha256sum --check --status
}
host="$machine-unknown-linux-gnu"
verify_download "https://static.rust-lang.org/rustup/archive/1.28.2/$host/rustup-init" rustup-init "$rustup_sha"
verify_download "https://static.rust-lang.org/dist/channel-rust-1.97.1.toml" rust-channel.toml 03569b1886ceb5c05276b50c8431ab111de944cd6140fe1fa7d821dd8e0f29cf
export CARGO_HOME=/work/cargo-home RUSTUP_HOME=/work/rustup CARGO_TARGET_DIR=/work/prepare-target
export PATH="$CARGO_HOME/bin:$PATH"
chmod 755 rustup-init
./rustup-init -y --no-modify-path --profile minimal --default-host "$host" --default-toolchain 1.97.1
"${pythons[0]}" -I - "$host" rust-channel.toml "$(rustc --print sysroot)/lib/rustlib" <<'PY'
import sys, tomllib
from pathlib import Path
host, upstream, rustlib = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
expected = tomllib.loads(upstream.read_text())
installed = tomllib.loads((rustlib / "multirust-channel-manifest.toml").read_text())
packages = ("rustc", "cargo", "rust-std")
if set((rustlib / "components").read_text().splitlines()) != {f"{name}-{host}" for name in packages}:
    sys.exit("unexpected installed Rust components")
if installed["date"] != expected["date"]:
    sys.exit("installed Rust distribution date differs")
for name in packages:
    target = expected["pkg"][name]["target"][host]
    actual = installed["pkg"][name]["target"][host]
    if any(actual[key] != target[key] for key in ("available", "url", "hash", "xz_url", "xz_hash")):
        sys.exit("installed Rust component differs from the pinned distribution")
PY
wheel=${maturin_url##*/}
verify_download "$maturin_url" "$wheel" "$maturin_sha"
"${pythons[0]}" -I -m venv /work/tools
/work/tools/bin/python -I -m pip --isolated install --no-index --no-deps "$wheel"
command -v cc ld readelf auditwheel > /dev/null
"${pythons[0]}" -I -B "$driver" build /work "${arguments[@]}" --repo /src --maturin /work/tools/bin/maturin --build-only

# Transfer the comparison executable separately so normal installed-wheel tests run without it on PATH.
mkdir -p transfer/built
cp build-environment.json build-tools.json comparison-coffer transfer/
cp built/build-manifest.json transfer/built/
for abi in "${abis[@]}"; do
    mkdir "transfer/built/$abi"
    cp "built/$abi/cofferdb-0.2.0-$abi-$abi-manylinux_2_28_$machine.whl" "transfer/built/$abi/"
done
