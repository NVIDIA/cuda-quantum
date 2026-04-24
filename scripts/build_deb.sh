#!/bin/bash

# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Builds a Debian binary package (.deb) from a staged CUDA-Q install tree.
#
# Produces one of two flavors:
#   core     -> cuda-quantum-cu${N}_${version}_${arch}.deb,
#               installing to /opt/nvidia/cudaq
#   realtime -> cuda-quantum-realtime-cu${N}_${version}_${arch}.deb,
#               installing to /opt/nvidia/cudaq-realtime
#
# Usage:
#   bash scripts/build_deb.sh -c 12 -i build/cuda_quantum_assets/cudaq -o out
#   bash scripts/build_deb.sh -c 13 -f realtime -i /realtime_assets -o out
#
# Options:
#   -c <cuda_major>   CUDA major version (12 or 13). Required.
#   -f <flavor>       core (default) or realtime.
#   -i <staged_dir>   Staged install tree to package. Required.
#                     For core, typically build/cuda_quantum_assets/cudaq
#                     produced by scripts/build_installer.sh. For realtime,
#                     the tree produced by realtime/scripts/build_installer.sh
#                     (typically /realtime_assets).
#   -o <output_dir>   Output directory for the .deb. Defaults to "out".
#   -V <version>      CUDA-Q version string for the Debian Version field.
#                     Defaults to 0.0.0 if unspecified.
#   -d                Docker mode (no behavioral difference today; kept for
#                     parity with build_installer.sh).
#   -v                Verbose output.
#
# The staged tree is expected to contain bin/, lib/, include/, targets/, and
# set_env.sh at its root. The script does not rewrite RPATHs or binaries --
# relative $ORIGIN RPATHs set at build time handle the relocation to
# /opt/nvidia/cudaq[-realtime].

set -euo pipefail

this_file_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$this_file_dir/.." && pwd)"

cuda_major=""
flavor="core"
staged_dir=""
output_dir="out"
cudaq_version="0.0.0"
docker_mode=false
verbose=false
# xz is the default for CI (smallest deb); gzip is ~10x faster to build and
# useful for local iteration. Valid values match dpkg-deb's -Z flag:
# none, gzip, bzip2, lzma, xz, zstd.
compression="xz"

__optind__=$OPTIND
OPTIND=1
while getopts ":c:f:i:o:V:z:dv" opt; do
  case $opt in
    c) cuda_major="$OPTARG" ;;
    f) flavor="$OPTARG" ;;
    i) staged_dir="$OPTARG" ;;
    o) output_dir="$OPTARG" ;;
    V) cudaq_version="$OPTARG" ;;
    z) compression="$OPTARG" ;;
    d) docker_mode=true ;;
    v) verbose=true ;;
    \?)
      echo "Invalid option -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument" >&2
      exit 1
      ;;
  esac
done
OPTIND=$__optind__

# Silence "unused variable" warnings from set -u; these are accepted but not
# consulted directly (parity with build_installer.sh interface).
: "${docker_mode}"

if [ -z "$cuda_major" ]; then
  echo "Error: CUDA major version required (-c 12 or -c 13)" >&2
  exit 1
fi
if [ "$cuda_major" != "12" ] && [ "$cuda_major" != "13" ]; then
  echo "Error: CUDA major version must be 12 or 13, got: $cuda_major" >&2
  exit 1
fi

if [ -z "$staged_dir" ]; then
  echo "Error: staged install tree required (-i <dir>)" >&2
  exit 1
fi
if [ ! -d "$staged_dir" ]; then
  echo "Error: staged directory does not exist: $staged_dir" >&2
  exit 1
fi
staged_dir="$(cd "$staged_dir" && pwd)"

other_cuda=$([ "$cuda_major" = "12" ] && echo 13 || echo 12)

case "$flavor" in
  core)
    pkg_name="cuda-quantum-cu${cuda_major}"
    conflicts_pkg="cuda-quantum-cu${other_cuda}"
    provides_pkg="cuda-quantum"
    install_prefix="/opt/nvidia/cudaq"
    description_short="NVIDIA CUDA-Q toolkit (CUDA ${cuda_major}.x)"
    profile_name="cuda-quantum-cu${cuda_major}.sh"
    ;;
  realtime)
    pkg_name="cuda-quantum-realtime-cu${cuda_major}"
    conflicts_pkg="cuda-quantum-realtime-cu${other_cuda}"
    provides_pkg="cuda-quantum-realtime"
    install_prefix="/opt/nvidia/cudaq-realtime"
    description_short="NVIDIA CUDA-Q realtime toolkit (CUDA ${cuda_major}.x)"
    profile_name="cuda-quantum-realtime-cu${cuda_major}.sh"
    ;;
  *)
    echo "Error: flavor must be 'core' or 'realtime', got: $flavor" >&2
    exit 1
    ;;
esac

case "$(uname -m)" in
  x86_64)  deb_arch="amd64" ;;
  aarch64) deb_arch="arm64" ;;
  *)
    echo "Error: unsupported architecture $(uname -m)" >&2
    exit 1
    ;;
esac

if $verbose; then
  echo "flavor          = $flavor"
  echo "cuda_major      = $cuda_major"
  echo "staged_dir      = $staged_dir"
  echo "output_dir      = $output_dir"
  echo "cudaq_version   = $cudaq_version"
  echo "package         = $pkg_name"
  echo "install_prefix  = $install_prefix"
  echo "deb_arch        = $deb_arch"
fi

command -v dpkg-deb >/dev/null 2>&1 || {
  echo "Error: dpkg-deb not found on PATH" >&2
  exit 1
}

mkdir -p "$output_dir"
output_dir="$(cd "$output_dir" && pwd)"

work_dir="$(mktemp -d -t cudaq-deb.XXXXXX)"
trap 'rm -rf "$work_dir"' EXIT

pkgroot="$work_dir/pkgroot"
mkdir -p "$pkgroot/DEBIAN" \
         "$pkgroot${install_prefix%/*}" \
         "$pkgroot/etc/profile.d" \
         "$pkgroot/usr/share/doc/${pkg_name}"

# Copy the staged tree into $install_prefix. Use cp -a to preserve symlinks,
# permissions, and timestamps so RPATHs and exec bits land intact.
cp -a "$staged_dir" "$pkgroot${install_prefix}"

# /etc/profile.d drop-in replaces the migrate_assets.sh /etc/profile edit.
cat > "$pkgroot/etc/profile.d/${profile_name}" << PROFILED
# Added by ${pkg_name} deb package.
export CUDAQ_INSTALL_PATH="${install_prefix}"
if [ -f "\${CUDAQ_INSTALL_PATH}/set_env.sh" ]; then
    . "\${CUDAQ_INSTALL_PATH}/set_env.sh"
fi
PROFILED
chmod 0644 "$pkgroot/etc/profile.d/${profile_name}"

# Debian policy requires /usr/share/doc/<pkg>/copyright and (for compressed
# changelogs) changelog.Debian.gz.
if [ -f "$repo_root/LICENSE" ]; then
  {
    echo "Source: https://github.com/NVIDIA/cuda-quantum"
    echo "Upstream-Name: CUDA-Q"
    echo
    echo "Files: *"
    echo "Copyright: 2022 - $(date +%Y) NVIDIA Corporation & Affiliates."
    echo "License: Apache-2.0"
    echo " See /usr/share/common-licenses/Apache-2.0 on Debian systems, or"
    echo " the LICENSE file in the upstream source distribution."
    echo
    sed 's/^/ /' "$repo_root/LICENSE"
  } > "$pkgroot/usr/share/doc/${pkg_name}/copyright"
  chmod 0644 "$pkgroot/usr/share/doc/${pkg_name}/copyright"
fi

# Minimal Debian changelog. Upstream does not maintain a Debian-format
# changelog; one entry per build is the standard pattern for vendor debs.
changelog="$pkgroot/usr/share/doc/${pkg_name}/changelog.Debian"
{
  echo "${pkg_name} (${cudaq_version}) unstable; urgency=medium"
  echo
  echo "  * Automated build from upstream CUDA-Q ${cudaq_version}."
  echo
  echo " -- NVIDIA Corporation <cuda-quantum@nvidia.com>  $(date -R 2>/dev/null || date -u '+%a, %d %b %Y %H:%M:%S +0000')"
} > "$changelog"
gzip -9n "$changelog"
chmod 0644 "${changelog}.gz"

# Compute installed-size (KiB) for the control file. Debian policy wants
# the size of the payload excluding DEBIAN/, in kilobytes.
installed_size=$(du -sk --exclude=DEBIAN "$pkgroot" | cut -f1)

cat > "$pkgroot/DEBIAN/control" << CONTROL
Package: ${pkg_name}
Version: ${cudaq_version}
Architecture: ${deb_arch}
Maintainer: NVIDIA Corporation <cuda-quantum@nvidia.com>
Installed-Size: ${installed_size}
Depends: libc6 (>= 2.35), libc6-dev, libstdc++6 (>= 12), libgomp1, libgcc-s1
Conflicts: ${conflicts_pkg}
Provides: ${provides_pkg}
Section: science
Priority: optional
Homepage: https://developer.nvidia.com/cuda-q
Description: ${description_short}
 CUDA-Q is an open-source platform for integrating and programming quantum
 processing units (QPUs), GPUs, and CPUs in one system.
 .
 This package bundles the toolkit together with the required LLVM, cuQuantum,
 and cuTensor runtimes, and installs to ${install_prefix}. Environment
 variables are configured via /etc/profile.d/${profile_name} for new login
 shells.
CONTROL
chmod 0644 "$pkgroot/DEBIAN/control"

cat > "$pkgroot/DEBIAN/postinst" << POSTINST
#!/bin/sh
set -e

INSTALL_PREFIX="${install_prefix}"

case "\$1" in
    configure)
        # Refresh the dynamic linker cache so bundled libs under
        # \$INSTALL_PREFIX/lib/ are picked up by any consumer that
        # dlopen()s them by soname. The deb does not add its lib dir to
        # /etc/ld.so.conf.d because all CUDA-Q consumers either use RPATH
        # or explicit LD_LIBRARY_PATH via the profile.d drop-in.
        ldconfig || true

        # Build the MPI plugin if a system MPI is discoverable. Mirrors the
        # behavior of scripts/migrate_assets.sh. Failures are non-fatal --
        # the user can rerun activate_custom_mpi.sh later.
        PLUGIN_DIR="\$INSTALL_PREFIX/distributed_interfaces"
        if [ -x "\$INSTALL_PREFIX/bin/nvq++" ] && [ -x "\$PLUGIN_DIR/activate_custom_mpi.sh" ]; then
            if command -v mpicc >/dev/null 2>&1 || command -v ompi_info >/dev/null 2>&1; then
                (cd "\$PLUGIN_DIR" && bash activate_custom_mpi.sh) || \\
                    echo "Warning: CUDA-Q MPI plugin build failed; rerun \$PLUGIN_DIR/activate_custom_mpi.sh after installing an MPI implementation." >&2
            fi
        fi
        ;;
esac

exit 0
POSTINST
chmod 0755 "$pkgroot/DEBIAN/postinst"

cat > "$pkgroot/DEBIAN/prerm" << PRERM
#!/bin/sh
set -e
exit 0
PRERM
chmod 0755 "$pkgroot/DEBIAN/prerm"

cat > "$pkgroot/DEBIAN/postrm" << POSTRM
#!/bin/sh
set -e

INSTALL_PREFIX="${install_prefix}"

case "\$1" in
    purge|remove)
        # Remove the compiled MPI plugin artifact that postinst may have
        # produced; the rest of the payload is under INSTALL_PREFIX and
        # already handled by dpkg.
        rm -f "\$INSTALL_PREFIX/distributed_interfaces/libcudaq_distributed_interface_mpi.so" \\
              "\$INSTALL_PREFIX/distributed_interfaces/mpi_comm_impl.o" 2>/dev/null || true
        ldconfig || true
        # On purge, clean up the install prefix if it is empty.
        if [ "\$1" = "purge" ] && [ -d "\$INSTALL_PREFIX" ]; then
            rmdir --ignore-fail-on-non-empty "\$INSTALL_PREFIX" 2>/dev/null || true
        fi
        ;;
esac

exit 0
POSTRM
chmod 0755 "$pkgroot/DEBIAN/postrm"

# mktemp -d yields 0700 on the parent; dpkg-deb rejects any directory under
# pkgroot that isn't world-readable + executable. Normalize all directories
# (but leave file bits alone so exec bits on binaries are preserved).
find "$pkgroot" -type d -exec chmod 0755 {} +

# Build the .deb. --root-owner-group sets root:root on all files without
# requiring actual root (works inside unprivileged CI containers).
deb_file="${output_dir}/${pkg_name}_${cudaq_version}_${deb_arch}.deb"
if $verbose; then
  dpkg-deb --build --root-owner-group "-Z${compression}" "$pkgroot" "$deb_file"
else
  dpkg-deb --build --root-owner-group "-Z${compression}" "$pkgroot" "$deb_file" >/dev/null
fi

echo "Built $deb_file"

# Best-effort lintian sanity pass. Warnings only; CUDA-Q bundles too many
# runtime libs for strict lintian compliance.
if command -v lintian >/dev/null 2>&1; then
  lintian --no-tag-display-limit "$deb_file" || true
fi
