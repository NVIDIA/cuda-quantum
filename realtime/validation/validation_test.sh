#!/usr/bin/env bash
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Orchestrate installation validation: resolve apps/plugins, start the RTH,
# pass its endpoint to playback, release their shared start gate, then report
# playback's packet/timing results and stop the RTH. Installed validate.sh is
# a thin wrapper over this runner; the same script serves local builds.
# Quoted option lists are tokenized without shell execution, and endpoint text
# is passed through unchanged to the selected playback provider.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# Empty paths mean resolve from the selected preset and build/install layout.
validation_app=""
playback_app=""
bridge=""
provider=""
execution=""
transport=roce
transport_selected=false
# Preserve the original tokens for delegation to the direct HSB workflow.
legacy_action=false
original_arguments=("$@")
legacy_device=""
legacy_fpga_ip=""
legacy_bridge_ip=""
legacy_mtu=""
# Default RoCE ring: 128 slots of 384 bytes. Platform-specific sizes can be
# selected with --page-size and --num-pages.
page_size=384
num_pages=128
page_size_selected=false
num_pages_selected=false
bin_dir=""
source_dir=""
# Common execution/workload controls. All pacing/capture budgets use ns;
# conversion to whole seconds happens only for process watchdogs.
dispatch=separate
execution_selected=false
dispatch_selected=false
gpu_id=0
threads=32
samples=100
message_size=32
message_send_rate=10000
quiet_ns=1000000000
timeout_ns=30000000000
output_dir=""
# Arrays preserve argument boundaries through parsing and process launch.
bridge_args=()
provider_args=()
fail() { echo "ERROR: $*" >&2; exit 1; }
usage() {
    cat <<'HELP'
Validate request/reply correctness with CUDA-Q realtime and report available RTT.

Usage:
  validate.sh [options]          Installed entry point
  validation_test.sh [options]  Local build entry point (realtime/build)

Quick starts:
  validate.sh --dispatch=unified \
    --bridge-arg="--device=mlx5_0 --peer-ip=192.168.0.2"
      HSB FPGA -> GPU over RoCE. Adjust the NIC and FPGA address.
  validate.sh --cpu --dispatch=unified \
    --bridge-arg="--device=mlx5_0 --peer-ip=192.168.0.2"
      HSB FPGA -> CPU over RoCE; no GPU execution.
  validate.sh --udp
      Local UDP validation on the CPU; no FPGA needed.

Selection:
  --roce / --udp            Transport preset (default: RoCE)
  --cpu / --gpu [N]         CPU or GPU execution; optional GPU ordinal
  --gpu-id=N               GPU ordinal (default: 0)
  --dispatch=MODE          separate (default), unified, or forward
  --threads=N              GPU dispatch block size (default: 32)

  No selection flags: RoCE + GPU bridge + HSB playback provider.
  --cpu: RoCE + CPU bridge + HSB playback provider.
  HSB playback requires the CUDA driver library (libcuda.so.1), including with
  --cpu. CPU RoCE requires libibverbs and a configured RDMA device.
  --udp: UDP bridge + UDP playback provider + CPU, even without --cpu.
  --roce and --udp are mutually exclusive, as are --cpu and --gpu.
  The built-in UDP bridge supports separate/unified CPU dispatch, not GPU
  or forward. Unsupported paths fail without silently changing execution.
  Selections and resolved plugin paths are printed to stdout.

Workload and timing:
  --num-messages=N         Request count (default: 100)
  --message-size=N         RPC header + payload bytes (default: 32 = 24 + 8)
  --message-send-rate=N    Positive integer messages/second (default: 10000)
  --quiet-ns=N             Capture quiet interval (default: 1000000000)
  --timeout-ns=N           Separate setup/run/cleanup budgets (default: 30000000000)

  The requested rate and configured message spacing are logged.
  HSB spacing is rounded up to whole microseconds, with a minimum of 1 µs.
  Durations above are integer nanoseconds. Quiet time must be positive and
  shorter than the timeout. RPC increment adds one to each payload byte.
  Forward mode instead checks an unchanged native echo, without RPC dispatch.

Plugin overrides and options:
  --bridge=LIB             RTH bridge .so (default: selected transport/CPU/GPU)
  --playback-provider=LIB  Playback .so (default: HSB for RoCE, UDP for UDP)
  --bridge-arg="OPTIONS"   Space-separated bridge options
  --playback-arg="OPTIONS" Space-separated playback-provider options

  Plugin paths are optional; built-in libraries are found in the installation
  or realtime/build. Built-in bridge settings follow the selected dispatch,
  GPU and message size. Explicit bridge options override preset defaults.
  Built-in RoCE accepts --device NAME or --device=NAME in bridge options.
  Custom bridges receive user options unchanged.

  Quote the whole list; quote values inside it when they contain spaces:
    --bridge-arg="--device=mlx5_0 --peer-ip=192.168.0.2"
    --playback-arg='--label="two words" --port=5000'
  The latter illustrates provider-defined options, not HSB options. HSB needs
  no playback arguments; it obtains connection settings from the RTH bridge.
  Lists may be repeated. Bash parses quoting without executing shell expressions
  or expanding variables. No Python installation is needed.
  Options after -- go directly to the bridge.
  Runner flags such as --cpu and --setup-network take no =value suffix.

Logs and executable locations:
  --output-dir=PATH        New directory for commands.sh, rth.log, playback.log
                          (default: a new temporary directory; logs are retained)
  --validation-app=PATH    Override the RTH executable
  --playback-app=PATH      Override the playback executable
  --help / -h              Show this help

Existing command compatibility:
  --device DEV            RoCE NIC name (default: auto-detect)
  --fpga-ip ADDR          FPGA address (default: 192.168.0.2)
  --gpu N                 Select GPU execution and ordinal N
  --unified / --forward   Aliases for the corresponding --dispatch mode
  --payload-size N        Payload bytes; message size becomes N + 24
  --page-size N           RoCE slot stride (default: 384 bytes)
  --num-pages N           RoCE ring slots (default: 128)
  --timeout N             Operation timeout in seconds
  --bin-dir DIR           Executable directory
  --cuda-quantum-dir DIR  Locate tools in DIR/realtime/build
  --save-temps            Accepted; logs are always retained
  --provider LIB          Alias for --playback-provider
  --provider-arg ARG      Compatibility option; forwards one argument verbatim

  --bridge-ip ADDR and --mtu N retain their network-setup/emulator scope.
  They do not change a hardware-only validation run; the script logs this.
  --build, --setup-network, --no-run, --emulate, --no-verify, --hsb-dir DIR,
  --jobs N, --emulator-ip ADDR, --control-port N and --app NAME delegate to
  the retained roce_test.sh with legacy options. The script logs that choice.
  These operations use the legacy tools, not the playback-provider workflow.
  --dispatch and --message-size translate to legacy options; --bridge-arg and
  arguments after -- retain their exact tokens (requires --app validation_app).
  Playback-provider options, --message-send-rate, --quiet-ns, --timeout-ns and
  --output-dir have no legacy equivalent and are rejected in this mode.

All value options accept --name=value or --name value.

Results:
  Packet correctness and timing availability are reported separately.
  Missing timing is logged and does not fail otherwise-correct validation.
  Exit 0: playback validation succeeded; timing may be unavailable.
  Exit 1: arguments, packet verification, or runner failure.
  Exit 2: run incomplete or timed out. Exit 3: playback-provider error.
  Exit 4: unsupported execution path. External watchdogs may return 124/137.
  After playback succeeds, RTH shutdown failures are warnings, not failures.
  VALIDATION PASS is printed after bounded RTH cleanup.
HELP
}
# Split quoted lists using Bash only. No eval, expansion or command execution.
split_options() {
    local input=$1 token="" quote="" character next_character
    # Distinguish an explicitly quoted empty argument from whitespace between
    # arguments. quote records whether characters are currently literal/grouped.
    local token_started=false index
    parsed_options=()
    for ((index = 0; index < ${#input}; ++index)); do
        character=${input:index:1}
        if [[ "$quote" == "'" ]]; then
            if [[ "$character" == "'" ]]; then quote="";
            else token+=$character; fi
        # Inside double quotes only the listed escapes lose their backslash;
        # dollar signs/backticks are copied, never expanded or executed.
        elif [[ "$quote" == '"' ]]; then
            case "$character" in
                '"') quote="" ;;
                '\')
                    next_character=${input:index+1:1}
                    case "$next_character" in
                        '"'|'\'|'$'|'`') token+=$next_character; index=$((index + 1)) ;;
                        $'\n') index=$((index + 1)) ;;
                        *) token+=$character ;;
                    esac ;;
                *) token+=$character ;;
            esac
        else
            case "$character" in
                ' '|$'\t'|$'\n')
                    if "$token_started"; then parsed_options+=("$token"); fi
                    token=""; token_started=false ;;
                "'"|'"') quote=$character; token_started=true ;;
                '\')
                    index=$((index + 1))
                    (( index < ${#input} )) || fail "Trailing backslash in option list"
                    character=${input:index:1}
                    if [[ "$character" != $'\n' ]]; then
                        token+=$character; token_started=true
                    fi ;;
                *) token+=$character; token_started=true ;;
            esac
        fi
    done
    # Finalize the last token even without trailing whitespace. An unmatched
    # quote is an error rather than a partially forwarded provider argument.
    [[ -z "$quote" ]] || fail "Unclosed quote in option list"
    if "$token_started"; then parsed_options+=("$token"); fi
}
select_execution() {
    [[ -z "$execution" || "$execution" == "$1" ]] || fail "--cpu and --gpu are mutually exclusive"
    execution=$1
    execution_selected=true
}
select_dispatch() {
    if "$dispatch_selected" && [[ "$dispatch" != "$1" ]]; then
        fail "Conflicting dispatch modes"
    fi
    dispatch=$1
    dispatch_selected=true
}
set_gpu_index() {
    local requested=$1
    [[ "$requested" =~ ^[0-9]+$ && ${#requested} -le 18 ]] || fail "Invalid GPU index: $requested"
    (( 10#$requested <= 2147483647 )) || fail "Invalid GPU index: $requested"
    gpu_id=$((10#$requested))
}
# Parse runner options only up to --. The remaining tokens belong to the bridge.
while (( $# )); do
    argument=$1
    shift
    option=${argument%%=*}
    value=""
    # Diagnose flag assignments before the value parser can mistake an empty
    # suffix for a missing argument, or legacy delegation can enable the flag.
    case "$option" in
        --help|-h|--roce|--udp|--cpu|--unified|--forward|--save-temps|--build|--setup-network|--no-run|--emulate|--no-verify|--)
            [[ "$argument" == "$option" ]] || fail "$option takes no value" ;;
    esac
    # Match presence-only flags exactly: stripping =value here could enable
    # actions such as --setup-network even when the supplied value is "no".
    case "$argument" in
        --help|-h) usage; exit 0 ;;
        --roce|--udp)
            if "$transport_selected" && [[ "$transport" != "${option#--}" ]]; then fail "--roce and --udp are mutually exclusive"; fi
            transport=${option#--}; transport_selected=true; continue ;;
        --cpu) select_execution cpu; continue ;;
        --gpu|--gpu=*)
            select_execution gpu
            if [[ "$argument" == *=* ]]; then gpu_id=${argument#*=};
            elif [[ "${1:-}" =~ ^[0-9]+$ ]]; then gpu_id=$1; shift; fi
            set_gpu_index "$gpu_id"
            continue ;;
        --unified|--forward) select_dispatch "${option#--}"; continue ;;
        --save-temps) continue ;; # Logs are always preserved by this runner.
        --build|--setup-network|--no-run|--emulate|--no-verify)
            legacy_action=true; continue ;;
        --) bridge_args+=("$@"); break ;;
    esac
    # Normalize --key=value and --key value only for runner-owned options.
    # Bridge tokens were already separated above and never enter this path.
    if [[ "$argument" == *=* ]]; then value=${argument#*=};
    else
        (( $# )) || fail "Missing value for $option"
        # Plugin option lists may themselves start with --. Other runner
        # values must not consume the next flag as an argument by mistake.
        case "$option" in
            --bridge-arg|--playback-arg|--provider-arg) ;;
            *) [[ "$1" != --* && "$1" != -h ]] || fail "Missing value for $option" ;;
        esac
        value=$1; shift
    fi
    case "$option" in
        --bridge-arg|--playback-arg|--provider-arg) ;;
        *) [[ -n "$value" ]] || fail "Missing value for $option" ;;
    esac
    case "$option" in
        --dispatch) select_dispatch "$value" ;;
        --bridge) bridge=$value ;;
        --playback-provider|--provider) provider=$value ;;
        --gpu-id) set_gpu_index "$value" ;;
        --threads) threads=$value ;;
        --num-messages) samples=$value ;;
        --message-size) message_size=$value ;;
        --payload-size)
            [[ "$value" =~ ^[0-9]+$ && ${#value} -le 9 ]] || fail "Invalid payload size"
            message_size=$((10#$value + 24)) ;;
        --message-send-rate) message_send_rate=$value ;;
        --quiet-ns) quiet_ns=$value ;;
        --timeout-ns) timeout_ns=$value ;;
        --timeout)
            [[ "$value" =~ ^[0-9]+$ && ${#value} -le 8 ]] || fail "Invalid timeout in seconds"
            timeout_ns=$((10#$value * 1000000000)) ;;
        --bridge-arg) split_options "$value"; bridge_args+=("${parsed_options[@]}") ;;
        --playback-arg) split_options "$value"; provider_args+=("${parsed_options[@]}") ;;
        --provider-arg) provider_args+=("$value") ;; # Forward one argument verbatim.
        --device) legacy_device=$value ;;
        --fpga-ip) legacy_fpga_ip=$value ;;
        --bridge-ip) legacy_bridge_ip=$value ;;
        --mtu) legacy_mtu=$value ;;
        --page-size) page_size=$value; page_size_selected=true ;;
        --num-pages) num_pages=$value; num_pages_selected=true ;;
        --bin-dir) bin_dir=$value ;;
        --cuda-quantum-dir) source_dir=$value ;;
        --hsb-dir|--jobs|--emulator-ip|--control-port|--app) legacy_action=true ;;
        --validation-app) validation_app=$value ;;
        --playback-app) playback_app=$value ;;
        --output-dir) output_dir=$value ;;
        *) fail "Unknown runner option: $argument" ;;
    esac
done
# Validate common values before either execution path can launch commands.
case "$dispatch" in separate|unified|forward) ;; *) fail "Invalid dispatch mode: $dispatch" ;; esac
# Bounds also prevent signed Bash arithmetic overflow in the process watchdogs.
for setting in samples message_size message_send_rate quiet_ns timeout_ns gpu_id threads page_size num_pages; do
    value=${!setting}
    [[ "$value" =~ ^[0-9]+$ && ${#value} -le 18 ]] || fail "Invalid integer: $setting"
    # Force decimal interpretation, including values with leading zeros.
    # Bash would otherwise treat a prefix such as 08 as an invalid octal value.
    printf -v "$setting" '%d' "$((10#$value))"
    [[ "$setting" == gpu_id ]] || (( ${!setting} > 0 )) || fail "$setting must be positive"
done
(( gpu_id <= 2147483647 && threads <= 1024 )) || fail "Invalid GPU ID or thread count"
(( message_size >= 24 )) || fail "--message-size must be at least 24 bytes for the RPC header"
# Build, emulator and network-setup actions use the direct-tool runner. Translate
# only options with an equivalent there; reject unsupported combinations early.
if "$legacy_action"; then
    [[ "$transport" == roce ]] || fail "Legacy build/network/emulator operations require --roce"
    legacy_arguments=()
    # Normalize only runner options. The first parse already collected opaque
    # bridge tokens from --bridge-arg and --; never split or reinterpret them.
    set -- "${original_arguments[@]}"
    while (( $# )); do
        argument=$1
        shift
        option=${argument%%=*}
        case "$option" in
            --) break ;;
            --roce) continue ;;
            --cpu|--unified|--forward|--save-temps|--build|--setup-network|--no-run|--emulate|--no-verify)
                legacy_arguments+=("$option"); continue ;;
            --gpu)
                legacy_arguments+=(--gpu)
                if [[ "$argument" == *=* ]]; then legacy_arguments+=("${argument#*=}");
                elif [[ "${1:-}" =~ ^[0-9]+$ ]]; then legacy_arguments+=("$1"); shift; fi
                continue ;;
        esac
        if [[ "$argument" == *=* ]]; then value=${argument#*=};
        else value=$1; shift; fi
        case "$option" in
            --validation-app|--playback-app|--bridge-arg) continue ;;
            --dispatch)
                case "$value" in
                    separate) ;;
                    unified|forward) legacy_arguments+=("--$value") ;;
                    *) fail "Invalid dispatch mode: $value" ;;
                esac ;;
            --message-size)
                [[ "$value" =~ ^[0-9]+$ && ${#value} -le 9 ]] || fail "Invalid message size"
                (( 10#$value >= 24 )) || fail "Message size must be >=24"
                legacy_arguments+=(--payload-size "$((10#$value - 24))") ;;
            --playback-provider|--provider|--playback-arg|--provider-arg|--message-send-rate|--quiet-ns|--timeout-ns|--output-dir)
                fail "$option has no equivalent in the legacy roce_test.sh workflow" ;;
            *) legacy_arguments+=("$option" "$value") ;;
        esac
    done
    # Installer wrappers supply app paths; the legacy script takes a bin
    # directory. Explicit binary/source locations take precedence over it.
    if [[ -z "$bin_dir" && -z "$source_dir" && -n "$validation_app" ]]; then
        legacy_arguments+=(--bin-dir "$(dirname -- "$validation_app")")
    fi
    # All runner options must precede the legacy parser's -- boundary.
    if (( ${#bridge_args[@]} )); then legacy_arguments+=(-- "${bridge_args[@]}"); fi
    echo "Compatibility: using roce_test.sh for legacy build/network/emulator or verification options"
    exec bash "$script_dir/roce_test.sh" "${legacy_arguments[@]}"
fi
# The playback-provider quiet period does not apply to the legacy workflow.
(( quiet_ns < timeout_ns )) || fail "--quiet-ns ($quiet_ns ns; default: 1000000000 ns = 1 s) must be shorter than --timeout-ns ($timeout_ns ns)"
if [[ -z "$execution" ]]; then
    if [[ "$transport" == udp ]]; then execution=cpu; else execution=gpu; fi
fi
selection_reason="explicit"
"$transport_selected" || selection_reason="default"
echo "Transport: $transport ($selection_reason preset; used for omitted plugin paths)"
selection_reason="explicit"
"$execution_selected" || selection_reason="$transport default"
echo "Execution: $execution ($selection_reason)"
[[ -z "$legacy_bridge_ip" ]] || echo "Compatibility: --bridge-ip is used only with --setup-network or --emulate"
[[ -z "$legacy_mtu" ]] || echo "Compatibility: --mtu is used only with --setup-network"
if [[ -n "$source_dir" ]]; then
    [[ -n "$bin_dir" ]] || bin_dir="$source_dir/realtime/build/validation"
fi
if [[ -n "$bin_dir" ]]; then
    validation_app="$bin_dir/validation_app"
    if [[ -x "$bin_dir/playback_app" ]]; then playback_app="$bin_dir/playback_app";
    else playback_app="$bin_dir/../validation/playback_app"; fi
fi
# Resolve packaged libraries beside the runner or under the local realtime build.
bridge_dirs=("$script_dir/../lib" "$script_dir/../../lib" "$script_dir/../build/lib")
playback_dirs=("$script_dir/../lib" "$script_dir" "$script_dir/../build/validation")
if [[ -n "$bin_dir" ]]; then
    bridge_dirs=("$bin_dir/../lib" "$bin_dir/../../lib" "${bridge_dirs[@]}")
    playback_dirs=("$bin_dir/../lib" "$bin_dir/../validation" "${playback_dirs[@]}")
fi
# Return an absolute path from the ordered candidate list. Prefer the chosen
# binary location so installed runs do not accidentally select build artifacts.
find_library() {
    local name=$1 directory
    shift
    for directory in "$@"; do
        if [[ -f "$directory/$name" ]]; then
            printf '%s/%s\n' "$(cd -- "$directory" && pwd)" "$name"
            return
        fi
    done
    fail "Cannot locate $name; build/install it or supply its plugin path"
}
if [[ -z "$bridge" ]]; then
    if [[ "$transport" == udp ]]; then
        [[ "$execution" == cpu ]] || { echo "UNSUPPORTED: built-in UDP bridge requires CPU execution"; exit 4; }
        bridge=$(find_library libcudaq-realtime-bridge-udp.so "${bridge_dirs[@]}")
    else
        bridge=$(find_library "libcudaq-realtime-bridge-$execution-roce.so" "${bridge_dirs[@]}")
    fi
fi
if [[ -z "$provider" ]]; then
    playback_kind=hsb
    [[ "$transport" != udp ]] || playback_kind=udp
    provider=$(find_library "libcudaq-playback-provider-$playback_kind.so" "${playback_dirs[@]}")
fi
# Convert rate to the existing spacing argument, rounding up so playback is
# never configured faster than requested. HSB requires whole microseconds.
spacing_ns=$(((1000000000 + message_send_rate - 1) / message_send_rate))
if [[ "${provider##*/}" == libcudaq-playback-provider-hsb.so ]]; then
    (( spacing_ns >= 1000 )) || fail "HSB requires at least 1000 ns between requests"
    spacing_ns=$((((spacing_ns + 999) / 1000) * 1000))
fi
rate_log="SEND_RATE requested_messages_per_second=$message_send_rate request_spacing_ns=$spacing_ns"
echo "$rate_log"
# Built-in adapters translate common settings. Explicit plugin options follow
# these defaults and can override them. Custom bridges receive only user options.
preset_args=()
case "${bridge##*/}" in
    libcudaq-realtime-bridge-gpu-roce.so|libcudaq-realtime-bridge-cpu-roce.so)
        # These parsers require --device=NAME. Normalize an explicit device
        # before auto-detection so a space-separated value cannot be ignored.
        normalized_bridge_args=()
        bridge_device_selected=false
        for ((argument_index = 0; argument_index < ${#bridge_args[@]}; ++argument_index)); do
            argument=${bridge_args[argument_index]}
            case "$argument" in
                --device)
                    argument_index=$((argument_index + 1))
                    device_name=${bridge_args[argument_index]:-}
                    [[ -n "$device_name" && "$device_name" != --* ]] || fail "Missing value for bridge --device"
                    argument="--device=$device_name"
                    bridge_device_selected=true ;;
                --device=*)
                    [[ -n "${argument#*=}" ]] || fail "Missing value for bridge --device"
                    bridge_device_selected=true ;;
            esac
            normalized_bridge_args+=("$argument")
        done
        bridge_args=("${normalized_bridge_args[@]}")
        if [[ -n "$legacy_device" ]]; then preset_args+=("--device=$legacy_device");
        elif ! "$bridge_device_selected"; then
            for device_path in /sys/class/infiniband/*; do
                [[ -e "$device_path" ]] || continue
                preset_args+=("--device=${device_path##*/}"); break
            done
        fi
        preset_args+=("--peer-ip=${legacy_fpga_ip:-192.168.0.2}" --remote-qp=2)
        if [[ "${bridge##*/}" == libcudaq-realtime-bridge-cpu-roce.so ]]; then
            preset_args+=(--qp_config=hsb_fpga "--frame-size=$message_size" "--slot-size=$page_size" "--num-slots=$num_pages")
        else
            preset_args+=("--gpu=$gpu_id" "--threads=$threads" "--payload-size=$((message_size - 24))" "--page-size=$page_size" "--num-pages=$num_pages")
        fi
        [[ "$dispatch" == separate ]] || preset_args+=("--$dispatch") ;;
    libcudaq-realtime-bridge-udp.so)
        # Leave the UDP provider's existing port and ring defaults unchanged.
        "$num_pages_selected" && preset_args+=("--num-slots=$num_pages")
        "$page_size_selected" && preset_args+=("--slot-size=$page_size")
        [[ "$dispatch" != unified ]] || preset_args+=(--unified) ;;
    *)
        [[ -z "$legacy_device" ]] || preset_args+=("--device=$legacy_device")
        [[ -z "$legacy_fpga_ip" ]] || preset_args+=("--peer-ip=$legacy_fpga_ip") ;;
esac
bridge_args=("${preset_args[@]}" "${bridge_args[@]}")
echo "RTH bridge: $bridge"
echo "Playback provider: $provider"
echo "Dispatch: $dispatch"
for candidate in "$script_dir/validation_app" "$script_dir/../bin/validation_app" "$script_dir/../build/validation/validation_app"; do
    if [[ -z "$validation_app" && -x "$candidate" ]]; then validation_app=$candidate; fi
done
for candidate in "$script_dir/playback_app" "$script_dir/../bin/playback_app" "$script_dir/../build/validation/playback_app"; do
    if [[ -z "$playback_app" && -x "$candidate" ]]; then playback_app=$candidate; fi
done
[[ -x "$validation_app" && -x "$playback_app" ]] || fail "Build realtime first, or supply both executable paths"
command -v timeout >/dev/null || fail "GNU timeout is required"
if [[ -z "$output_dir" ]]; then
    output_dir=$(mktemp -d "${TMPDIR:-/tmp}/cudaq-validation.XXXXXXXX")
else
    mkdir -- "$output_dir" || fail "Use a new output directory to preserve earlier results"
fi
output_dir=$(cd -- "$output_dir" && pwd)
echo "Logs: $output_dir"
# Each app has internal deadlines; external watchdogs also bound blocking calls
# in a faulty plugin. Budget separately for create, the start gate, run, and destroy.
phase_seconds=$(((timeout_ns + 999999999) / 1000000000))
playback_seconds=$((4 * phase_seconds + 5))
rth_seconds=$((5 * phase_seconds + 15))
# These PIDs identify timeout supervisors, which forward signals to the apps
# and enforce a final kill deadline even if a plugin blocks during shutdown.
rth_pid=""
playback_pid=""
cleanup() {
    local status=$? child log_file
    # Keep the original failure/signal status even if stopping another child
    # also fails; disabling traps prevents recursive cleanup during exit.
    trap - EXIT INT TERM
    for child in "$playback_pid" "$rth_pid"; do
        if [[ -n "$child" ]] && kill -0 "$child" 2>/dev/null; then
            kill -TERM "$child" 2>/dev/null || true
            # timeout forwards TERM to the app and escalates to KILL after 5s.
            wait "$child" || true
        fi
    done
    # Print failure logs after both apps stop so shutdown diagnostics are
    # included. Keep logging in one place and preserve the original exit code.
    if (( status != 0 )); then
        for log_file in "$output_dir/rth.log" "$output_dir/playback.log"; do
            if [[ -f "$log_file" ]]; then
                { printf '\n=== %s ===\n' "${log_file##*/}"; cat "$log_file"; } || true
            fi
        done
    fi
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
# Publish the RTH endpoint first. Provider create may be needed to unblock the
# RTH connection handshake, so do not wait for full RTH readiness yet.
rth_command=("$validation_app" "--$execution" "--dispatch=$dispatch" "--gpu-id=$gpu_id" "--threads=$threads" "--timeout=$rth_seconds" "--bridge=$bridge" -- "${bridge_args[@]}")
if [[ "$execution" == gpu ]]; then
    # The persistent dispatcher starts before the bridge's RX/TX kernels.
    # Lazy loading of those kernels may require CUDA context synchronization,
    # which can block waiting for the persistent dispatcher to finish.
    # Force eager loading before CUDA initializes; preloading the dispatcher
    # alone does not cover bridge kernels. Apply this only to the GPU RTH process.
    rth_command=(env CUDA_MODULE_LOADING=EAGER "${rth_command[@]}")
fi
printf '%q ' "${rth_command[@]}" > "$output_dir/commands.sh"
printf '\n' >> "$output_dir/commands.sh"
timeout --signal=TERM --kill-after=5s "${rth_seconds}s" "${rth_command[@]}" > "$output_dir/rth.log" 2>&1 &
rth_pid=$!
ready_deadline=$((SECONDS + phase_seconds + 1))
until grep -q '^CUDAQ_VALIDATION_ENDPOINT ' "$output_dir/rth.log"; do
    if ! kill -0 "$rth_pid" 2>/dev/null; then
        status=0; wait "$rth_pid" || status=$?
        # Exiting successfully without an endpoint is still a startup failure;
        # otherwise preserve useful app codes such as unsupported-path (4).
        (( status != 0 )) || status=1
        exit "$status"
    fi
    if (( SECONDS >= ready_deadline )); then
        fail "RTH endpoint deadline expired"
    fi
    sleep 0.05
done
# A bridge may publish no endpoint. Pass the text unchanged if it does.
endpoint=$(sed -n 's/^CUDAQ_VALIDATION_ENDPOINT //p' "$output_dir/rth.log")
grep '^CUDAQ_VALIDATION_CASE ' "$output_dir/rth.log" || true
playback_command=("$playback_app" "--playback-provider=$provider" "--start-file=$output_dir/start" "--num-messages=$samples" "--message-size=$message_size" "--spacing-ns=$spacing_ns" "--quiet-ns=$quiet_ns" "--timeout-ns=$timeout_ns")
[[ -z "$endpoint" ]] || playback_command+=("--endpoint=$endpoint")
[[ "$dispatch" != forward ]] || playback_command+=(--forward)
playback_command+=(-- "${provider_args[@]}")
printf '%q ' "${playback_command[@]}" >> "$output_dir/commands.sh"
printf '\n' >> "$output_dir/commands.sh"
printf '%s\n' "$rate_log" > "$output_dir/playback.log"
timeout --signal=TERM --kill-after=5s "${playback_seconds}s" "${playback_command[@]}" >> "$output_dir/playback.log" 2>&1 &
playback_pid=$!
# Producer create and RTH connect may perform a rendezvous. Wait for both, then
# release start; no test request can be sent during their preparation.
ready_deadline=$((SECONDS + phase_seconds + 1))
until grep -q '^=== App Ready ===$' "$output_dir/rth.log" &&
      grep -q '^CUDAQ_PLAYBACK_READY$' "$output_dir/playback.log"; do
    for child in "$rth_pid" "$playback_pid"; do
        if ! kill -0 "$child" 2>/dev/null; then
            status=0; wait "$child" || status=$?
            (( status != 0 )) || status=1
            exit "$status"
        fi
    done
    if (( SECONDS >= ready_deadline )); then
        fail "RTH/playback readiness deadline expired"
    fi
    sleep 0.05
done
# Only file existence matters. Creating it releases the already-prepared
# producer; no endpoint or control command is encoded into its contents.
: > "$output_dir/start"
status=0
# Detect an RTH crash during playback rather than waiting for capture to expire.
while kill -0 "$playback_pid" 2>/dev/null; do
    if ! kill -0 "$rth_pid" 2>/dev/null; then
        status=0; wait "$rth_pid" || status=$?
        rth_pid=""
        echo "ERROR: RTH exited during playback (exit code $status)" >&2
        # Preserve the app/signal status. A clean exit before playback finishes
        # is still a validation failure; cleanup prints both process logs.
        (( status != 0 )) || status=1
        exit "$status"
    fi
    sleep 0.05
done
wait "$playback_pid" || status=$?
playback_pid=""
(( status == 0 )) || exit "$status"
cat "$output_dir/playback.log"
# Playback has verified every requested reply. Try normal RTH shutdown, then
# let timeout bound cleanup. A shutdown error cannot invalidate those samples.
kill -INT "$rth_pid" 2>/dev/null || true
status=0
wait "$rth_pid" || status=$?
rth_pid=""
cat "$output_dir/rth.log"
if (( status != 0 )); then
    echo "WARNING: RTH shutdown returned $status after playback validation succeeded; validation remains successful"
fi
echo "VALIDATION PASS execution=$execution dispatch=$dispatch samples=$samples"
