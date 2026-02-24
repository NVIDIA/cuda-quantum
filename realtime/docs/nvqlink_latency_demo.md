# Steps to execute the NVQLink latency demo

The source Verilog code can be found [here](https://edge.urm.nvidia.com/artifactory/sw-holoscan-thirdparty-generic-local/QEC/).

More details about how the `Holoscan Sensor Bridge` (`HSB`) IP can be incorporated
can be found [here](https://docs.nvidia.com/holoscan/sensor-bridge/latest/fpga_index.html)

Furthermore, for this experiment, we need the Integrated Logic Analyzer (`ILA`)
to keep the captured measurements. See the "Hololink IP:
Connecting an `APB` `ILA` for Debug" section below.

## Steps to do the experiment

1. Load the bit-file into the FPGA.
2. Setup the host to run the experiment.
Mainly the IP address of the NIC needs to be set to `192.168.0.101`.
More details can be found at the
*Data Channel Enumeration and IP Address Configuration* section of [this document](https://docs.nvidia.com/holoscan/sensor-bridge/latest/architecture.html)
3. Download the accompanying software from [GitHub](https://github.com/nvidia-holoscan/holoscan-sensor-bridge/tree/nvqlink)

   Then generate the docker:

   ```sh
   sudo sh ./docker/build.sh --dgpu
   sudo sh ./docker/demo.sh
   ```

To run the test, here is an example for 32B messages reported in the paper:

```sh
python3 ./examples/gpunetio_loopback.py --frame-size=32 --hololink=192.168.0.2 --rx-ibv-name=mlx5_0 --tx-ibv-name=mlx5_0 --mtu=256
```

Then to capture the data from the experiment and run the latency calculation:

```sh
python3 ila.py
python3 latency_analysis.py
```

(These two python scripts can be found next to the Verilog source code).

## Hololink IP: Connecting an `APB` `ILA` for Debug

This guide describes how to attach an Integrated Logic Analyzer (`ILA`)
to one of the Hololink IP's `APB` register interfaces for real-time signal capture
and debugging over Ethernet.

### Overview

The Hololink IP exposes multiple `APB` register interfaces via the `REG_INST`
parameter (defined in `HOLOLINK_def.svh`).
These interfaces can be used to connect custom user logic, including `ILA`'s,
for monitoring internal signals.

In this example, we connect the `s_apb_ila` module to **`APB[2]`**
and configure it to capture `PTP` timestamps, frame information,
and other debug signals.

### `APB` Interface Signals from Hololink

The Hololink IP provides the following `APB` signals for user register interfaces:

```systemverilog
// From HOLOLINK_top outputs
logic [`REG_INST-1:0] apb_psel;      // Per-interface select
logic                 apb_penable;   // Common enable
logic [31:0]          apb_paddr;     // Common address bus
logic [31:0]          apb_pwdata;    // Common write data
logic                 apb_pwrite;    // Common write enable

// To HOLOLINK_top inputs
logic [`REG_INST-1:0] apb_pready;    // Per-interface ready
logic [31:0]          apb_prdata [`REG_INST-1:0];  // Per-interface read data
logic [`REG_INST-1:0] apb_pserr;     // Per-interface error
```

### Step 1: Tie Off Unused `APB` Interfaces

For any `APB` interfaces not in use, tie off the signals appropriately:

```systemverilog
// Tie off unused APB bus signals
assign apb_pserr[7:3]  = '0;
assign apb_pserr[1:0]  = '0;
assign apb_pready[7:3] = '1;
assign apb_pready[1:0] = '0;
```

> **Note:** `APB[2]` is left unassigned here since it will be connected to the `ILA`.

---

### Step 2: Create `APB` Interface Structs for the `ILA`

The `s_apb_ila` module uses the `apb_m2s` and `apb_s2m` struct types from `apb_pkg`.
Declare the interface signals:

```systemverilog
import apb_pkg::*;

apb_m2s ila_apb_m2s;
apb_s2m ila_apb_s2m;
```

---

### Step 3: Instantiate the `s_apb_ila` Module

The `s_apb_ila` module is part of the Hololink IP library (`lib_apb/s_apb_ila.sv`).

```systemverilog
localparam ILA_DATA_WIDTH = 256;

s_apb_ila #(
  .DEPTH            ( 65536                          ),
  .W_DATA           ( ILA_DATA_WIDTH                 )
) u_apb_ila (
  // APB Interface (slow clock domain)
  .i_aclk           ( apb_clk                        ),
  .i_arst           ( apb_rst                        ),
  .i_apb_m2s        ( ila_apb_m2s                    ),
  .o_apb_s2m        ( ila_apb_s2m                    ),
  
  // User Capture Interface (fast clock domain)
  .i_pclk           ( hif_clk                        ),
  .i_prst           ( hif_rst                        ),
  .i_trigger        ( '1                             ),  // Always triggered
  .i_enable         ( '1                             ),  // Always enabled
  .i_wr_data        ( ila_wr_data                    ),  // Data to capture
  .i_wr_en          ( ptp_ts_en                      ),  // Write enable
  .o_ctrl_reg       (                                )   // Optional control output
);
```

---

### Step 4: Connect `APB[2]` to the `ILA`

Map the Hololink `APB` signals to the `ILA`'s struct interface:

```systemverilog
// APB Master-to-Slave signals (from Hololink to ILA)
assign ila_apb_m2s.psel    = apb_psel[2];     // Select APB interface 2
assign ila_apb_m2s.penable = apb_penable;
assign ila_apb_m2s.paddr   = apb_paddr;
assign ila_apb_m2s.pwdata  = apb_pwdata;
assign ila_apb_m2s.pwrite  = apb_pwrite;

// APB Slave-to-Master signals (from ILA back to Hololink)
assign apb_pready[2] = ila_apb_s2m.pready;
assign apb_prdata[2] = ila_apb_s2m.prdata;
assign apb_pserr[2]  = ila_apb_s2m.pserr;
```

---

### Step 5: Define the Write Data Vector

Structure the `ila_wr_data` signal to capture the signals of interest.
Here's the example configuration used:

```systemverilog
localparam ILA_DATA_WIDTH = 256;
logic [ILA_DATA_WIDTH-1:0] ila_wr_data;

// Bit assignments
assign ila_wr_data[63:0]    = ptp_ts[63:0];                     // PTP timestamp from sensor frame
assign ila_wr_data[127:64]  = {ptp_sec_sync_usr[31:0],          // Synchronized PTP seconds
                               ptp_nsec_sync_usr[31:0]};        // Synchronized PTP nanoseconds
assign ila_wr_data[139:128] = frame_cnt;                        // 12-bit frame counter
assign ila_wr_data[140]     = sof;                              // Start of frame
assign ila_wr_data[141]     = eof;                              // End of frame
assign ila_wr_data[255:142] = 'h123456789ABCDEF;                // Debug pattern (filler)
```

#### Write Data Bit Map Summary

| Bits | Width | Signal | Description |
|------|-------|--------|-------------|
| [63:0] | 64 | `ptp_ts` | `PTP` timestamp extracted from sensor TX data |
| [127:64] | 64 | `{ptp_sec, ptp_nsec}` | Synchronized `PTP` time (seconds + nanoseconds) from Hololink |
| [139:128] | 12 | `frame_cnt` | Frame counter extracted from sensor TX data |
| [140] | 1 | `sof` | Start of frame indicator |
| [141] | 1 | `eof` | End of frame indicator |
| [255:142] | 114 | Debug pattern | Fixed pattern for debugging |

> **Note:** `ptp_sec_sync_usr` and `ptp_nsec_sync_usr` are the `PTP` time outputs
from Hololink (`o_ptp_sec`, `o_ptp_nanosec`) synchronized to
the host interface clock domain.

---

### Step 6: Supporting Logic

#### Frame Detection

```systemverilog
logic sof, eof;
assign sof = sif_tx_axis_tvalid[0];   // SOF on first valid
assign eof = sif_tx_axis_tlast[0];    // EOF on last
```

#### Timestamp Capture

```systemverilog
logic [79:0]  ptp_ts;
logic         ptp_ts_en;
logic [11:0]  frame_cnt;

always_ff @(posedge hif_clk) begin
  if (hif_rst) begin
    ptp_ts    <= '0;
    ptp_ts_en <= '0;
    frame_cnt <= '0;
  end
  else begin
    ptp_ts    <= (sof) ? sif_tx_axis_tdata[0][79:0] : ptp_ts;
    frame_cnt <= (sof) ? sif_tx_axis_tdata[0][91:80] : frame_cnt;
    ptp_ts_en <= sof;
  end
end
```

---

### Sensor RX Interface Tie-Off

In this configuration, only the **Sensor TX interface** is used
(for receiving data from the host).
The Sensor RX interface is not used and should be tied off as follows:

```systemverilog
// Sensor Rx Streaming Interface - Tie off (not used)
.i_sif_axis_tvalid ( '0           ),
.i_sif_axis_tlast  ( '0           ),
.i_sif_axis_tdata  ( '{default:0} ),
.i_sif_axis_tkeep  ( '{default:0} ),
.i_sif_axis_tuser  ( '{default:0} ),
.o_sif_axis_tready (              ),  // Leave unconnected
```

The Sensor TX interface (`o_sif_axis_*`) should have `i_sif_axis_tready`
tied high to always accept data:

```systemverilog
.i_sif_axis_tready ( '1 ),
```

---

Once integrated, the `ILA` data can be accessed via `APB` register
reads from the host over Ethernet using the Hololink control plane.
