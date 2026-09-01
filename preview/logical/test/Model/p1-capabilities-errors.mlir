// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt --split-input-file --verify-diagnostics %s

module {
  lvm.domain @machine {
    lvm.space @left {
      capabilities = [#lvm.capability<"qlx.machine/logical_compute">]
    }
    lvm.space @right {
      capabilities = [#lvm.capability<"qlx.machine/logical_compute">]
    }
    // expected-error@+1 {{capabilities entries must be #lvm.capability attributes}}
    lvm.channel @link {capabilities = ["qlx.machine/resource_transfer"], from = @left, to = @right}
  }
}

// -----

module {
  lvm.domain @machine {
    // expected-error@+1 {{reserved QLX machine capabilities must use the qlx.machine namespace}}
    lvm.space @compute {
      capabilities = [#lvm.capability<"qlx.invalid/logical_compute">]
    }
  }
}

// -----

module {
  lvm.domain @machine {
    lvm.space @left {
      capabilities = [#lvm.capability<"qlx.machine/logical_compute">]
    }
    lvm.space @right {
      capabilities = [#lvm.capability<"qlx.machine/logical_compute">]
    }
    // expected-error@+1 {{contains duplicate capability 'qlx.machine/resource_transfer'}}
    lvm.channel @link {
      capabilities = [
        #lvm.capability<"qlx.machine/resource_transfer">,
        #lvm.capability<"qlx.machine/resource_transfer">
      ],
      from = @left,
      to = @right
    }
  }
}
