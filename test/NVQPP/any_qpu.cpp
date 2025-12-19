/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.
 *
 * This source code and the accompanying materials are made available under
 * the terms of the Apache License 2.0 which accompanies this distribution.
 ******************************************************************************/

// RUN: rm -rf %t && mkdir -p %t
// RUN: python3 %S/../split-file.py %s %t
// RUN: cd %t
// RUN: nvq++ -shared -fPIC qpu1.cpp -o libqpu1.so
// RUN: nvq++ -shared -fPIC qpu2.cpp -o libqpu2.so
// RUN: nvq++ -shared -fPIC old_qpu.cpp -o libold_qpu.so
// RUN: nvq++ proto.cpp libqpu1.so libqpu2.so libold_qpu.so -o proto
// RUN: nvq++ proto.cpp libqpu1.so libold_qpu.so -o proto_missing
// RUN: ./proto | FileCheck %s --check-prefix=FULL
// RUN: ./proto_missing | FileCheck %s --check-prefix=MISSING

// FULL: Constructed old_qpu
// FULL: Constructed qpu1
// FULL: Constructed qpu2
// FULL: I am old old_qpu
// FULL: I am qpu1
// FULL: I am qpu2
// FULL: Constructed qpu1
// FULL: Deleted old_qpu
// FULL: Deleted qpu1
// FULL: Deleted qpu1
// FULL: Deleted qpu2

// MISSING: Constructed old_qpu
// MISSING: Constructed qpu1
// MISSING: Could not find qpu2 in registry
// MISSING: I am old old_qpu
// MISSING: I am qpu1
// MISSING: Constructed qpu1
// MISSING: Deleted old_qpu
// MISSING: Deleted qpu1
// MISSING: Deleted qpu1

//--- any_qpu.h
#pragma once

#include "cudaq/version2/cudaq/utils/type_erased_registry.h"
#include <functional>
#include <iostream>
#include <memory>
#include <string>

struct qpu_dispatch_table {
  std::function<std::shared_ptr<void>()> create;
  std::function<std::string(void *)> name;
  std::function<void(void *)> print;

  template <typename T> void build() {
    create = []() { return std::make_shared<T>(); };
    name = [](void *i) { return static_cast<T *>(i)->name(); };
    print = [](void *i) {
      static_cast<T *>(i)->print();
      return true;
    };
  }
};

class any_qpu {
  std::shared_ptr<void> instance;
  const qpu_dispatch_table *vtable;

public:
  any_qpu(std::shared_ptr<void> inst, const qpu_dispatch_table *vt)
      : instance(std::move(inst)), vtable(vt) {}

  std::string name() { return vtable->name(instance.get()); }
  void print() { vtable->print(instance.get()); }
};

using QPURegistry =
    cudaq::registry::TypeErasedRegistry<any_qpu, qpu_dispatch_table>;

#define CUDAQ_REGISTER_QPU_TYPE(TYPE)                                            \
  static cudaq::registry::TypeErasedRegistrar<QPURegistry, TYPE>                \
      registrar_##TYPE(#TYPE);

//--- new_base.h
#pragma once

#include <iostream>

template <typename Derived> class new_base {
public:
  new_base() {
    std::cout << "Constructed " << static_cast<const Derived *>(this)->name()
              << std::endl;
  }
  void print() const {
    std::cout << "I am " << static_cast<const Derived *>(this)->name()
              << std::endl;
  }
  ~new_base() {
    std::cout << "Deleted " << static_cast<const Derived *>(this)->name()
              << std::endl;
  }
};

//--- old_base.h
#pragma once

#include <iostream>
#include <string>

class old_base {
public:
  virtual std::string name() const = 0;
  void print() const {
    std::cout << "I am old " << name() << std::endl;
  }
  virtual ~old_base() = default;
};

//--- qpu1.cpp
#include "any_qpu.h"
#include "new_base.h"

class qpu1 : public new_base<qpu1> {
public:
  std::string name() const { return "qpu1"; }
};

CUDAQ_REGISTER_QPU_TYPE(qpu1)

//--- qpu2.cpp
#include "any_qpu.h"
#include "new_base.h"

class qpu2 : public new_base<qpu2> {
public:
  std::string name() const { return "qpu2"; }
};

CUDAQ_REGISTER_QPU_TYPE(qpu2)

//--- old_qpu.cpp
#include "any_qpu.h"
#include "old_base.h"

class old_qpu : public old_base {
public:
  std::string name() const override { return "old_qpu"; }
  old_qpu() { std::cout << "Constructed old_qpu" << std::endl; }
  ~old_qpu() { std::cout << "Deleted old_qpu" << std::endl; }
};

CUDAQ_REGISTER_QPU_TYPE(old_qpu)

//--- proto.cpp
#include "any_qpu.h"
#include <iostream>
#include <memory>
#include <string>
#include <vector>

CUDAQ_INSTANTIATE_TYPE_ERASED_REGISTRY(QPURegistry)

static void add_qpu_to_vector(std::vector<std::unique_ptr<any_qpu>> &vec,
                              const std::string &name) {
  auto qpu = QPURegistry::get().instantiate(name);
  if (qpu)
    vec.push_back(std::move(qpu));
  else
    std::cout << "Could not find " << name << " in registry" << std::endl;
}

int main() {
  std::vector<std::unique_ptr<any_qpu>> qpus;
  add_qpu_to_vector(qpus, "old_qpu");
  add_qpu_to_vector(qpus, "qpu1");
  add_qpu_to_vector(qpus, "qpu2");

  for (const auto &qpu : qpus)
    qpu->print();

  qpus[0] = QPURegistry::get().instantiate("qpu1");
  return 0;
}

