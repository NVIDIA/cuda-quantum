/********************************************************** -*- tablegen -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#ifndef CUDAQ_DIALECT_CC_TYPES
#define CUDAQ_DIALECT_CC_TYPES

include "cudaq/Optimizer/Dialect/CC/CCDialect.td"
include "mlir/Interfaces/DataLayoutInterfaces.td"
include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// BaseType
//===----------------------------------------------------------------------===//

class CCType<string name, string typeMnemonic, list<Trait> traits = [],
             string base = "mlir::Type">
    : TypeDef<CCDialect, name, traits, base> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// PointerType
//===----------------------------------------------------------------------===//

def cc_PointerType : CCType<"Pointer", "ptr"> {
  let summary = "A C++ pointer/reference type.";
  let description = [{
    The bridge can use this type to represent values that are addresses in
    memory. That includes declarations such as `T*` (pointer to T), `T&`
    (lvalue reference to T), and `T&&` (rvalue reference to T).

    This is a more permissive type than the LLVMPointerType, which has
    restrictions on its element type that makes it harder to use at time.
    Ultimately, !cc.ptr types will become LLVM IR pointers.
  }];

  let parameters = (ins "mlir::Type":$elementType);
  let assemblyFormat = "`<` qualified($elementType) `>`";

  let builders = [
    TypeBuilderWithInferredContext<(ins "mlir::Type":$elementType), [{
      return Base::get(elementType.getContext(), elementType);
    }]>,
  ];
}

//===----------------------------------------------------------------------===//
// StructType
//===----------------------------------------------------------------------===//

def cc_StructType : CCType<"Struct", "struct",
    [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>]> {

  let summary = "Primitive aggregate C/C++ type.";
  let description = [{
    A struct is first-class type for an aggregate tuple value composed of
    multiple ordered subvalues of any type. A value of type struct need not be
    a memory object like an array.

    A subobject is indexed by position from `0` to `N-1`, where `N` is the
    number of members in the struct.
  }];

  let parameters = (ins
    "mlir::StringAttr":$name,
    ArrayRefParameter<"mlir::Type">:$members,
    "bool":$opaque,
    "bool":$packed
  );

  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// ArrayType
//===----------------------------------------------------------------------===//

def cc_ArrayType : CCType<"Array", "array"> {
  let summary = "Primitive C-like, one-dimensional sequential type.";
  let description = [{
    A value in memory of type `!cc.array` is similar to a `!cc.pointer` but
    allows 0-based indexing into a sequence of values. A `!cc.pointer` by
    contrast points to a single object.

    A value of type `!cc.array` should ordinarily be an object in memory. This
    is transitive in nature, so any aggregate that recursively has a member of
    `!cc.array` should itself reside in memory. More precisely, loading an array
    into a register is not portable.

    See also the StdVectorType type as another aggregate sequence of values.
  }];

  let parameters = (ins
    "mlir::Type":$elementType,
    "std::size_t":$size
  );

  let assemblyFormat = [{
    `<` qualified($elementType) `x` $size `>`
  }];
}


//===----------------------------------------------------------------------===//
// LambdaType
//===----------------------------------------------------------------------===//

def cc_LambdaType : CCType<"Lambda", "lambda"> {
  let summary = "Proxy for a C++ lambda instance in cc";
  let description = [{
    A kernel may be a template function that takes a lambda as an argument.
    The type of these lambda instances is this proxy lambda type. The
    instantiated template instance will resolve the lambda instance, and this
    resolved instance of the lambda will be lowered in the specialized template
    function.
  }];
  
  let parameters = (ins "mlir::FunctionType":$signature);

  let assemblyFormat = "`<` $signature `>`";

  let builders = [
    TypeBuilderWithInferredContext<(ins "mlir::FunctionType":$signature), [{
      return Base::get(signature.getContext(), signature);
    }]>,
  ];

  let extraClassDeclaration = [{
    /// Create a lambda type without a signature. This is for prototyping only.
    [[deprecated]] static LambdaType getNoSignature(mlir::MLIRContext *ctx);
  }];
}

//===----------------------------------------------------------------------===//
// StdVectorType
//===----------------------------------------------------------------------===//

def cc_StdVectorType : CCType<"Stdvec", "stdvec"> {
  let summary = "Proxy for a std::vector<T> in cc";
  let description = [{
    This captures a C++ std::vector<T> template type as an abstract CC
    type. The intention is to allow us to reason about std::vectors at the
    CC level before decomposing these into more refined data structures.

    ```c++
    void foo(std::vector<int> vec) ...
    ```
    will be lowered from the AST to CC as
    ```mlir
    func.func @foo(%arg0 : !cc.stdvec<i32>) ...
    ```
    assuming `int` has a length of 32 bits for the target.
  }];

  let parameters = (ins "mlir::Type":$elementType);

  let assemblyFormat = "`<` qualified($elementType) `>`";

  let builders = [
    TypeBuilderWithInferredContext<(ins "mlir::Type":$elementType), [{
      return Base::get(elementType.getContext(), elementType);
    }]>,
  ];
}

#endif // CUDAQ_DIALECT_CC_TYPES_TD
