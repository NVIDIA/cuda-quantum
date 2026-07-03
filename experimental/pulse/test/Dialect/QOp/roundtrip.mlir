// RUN: cudaq-pulse-opt %s | cudaq-pulse-opt | FileCheck %s

// CHECK-LABEL: func @test_qop_basic
func.func @test_qop_basic() {
  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64

  // CHECK: qop.spin
  %sx = qop.spin(%c0) {kind = #qop<handler_kind spin_x>} : !qop.handler
  %sz = qop.spin(%c1) {kind = #qop<handler_kind spin_z>} : !qop.handler

  // CHECK: qop.const_scalar
  %coeff = qop.const_scalar {real = 0.3 : f64, imag = 0.0 : f64} : !qop.scalar

  // CHECK: qop.make_product
  %term = qop.make_product(%coeff, %sx) : !qop.product

  // CHECK: qop.make_sum
  %H = qop.make_sum(%term) : !qop.op

  // CHECK: qop.dagger
  %Hd = qop.dagger %H : !qop.op
  return
}
