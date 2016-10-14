; REQUIRES: asserts
; RUN: llc < %s -mtriple=arm64-linux-gnu -mcpu=cortex-a53 -pre-RA-sched=source -enable-misched -verify-misched -debug-only=misched -o - 2>&1 > /dev/null | FileCheck %s
;
; For Cortex-A53, shiftable operands that are not actually shifted
; are not needed for an additional two cycles.
;
; CHECK: ********** MI Scheduling **********
; CHECK: shiftable
; CHECK: SU(2):   %vreg2<def> = SUBXri %vreg1, 20, 0
; CHECK:   Successors:
; CHECK-NEXT:    data SU(4): Latency=1 Reg=%vreg2
; CHECK-NEXT:    data SU(3): Latency=2 Reg=%vreg2
; CHECK: ********** INTERVALS **********
define i64 @shiftable(i64 %A, i64 %B) {
        %tmp0 = sub i64 %B, 20
        %tmp1 = shl i64 %tmp0, 5;
        %tmp2 = add i64 %A, %tmp1;
        %tmp3 = add i64 %A, %tmp0
        %tmp4 = mul i64 %tmp2, %tmp3

        ret i64 %tmp4
}
