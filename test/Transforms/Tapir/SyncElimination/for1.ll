; RUN: opt < %s -sync-elimination -S | FileCheck %s

; ModuleID = 'for1.cpp'
source_filename = "for1.cpp"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind uwtable
define void @_Z4funcv() #0 {
entry:
  %i = alloca i32, align 4
  %__end = alloca i32, align 4
  %i1 = alloca i32, align 4
  %__end2 = alloca i32, align 4
  store i32 0, i32* %i, align 4
  store i32 10, i32* %__end, align 4
  br label %pfor.cond

pfor.cond:                                        ; preds = %pfor.inc, %entry
  %0 = load i32, i32* %i, align 4
  %1 = load i32, i32* %__end, align 4
  %cmp = icmp slt i32 %0, %1
  br i1 %cmp, label %pfor.detach, label %pfor.end

pfor.detach:                                      ; preds = %pfor.cond
  detach label %pfor.body, label %pfor.inc

pfor.body:                                        ; preds = %pfor.detach
  %2 = bitcast i32 undef to i32
  br label %pfor.preattach

pfor.preattach:                                   ; preds = %pfor.body
  reattach label %pfor.inc

pfor.inc:                                         ; preds = %pfor.preattach, %pfor.detach
  %3 = load i32, i32* %i, align 4
  %inc = add nsw i32 %3, 1
  store i32 %inc, i32* %i, align 4
  br label %pfor.cond, !llvm.loop !1

pfor.end:                                         ; preds = %pfor.cond
  sync label %pfor.end.continue

pfor.end.continue:                                ; preds = %pfor.end
  store i32 0, i32* %i1, align 4
  store i32 10, i32* %__end2, align 4
  br label %pfor.cond3

; CHECK: pfor.end
; CHECK-NOT: sync
; CHECK: pfor.cond

pfor.cond3:                                       ; preds = %pfor.inc8, %pfor.end.continue
  %4 = load i32, i32* %i1, align 4
  %5 = load i32, i32* %__end2, align 4
  %cmp4 = icmp slt i32 %4, %5
  br i1 %cmp4, label %pfor.detach5, label %pfor.end10

pfor.detach5:                                     ; preds = %pfor.cond3
  detach label %pfor.body6, label %pfor.inc8

pfor.body6:                                       ; preds = %pfor.detach5
  %6 = bitcast i32 undef to i32
  br label %pfor.preattach7

pfor.preattach7:                                  ; preds = %pfor.body6
  reattach label %pfor.inc8

pfor.inc8:                                        ; preds = %pfor.preattach7, %pfor.detach5
  %7 = load i32, i32* %i1, align 4
  %inc9 = add nsw i32 %7, 1
  store i32 %inc9, i32* %i1, align 4
  br label %pfor.cond3, !llvm.loop !3

pfor.end10:                                       ; preds = %pfor.cond3
  sync label %pfor.end.continue11

pfor.end.continue11:                              ; preds = %pfor.end10
  ret void
}

attributes #0 = { noinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.ident = !{!0}

!0 = !{!"clang version 4.0.0 (git@github.com:wsmoses/Cilk-Clang cc78c4b6082bb80687e64c8104bf9744e6fa8fdc) (git@github.com:isundaylee/Parallel-IR.git deb1da93d0b64458c5890d1a05a941d18a7c9e49)"}
!1 = distinct !{!1, !2}
!2 = !{!"tapir.loop.spawn.strategy", i32 1}
!3 = distinct !{!3, !2}
