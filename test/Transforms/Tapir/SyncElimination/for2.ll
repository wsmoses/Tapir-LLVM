; RUN: opt < %s -sync-elimination -S | FileCheck %s

; ModuleID = 'for2.cpp'
source_filename = "for2.cpp"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind uwtable
define void @_Z4funcv() #0 {
entry:
  %i = alloca i32, align 4
  %__end = alloca i32, align 4
  store i32 0, i32* %i, align 4
  store i32 100, i32* %__end, align 4
  br label %pfor.cond

pfor.cond:                                        ; preds = %pfor.inc7, %entry
  %0 = load i32, i32* %i, align 4
  %1 = load i32, i32* %__end, align 4
  %cmp = icmp slt i32 %0, %1
  br i1 %cmp, label %pfor.detach, label %pfor.end9

pfor.detach:                                      ; preds = %pfor.cond
  detach label %pfor.body, label %pfor.inc7

pfor.body:                                        ; preds = %pfor.detach
  %j = alloca i32, align 4
  %__end1 = alloca i32, align 4
  %2 = bitcast i32 undef to i32
  store i32 0, i32* %j, align 4
  store i32 3, i32* %__end1, align 4
  br label %pfor.cond2

pfor.cond2:                                       ; preds = %pfor.inc, %pfor.body
  %3 = load i32, i32* %j, align 4
  %4 = load i32, i32* %__end1, align 4
  %cmp3 = icmp slt i32 %3, %4
  br i1 %cmp3, label %pfor.detach4, label %pfor.end

pfor.detach4:                                     ; preds = %pfor.cond2
  detach label %pfor.body5, label %pfor.inc

pfor.body5:                                       ; preds = %pfor.detach4
  %5 = bitcast i32 undef to i32
  br label %pfor.preattach

pfor.preattach:                                   ; preds = %pfor.body5
  reattach label %pfor.inc

pfor.inc:                                         ; preds = %pfor.preattach, %pfor.detach4
  %6 = load i32, i32* %j, align 4
  %inc = add nsw i32 %6, 1
  store i32 %inc, i32* %j, align 4
  br label %pfor.cond2, !llvm.loop !1

pfor.end:                                         ; preds = %pfor.cond2
  sync label %pfor.end.continue

; CHECK: pfor.body5
; CHECK-NOT: sync label %pfor.end.continue
; CHECK: pfor.inc7

pfor.end.continue:                                ; preds = %pfor.end
  br label %pfor.preattach6

pfor.preattach6:                                  ; preds = %pfor.end.continue
  reattach label %pfor.inc7

pfor.inc7:                                        ; preds = %pfor.preattach6, %pfor.detach
  %7 = load i32, i32* %i, align 4
  %inc8 = add nsw i32 %7, 1
  store i32 %inc8, i32* %i, align 4
  br label %pfor.cond, !llvm.loop !3

pfor.end9:                                        ; preds = %pfor.cond
  sync label %pfor.end.continue10

pfor.end.continue10:                              ; preds = %pfor.end9
  ret void
}

attributes #0 = { noinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.ident = !{!0}

!0 = !{!"clang version 4.0.0 (git@github.com:wsmoses/Cilk-Clang cc78c4b6082bb80687e64c8104bf9744e6fa8fdc) (git@github.com:isundaylee/Parallel-IR.git 7fe3eb9fe02405d835a6ba095ae507e93c686b2d)"}
!1 = distinct !{!1, !2}
!2 = !{!"tapir.loop.spawn.strategy", i32 1}
!3 = distinct !{!3, !2}
