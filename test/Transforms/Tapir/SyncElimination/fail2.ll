; RUN: opt < %s -sync-elimination -S | FileCheck %s

; ModuleID = 'fail2.cpp'
source_filename = "fail2.cpp"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind uwtable
define void @_Z4funcPiS_(i32* %a, i32* %b) #0 {
entry:
  %a.addr = alloca i32*, align 8
  %b.addr = alloca i32*, align 8
  store i32* %a, i32** %a.addr, align 8
  store i32* %b, i32** %b.addr, align 8
  detach label %det.achd, label %det.cont

det.achd:                                         ; preds = %entry
  %0 = bitcast i32 undef to i32
  %1 = load i32*, i32** %a.addr, align 8
  store i32 1, i32* %1, align 4
  reattach label %det.cont

det.cont:                                         ; preds = %det.achd, %entry
  sync label %sync.continue
; CHECK: sync label %sync.continue

sync.continue:                                    ; preds = %det.cont
  %2 = load i32*, i32** %b.addr, align 8
  store i32 2, i32* %2, align 4
  sync label %sync.continue1
; CHECK-NOT: sync label %sync.continue1

sync.continue1:                                   ; preds = %sync.continue
  ret void
; CHECK: ret void
}

attributes #0 = { noinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.ident = !{!0}

!0 = !{!"clang version 4.0.0 (git@github.com:wsmoses/Cilk-Clang cc78c4b6082bb80687e64c8104bf9744e6fa8fdc) (git@github.com:isundaylee/Parallel-IR.git 438f1fb17778bfbb5678b08082e677c288d81c89)"}
