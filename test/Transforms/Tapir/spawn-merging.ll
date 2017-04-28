; ModuleID = 'spawn-merging.cpp'
source_filename = "spawn-merging.cpp"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind uwtable
define void @_Z4funcv() #0 {
entry:
  detach label %det.achd, label %det.cont

det.achd:                                         ; preds = %entry
  %0 = bitcast i32 undef to i32
  reattach label %det.cont

det.cont:                                         ; preds = %det.achd, %entry
  detach label %det.achd1, label %det.cont2

det.achd1:                                        ; preds = %det.cont
  %1 = bitcast i32 undef to i32
  reattach label %det.cont2

det.cont2:                                        ; preds = %det.achd1, %det.cont
  ret void
}

attributes #0 = { noinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.ident = !{!0}

!0 = !{!"clang version 4.0.0 (git@github.com:wsmoses/Cilk-Clang cc78c4b6082bb80687e64c8104bf9744e6fa8fdc) (git@github.com:wsmoses/Parallel-IR.git d5331895cb2d1437b7788469ac72c731b65a949b)"}
