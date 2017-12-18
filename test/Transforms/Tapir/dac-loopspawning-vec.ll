; Test that Tapir's loop spawning pass transforms this simple loop
; into recursive divide-and-conquer.
; RUN: opt < %s -loop-spawning -S -cilk-target=1 | FileCheck %s

; Function Attrs: nounwind uwtable
define void @brokenCompiler(i8* nocapture %Flags, i64 %n) local_unnamed_addr #0 {
entry:
; CHECK: %0 = call i32 @__cilkrts_get_nworkers()
  %syncreg = tail call token @llvm.syncregion.start()
  br label %vector.body

vector.body:                                      ; preds = %vec.inc, %entry
  %index = phi i64 [ 0, %entry ], [ %index.next, %vec.inc ]
  %index.next = add nuw nsw i64 %index, 32
  %eq = icmp eq i64 %index.next, %n
  detach within %syncreg, label %vec.detached, label %vec.inc

vec.detached:                                     ; preds = %vector.body
  %gep = getelementptr inbounds i8, i8* %Flags, i64 %index
  store i8 0, i8* %gep
  reattach within %syncreg, label %vec.inc

vec.inc:                                          ; preds = %vec.detached, %vector.body
  br i1 %eq, label %middle.block, label %vector.body, !llvm.loop !5

middle.block:                                     ; preds = %vec.inc
  br label %pfor.detach.preheader

pfor.detach.preheader:                            ; preds = %middle.block
  sync within %syncreg, label %pfor.end.continue

pfor.end.continue:                                ; preds = %pfor.detach.preheader
  ret void
}

; Function Attrs: argmemonly nounwind
declare token @llvm.syncregion.start() #1

attributes #0 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 5.0.0 (https://github.com/wsmoses/Cilk-Clang 9e81b3be8a7749cb8feea3f6bad30df9b7ba1e75) (git@github.com:wsmoses/Parallel-IR f48aa20dd791783172bb739aca51263e439c5ba3)"}
!2 = !{!3, !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
!5 = distinct !{!5, !6} 
!6 = !{!"tapir.loop.spawn.strategy", i32 1}
