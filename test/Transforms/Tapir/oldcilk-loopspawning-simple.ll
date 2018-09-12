; Test that Tapir's loop spawning pass transforms this simple loop
; into recursive divide-and-conquer.

; RUN: opt < %s -loop-spawning -S -ls-tapir-target=cilklegacy | FileCheck %s

; Function Attrs: nounwind uwtable
define void @foo(i32 %n) local_unnamed_addr #0 {
; CHECK-LABEL: @foo(
entry:
  %syncreg = call token @llvm.syncregion.start()
  %cmp5 = icmp sgt i32 %n, 0
  br i1 %cmp5, label %pfor.detach.preheader, label %pfor.cond.cleanup

pfor.detach.preheader:                            ; preds = %entry
; CHECK: pfor.detach.preheader:
; CHECK: [[LIMIT:%[0-9]+]] = add [[TYPE:i[0-9]+]] %n, -1
; CHECK: [[CLOSUREALLOC:%[0-9]+]] = alloca [[CLOSURETYPE:%[0-9]+]]
; CHECK-NEXT: br label %pfor.detach.preheader.split
; CHECK: [[CLOSURECAST:%[0-9]+]] = bitcast [[CLOSURETYPE]]* [[CLOSUREALLOC]] to i8*
; CHECK: call void @__cilkrts_cilk_for_32(void (i8*, i32, i32)* bitcast (void (%0*, i32, i32)* @[[OUTLINED:[a-zA-Z0-9._]+]] to void (i8*, i32, i32)*), i8* [[CLOSURECAST]], i32 %n, i32 [[GRAIN:%[0-9]+]])
; CHECK-NEXT: br label %pfor.cond.cleanup.loopexit
  br label %pfor.detach

pfor.cond.cleanup.loopexit:                       ; preds = %pfor.inc
  br label %pfor.cond.cleanup

pfor.cond.cleanup:                                ; preds = %pfor.cond.cleanup.loopexit, %entry
  sync within %syncreg, label %0

; <label>:0:                                      ; preds = %pfor.cond.cleanup
  ret void

pfor.detach:                                      ; preds = %pfor.detach.preheader, %pfor.inc
  %i.06 = phi i32 [ %inc, %pfor.inc ], [ 0, %pfor.detach.preheader ]
  detach within %syncreg, label %pfor.body, label %pfor.inc

pfor.body:                                        ; preds = %pfor.detach
  tail call void @bar(i32 %i.06) #2
  reattach within %syncreg, label %pfor.inc

pfor.inc:                                         ; preds = %pfor.body, %pfor.detach
  %inc = add nuw nsw i32 %i.06, 1
  %exitcond = icmp eq i32 %inc, %n
  br i1 %exitcond, label %pfor.cond.cleanup.loopexit, label %pfor.detach, !llvm.loop !1
}

; CHECK: define internal void @[[OUTLINED]](%0* %.ls, i32 %start.ls, i32 %.ls1) local_unnamed_addr
; CHECK-NEXT: pfor.detach.preheader.split.ls:
; CHECK-NEXT:  br label %pfor.detach.preheader.split.ls2

; CHECK: pfor.cond.cleanup.loopexit.ls:                    ; preds = %pfor.inc.ls
; CHECK-NEXT:  ret void

; CHECK: pfor.detach.ls:                                   ; preds = %pfor.detach.preheader.split.ls2, %pfor.inc.ls
; CHECK-NEXT:   %i.06.ls = phi i32 [ %inc.ls, %pfor.inc.ls ], [ %start.ls, %pfor.detach.preheader.split.ls2 ]
; CHECK-NEXT:   br label %pfor.body.ls

; CHECK: pfor.body.ls:                                     ; preds = %pfor.detach.ls
; CHECK-NEXT:   tail call void @bar(i32 %i.06.ls) #4
; CHECK-NEXT:   br label %pfor.inc.ls

; CHECK: pfor.inc.ls:                                      ; preds = %pfor.body.ls
; CHECK-NEXT:   %0 = icmp ult i32 %i.06.ls, %.ls1
; CHECK-NEXT:   %inc.ls = add nuw nsw i32 %i.06.ls, 1
; CHECK-NEXT:   br i1 %0, label %pfor.detach.ls, label %pfor.cond.cleanup.loopexit.ls

; CHECK: pfor.detach.preheader.split.ls2:
; CHECK-NEXT: br label %pfor.detach.ls

declare void @bar(i32) local_unnamed_addr #1

; Function Attrs: argmemonly nounwind
declare token @llvm.syncregion.start() #3

attributes #0 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind }
attributes #3 = { argmemonly nounwind }

!1 = distinct !{!1, !2}
!2 = !{!"tapir.loop.spawn.strategy", i32 1}
