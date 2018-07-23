; Test that we can control unrolling for different tapir backends

; RUN: opt < %s -loop-unroll -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind uwtable
define void @dac(i32 %n, double* nocapture %a) local_unnamed_addr #0 {
; CHECK-LABEL: dac
; CHECK: detach within
; CHECK-NOT: detach within

entry:
  %syncreg = tail call token @llvm.syncregion.start()
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %pfor.detach.lr.ph, label %pfor.cond.cleanup

pfor.detach.lr.ph:                                ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %pfor.detach

pfor.cond.cleanup:                                ; preds = %pfor.inc, %entry
  sync within %syncreg, label %pfor.end.continue

pfor.end.continue:                                ; preds = %pfor.cond.cleanup
  ret void

pfor.detach:                                      ; preds = %pfor.inc, %pfor.detach.lr.ph
  %indvars.iv = phi i64 [ 0, %pfor.detach.lr.ph ], [ %indvars.iv.next, %pfor.inc ]
  detach within %syncreg, label %pfor.body, label %pfor.inc

pfor.body:                                        ; preds = %pfor.detach
  %0 = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %0 to double
  %arrayidx = getelementptr inbounds double, double* %a, i64 %indvars.iv
  store double %conv, double* %arrayidx, align 8, !tbaa !2
  reattach within %syncreg, label %pfor.inc

pfor.inc:                                         ; preds = %pfor.body, %pfor.detach
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond, label %pfor.cond.cleanup, label %pfor.detach, !llvm.loop !6
}

define void @gpu(i32 %n, double* nocapture %a) local_unnamed_addr #0 {
; CHECK-LABEL: gpu
; CHECK: detach within
; CHECK-NOT: detach within

entry:
  %syncreg = tail call token @llvm.syncregion.start()
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %pfor.detach.lr.ph, label %pfor.cond.cleanup

pfor.detach.lr.ph:                                ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %pfor.detach

pfor.cond.cleanup:                                ; preds = %pfor.inc, %entry
  sync within %syncreg, label %pfor.end.continue

pfor.end.continue:                                ; preds = %pfor.cond.cleanup
  ret void

pfor.detach:                                      ; preds = %pfor.inc, %pfor.detach.lr.ph
  %indvars.iv = phi i64 [ 0, %pfor.detach.lr.ph ], [ %indvars.iv.next, %pfor.inc ]
  detach within %syncreg, label %pfor.body, label %pfor.inc

pfor.body:                                        ; preds = %pfor.detach
  %0 = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %0 to double
  %arrayidx = getelementptr inbounds double, double* %a, i64 %indvars.iv
  store double %conv, double* %arrayidx, align 8, !tbaa !2
  reattach within %syncreg, label %pfor.inc

pfor.inc:                                         ; preds = %pfor.body, %pfor.detach
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond, label %pfor.cond.cleanup, label %pfor.detach, !llvm.loop !8
}

define void @seq(i32 %n, double* nocapture %a) local_unnamed_addr #0 {
; CHECK-LABEL: seq
; CHECK: detach within
; CHECK: detach within
; CHECK: detach within
; CHECK: detach within
; CHECK: detach within

entry:
  %syncreg = tail call token @llvm.syncregion.start()
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %pfor.detach.lr.ph, label %pfor.cond.cleanup

pfor.detach.lr.ph:                                ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %pfor.detach

pfor.cond.cleanup:                                ; preds = %pfor.inc, %entry
  sync within %syncreg, label %pfor.end.continue

pfor.end.continue:                                ; preds = %pfor.cond.cleanup
  ret void

pfor.detach:                                      ; preds = %pfor.inc, %pfor.detach.lr.ph
  %indvars.iv = phi i64 [ 0, %pfor.detach.lr.ph ], [ %indvars.iv.next, %pfor.inc ]
  detach within %syncreg, label %pfor.body, label %pfor.inc

pfor.body:                                        ; preds = %pfor.detach
  %0 = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %0 to double
  %arrayidx = getelementptr inbounds double, double* %a, i64 %indvars.iv
  store double %conv, double* %arrayidx, align 8, !tbaa !2
  reattach within %syncreg, label %pfor.inc

pfor.inc:                                         ; preds = %pfor.body, %pfor.detach
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond, label %pfor.cond.cleanup, label %pfor.detach, !llvm.loop !10
}

define void @none(i32 %n, double* nocapture %a) local_unnamed_addr #0 {
; CHECK-LABEL: none
; CHECK: detach within
; CHECK: detach within
; CHECK: detach within
; CHECK: detach within
; CHECK: detach within
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %pfor.detach.lr.ph, label %pfor.cond.cleanup

pfor.detach.lr.ph:                                ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %pfor.detach

pfor.cond.cleanup:                                ; preds = %pfor.inc, %entry
  sync within %syncreg, label %pfor.end.continue

pfor.end.continue:                                ; preds = %pfor.cond.cleanup
  ret void

pfor.detach:                                      ; preds = %pfor.inc, %pfor.detach.lr.ph
  %indvars.iv = phi i64 [ 0, %pfor.detach.lr.ph ], [ %indvars.iv.next, %pfor.inc ]
  detach within %syncreg, label %pfor.body, label %pfor.inc

pfor.body:                                        ; preds = %pfor.detach
  %0 = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %0 to double
  %arrayidx = getelementptr inbounds double, double* %a, i64 %indvars.iv
  store double %conv, double* %arrayidx, align 8, !tbaa !2
  reattach within %syncreg, label %pfor.inc

pfor.inc:                                         ; preds = %pfor.body, %pfor.detach
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond, label %pfor.cond.cleanup, label %pfor.detach
}

; Function Attrs: argmemonly nounwind
declare token @llvm.syncregion.start() #1

attributes #0 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 5.0.0 (git@github.com:wsmoses/cilk-clang 5cfdd723a552d2ef151fd8990dec559fa7bd4795) (git@github.com:wsmoses/parallel-ir dfb187fa0b106c5a4f1d96ac14368946cbf50b60)"}
!2 = !{!3, !3, i64 0}
!3 = !{!"double", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
!6 = distinct !{!6, !7}
!7 = !{!"tapir.loop.spawn.strategy", i32 1}
!8 = distinct !{!8, !9}
!9 = !{!"tapir.loop.spawn.strategy", i32 2}
!10 = distinct !{!10, !11}
!11 = !{!"tapir.loop.spawn.strategy", i32 0}
