; RUN: opt < %s  -loop-vectorize-rhino -force-vector-width=4 -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

@A = global [100000 x i32] zeroinitializer, align 16
@B = global [100000 x i32] zeroinitializer, align 16

; void f(int * A, int * B, int n) {
;     cilk_for (int i=0; i<n; i++) {
;         A[i] = B[i];
;     }
; }

; CHECK-LABEL: @f(
; CHECK: vec.detached
; CHECK-NOT: memcheck
; Function Attrs: nounwind uwtable
define void @f(i32* nocapture %A, i32* nocapture readonly %B, i32 %n) local_unnamed_addr #0 {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  %cmp12 = icmp sgt i32 %n, 0
  br i1 %cmp12, label %pfor.detach.lr.ph, label %pfor.cond.cleanup

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
  %arrayidx = getelementptr inbounds i32, i32* %B, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4, !tbaa !2
  %arrayidx4 = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  store i32 %0, i32* %arrayidx4, align 4, !tbaa !2
  reattach within %syncreg, label %pfor.inc

pfor.inc:                                         ; preds = %pfor.body, %pfor.detach
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond, label %pfor.cond.cleanup, label %pfor.detach, !llvm.loop !6
}

; Function Attrs: argmemonly nounwind
declare token @llvm.syncregion.start() #1

; Function Attrs: norecurse nounwind uwtable
define i32 @main(i32 %argc, i8** nocapture readnone %argv) local_unnamed_addr #2 {
entry:
  tail call void @f(i32* getelementptr inbounds ([100000 x i32], [100000 x i32]* @A, i64 0, i64 0), i32* getelementptr inbounds ([100000 x i32], [100000 x i32]* @B, i64 0, i64 0), i32 100000)
  ret i32 0
}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 5.0.0 (https://github.com/wsmoses/Cilk-Clang.git eaf246ef85cae33736dc7b015af97267045a6230) (git@github.com:wsmoses/Parallel-IR.git ca578abf2ded623076a35ebe6dd37816c0c41ede)"}
!2 = !{!3, !3, i64 0}
!3 = !{!"int", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C++ TBAA"}
!6 = distinct !{!6, !7}
!7 = !{!"tapir.loop.spawn.strategy", i32 1}
