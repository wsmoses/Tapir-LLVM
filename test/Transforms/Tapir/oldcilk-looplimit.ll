; Test that Tapir's loop spawning pass correctly transforms a loop
; that reads its original end iteration count.

; RUN: opt < %s -loop-spawning -S -ls-tapir-target=cilklegacy | FileCheck %s

source_filename = "looplimittest.c"

@.str = private unnamed_addr constant [13 x i8] c"Limit is %d\0A\00", align 1
@str = private unnamed_addr constant [9 x i8] c"Starting\00"
@str.3 = private unnamed_addr constant [9 x i8] c"Finished\00"

; Function Attrs: noinline nounwind uwtable
define void @foo(i32 %limit) local_unnamed_addr #0 {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  %cmp9 = icmp slt i32 %limit, 0
  br i1 %cmp9, label %pfor.cond.cleanup, label %pfor.detach

; CHECK: pfor.detach.preheader:
; CHECK: [[CLOSUREALLOC:%[0-9]+]] = alloca [[CLOSURETYPE:%[0-9]+]]
; CHECK-NEXT: [[GEP:%[0-9]+]] = getelementptr [[CLOSURETYPE]], [[CLOSURETYPE]]* [[CLOSUREALLOC]], i32 0, i32 0
; CHECK-NEXT: store i32 %limit, i32* [[GEP]]
; CHECK-NEXT: br label %pfor.detach.preheader.split
; CHECK: [[LIMIT:%[0-9]+]] = add [[TYPE:i[0-9]+]] %limit, 1
; CHECK: [[CLOSURECAST:%[0-9]+]] = bitcast [[CLOSURETYPE]]* [[CLOSUREALLOC]] to i8*
; CHECK: call void @__cilkrts_cilk_for_32(void (i8*, i32, i32)* bitcast (void (%0*, i32, i32)* @[[OUTLINED:[a-zA-Z0-9._]+]] to void (i8*, i32, i32)*), i8* [[CLOSURECAST]], i32 [[LIMIT]], i32 [[GRAIN:%[0-9]+]])
; CHECK-NEXT: br label %pfor.cond.cleanup.loopexit

pfor.cond.cleanup:                                ; preds = %pfor.inc, %entry
  sync within %syncreg, label %pfor.end.continue

pfor.end.continue:                                ; preds = %pfor.cond.cleanup
  ret void

; CHECK: define internal void @[[OUTLINED]](
; CHECK: [[CLOSURETYPE]]* [[closure:%[a-zA-Z0-9._]+]]
; CHECK: [[TYPE]] [[START:%[a-zA-Z0-9._]+]]
; CHECK: [[TYPE]] [[END:%[a-zA-Z0-9._]+]]
; CHECK: pfor.detach.preheader.split.ls:
; CHECK-NEXT: %0 = getelementptr [[CLOSURETYPE]], [[CLOSURETYPE]]* [[closure]], i32 0, i32 0
; CHECK-NEXT: [[LIM:%[0-9]+]] = load i32, i32* %0
; CHECK-NEXT br label %pfor.detach.preheader.split.ls1


pfor.detach:                                      ; preds = %entry, %pfor.inc
  %__begin.010 = phi i32 [ %inc, %pfor.inc ], [ 0, %entry ]
  detach within %syncreg, label %pfor.body, label %pfor.inc

pfor.body:                                        ; preds = %pfor.detach
; CHECK: pfor.body.ls:
  %call = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str, i64 0, i64 0), i32 %limit)
; CHECK-NEXT: %call.ls = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str, i64 0, i64 0), i32 [[LIM]])
  reattach within %syncreg, label %pfor.inc
; CHECK: br label %[[INC:[a-zA-Z0-9._]+]]

pfor.inc:                                         ; preds = %pfor.body, %pfor.detach
; CHECK: pfor.inc.ls: ; preds = %pfor.body.ls
; CHECK-NEXT: [[LOCALCMP:%[0-9]+]] = icmp ult i32 %__begin.010.ls, [[END]]
  %inc = add nuw nsw i32 %__begin.010, 1
; CHECK-NEXT: add {{.*}} %__begin.010.ls, 1
  %exitcond = icmp eq i32 %__begin.010, %limit
; CHECK: br i1 [[LOCALCMP]]
  br i1 %exitcond, label %pfor.cond.cleanup, label %pfor.detach, !llvm.loop !2
}

; Function Attrs: argmemonly nounwind
declare token @llvm.syncregion.start() #1

; Function Attrs: nounwind
declare i32 @printf(i8* nocapture readonly, ...) local_unnamed_addr #2

; Function Attrs: nounwind
declare i32 @puts(i8* nocapture readonly) #4

attributes #0 = { noinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind }

!2 = distinct !{!2, !3}
!3 = !{!"tapir.loop.spawn.strategy", i32 1}
