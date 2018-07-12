; Simple test to verify that Tapir lowering to the qthreads target
; inserts the correct calls.
;
; RUN: opt < %s -tapir2target -tapir-target=qthreads -S | FileCheck %s

source_filename = "fib.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind uwtable
define i32 @fib(i32 %n) local_unnamed_addr #0 {
entry:
  %x = alloca i32, align 4
  %syncreg = tail call token @llvm.syncregion.start()
  %cmp = icmp slt i32 %n, 2
  br i1 %cmp, label %return, label %if.end
; CHECK: call i8* @qt_sinc_create(

if.end:                                           ; preds = %entry
  %x.0.x.0..sroa_cast = bitcast i32* %x to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %x.0.x.0..sroa_cast)
  detach within %syncreg, label %det.achd, label %det.cont
; CHECK: call void @qt_sinc_expect(
; CHECK: call i32 @qthread_fork_copyargs(i64 (i8*)*

det.achd:                                         ; preds = %if.end
  %sub = add nsw i32 %n, -1
  %call = tail call i32 @fib(i32 %sub)
  store i32 %call, i32* %x, align 4
  reattach within %syncreg, label %det.cont

det.cont:                                         ; preds = %det.achd, %if.end
  %sub1 = add nsw i32 %n, -2
  %call2 = tail call i32 @fib(i32 %sub1)
  sync within %syncreg, label %sync.continue
; CHECK: call void @qt_sinc_wait(

sync.continue:                                    ; preds = %det.cont
  %x.0.load9 = load i32, i32* %x, align 4
  %add = add nsw i32 %x.0.load9, %call2
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %x.0.x.0..sroa_cast)
  br label %return

return:                                           ; preds = %entry, %sync.continue
  %retval.0 = phi i32 [ %add, %sync.continue ], [ %n, %entry ]
  ret i32 %retval.0
; CHECK: call void @qt_sinc_destroy(
}

; CHECK: call void @qt_sinc_submit(

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #1

; Function Attrs: argmemonly nounwind
declare token @llvm.syncregion.start() #1

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #1

attributes #0 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 5.0.0 (git@github.com:wsmoses/Cilk-Clang.git 245c29d5cb99796c4107fd83f9bbe668c130b275) (git@github.com:wsmoses/Tapir-LLVM.git 4c29285ec4342f4eaba95987b05f84f964af9008)"}
