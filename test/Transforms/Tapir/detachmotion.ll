; RUN: opt < %s -nesteddetach -S | FileCheck %s

; ModuleID = 'test2.c'
source_filename = "test2.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind uwtable
; CHECK-NOT: syncreg1
define void @foo() local_unnamed_addr #0 {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  detach within %syncreg, label %det.achd, label %det.cont3

det.achd:                                         ; preds = %entry
  %syncreg1 = tail call token @llvm.syncregion.start()
  detach within %syncreg1, label %det.achd2, label %det.cont

det.achd2:                                        ; preds = %det.achd
  tail call void (...) @a() #3
  reattach within %syncreg1, label %det.cont

det.cont:                                         ; preds = %det.achd2, %det.achd
  reattach within %syncreg, label %det.cont3

det.cont3:                                        ; preds = %det.cont, %entry
  tail call void (...) @b() #3
  sync within %syncreg, label %sync.continue

sync.continue:                                    ; preds = %det.cont3
  ret void
}

; Function Attrs: argmemonly nounwind
declare token @llvm.syncregion.start() #1

declare void @a(...) local_unnamed_addr #2

declare void @b(...) local_unnamed_addr #2

attributes #0 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 5.0.0 (https://github.com/wsmoses/Tapir-Clang 2637f015d66418964aa0225534c004dd71a174b8) (git@github.com:wsmoses/Tapir-LLVM.git 89bb9a7ebb000b3e1a44405872cfef37e7ae679e)"}
