; RUN: opt < %s -smallblock -S | FileCheck %s

; Function Attrs: nounwind uwtable
define void @foo(i32 %x, i32 %y) local_unnamed_addr #0 {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  %tobool = icmp eq i32 %x, 0
  %z = alloca i32, align 4
  %q = alloca i32, align 4
; CHECK: detach
  detach within %syncreg, label %det.achd, label %det.cont

det.achd:                                         ; preds = %entry
  %foo = add i32 %x, %y
  tail call void (...) @a() #3
  store i32 %foo, i32* %z
; CHECK: reattach
  reattach within %syncreg, label %det.cont

det.cont:                                         ; preds = %det.achd, %entry
  tail call void (...) @c() #3
  sync within %syncreg, label %sync.continue

sync.continue:                                    ; preds = %det.cont
  ret void
}

; Function Attrs: argmemonly nounwind
declare token @llvm.syncregion.start() #1

declare void @a(...) local_unnamed_addr #2

declare void @b(...) local_unnamed_addr #2

declare void @c(...) local_unnamed_addr #2

attributes #0 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 5.0.0 (https://github.com/wsmoses/Tapir-Clang 2637f015d66418964aa0225534c004dd71a174b8) (git@github.com:wsmoses/Tapir-LLVM.git 5c2fdbbfc263e76384b49a7183b12a6b7132873d)"}
