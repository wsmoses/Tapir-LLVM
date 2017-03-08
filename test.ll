; ModuleID = 'test.c'
source_filename = "test.c"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.12.0"

@global = external global double, align 8

; Function Attrs: noinline nounwind ssp uwtable
define void @test() #0 {
entry:
  %n = alloca double, align 8
  detach label %det.achd, label %det.cont

det.achd:                                         ; preds = %entry
  %0 = bitcast i32 undef to i32
  %call = call double (...) @foo() #3
  store double %call, double* %n, align 8
  reattach label %det.cont

det.cont:                                         ; preds = %det.achd, %entry
  %call1 = call i32 (...) @cond() #3
  %tobool = icmp ne i32 %call1, 0
  br i1 %tobool, label %if.then, label %if.end

if.then:                                          ; preds = %det.cont
  sync label %sync.continue

sync.continue:                                    ; preds = %if.then
  br label %if.end

if.end:                                           ; preds = %sync.continue, %det.cont
  %call2 = call double (...) @moo() #3
  store double %call2, double* @global, align 8
  sync label %sync.continue3

sync.continue3:                                   ; preds = %if.end
  %1 = load double, double* %n, align 8
  call void @bat(double %1)
  ret void
}

; Function Attrs: nounwind readnone
declare double @foo(...) #1

; Function Attrs: nounwind readnone
declare i32 @cond(...) #1

; Function Attrs: nounwind readnone
declare double @moo(...) #1

declare void @bat(double) #2

attributes #0 = { noinline nounwind ssp uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+fxsr,+mmx,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+fxsr,+mmx,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+fxsr,+mmx,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind readnone }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"PIC Level", i32 2}
!1 = !{!"clang version 4.0.0 (git@github.com:wsmoses/Cilk-Clang cc78c4b6082bb80687e64c8104bf9744e6fa8fdc) (git@github.com:wsmoses/Parallel-IR 4611d796dea964dea884c34cadcef14b256fbe56)"}
