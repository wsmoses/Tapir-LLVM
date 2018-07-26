; RUN: opt < %s -loop-spawning -ls-tapir-target=cilk -simplifycfg -S | FileCheck %s

; CHECK: define internal fastcc void @foo_pfor.detach.ls(i64 %start.ls, i64 %.ls, i64 %grainsize.ls) #0 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*)

; ModuleID = 'newstart.ll'
source_filename = "sret-test.cpp"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%"class.std::vector.0" = type { %"struct.std::_Vector_base.1" }
%"struct.std::_Vector_base.1" = type { %"struct.std::_Vector_base<std::tuple<int, double, int>, std::allocator<std::tuple<int, double, int> > >::_Vector_impl" }
%"struct.std::_Vector_base<std::tuple<int, double, int>, std::allocator<std::tuple<int, double, int> > >::_Vector_impl" = type { %"class.std::tuple"*, %"class.std::tuple"*, %"class.std::tuple"* }
%"class.std::tuple" = type { %"struct.std::_Tuple_impl.base", [4 x i8] }
%"struct.std::_Tuple_impl.base" = type <{ %"struct.std::_Tuple_impl.5", %"struct.std::_Head_base.8" }>
%"struct.std::_Tuple_impl.5" = type { %"struct.std::_Tuple_impl.6", %"struct.std::_Head_base.7" }
%"struct.std::_Tuple_impl.6" = type { %"struct.std::_Head_base" }
%"struct.std::_Head_base" = type { i32 }
%"struct.std::_Head_base.7" = type { double }
%"struct.std::_Head_base.8" = type { i32 }
%"class.std::vector" = type { %"struct.std::_Vector_base" }
%"struct.std::_Vector_base" = type { %"struct.std::_Vector_base<params, std::allocator<params> >::_Vector_impl" }
%"struct.std::_Vector_base<params, std::allocator<params> >::_Vector_impl" = type { %struct.params*, %struct.params*, %struct.params* }
%struct.params = type { i32, i32, float, float, float, i32 }

; Function Attrs: uwtable
define void @foo(%"class.std::vector.0"* noalias sret %agg.result, i64 %numiters, i64 %numiters2, i64 %numiters3, i32 %trials, %"class.std::vector"* nocapture readonly dereferenceable(24) %ps) #0 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  br label %pfor.detach

pfor.detach:                                      ; preds = %pfor.inc78, %entry
  %indvars.iv395 = phi i64 [ 0, %entry ], [ %indvars.iv.next396, %pfor.inc78 ]
  detach within %syncreg, label %pfor.body, label %pfor.inc78

pfor.body:                                        ; preds = %pfor.detach
  %cmp.i.i = call i1 @a()
  br i1 %cmp.i.i, label %if.else.i.i, label %exit2601

if.else.i.i:                                      ; preds = %pfor.body
  invoke void @invokable2()
          to label %exit260 unwind label %lpad64

lpad64:                                           ; preds = %if.else.i.i
  %lpad64v0 = landingpad { i8*, i32 }
          cleanup
  br label %invoke.cont.i

exit260:                                          ; preds = %if.else.i.i
  reattach within %syncreg, label %pfor.inc78

exit2601:                                         ; preds = %pfor.body
  reattach within %syncreg, label %pfor.inc78

pfor.inc78:                                       ; preds = %exit2601, %exit260, %pfor.detach
  %indvars.iv.next396 = add nuw nsw i64 %indvars.iv395, 1
  %cmp = icmp slt i64 %indvars.iv.next396, %numiters
  br i1 %cmp, label %pfor.detach, label %pfor.cond.cleanup, !llvm.loop !2

pfor.cond.cleanup:                                ; preds = %pfor.inc78
  sync within %syncreg, label %for.body90

for.body90:                                       ; preds = %pfor.cond.cleanup
  invoke void @invokable()
          to label %exit220 unwind label %lpad103

lpad103:                                          ; preds = %for.body90
  %lpad103v0 = landingpad { i8*, i32 }
          cleanup
  %lpad103v1 = extractvalue { i8*, i32 } %lpad103v0, 0
  %lpad103v2 = extractvalue { i8*, i32 } %lpad103v0, 1
  br label %invoke.cont.i

invoke.cont.i:                                    ; preds = %lpad103, %lpad64
  %ehselector.slot.0 = phi i32 [ %lpad103v2, %lpad103 ], [ undef, %lpad64 ]
  %exn.slot.0 = phi i8* [ %lpad103v1, %lpad103 ], [ undef, %lpad64 ]
  %lpad.val = insertvalue { i8*, i32 } undef, i8* %exn.slot.0, 0
  %lpad.val117 = insertvalue { i8*, i32 } %lpad.val, i32 %ehselector.slot.0, 1
  resume { i8*, i32 } %lpad.val117

exit220:                                          ; preds = %for.body90
  ret void
}

declare i1 @a()

declare i32 @__gxx_personality_v0(...)

; Function Attrs: argmemonly nounwind
declare token @llvm.syncregion.start() #1

; Function Attrs: uwtable
declare void @invokable() #0

; Function Attrs: uwtable
declare void @invokable2() #0

attributes #0 = { uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 5.0.0 (git@github.com:wsmoses/Tapir-Clang.git 245c29d5cb99796c4107fd83f9bbe668c130b275) (git@github.com:wsmoses/Tapir-LLVM.git 7352407d063c8bac796926ca618e14d8eca87735)"}
!2 = distinct !{!2, !3}
!3 = !{!"tapir.loop.spawn.strategy", i32 1}
