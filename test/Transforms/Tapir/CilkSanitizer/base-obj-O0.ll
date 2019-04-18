; Thanks to Endrias Kahssay for contributing this test case.
;
; RUN: opt < %s -csan -S -o - | FileCheck %s
; RUN: opt < %s -aa-pipeline=basic-aa -passes='cilksan' -S -o - | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@.str = private unnamed_addr constant [11 x i8] c"index %d \0A\00", align 1
@.str.1 = private unnamed_addr constant [11 x i8] c"entry %d \0A\00", align 1
@.str.2 = private unnamed_addr constant [9 x i8] c"sum %d \0A\00", align 1

; Function Attrs: noinline nounwind optnone uwtable
define i32 @hello(i64* %a, i32 %index) #0 !dbg !9 {
entry:
  %retval = alloca i32, align 4
  %a.addr = alloca i64*, align 8
  %index.addr = alloca i32, align 4
  store i64* %a, i64** %a.addr, align 8
  call void @llvm.dbg.declare(metadata i64** %a.addr, metadata !19, metadata !DIExpression()), !dbg !20
  store i32 %index, i32* %index.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %index.addr, metadata !21, metadata !DIExpression()), !dbg !22
  %0 = load i32, i32* %index.addr, align 4, !dbg !23
  %call = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str, i32 0, i32 0), i32 %0), !dbg !24
  %1 = load i64*, i64** %a.addr, align 8, !dbg !25
  %2 = load i32, i32* %index.addr, align 4, !dbg !26
  %idxprom = sext i32 %2 to i64, !dbg !25
  %arrayidx = getelementptr inbounds i64, i64* %1, i64 %idxprom, !dbg !25
  store i64 4, i64* %arrayidx, align 8, !dbg !27
  %3 = load i32, i32* %retval, align 4, !dbg !28
  ret i32 %3, !dbg !28
}

; CHECK-LABEL: define i32 @hello
; CHECK: call void @__csan_store(i64
; CHECK-NEXT: store i64 4, i64* %arrayidx

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare i32 @printf(i8*, ...) #2

; Function Attrs: noinline nounwind optnone uwtable
define i32 @main() #0 !dbg !29 {
entry:
  %retval = alloca i32, align 4
  %size = alloca i32, align 4
  %my_array = alloca i64*, align 8
  %syncreg = call token @llvm.syncregion.start()
  %__init = alloca i32, align 4
  %__begin = alloca i32, align 4
  %__end = alloca i32, align 4
  %cleanup.dest.slot = alloca i32
  %syncreg14 = call token @llvm.syncregion.start()
  %sum = alloca i32, align 4
  %i16 = alloca i32, align 4
  store i32 0, i32* %retval, align 4
  call void @llvm.dbg.declare(metadata i32* %size, metadata !31, metadata !DIExpression()), !dbg !32
  store i32 10, i32* %size, align 4, !dbg !32
  call void @llvm.dbg.declare(metadata i64** %my_array, metadata !33, metadata !DIExpression()), !dbg !34
  %0 = load i32, i32* %size, align 4, !dbg !35
  %conv = sext i32 %0 to i64, !dbg !35
  %mul = mul i64 %conv, 8, !dbg !36
  %call = call noalias i8* @malloc(i64 %mul) #5, !dbg !37
  %1 = bitcast i8* %call to i64*, !dbg !37
  store i64* %1, i64** %my_array, align 8, !dbg !34
  call void @llvm.dbg.declare(metadata i32* %__init, metadata !38, metadata !DIExpression()), !dbg !40
  store i32 0, i32* %__init, align 4, !dbg !41
  call void @llvm.dbg.declare(metadata i32* %__begin, metadata !42, metadata !DIExpression()), !dbg !40
  store i32 0, i32* %__begin, align 4, !dbg !41
  call void @llvm.dbg.declare(metadata i32* %__end, metadata !43, metadata !DIExpression()), !dbg !40
  %2 = load i32, i32* %size, align 4, !dbg !44
  %sub = sub nsw i32 %2, 0, !dbg !41
  %sub1 = sub nsw i32 %sub, 1, !dbg !41
  %div = sdiv i32 %sub1, 1, !dbg !41
  %add = add nsw i32 %div, 1, !dbg !41
  store i32 %add, i32* %__end, align 4, !dbg !41
  br label %pfor.cond, !dbg !41

pfor.cond:                                        ; preds = %pfor.inc, %entry
  %3 = load i32, i32* %__begin, align 4, !dbg !45
  %4 = load i32, i32* %__end, align 4, !dbg !45
  %cmp = icmp slt i32 %3, %4, !dbg !45
  br i1 %cmp, label %pfor.detach, label %pfor.cond.cleanup, !dbg !41

pfor.cond.cleanup:                                ; preds = %pfor.cond
  store i32 2, i32* %cleanup.dest.slot, align 4
  sync within %syncreg, label %sync.continue, !dbg !45

pfor.detach:                                      ; preds = %pfor.cond
  %5 = load i32, i32* %__init, align 4, !dbg !45
  %6 = load i32, i32* %__begin, align 4, !dbg !45
  %mul3 = mul nsw i32 %6, 1, !dbg !45
  %add4 = add nsw i32 %5, %mul3, !dbg !45
  detach within %syncreg, label %pfor.body.entry, label %pfor.inc, !dbg !41

pfor.body.entry:                                  ; preds = %pfor.detach
  %i = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %i, metadata !47, metadata !DIExpression()), !dbg !48
  store i32 %add4, i32* %i, align 4, !dbg !41
  br label %pfor.body, !dbg !41

pfor.body:                                        ; preds = %pfor.body.entry
  %call5 = call i32 @rand() #5, !dbg !49
  %add6 = add nsw i32 %call5, 4, !dbg !51
  %conv7 = sitofp i32 %add6 to double, !dbg !49
  %7 = call double @llvm.ceil.f64(double %conv7), !dbg !52
  %conv8 = fptosi double %7 to i32, !dbg !53
  %conv9 = sext i32 %conv8 to i64, !dbg !53
  %8 = load i64*, i64** %my_array, align 8, !dbg !54
  %9 = load i32, i32* %i, align 4, !dbg !55
  %idxprom = sext i32 %9 to i64, !dbg !54
  %arrayidx = getelementptr inbounds i64, i64* %8, i64 %idxprom, !dbg !54
  store i64 %conv9, i64* %arrayidx, align 8, !dbg !56
  %10 = load i64*, i64** %my_array, align 8, !dbg !57
  %11 = load i32, i32* %i, align 4, !dbg !58
  %idxprom10 = sext i32 %11 to i64, !dbg !57
  %arrayidx11 = getelementptr inbounds i64, i64* %10, i64 %idxprom10, !dbg !57
  %12 = load i64, i64* %arrayidx11, align 8, !dbg !57
  %call12 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.1, i32 0, i32 0), i64 %12), !dbg !59
  br label %pfor.preattach, !dbg !60

pfor.preattach:                                   ; preds = %pfor.body
  reattach within %syncreg, label %pfor.inc, !dbg !60

pfor.inc:                                         ; preds = %pfor.preattach, %pfor.detach
  %13 = load i32, i32* %__begin, align 4, !dbg !45
  %inc = add nsw i32 %13, 1, !dbg !45
  store i32 %inc, i32* %__begin, align 4, !dbg !45
  br label %pfor.cond, !dbg !45, !llvm.loop !61

sync.continue:                                    ; preds = %pfor.cond.cleanup
  br label %pfor.end

pfor.end:                                         ; preds = %sync.continue
  %14 = load i64*, i64** %my_array, align 8, !dbg !64
  detach within %syncreg14, label %det.achd, label %det.cont, !dbg !65

det.achd:                                         ; preds = %pfor.end
  %call15 = call i32 @hello(i64* %14, i32 3), !dbg !65
  reattach within %syncreg14, label %det.cont, !dbg !65

det.cont:                                         ; preds = %det.achd, %pfor.end
  call void @llvm.dbg.declare(metadata i32* %sum, metadata !66, metadata !DIExpression()), !dbg !67
  store i32 0, i32* %sum, align 4, !dbg !67
  call void @llvm.dbg.declare(metadata i32* %i16, metadata !68, metadata !DIExpression()), !dbg !70
  store i32 0, i32* %i16, align 4, !dbg !70
  br label %for.cond, !dbg !71

for.cond:                                         ; preds = %for.inc, %det.cont
  %15 = load i32, i32* %i16, align 4, !dbg !72
  %16 = load i32, i32* %size, align 4, !dbg !74
  %sub17 = sub nsw i32 %16, 1, !dbg !75
  %cmp18 = icmp slt i32 %15, %sub17, !dbg !76
  br i1 %cmp18, label %for.body, label %for.end, !dbg !77

for.body:                                         ; preds = %for.cond
  %17 = load i64*, i64** %my_array, align 8, !dbg !78
  %18 = load i32, i32* %i16, align 4, !dbg !80
  %add20 = add nsw i32 %18, 1, !dbg !81
  %idxprom21 = sext i32 %add20 to i64, !dbg !78
  %arrayidx22 = getelementptr inbounds i64, i64* %17, i64 %idxprom21, !dbg !78
  %19 = load i64, i64* %arrayidx22, align 8, !dbg !78
  %20 = load i32, i32* %sum, align 4, !dbg !82
  %conv23 = sext i32 %20 to i64, !dbg !82
  %add24 = add nsw i64 %conv23, %19, !dbg !82
  %conv25 = trunc i64 %add24 to i32, !dbg !82
  store i32 %conv25, i32* %sum, align 4, !dbg !82
  br label %for.inc, !dbg !83

for.inc:                                          ; preds = %for.body
  %21 = load i32, i32* %i16, align 4, !dbg !84
  %inc26 = add nsw i32 %21, 1, !dbg !84
  store i32 %inc26, i32* %i16, align 4, !dbg !84
  br label %for.cond, !dbg !85, !llvm.loop !86

for.end:                                          ; preds = %for.cond
  %22 = load i32, i32* %sum, align 4, !dbg !88
  %call27 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([9 x i8], [9 x i8]* @.str.2, i32 0, i32 0), i32 %22), !dbg !89
  store i32 0, i32* %retval, align 4, !dbg !90
  sync within %syncreg14, label %sync.continue28, !dbg !91

sync.continue28:                                  ; preds = %for.end
  %23 = load i32, i32* %retval, align 4, !dbg !91
  ret i32 %23, !dbg !91
}

; CHECK-LABEL: define i32 @main
; CHECK: {{^for.body}}:
; CHECK-NEXT: %[[LOCALMYARRAY:.+]] = load i64*, i64** %my_array
; CHECK: %[[LOCALARRAYIDX:.+]] = getelementptr inbounds i64, i64* %[[LOCALMYARRAY]]
; CHECK: call void @__csan_load
; CHECK-NEXT: load i64, i64* %[[LOCALARRAYIDX]]

; Function Attrs: nounwind
declare noalias i8* @malloc(i64) #3

; Function Attrs: argmemonly nounwind
declare token @llvm.syncregion.start() #4

; Function Attrs: nounwind
declare i32 @rand() #3

; Function Attrs: nounwind readnone speculatable
declare double @llvm.ceil.f64(double) #1

attributes #0 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { argmemonly nounwind }
attributes #5 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!5, !6, !7}
!llvm.ident = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 6.0.0 (git@github.com:wsmoses/Tapir-Clang.git 9a357f1e72cd838b73fc5e630cd360676512aef3) (git@github.mit.edu:SuperTech/Tapir-CSI-llvm.git b49ddd4ce6d6711d88c4a75339dc25fa09235be1)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3)
!1 = !DIFile(filename: "endrias-20190416.c", directory: "/data/compilers/tests/adhoc/cilksan")
!2 = !{}
!3 = !{!4}
!4 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!5 = !{i32 2, !"Dwarf Version", i32 4}
!6 = !{i32 2, !"Debug Info Version", i32 3}
!7 = !{i32 1, !"wchar_size", i32 4}
!8 = !{!"clang version 6.0.0 (git@github.com:wsmoses/Tapir-Clang.git 9a357f1e72cd838b73fc5e630cd360676512aef3) (git@github.mit.edu:SuperTech/Tapir-CSI-llvm.git b49ddd4ce6d6711d88c4a75339dc25fa09235be1)"}
!9 = distinct !DISubprogram(name: "hello", scope: !1, file: !1, line: 7, type: !10, isLocal: false, isDefinition: true, scopeLine: 7, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !2)
!10 = !DISubroutineType(types: !11)
!11 = !{!4, !12, !4}
!12 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !13, size: 64)
!13 = !DIDerivedType(tag: DW_TAG_typedef, name: "entry", file: !1, line: 5, baseType: !14)
!14 = !DIDerivedType(tag: DW_TAG_typedef, name: "int64_t", file: !15, line: 27, baseType: !16)
!15 = !DIFile(filename: "/usr/include/bits/stdint-intn.h", directory: "/data/compilers/tests/adhoc/cilksan")
!16 = !DIDerivedType(tag: DW_TAG_typedef, name: "__int64_t", file: !17, line: 43, baseType: !18)
!17 = !DIFile(filename: "/usr/include/bits/types.h", directory: "/data/compilers/tests/adhoc/cilksan")
!18 = !DIBasicType(name: "long int", size: 64, encoding: DW_ATE_signed)
!19 = !DILocalVariable(name: "a", arg: 1, scope: !9, file: !1, line: 7, type: !12)
!20 = !DILocation(line: 7, column: 19, scope: !9)
!21 = !DILocalVariable(name: "index", arg: 2, scope: !9, file: !1, line: 7, type: !4)
!22 = !DILocation(line: 7, column: 26, scope: !9)
!23 = !DILocation(line: 8, column: 25, scope: !9)
!24 = !DILocation(line: 8, column: 3, scope: !9)
!25 = !DILocation(line: 9, column: 3, scope: !9)
!26 = !DILocation(line: 9, column: 5, scope: !9)
!27 = !DILocation(line: 9, column: 12, scope: !9)
!28 = !DILocation(line: 10, column: 1, scope: !9)
!29 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 12, type: !30, isLocal: false, isDefinition: true, scopeLine: 12, isOptimized: false, unit: !0, variables: !2)
!30 = !DISubroutineType(types: !3)
!31 = !DILocalVariable(name: "size", scope: !29, file: !1, line: 13, type: !4)
!32 = !DILocation(line: 13, column: 10, scope: !29)
!33 = !DILocalVariable(name: "my_array", scope: !29, file: !1, line: 14, type: !12)
!34 = !DILocation(line: 14, column: 14, scope: !29)
!35 = !DILocation(line: 14, column: 32, scope: !29)
!36 = !DILocation(line: 14, column: 36, scope: !29)
!37 = !DILocation(line: 14, column: 25, scope: !29)
!38 = !DILocalVariable(name: "__init", scope: !39, type: !4, flags: DIFlagArtificial)
!39 = distinct !DILexicalBlock(scope: !29, file: !1, line: 15, column: 6)
!40 = !DILocation(line: 0, scope: !39)
!41 = !DILocation(line: 15, column: 6, scope: !39)
!42 = !DILocalVariable(name: "__begin", scope: !39, type: !4, flags: DIFlagArtificial)
!43 = !DILocalVariable(name: "__end", scope: !39, type: !4, flags: DIFlagArtificial)
!44 = !DILocation(line: 15, column: 30, scope: !39)
!45 = !DILocation(line: 15, column: 6, scope: !46)
!46 = distinct !DILexicalBlock(scope: !39, file: !1, line: 15, column: 6)
!47 = !DILocalVariable(name: "i", scope: !46, file: !1, line: 15, type: !4)
!48 = !DILocation(line: 15, column: 19, scope: !46)
!49 = !DILocation(line: 16, column: 32, scope: !50)
!50 = distinct !DILexicalBlock(scope: !46, file: !1, line: 15, column: 40)
!51 = !DILocation(line: 16, column: 39, scope: !50)
!52 = !DILocation(line: 16, column: 27, scope: !50)
!53 = !DILocation(line: 16, column: 21, scope: !50)
!54 = !DILocation(line: 16, column: 7, scope: !50)
!55 = !DILocation(line: 16, column: 16, scope: !50)
!56 = !DILocation(line: 16, column: 19, scope: !50)
!57 = !DILocation(line: 17, column: 31, scope: !50)
!58 = !DILocation(line: 17, column: 40, scope: !50)
!59 = !DILocation(line: 17, column: 9, scope: !50)
!60 = !DILocation(line: 18, column: 6, scope: !50)
!61 = distinct !{!61, !41, !62, !63}
!62 = !DILocation(line: 18, column: 6, scope: !39)
!63 = !{!"tapir.loop.spawn.strategy", i32 1}
!64 = !DILocation(line: 19, column: 23, scope: !29)
!65 = !DILocation(line: 19, column: 6, scope: !29)
!66 = !DILocalVariable(name: "sum", scope: !29, file: !1, line: 20, type: !4)
!67 = !DILocation(line: 20, column: 10, scope: !29)
!68 = !DILocalVariable(name: "i", scope: !69, file: !1, line: 21, type: !4)
!69 = distinct !DILexicalBlock(scope: !29, file: !1, line: 21, column: 6)
!70 = !DILocation(line: 21, column: 14, scope: !69)
!71 = !DILocation(line: 21, column: 10, scope: !69)
!72 = !DILocation(line: 21, column: 21, scope: !73)
!73 = distinct !DILexicalBlock(scope: !69, file: !1, line: 21, column: 6)
!74 = !DILocation(line: 21, column: 25, scope: !73)
!75 = !DILocation(line: 21, column: 29, scope: !73)
!76 = !DILocation(line: 21, column: 23, scope: !73)
!77 = !DILocation(line: 21, column: 6, scope: !69)
!78 = !DILocation(line: 22, column: 15, scope: !79)
!79 = distinct !DILexicalBlock(scope: !73, file: !1, line: 21, column: 37)
!80 = !DILocation(line: 22, column: 24, scope: !79)
!81 = !DILocation(line: 22, column: 25, scope: !79)
!82 = !DILocation(line: 22, column: 12, scope: !79)
!83 = !DILocation(line: 24, column: 6, scope: !79)
!84 = !DILocation(line: 21, column: 34, scope: !73)
!85 = !DILocation(line: 21, column: 6, scope: !73)
!86 = distinct !{!86, !77, !87}
!87 = !DILocation(line: 24, column: 6, scope: !69)
!88 = !DILocation(line: 25, column: 27, scope: !29)
!89 = !DILocation(line: 25, column: 7, scope: !29)
!90 = !DILocation(line: 27, column: 7, scope: !29)
!91 = !DILocation(line: 28, column: 1, scope: !29)
