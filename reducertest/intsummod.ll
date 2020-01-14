; ModuleID = 'intsum.c'
source_filename = "intsum.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.anon = type { %struct.__cilkrts_hyperobject_base, i32, [60 x i8] }
%struct.__cilkrts_hyperobject_base = type { %struct.cilk_c_monoid, i64, i64, i64 }
%struct.cilk_c_monoid = type { void (i8*, i8*, i8*)*, void (i8*, i8*)*, void (i8*, i8*)*, i8* (i8*, i64)*, void (i8*, i8*)* }
%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i64, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i8*, i8*, i64, i32, [20 x i8] }
%struct._IO_marker = type { %struct._IO_marker*, %struct._IO_FILE*, i32 }
%struct.timespec = type { i64, i64 }

@.str = private unnamed_addr constant [18 x i8] c"ktiming_getmark()\00", align 1
@__const.compute_sum.my_int_sum_reducer = private unnamed_addr constant %struct.anon { %struct.__cilkrts_hyperobject_base { %struct.cilk_c_monoid { void (i8*, i8*, i8*)* @reduce_intsum, void (i8*, i8*)* @identity_intsum, void (i8*, i8*)* @__cilkrts_hyperobject_noop_destroy, i8* (i8*, i64)* @__cilkrts_hyperobject_alloc, void (i8*, i8*)* @__cilkrts_hyperobject_dealloc }, i64 0, i64 64, i64 4 }, i32 0, [60 x i8] undef }, align 64
@stderr = external dso_local local_unnamed_addr global %struct._IO_FILE*, align 8
@.str.1 = private unnamed_addr constant [39 x i8] c"Usage: ilist_dac [<cilk-options>] <n>\0A\00", align 1
@.str.2 = private unnamed_addr constant [26 x i8] c"Result: %d/%d successes!\0A\00", align 1
@.str.3 = private unnamed_addr constant [22 x i8] c"Running time %d: %gs\0A\00", align 1
@.str.4 = private unnamed_addr constant [28 x i8] c"Running time average: %g s\0A\00", align 1
@.str.5 = private unnamed_addr constant [26 x i8] c"Std. dev: %g s (%2.3f%%)\0A\00", align 1

; Function Attrs: nounwind uwtable
define dso_local i64 @ktiming_getmark() local_unnamed_addr #0 {
entry:
  %temp = alloca %struct.timespec, align 8
  %0 = bitcast %struct.timespec* %temp to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %0) #8
  %call = call i32 @clock_gettime(i32 1, %struct.timespec* nonnull %temp) #8
  %cmp = icmp eq i32 %call, 0
  br i1 %cmp, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  call void @perror(i8* getelementptr inbounds ([18 x i8], [18 x i8]* @.str, i64 0, i64 0)) #9
  call void @exit(i32 -1) #10
  unreachable

if.end:                                           ; preds = %entry
  %tv_nsec = getelementptr inbounds %struct.timespec, %struct.timespec* %temp, i64 0, i32 1
  %1 = load i64, i64* %tv_nsec, align 8, !tbaa !2
  %tv_sec = getelementptr inbounds %struct.timespec, %struct.timespec* %temp, i64 0, i32 0
  %2 = load i64, i64* %tv_sec, align 8, !tbaa !7
  %mul2 = mul i64 %2, 1000000000
  %add = add i64 %mul2, %1
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %0) #8
  ret i64 %add
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #1

; Function Attrs: nounwind
declare dso_local i32 @clock_gettime(i32, %struct.timespec*) local_unnamed_addr #2

; Function Attrs: nounwind
declare dso_local void @perror(i8* nocapture readonly) local_unnamed_addr #2

; Function Attrs: noreturn nounwind
declare dso_local void @exit(i32) local_unnamed_addr #3

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #1

; Function Attrs: argmemonly norecurse nounwind readonly uwtable
define dso_local i64 @ktiming_diff_usec(i64* nocapture readonly %start, i64* nocapture readonly %end) local_unnamed_addr #4 {
entry:
  %0 = load i64, i64* %end, align 8, !tbaa !8
  %1 = load i64, i64* %start, align 8, !tbaa !8
  %sub = sub i64 %0, %1
  ret i64 %sub
}

; Function Attrs: argmemonly norecurse nounwind readonly uwtable
define dso_local double @ktiming_diff_sec(i64* nocapture readonly %start, i64* nocapture readonly %end) local_unnamed_addr #4 {
entry:
  %0 = load i64, i64* %end, align 8, !tbaa !8
  %1 = load i64, i64* %start, align 8, !tbaa !8
  %sub.i = sub i64 %0, %1
  %conv = uitofp i64 %sub.i to double
  %div = fdiv double %conv, 1.000000e+09
  ret double %div
}

; Function Attrs: nounwind uwtable
define dso_local void @print_runtime(i64* nocapture readonly %tm_elapsed, i32 %size) local_unnamed_addr #0 {
entry:
  tail call fastcc void @print_runtime_helper(i64* %tm_elapsed, i32 %size, i32 0)
  ret void
}

; Function Attrs: nounwind uwtable
define internal fastcc void @print_runtime_helper(i64* nocapture readonly %usec_elapsed, i32 %size, i32 %summary) unnamed_addr #0 {
entry:
  %cmp73 = icmp sgt i32 %size, 0
  br i1 %cmp73, label %for.body.lr.ph, label %if.end28.thread

for.body.lr.ph:                                   ; preds = %entry
  %tobool = icmp eq i32 %summary, 0
  %wide.trip.count80 = zext i32 %size to i64
  br label %for.body

for.body:                                         ; preds = %for.inc, %for.body.lr.ph
  %indvars.iv77 = phi i64 [ 0, %for.body.lr.ph ], [ %1, %for.inc ]
  %total.074 = phi i64 [ 0, %for.body.lr.ph ], [ %add, %for.inc ]
  %arrayidx = getelementptr inbounds i64, i64* %usec_elapsed, i64 %indvars.iv77
  %0 = load i64, i64* %arrayidx, align 8, !tbaa !8
  %add = add i64 %0, %total.074
  %1 = add nuw nsw i64 %indvars.iv77, 1
  br i1 %tobool, label %if.then, label %for.inc

if.then:                                          ; preds = %for.body
  %conv = uitofp i64 %0 to double
  %mul = fmul double %conv, 1.000000e-09
  %2 = trunc i64 %1 to i32
  %call = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([22 x i8], [22 x i8]* @.str.3, i64 0, i64 0), i32 %2, double %mul)
  br label %for.inc

for.inc:                                          ; preds = %for.body, %if.then
  %exitcond81 = icmp eq i64 %1, %wide.trip.count80
  br i1 %exitcond81, label %for.end, label %for.body

for.end:                                          ; preds = %for.inc
  %conv4 = sext i32 %size to i64
  %div = udiv i64 %add, %conv4
  %conv5 = uitofp i64 %div to double
  %cmp6 = icmp sgt i32 %size, 1
  br i1 %cmp6, label %for.body12.preheader, label %if.end28.thread

for.body12.preheader:                             ; preds = %for.end
  %3 = add nsw i64 %wide.trip.count80, -1
  %xtraiter = and i64 %wide.trip.count80, 3
  %4 = icmp ult i64 %3, 3
  br i1 %4, label %if.end28.unr-lcssa, label %for.body12.preheader.new

for.body12.preheader.new:                         ; preds = %for.body12.preheader
  %unroll_iter = sub nsw i64 %wide.trip.count80, %xtraiter
  br label %for.body12

if.end28.thread:                                  ; preds = %entry, %for.end
  %conv586 = phi double [ %conv5, %for.end ], [ 0.000000e+00, %entry ]
  %mul2967 = fmul double %conv586, 1.000000e-09
  %call3068 = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([28 x i8], [28 x i8]* @.str.4, i64 0, i64 0), double %mul2967)
  br label %if.end39

for.body12:                                       ; preds = %for.body12, %for.body12.preheader.new
  %indvars.iv = phi i64 [ 0, %for.body12.preheader.new ], [ %indvars.iv.next.3, %for.body12 ]
  %dev_sq_sum.072 = phi double [ 0.000000e+00, %for.body12.preheader.new ], [ %add21.3, %for.body12 ]
  %niter = phi i64 [ %unroll_iter, %for.body12.preheader.new ], [ %niter.nsub.3, %for.body12 ]
  %arrayidx14 = getelementptr inbounds i64, i64* %usec_elapsed, i64 %indvars.iv
  %5 = load i64, i64* %arrayidx14, align 8, !tbaa !8
  %conv15 = uitofp i64 %5 to double
  %sub = fsub double %conv5, %conv15
  %mul20 = fmul double %sub, %sub
  %add21 = fadd double %dev_sq_sum.072, %mul20
  %indvars.iv.next = or i64 %indvars.iv, 1
  %arrayidx14.1 = getelementptr inbounds i64, i64* %usec_elapsed, i64 %indvars.iv.next
  %6 = load i64, i64* %arrayidx14.1, align 8, !tbaa !8
  %conv15.1 = uitofp i64 %6 to double
  %sub.1 = fsub double %conv5, %conv15.1
  %mul20.1 = fmul double %sub.1, %sub.1
  %add21.1 = fadd double %add21, %mul20.1
  %indvars.iv.next.1 = or i64 %indvars.iv, 2
  %arrayidx14.2 = getelementptr inbounds i64, i64* %usec_elapsed, i64 %indvars.iv.next.1
  %7 = load i64, i64* %arrayidx14.2, align 8, !tbaa !8
  %conv15.2 = uitofp i64 %7 to double
  %sub.2 = fsub double %conv5, %conv15.2
  %mul20.2 = fmul double %sub.2, %sub.2
  %add21.2 = fadd double %add21.1, %mul20.2
  %indvars.iv.next.2 = or i64 %indvars.iv, 3
  %arrayidx14.3 = getelementptr inbounds i64, i64* %usec_elapsed, i64 %indvars.iv.next.2
  %8 = load i64, i64* %arrayidx14.3, align 8, !tbaa !8
  %conv15.3 = uitofp i64 %8 to double
  %sub.3 = fsub double %conv5, %conv15.3
  %mul20.3 = fmul double %sub.3, %sub.3
  %add21.3 = fadd double %add21.2, %mul20.3
  %indvars.iv.next.3 = add nuw nsw i64 %indvars.iv, 4
  %niter.nsub.3 = add i64 %niter, -4
  %niter.ncmp.3 = icmp eq i64 %niter.nsub.3, 0
  br i1 %niter.ncmp.3, label %if.end28.unr-lcssa, label %for.body12

if.end28.unr-lcssa:                               ; preds = %for.body12, %for.body12.preheader
  %add21.lcssa.ph = phi double [ undef, %for.body12.preheader ], [ %add21.3, %for.body12 ]
  %indvars.iv.unr = phi i64 [ 0, %for.body12.preheader ], [ %indvars.iv.next.3, %for.body12 ]
  %dev_sq_sum.072.unr = phi double [ 0.000000e+00, %for.body12.preheader ], [ %add21.3, %for.body12 ]
  %lcmp.mod = icmp eq i64 %xtraiter, 0
  br i1 %lcmp.mod, label %if.end28, label %for.body12.epil

for.body12.epil:                                  ; preds = %if.end28.unr-lcssa, %for.body12.epil
  %indvars.iv.epil = phi i64 [ %indvars.iv.next.epil, %for.body12.epil ], [ %indvars.iv.unr, %if.end28.unr-lcssa ]
  %dev_sq_sum.072.epil = phi double [ %add21.epil, %for.body12.epil ], [ %dev_sq_sum.072.unr, %if.end28.unr-lcssa ]
  %epil.iter = phi i64 [ %epil.iter.sub, %for.body12.epil ], [ %xtraiter, %if.end28.unr-lcssa ]
  %arrayidx14.epil = getelementptr inbounds i64, i64* %usec_elapsed, i64 %indvars.iv.epil
  %9 = load i64, i64* %arrayidx14.epil, align 8, !tbaa !8
  %conv15.epil = uitofp i64 %9 to double
  %sub.epil = fsub double %conv5, %conv15.epil
  %mul20.epil = fmul double %sub.epil, %sub.epil
  %add21.epil = fadd double %dev_sq_sum.072.epil, %mul20.epil
  %indvars.iv.next.epil = add nuw nsw i64 %indvars.iv.epil, 1
  %epil.iter.sub = add i64 %epil.iter, -1
  %epil.iter.cmp = icmp eq i64 %epil.iter.sub, 0
  br i1 %epil.iter.cmp, label %if.end28, label %for.body12.epil, !llvm.loop !9

if.end28:                                         ; preds = %for.body12.epil, %if.end28.unr-lcssa
  %add21.lcssa = phi double [ %add21.lcssa.ph, %if.end28.unr-lcssa ], [ %add21.epil, %for.body12.epil ]
  %sub25 = add nsw i32 %size, -1
  %conv26 = sitofp i32 %sub25 to double
  %div27 = fdiv double %add21.lcssa, %conv26
  %mul29 = fmul double %conv5, 1.000000e-09
  %call30 = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([28 x i8], [28 x i8]* @.str.4, i64 0, i64 0), double %mul29)
  %cmp31 = fcmp une double %div27, 0.000000e+00
  br i1 %cmp31, label %if.then33, label %if.end39

if.then33:                                        ; preds = %if.end28
  %mul34 = fmul double %div27, 1.000000e-09
  %div35 = fdiv double %div27, %conv5
  %mul36 = fmul double %div35, 1.000000e-09
  %mul37 = fmul double %mul36, 1.000000e+02
  %call38 = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([26 x i8], [26 x i8]* @.str.5, i64 0, i64 0), double %mul34, double %mul37)
  br label %if.end39

if.end39:                                         ; preds = %if.end28.thread, %if.then33, %if.end28
  ret void
}

; Function Attrs: nounwind uwtable
define dso_local void @print_runtime_summary(i64* nocapture readonly %tm_elapsed, i32 %size) local_unnamed_addr #0 {
entry:
  tail call fastcc void @print_runtime_helper(i64* %tm_elapsed, i32 %size, i32 1)
  ret void
}

; Function Attrs: argmemonly norecurse nounwind uwtable writeonly
define dso_local void @identity_intsum(i8* nocapture readnone %reducer, i8* nocapture %sum) #5 {
entry:
  %0 = bitcast i8* %sum to i32*
  store i32 0, i32* %0, align 4, !tbaa !11
  ret void
}

; Function Attrs: argmemonly norecurse nounwind uwtable
define dso_local void @reduce_intsum(i8* nocapture readnone %reducer, i8* nocapture %left, i8* nocapture readonly %right) #6 {
entry:
  %0 = bitcast i8* %right to i32*
  %1 = load i32, i32* %0, align 4, !tbaa !11
  %2 = bitcast i8* %left to i32*
  %3 = load i32, i32* %2, align 4, !tbaa !11
  %add = add nsw i32 %3, %1
  store i32 %add, i32* %2, align 4, !tbaa !11
  ret void
}

; Function Attrs: nounwind uwtable
define dso_local i32 @compute_sum(i32 %limit) local_unnamed_addr #0 {
entry:
  %reducer = alloca reducer i32, !reduce !{void (i8*, i8*, i8*)* @reduce_intsum}, !identity !{void (i8*, i8*)* @identity_intsum}, !destroy !{void (i8*, i8*)* @__cilkrts_hyperobject_noop_destroy}, !alloc !{i8* (i8*, i64)* @__cilkrts_hyperobject_alloc}, !dealloc !{void (i8*, i8*)* @__cilkrts_hyperobject_dealloc}
  %syncreg = tail call token @llvm.syncregion.start()
  store i32 0, i32* %reducer, align 4, !tbaa !11
  %cmp = icmp sgt i32 %limit, 0
  br i1 %cmp, label %pfor.cond, label %cleanup

pfor.cond:                                        ; preds = %entry, %pfor.inc
  %__begin.0 = phi i32 [ %inc, %pfor.inc ], [ 0, %entry ]
  detach within %syncreg, label %pfor.body, label %pfor.inc

pfor.body:                                        ; preds = %pfor.cond
  %a3 = load i32, i32* %reducer, align 4, !tbaa !11
  %add6 = add nsw i32 %a3, 1
  store i32 %add6, i32* %reducer, align 4, !tbaa !11
  reattach within %syncreg, label %pfor.inc

pfor.inc:                                         ; preds = %pfor.body, %pfor.cond
  %inc = add nuw nsw i32 %__begin.0, 1
  %exitcond = icmp eq i32 %inc, %limit
  br i1 %exitcond, label %pfor.cond.cleanup, label %pfor.cond, !llvm.loop !13

pfor.cond.cleanup:                                ; preds = %pfor.inc
  sync within %syncreg, label %cleanup

cleanup:                                          ; preds = %pfor.cond.cleanup, %entry
  %a5 = load i32, i32* %reducer, align 4, !tbaa !11
  ret i32 %a5
}

declare dso_local void @__cilkrts_hyperobject_noop_destroy(i8*, i8*) #7

declare dso_local i8* @__cilkrts_hyperobject_alloc(i8*, i64) #7

declare dso_local void @__cilkrts_hyperobject_dealloc(i8*, i8*) #7

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1) #1

declare dso_local void @__cilkrts_hyper_create(%struct.__cilkrts_hyperobject_base*) local_unnamed_addr #7

declare dso_local i8* @__cilkrts_hyper_lookup(%struct.__cilkrts_hyperobject_base*) local_unnamed_addr #7

; Function Attrs: argmemonly nounwind
declare token @llvm.syncregion.start() #1

declare dso_local void @__cilkrts_hyper_destroy(%struct.__cilkrts_hyperobject_base*) local_unnamed_addr #7

; Function Attrs: nounwind uwtable
define dso_local i32 @main(i32 %argc, i8** nocapture readonly %args) local_unnamed_addr #0 {
entry:
  %temp.i21 = alloca <2 x i64>, align 16
  %tmpcast44 = bitcast <2 x i64>* %temp.i21 to %struct.timespec*
  %my_int_sum_reducer.i = alloca %struct.anon, align 64
  %syncreg.i = tail call token @llvm.syncregion.start()
  %temp.i = alloca <2 x i64>, align 16
  %cmp = icmp eq i32 %argc, 2
  br i1 %cmp, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  %0 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !15
  %1 = tail call i64 @fwrite(i8* getelementptr inbounds ([39 x i8], [39 x i8]* @.str.1, i64 0, i64 0), i64 38, i64 1, %struct._IO_FILE* %0) #9
  tail call void @exit(i32 1) #10
  unreachable

if.end:                                           ; preds = %entry
  %tmpcast = bitcast <2 x i64>* %temp.i to %struct.timespec*
  %arrayidx = getelementptr inbounds i8*, i8** %args, i64 1
  %2 = load i8*, i8** %arrayidx, align 8, !tbaa !15
  %call.i = tail call i64 @strtol(i8* nocapture nonnull %2, i8** null, i32 10) #8
  %conv.i = trunc i64 %call.i to i32
  %3 = bitcast <2 x i64>* %temp.i to i8*
  %4 = bitcast %struct.anon* %my_int_sum_reducer.i to i8*
  %__cilkrts_hyperbase.i = getelementptr inbounds %struct.anon, %struct.anon* %my_int_sum_reducer.i, i64 0, i32 0
  %5 = bitcast <2 x i64>* %temp.i21 to i8*
  %mul = shl nsw i32 %conv.i, 1
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %3) #8
  %call.i18 = call i32 @clock_gettime(i32 1, %struct.timespec* nonnull %tmpcast) #8
  %cmp.i = icmp eq i32 %call.i18, 0
  br i1 %cmp.i, label %ktiming_getmark.exit, label %if.then.i

if.then.i:                                        ; preds = %if.end
  call void @perror(i8* getelementptr inbounds ([18 x i8], [18 x i8]* @.str, i64 0, i64 0)) #11
  call void @exit(i32 -1) #10
  unreachable

ktiming_getmark.exit:                             ; preds = %if.end
  %cmp.i20 = icmp sgt i32 %conv.i, 0
  %6 = load <2 x i64>, <2 x i64>* %temp.i, align 16, !tbaa !8
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %3) #8
  call void @llvm.lifetime.start.p0i8(i64 128, i8* nonnull %4) #8
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 64 %4, i8* align 64 bitcast (%struct.anon* @__const.compute_sum.my_int_sum_reducer to i8*), i64 128, i1 false) #8
  call void @__cilkrts_hyper_create(%struct.__cilkrts_hyperobject_base* nonnull %__cilkrts_hyperbase.i) #8
  %call.i19 = call i8* @__cilkrts_hyper_lookup(%struct.__cilkrts_hyperobject_base* nonnull %__cilkrts_hyperbase.i) #8
  %7 = bitcast i8* %call.i19 to i32*
  store i32 0, i32* %7, align 4, !tbaa !11
  br i1 %cmp.i20, label %pfor.cond.i, label %compute_sum.exit

pfor.cond.i:                                      ; preds = %ktiming_getmark.exit, %pfor.inc.i
  %__begin.0.i = phi i32 [ %inc.i, %pfor.inc.i ], [ 0, %ktiming_getmark.exit ]
  detach within %syncreg.i, label %pfor.body.i, label %pfor.inc.i

pfor.body.i:                                      ; preds = %pfor.cond.i
  %call5.i = call i8* @__cilkrts_hyper_lookup(%struct.__cilkrts_hyperobject_base* nonnull %__cilkrts_hyperbase.i) #8
  %8 = bitcast i8* %call5.i to i32*
  %9 = load i32, i32* %8, align 4, !tbaa !11
  %add6.i = add nsw i32 %9, 1
  store i32 %add6.i, i32* %8, align 4, !tbaa !11
  reattach within %syncreg.i, label %pfor.inc.i

pfor.inc.i:                                       ; preds = %pfor.body.i, %pfor.cond.i
  %inc.i = add nuw nsw i32 %__begin.0.i, 1
  %exitcond.i = icmp eq i32 %inc.i, %conv.i
  br i1 %exitcond.i, label %pfor.cond.cleanup.i, label %pfor.cond.i, !llvm.loop !13

pfor.cond.cleanup.i:                              ; preds = %pfor.inc.i
  sync within %syncreg.i, label %compute_sum.exit

compute_sum.exit:                                 ; preds = %ktiming_getmark.exit, %pfor.cond.cleanup.i
  %call10.i = call i8* @__cilkrts_hyper_lookup(%struct.__cilkrts_hyperobject_base* nonnull %__cilkrts_hyperbase.i) #8
  %10 = bitcast i8* %call10.i to i32*
  %11 = load i32, i32* %10, align 4, !tbaa !11
  call void @__cilkrts_hyper_destroy(%struct.__cilkrts_hyperobject_base* nonnull %__cilkrts_hyperbase.i) #8
  call void @llvm.lifetime.end.p0i8(i64 128, i8* nonnull %4) #8
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %5) #8
  %call.i22 = call i32 @clock_gettime(i32 1, %struct.timespec* nonnull %tmpcast44) #8
  %cmp.i23 = icmp eq i32 %call.i22, 0
  br i1 %cmp.i23, label %ktiming_getmark.exit29, label %if.then.i24

if.then.i24:                                      ; preds = %compute_sum.exit
  call void @perror(i8* getelementptr inbounds ([18 x i8], [18 x i8]* @.str, i64 0, i64 0)) #11
  call void @exit(i32 -1) #10
  unreachable

ktiming_getmark.exit29:                           ; preds = %compute_sum.exit
  %cmp5 = icmp eq i32 %11, %mul
  %cond = zext i1 %cmp5 to i32
  %12 = load <2 x i64>, <2 x i64>* %temp.i21, align 16, !tbaa !8
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %5) #8
  %13 = sub <2 x i64> %12, %6
  %14 = extractelement <2 x i64> %13, i32 0
  %reass.mul = mul i64 %14, 1000000000
  %15 = extractelement <2 x i64> %13, i32 1
  %sub.i = add i64 %15, %reass.mul
  %call9 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([26 x i8], [26 x i8]* @.str.2, i64 0, i64 0), i32 %cond, i32 1)
  %conv.i31 = uitofp i64 %sub.i to double
  %mul.i = fmul double %conv.i31, 1.000000e-09
  %call.i32 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([22 x i8], [22 x i8]* @.str.3, i64 0, i64 0), i32 1, double %mul.i) #8
  %call3068.i = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([28 x i8], [28 x i8]* @.str.4, i64 0, i64 0), double %mul.i) #8
  ret i32 0
}

; Function Attrs: nounwind
declare dso_local i32 @printf(i8* nocapture readonly, ...) local_unnamed_addr #2

; Function Attrs: nounwind
declare dso_local i64 @strtol(i8* readonly, i8** nocapture, i32) local_unnamed_addr #2

; Function Attrs: nounwind
declare i64 @fwrite(i8* nocapture, i64, i64, %struct._IO_FILE* nocapture) local_unnamed_addr #8

attributes #0 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { noreturn nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { argmemonly norecurse nounwind readonly uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { argmemonly norecurse nounwind uwtable writeonly "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #6 = { argmemonly norecurse nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #7 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #8 = { nounwind }
attributes #9 = { cold }
attributes #10 = { noreturn nounwind }
attributes #11 = { cold nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 8.0.1 (git@github.com:wsmoses/Tapir-Clang 863c62072fe610ba33e3850a440ceb3c30059beb) (git@github.com:wsmoses/Tapir-LLVM f4afca879ec0ac46034779a8dbb61a695e4d8862)"}
!2 = !{!3, !4, i64 8}
!3 = !{!"timespec", !4, i64 0, !4, i64 8}
!4 = !{!"long", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C/C++ TBAA"}
!7 = !{!3, !4, i64 0}
!8 = !{!4, !4, i64 0}
!9 = distinct !{!9, !10}
!10 = !{!"llvm.loop.unroll.disable"}
!11 = !{!12, !12, i64 0}
!12 = !{!"int", !5, i64 0}
!13 = distinct !{!13, !14}
!14 = !{!"tapir.loop.spawn.strategy", i32 1}
!15 = !{!16, !16, i64 0}
!16 = !{!"any pointer", !5, i64 0}

