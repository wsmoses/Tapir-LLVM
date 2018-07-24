/// Used to store information PPCG wants for kills. This information is
/// used by live range reordering.
///
/// @see computeLiveRangeReordering
/// @see GPUNodeBuilder::createPPCGScop
/// @see GPUNodeBuilder::createPPCGProg
struct MustKillsInfo {
  /// Collection of all kill statements that will be sequenced at the end of
  /// PPCGScop->schedule.
  ///
  /// The nodes in `KillsSchedule` will be merged using `isl_schedule_set`
  /// which merges schedules in *arbitrary* order.
  /// (we don't care about the order of the kills anyway).
  isl::schedule KillsSchedule;
  /// Map from kill statement instances to scalars that need to be
  /// killed.
  ///
  /// We currently derive kill information for:
  ///  1. phi nodes. PHI nodes are not alive outside the scop and can
  ///     consequently all be killed.
  ///  2. Scalar arrays that are not used outside the Scop. This is
  ///     checked by `isScalarUsesContainedInScop`.
  /// [params] -> { [Stmt_phantom[] -> ref_phantom[]] -> scalar_to_kill[] }
  isl::union_map TaggedMustKills;

  MustKillsInfo() : KillsSchedule(nullptr), TaggedMustKills(nullptr){};
};

/// Check if SAI's uses are entirely contained within Scop S.
/// If a scalar is used only with a Scop, we are free to kill it, as no data
/// can flow in/out of the value any more.
/// @see computeMustKillsInfo
static bool isScalarUsesContainedInScop(const Scop &S,
                                        const ScopArrayInfo *SAI) {
  assert(SAI->isValueKind() && "this function only deals with scalars."
                               " Dealing with arrays required alias analysis");

  const Region &R = S.getRegion();
  for (User *U : SAI->getBasePtr()->users()) {
    Instruction *I = dyn_cast<Instruction>(U);
    assert(I && "invalid user of scop array info");
    if (!R.contains(I))
      return false;
  }
  return true;
}

/// Compute must-kills needed to enable live range reordering with PPCG.
///
/// @params S The Scop to compute live range reordering information
/// @returns live range reordering information that can be used to setup
/// PPCG.
static MustKillsInfo computeMustKillsInfo(const Scop &S) {
  const isl::space ParamSpace(isl::manage(S.getParamSpace()));
  MustKillsInfo Info;

  // 1. Collect all ScopArrayInfo that satisfy *any* of the criteria:
  //      1.1 phi nodes in scop.
  //      1.2 scalars that are only used within the scop
  SmallVector<isl::id, 4> KillMemIds;
  for (ScopArrayInfo *SAI : S.arrays()) {
    if (SAI->isPHIKind() ||
        (SAI->isValueKind() && isScalarUsesContainedInScop(S, SAI)))
      KillMemIds.push_back(isl::manage(SAI->getBasePtrId()));
  }

  Info.TaggedMustKills = isl::union_map::empty(isl::space(ParamSpace));

  // Initialising KillsSchedule to `isl_set_empty` creates an empty node in the
  // schedule:
  //     - filter: "[control] -> { }"
  // So, we choose to not create this to keep the output a little nicer,
  // at the cost of some code complexity.
  Info.KillsSchedule = nullptr;

  for (isl::id &ToKillId : KillMemIds) {
    isl::id KillStmtId = isl::id::alloc(
        S.getIslCtx(),
        std::string("SKill_phantom_").append(ToKillId.get_name()), nullptr);

    // NOTE: construction of tagged_must_kill:
    // 2. We need to construct a map:
    //     [param] -> { [Stmt_phantom[] -> ref_phantom[]] -> scalar_to_kill[] }
    // To construct this, we use `isl_map_domain_product` on 2 maps`:
    // 2a. StmtToScalar:
    //         [param] -> { Stmt_phantom[] -> scalar_to_kill[] }
    // 2b. PhantomRefToScalar:
    //         [param] -> { ref_phantom[] -> scalar_to_kill[] }
    //
    // Combining these with `isl_map_domain_product` gives us
    // TaggedMustKill:
    //     [param] -> { [Stmt[] -> phantom_ref[]] -> scalar_to_kill[] }

    // 2a. [param] -> { Stmt[] -> scalar_to_kill[] }
    isl::map StmtToScalar = isl::map::universe(isl::space(ParamSpace));
    StmtToScalar = StmtToScalar.set_tuple_id(isl::dim::in, isl::id(KillStmtId));
    StmtToScalar = StmtToScalar.set_tuple_id(isl::dim::out, isl::id(ToKillId));

    isl::id PhantomRefId = isl::id::alloc(
        S.getIslCtx(), std::string("ref_phantom") + ToKillId.get_name(),
        nullptr);

    // 2b. [param] -> { phantom_ref[] -> scalar_to_kill[] }
    isl::map PhantomRefToScalar = isl::map::universe(isl::space(ParamSpace));
    PhantomRefToScalar =
        PhantomRefToScalar.set_tuple_id(isl::dim::in, PhantomRefId);
    PhantomRefToScalar =
        PhantomRefToScalar.set_tuple_id(isl::dim::out, ToKillId);

    // 2. [param] -> { [Stmt[] -> phantom_ref[]] -> scalar_to_kill[] }
    isl::map TaggedMustKill = StmtToScalar.domain_product(PhantomRefToScalar);
    Info.TaggedMustKills = Info.TaggedMustKills.unite(TaggedMustKill);

    // 3. Create the kill schedule of the form:
    //     "[param] -> { Stmt_phantom[] }"
    // Then add this to Info.KillsSchedule.
    isl::space KillStmtSpace = ParamSpace;
    KillStmtSpace = KillStmtSpace.set_tuple_id(isl::dim::set, KillStmtId);
    isl::union_set KillStmtDomain = isl::set::universe(KillStmtSpace);

    isl::schedule KillSchedule = isl::schedule::from_domain(KillStmtDomain);
    if (Info.KillsSchedule)
      Info.KillsSchedule = Info.KillsSchedule.set(KillSchedule);
    else
      Info.KillsSchedule = KillSchedule;
  }

  return Info;
}