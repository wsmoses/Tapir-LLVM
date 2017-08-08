//===------------- LoopFuse.cpp - Loop Fusion Pass ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// Fuse two adjacent loops to improve cache locality. Loops are multi-versioned
/// and unconditionally fused along one version to check for dependence
/// legality. Legality decides whether to keep the original version or the fused
/// version or both versions with runtime checks. LoopAccessLegacyAnalysis is used to
/// check dependence legality.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/LoopFuse.h"

#define DEBUG_TYPE "loop-fuse"

using namespace llvm;

static cl::opt<bool>
    LFuseVerify("loop-fuse-verify", cl::Hidden,
                cl::desc("Turn on DominatorTree and LoopInfo verification "
                         "after Loop Fusion"),
                cl::init(false));

STATISTIC(NumLoopsFused, "Number of loops fused");

// Replace IncomingBlocks in PHI nodes of @Br successors from Br's parent to
// @To.
void LoopFuse::RewritePHI(BranchInst *Br, BasicBlock *To) {
  assert((Br && To));
  for (auto *S : Br->successors()) {
    auto I = S->begin();
    while (PHINode *P = dyn_cast<PHINode>(&*I)) {
      P->setIncomingBlock(P->getBasicBlockIndex(Br->getParent()), To);
      ++I;
    }
  }
}

//===----------------------------------------------------------------------===//
//                     Loop Fusion Implementation.
// The idea to check fusion legality is by first fusing the loops and then look
// for fusion preventing dependences. This is done by versioning the loops
// first. The check is done on versioned loops and one of the version is
// discarded based on legality's success.
//===----------------------------------------------------------------------===//

/* Fuse loops @L1 and @L2. Remove ConnectingBlock (CB) and connect L1Latch to
   L2Header. Loop from L2Latch to L1Header. Make L1's indvar as indvar for the
   fused loop. Update LI by moving L2Blocks into L1 and call L1 as FusedLoop.
   Return FusedLoop.
    L1
    |               L1Blocks
    CB     -->      |       \
    |               L2Blocks/
    L2                    |/
*/
Loop *LoopFuse::FuseLoops(Loop &L1, Loop &L2) {
  PHINode *P1 = L1.getCanonicalInductionVariable();
  PHINode *P2 = L2.getCanonicalInductionVariable();

  BranchInst *Br1 = dyn_cast<BranchInst>(L1.getLoopLatch()->getTerminator());
  BranchInst *Br2 = dyn_cast<BranchInst>(L2.getLoopLatch()->getTerminator());

  // Make Br2 to branch to L1 header based on Br1's condition.
  unsigned LoopBack = 0;
  if (Br2->getSuccessor(1) == L2.getHeader())
    LoopBack = 1;
  assert((Br2->getSuccessor(LoopBack) == L2.getHeader()));
  Br2->setSuccessor(LoopBack, L1.getHeader());
  Br2->setCondition(Br1->getCondition());
  RewritePHI(Br1, Br2->getParent());

  // Zap L2 preheader and unconditionally branch from L1 latch to L2 header.
  // L2 preheader is a connecting block and it is known to contain only an
  // unconditional branch to L2 header.
  BasicBlock *L2PH = L2.getLoopPreheader(), *L2Header = L2.getHeader();
  BranchInst *L2PHBr = dyn_cast<BranchInst>(L2PH->getTerminator());
  RewritePHI(L2PHBr, Br1->getParent());
  DT->changeImmediateDominator(L2Header, L1.getLoopLatch());

  BranchInst::Create(L2Header, Br1);
  Br1->eraseFromParent();
  L2PH->dropAllReferences();
  L2PHBr->eraseFromParent();
  L2PH->eraseFromParent();
  DT->eraseNode(L2PH);
  LI->removeBlock(L2PH);

  P2->replaceAllUsesWith(P1);
  P2->eraseFromParent();

  // Update LI.
  // Move all blocks from L2 to L1.
  SmallVector<BasicBlock *, 2> L2BBs;
  for (auto bb = L2.block_begin(), bbe = L2.block_end(); bb != bbe; ++bb)
    L2BBs.push_back(*bb);
  for (auto *bb : L2BBs) {
    LI->removeBlock(bb);
    L1.addBasicBlockToLoop(bb, *LI);
  }
  // Remove L2.
  SE->forgetLoop(&L2);
  LI->markAsRemoved(&L2);

  // Update DT: DT changed only at L2PH zap and was updated during zapping.

  return &L1;
}

/*  Version the given loops along a parallel path and fuse the cloned loops.
    Check the dependence legality of the fused loop.

    L1PH                       BooleanBB                       BooleanBB
    |                             /\                              /\
    L1                        L1PH  L1PH.clone                L1PH  FusedPH
    |                version  |     |            Fuse along      |  |
    CB (L1Exit/L2PH)  ---->   L1    L1.clone     -------->      L1  L1Blocks
    |                         |     |            versioned       |  |       \
    L2                        CB    CB.clone     path           CB  L2Blocks |
    |                         |     |                            |  |      |/
    L2Exit                    L2    L2.clone                    L2  |
                                \  /                              \ /
                                L2Exit                         CommonExit
   CB is ConnectingBlock.
*/
bool LoopFuse::DependenceLegal(Loop &L1, Loop &L2) {

  // Version to fuse. LoopVersioning is not used here because:
  // a. Runtime checks are inserted later.
  // b. Intermediate VMap updates are required.
  // Moreover it is convenient for now to just clone and remap.
  BasicBlock *BooleanBB = L1.getLoopPreheader();
  BasicBlock *L1PH = SplitEdge(BooleanBB, L1.getHeader(), DT, LI);

  ValueToValueMapTy VMap1;
  SmallVector<BasicBlock *, 2> ClonedBBs1;
  Loop *ClonedLoop1 =
      cloneLoopWithPreheader(L1.getExitBlock(), BooleanBB, &L1, VMap1,
                             Twine(".L1clone"), LI, DT, ClonedBBs1);

  ValueToValueMapTy VMap2;
  SmallVector<BasicBlock *, 2> ClonedBBs2;
  Loop *ClonedLoop2 =
      cloneLoopWithPreheader(L2.getExitBlock(), L1.getExitBlock(), &L2, VMap2,
                             Twine(".L2clone"), LI, DT, ClonedBBs2);
  remapInstructionsInBlocks(ClonedBBs2, VMap2);
  VMap1[L1.getExitBlock()] = ClonedLoop2->getLoopPreheader();
  remapInstructionsInBlocks(ClonedBBs1, VMap1);

  // Build the custom VMap by concatenating VMap1 and VMap2.
  for (auto V : VMap1)
    VMap[V->first] = V->second;
  for (auto V : VMap2)
    VMap[V->first] = V->second;

  // VMap.size() != VMap1.size() + VMap2.size() because of redundants and
  // L1Exit update in VMap1 above.

  // Branch to either of the versions - using a boolean flag.
  Instruction *Term = BooleanBB->getTerminator();
  FusionSwitcher =
      BranchInst::Create(L1PH, ClonedLoop1->getLoopPreheader(),
                         ConstantInt::getTrue(L1PH->getContext()), Term);
  Term->eraseFromParent();

  // The two versions join back at L2 exit. Update DT.
  if (DT->dominates(L2.getLoopLatch(), L2.getExitBlock()))
    DT->changeImmediateDominator(L2.getExitBlock(), BooleanBB);

  DEBUG(dbgs() << "ClonedLoop1: " << *ClonedLoop1 << "\n");
  DEBUG(dbgs() << "ClonedLoop2: " << *ClonedLoop2 << "\n");

  FusedLoop = FuseLoops(*ClonedLoop1, *ClonedLoop2);
  DEBUG(dbgs() << "FusedLoop: " << *FusedLoop << "\n");

  // Check dependences.
  DEBUG(dbgs() << "Loop fused on versioned path. Checking dependences...\n");
  LAI = &LAA->getInfo(FusedLoop);
  DEBUG(LAI->print(dbgs()));

  auto Dependences = LAI->getDepChecker().getDependences();
  // TODO@jiahao: Investigate.
  // if (!Dependences || Dependences->empty()) {
  //   DEBUG(dbgs() << "Failed to get dependences to check fusion legality!"
  //                << " Skipping...\n");
  //   return false;
  // }

  // Fusion is illegal if there is a backward dependence between memory accesses
  // whose source was in L1 and sink was in L2. ClonedBBs1 and ClonedBBs2
  // contain cloned BBs from L1 and L2 respectively. They are used to check the
  // containment of srouce and sink.
  for (auto &Dep : *Dependences) {
    if (Dep.isPossiblyBackward()) {
      Instruction *Source = Dep.getSource(*LAI);
      Instruction *Sink = Dep.getDestination(*LAI);
      if (std::find(ClonedBBs1.begin(), ClonedBBs1.end(),
                    Source->getParent()) == ClonedBBs1.end())
        continue;
      if (std::find(ClonedBBs2.begin(), ClonedBBs2.end(), Sink->getParent()) ==
          ClonedBBs2.end())
        continue;
      DEBUG(dbgs() << "Loop carried backward dependence prevents fusion!\n");
      return false;
    }
  }
  DEBUG(dbgs() << "Loops are dependence legal to fuse!\n");
  return true;
}

// Return true if any of the defs made in @L1 is used inside @L2.
bool LoopFuse::DefsUsedAcrossLoops(Loop &L1, Loop &L2) {
  auto DefsUsedOutsideL1 = findDefsUsedOutsideOfLoop(&L1);
  for (auto *D : DefsUsedOutsideL1) {
    for (auto *U : D->users()) {
      if (L2.contains(dyn_cast<Instruction>(U)->getParent()))
        return true;
    }
  }
  return false;
}

bool LoopFuse::IsLegalAndProfitable(Loop &L1, Loop &L2) {
  // Basic legality.
  if (!L1.empty() || !L2.empty()) {
    // TODO: Update cloneLoopWithPreheader() to update LoopInfo for subloops
    // too and LoopFusion can be done for loops at any depth.
    DEBUG(dbgs() << "Not innermost loops! Skipping...\n");
    return false;
  }

  if (L1.getLoopDepth() != L2.getLoopDepth()) {
    DEBUG(dbgs() << "Loops not at same depth! Skipping...\n");
    return false;
  }

  if (!L1.getLoopPreheader() || !L2.getLoopPreheader()) {
    DEBUG(dbgs() << "No preheader! Skipping...\n");
    return false;
  }

  if (!L1.getExitBlock() || !L2.getExitBlock()) {
    DEBUG(dbgs() << "Single exit block not found! Skipping...\n");
    return false;
  }

  // Can fuse only bottom-tested loops and loops with latch being the single
  // exiting block.
  if ((L1.getExitingBlock() != L1.getLoopLatch()) ||
      (L2.getExitingBlock() != L2.getLoopLatch())) {
    DEBUG(dbgs() << "Not a bottom-tested loop! Skipping...\n");
    return false;
  }

  // Can fuse only adjacent loops. Adjacency is defined by:
  // a. L1Exit has single entry only from L1Latch.
  // b. L1Exit and L2Preheader are same i.e the block forms the ConnectingBlock.
  // c. ConnectingBlock just branches unconditionally to L2Header.
  auto *Br = dyn_cast<BranchInst>(L1.getExitBlock()->begin());
  if ((L1.getExitBlock()->getSinglePredecessor() != L1.getLoopLatch()) ||
      (L1.getExitBlock() != L2.getLoopPreheader()) ||
      (!Br || Br->isConditional())) {
    DEBUG(dbgs() << "Loops not adjacent! Skipping...\n");
    return false;
  }

  // Indvars of both loops is known and canonicalized.
  PHINode *P1 = L1.getCanonicalInductionVariable();
  PHINode *P2 = L2.getCanonicalInductionVariable();
  if (!P1 || !P2) {
    DEBUG(dbgs() << "Unknown induction variables! Skipping...\n");
    return false;
  }

  // P1 and P2 are canonical indvars. Backedge taken count check is enough to
  // ascertain both loops have same iteration space.
  if (SE->getBackedgeTakenCount(&L1) != SE->getBackedgeTakenCount(&L2))
    return false;

  // Cannot fuse if there are uses of L1 defs in L2.
  if (DefsUsedAcrossLoops(L1, L2))
    return false;

  // Dependene based legality.
  if (!DependenceLegal(L1, L2))
    return false;

  // TODO: Add profitability measures.

  return true;
}

// Remove Loop @L completely by deleting the BBs and also from @LI, @DT and @SE
// including preheader. Finally connect the single predecessor (the BooleanBB
// that contains FusionSwitcher) of preheader to loop exit.
void LoopFuse::RemoveLoopCompletelyWithPreheader(Loop &L) {
  DEBUG(dbgs() << "Removing loop: " << L << "\n");
  BasicBlock *PH = L.getLoopPreheader();
  BasicBlock *Exit = L.getExitBlock();
  assert(Exit && "Expected Exit bb and single pred to preheader!");

  // No need to RewritePHIs of Exit block given the Loop is deleted because the
  // uses remain same if FusedLoop is removed OR uses are already replaced if
  // original loops are deleted.

  // Branch to Exit block from FusionSwitcher.
  unsigned SuccNum = 0;
  if (FusionSwitcher->getSuccessor(1) == PH)
    SuccNum = 1;
  assert((FusionSwitcher->getSuccessor(SuccNum) == PH));
  FusionSwitcher->setSuccessor(SuccNum, Exit);
  if (DT->dominates(L.getLoopLatch(), Exit)) // L1 removal case.
    // Exit blocks iDom is FusionSwitcher's block due to versioning.
    DT->changeImmediateDominator(Exit, FusionSwitcher->getParent());

  // Erase each of the loop blocks. Update SE, DT and LI.
  SE->forgetLoop(&L);
  PH->dropAllReferences();
  for (auto bb = L.block_begin(), bbe = L.block_end(); bb != bbe; ++bb) {
    DT->changeImmediateDominator(*bb, PH);
    (*bb)->dropAllReferences();
  }

  PH->eraseFromParent();
  for (auto bb = L.block_begin(), bbe = L.block_end(); bb != bbe; ++bb) {
    // Now nuke bb and its DT.
    (*bb)->eraseFromParent();
    DT->eraseNode(*bb);
  }
  DT->eraseNode(PH);

  SmallVector<BasicBlock *, 2> LBBs;
  for (auto bb = L.block_begin(), bbe = L.block_end(); bb != bbe; ++bb)
    LBBs.push_back(*bb);
  for (auto *bb : LBBs)
    LI->removeBlock(bb);
  if (LI->getLoopFor(PH))
    LI->removeBlock(PH);

  LI->markAsRemoved(&L);
}

// Remove FusionSwitcher and branch directly to given loop @L's header. This
// removes loop's preheader and make FusionSwitcher's block as preheader.
void LoopFuse::RemoveFusionSwitcher(Loop &L) {
  assert(FusionSwitcher->isConditional());
  DEBUG(dbgs() << "Removing FusionSwitcher: " << *FusionSwitcher << "\n");

  BasicBlock *PH = L.getLoopPreheader();
  assert((PH->size() == 1));

  BranchInst *PHBr = dyn_cast<BranchInst>(PH->getTerminator());
  assert(PHBr->isUnconditional());

  RewritePHI(PHBr, FusionSwitcher->getParent());

  PHBr->removeFromParent();
  PHBr->insertBefore(FusionSwitcher);
  DT->changeImmediateDominator(L.getHeader(), FusionSwitcher->getParent());

  FusionSwitcher->eraseFromParent();
  PH->eraseFromParent();
  DT->eraseNode(PH);
  if (LI->getLoopFor(PH))
    LI->removeBlock(PH);
}

// Update the uses of defs that reach outside original loop with the defs made
// made in fused loop.
void LoopFuse::UpdateUsesOutsideLoop(Loop &L) {
  for (auto *D : findDefsUsedOutsideOfLoop(&L)) {
    auto VI = VMap.find(D);
    if (VI == VMap.end())
      continue;

    for (auto *U : D->users()) {
      if (!L.contains(dyn_cast<Instruction>(U)->getParent())) {
        if (auto *P = dyn_cast<PHINode>(U)) {
          // Replace U in PHI with <VMap(D), FusedLoopLatch>
          for (unsigned i = 0, e = P->getNumIncomingValues(); i != e; ++i) {
            if (P->getIncomingValue(i) == U) {
              P->removeIncomingValue(i);
              P->addIncoming(VI->second, FusedLoop->getLoopLatch());
            }
          }
        } else
          U->replaceUsesOfWith(D, VI->second);
      }
    }
  }
}

// Add/update phi for defs that reach uses outside the loop from original loop
// @L and from fused loop. Insert the phis into fused loop's exit block, which
// is also the exit block of original L2 loop. @OrigIncomingBlock refers to the
// block from where a def is reached outside of loop - L2 latch.
// TODO: This routine is similar to LoopVersioning's addPHINodes(), but
// rewritten here as access to internal data structures differ.
void LoopFuse::AddPHIsOutsideLoop(Loop &L, BasicBlock *OrigIncomingBlock) {
  BasicBlock *PHIBlock = FusedLoop->getExitBlock();
  assert(PHIBlock && "Unable to find FusedLoop's ExitBlock!");

  for (auto *Inst : findDefsUsedOutsideOfLoop(&L)) {
    PHINode *PN = nullptr;
    auto FusedInst = VMap.find(Inst);
    assert((FusedInst != VMap.end()) &&
           "Expected an equivalent instruction in fused loop!");
    // Update/add phi node for this Inst.
    bool FoundInst = false;
    for (auto I = PHIBlock->begin(); !FoundInst && (PN = dyn_cast<PHINode>(I));
         ++I) {
      for (unsigned i = 0, e = PN->getNumIncomingValues(); !FoundInst && i != e;
           ++i)
        if (PN->getIncomingValue(i) == Inst)
          FoundInst = true;
    }
    if (!PN) {
      PN = PHINode::Create(Inst->getType(), 2, Inst->getName() + ".lfuse",
                           &PHIBlock->front());

      for (auto *U : Inst->users())
        if (!L.contains(dyn_cast<Instruction>(U)->getParent()))
          U->replaceUsesOfWith(Inst, PN);

      PN->addIncoming(Inst, OrigIncomingBlock);
    }
    // Add incoming value from fused loop.
    PN->addIncoming(FusedInst->second, FusedLoop->getLoopLatch());
  }
}

bool LoopFuse::run(Loop &L1, Loop &L2) {
  assert((LI && LAA && DT && SE));
  DEBUG(dbgs() << "\nTrying to fuse:\n" << L1 << "AND\n" << L2 << "\n");

  FusionSwitcher = nullptr;
  FusedLoop = nullptr;
  VMap.clear();
  bool Changed = false;
  if (IsLegalAndProfitable(L1, L2)) {
    assert((FusedLoop && FusionSwitcher));
    auto *RuntimePtrChecks = LAI->getRuntimePointerChecking();
    if (RuntimePtrChecks->Need) {
      // Add runtime checks and add/update phis in exit block for the defs
      // reaching from two versions.
      Instruction *FirstCheck, *LastCheck;
      std::tie(FirstCheck, LastCheck) = LAI->addRuntimeChecks(FusionSwitcher);
      // TODO: Add SCEVRuntime checks?
      FusionSwitcher->setCondition(LastCheck);

      AddPHIsOutsideLoop(L1, L2.getLoopLatch());
      AddPHIsOutsideLoop(L2, L2.getLoopLatch());
      FusionKind = VERSIONED_FUSION;

    } else {
      // Remove original loops and retain FusedLoop. Also update the uses of
      // defs from original loops with the defs from fused loop.
      UpdateUsesOutsideLoop(L1);
      UpdateUsesOutsideLoop(L2);
      RemoveLoopCompletelyWithPreheader(L1);
      RemoveLoopCompletelyWithPreheader(L2);

      // Remove FusionSwitcher and directly point to FusedLoop header.
      if (DT->dominates(FusionSwitcher->getParent(), FusedLoop->getExitBlock()))
        DT->changeImmediateDominator(FusedLoop->getExitBlock(),
                                     FusedLoop->getLoopLatch());
      RemoveFusionSwitcher(*FusedLoop);
      FusionKind = PURE_FUSION;
    }
    ++NumLoopsFused;
    Changed = true;

  } else {
    if (FusedLoop) {
      // Loops were versioned to check legality. Rollback to original state.
      RemoveLoopCompletelyWithPreheader(*FusedLoop);

      // Remove FusionSwitcher and directly point to L1 header.
      if (DT->dominates(FusionSwitcher->getParent(), L2.getExitBlock()))
        DT->changeImmediateDominator(L2.getExitBlock(), L2.getLoopLatch());
      RemoveFusionSwitcher(L1);
      FusionKind = REVERTED_FUSION;
    }
  }

  if (LFuseVerify) {
    LI->verify(*DT);
    DT->verifyDomTree();
  }

  return Changed;
}

void PopulateInnermostLoopsOf(Loop &L, SmallVectorImpl<Loop *> &Loops) {
  if (L.empty())
    Loops.push_back(&L);
  for (auto I = L.begin(), E = L.end(); I != E; ++I)
    PopulateInnermostLoopsOf(**I, Loops);
}

bool LoopFuse::runOnFunction(Function &F) {
  LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  LAA = &getAnalysis<LoopAccessLegacyAnalysis>();
  DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  SE = &getAnalysis<ScalarEvolutionWrapperPass>().getSE();

  // Populate innermost loops and try a n^2 combination of loop fusion.
  bool Changed = false;
  SmallVector<Loop *, 2> Loops;
  for (auto L = LI->begin(), Le = LI->end(); L != Le; ++L)
    PopulateInnermostLoopsOf(**L, Loops);

  auto L1 = Loops.begin(), L1e = Loops.end();
  while (L1 != L1e) {
    auto L2 = Loops.begin(), L2e = Loops.end();
    while (L2 != L2e) {
      if (L1 == L2) {
        ++L2;
        continue;
      }
      if (run(**L1, **L2)) {
        // Remove L1 and L2 from Loops and add FusedLoop.
        Loops.erase(L1);
        Loops.erase(L2);
        Loops.push_back(FusedLoop);
        L1 = L2 = Loops.begin();
        L1e = L2e = Loops.end();
        Changed = true;
      } else
        ++L2;
    }
    ++L1;
  }

  if (LFuseVerify) {
    LI->verify(*DT);
    DT->verifyDomTree();
    assert((!verifyFunction(F, &dbgs())) && "Function verification failed!");
  }

  return Changed;
}

char LoopFuse::ID;

INITIALIZE_PASS_BEGIN(LoopFuse, "loop-fuse", "Loop Fusion", false, false)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopAccessLegacyAnalysis)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolutionWrapperPass)
INITIALIZE_PASS_END(LoopFuse, "loop-fuse", "Loop Fusion", false, false)

namespace llvm {
FunctionPass *createLoopFusePass() { return new LoopFuse(); }
}
