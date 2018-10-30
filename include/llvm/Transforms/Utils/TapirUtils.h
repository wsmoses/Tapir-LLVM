//===-- TapirUtils.h - Utility methods for Tapir ---------------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file utility methods for handling code containing Tapir instructions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_TAPIRUTILS_H
#define LLVM_TRANSFORMS_UTILS_TAPIRUTILS_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"

namespace llvm {

class BasicBlock;
class DetachInst;
class DominatorTree;
class TerminatorInst;
class Loop;

/// Move static allocas in a block into the specified entry block.  Leave
/// lifetime markers behind for those static allocas.  Returns true if the
/// cloned block still contains dynamic allocas, which cannot be moved.
bool MoveStaticAllocasInBlock(
    BasicBlock *Entry, BasicBlock *Block,
    SmallVectorImpl<Instruction *> &ExitPoints);

/// Serialize the sub-CFG detached by the specified detach
/// instruction.  Removes the detach instruction and returns a pointer
/// to the branch instruction that replaces it.
BranchInst* SerializeDetachedCFG(DetachInst *DI, DominatorTree *DT = nullptr);

/// Get the entry basic block to the detached context that contains
/// the specified block.
const BasicBlock *GetDetachedCtx(const BasicBlock *BB);
BasicBlock *GetDetachedCtx(BasicBlock *BB);

/// isCriticalContinueEdge - Return true if the specified edge is a critical
/// detach-continue edge.  Critical detach-continue edges are critical edges -
/// from a block with multiple successors to a block with multiple predecessors
/// - even after ignoring all reattach edges.
bool isCriticalContinueEdge(const TerminatorInst *TI, unsigned SuccNum);

/// Checks if this loop is a Tapir loop.  Right now we check that the loop is
/// in a canonical form:
/// 1) The header detaches the body.
/// 2) The loop contains a single latch.
/// 3) The body reattaches to the latch (which is necessary for a valid
///    detached CFG).
/// 4) The loop only branches to the exit block from the header or the latch.
bool isCanonicalTapirLoop(const Loop *L, bool print = false);

/// canDetach - Return true if the given function can perform a detach, false
/// otherwise.
bool canDetach(const Function *F);

}  // end llvm namespace

#endif
