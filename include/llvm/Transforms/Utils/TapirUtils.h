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


/// Utility class for getting and setting loop spawning hints in the form
/// of loop metadata.
/// This class keeps a number of loop annotations locally (as member variables)
/// and can, upon request, write them back as metadata on the loop. It will
/// initially scan the loop for existing metadata, and will update the local
/// values based on information in the loop.
class LoopSpawningHints {
public:
  enum SpawningStrategy {
    ST_SEQ,
    ST_DAC,
    ST_END,
  };

private:
  enum HintKind { HK_STRATEGY, HK_GRAINSIZE };

  /// Hint - associates name and validation with the hint value.
  struct Hint {
    const char *Name;
    unsigned Value; // This may have to change for non-numeric values.
    HintKind Kind;

    Hint(const char *Name, unsigned Value, HintKind Kind)
        : Name(Name), Value(Value), Kind(Kind) {}

    bool validate(unsigned Val);
  };

  /// Spawning strategy
  Hint Strategy;
  /// Grainsize
  Hint Grainsize;

  /// Return the loop metadata prefix.
  static inline StringRef Prefix() { return "tapir.loop."; }

public:
  static inline std::string printStrategy(enum SpawningStrategy Strat) {
    switch(Strat) {
    case LoopSpawningHints::ST_SEQ:
      return "Spawn iterations sequentially";
    case LoopSpawningHints::ST_DAC:
      return "Use divide-and-conquer";
    case LoopSpawningHints::ST_END:
    default:
      return "Unknown";
    }
  }

  LoopSpawningHints(const Loop *L);

  // /// Dumps all the hint information.
  // std::string emitRemark() const {
  //   LoopSpawningReport R;
  //   R << "Strategy = " << printStrategy(getStrategy());

  //   return R.str();
  // }

  SpawningStrategy getStrategy() const;

  unsigned getGrainsize() const;

private:
  /// Find hints specified in the loop metadata and update local values.
  void getHintsFromMetadata();

  /// Checks string hint with one operand and set value if valid.
  void setHint(StringRef Name, Metadata *Arg);

  /// Create a new hint from name / value pair.
  MDNode *createHintMetadata(StringRef Name, unsigned V) const;

  /// Matches metadata with hint name.
  bool matchesHintMetadataName(MDNode *Node, ArrayRef<Hint> HintTypes);

  /// Sets current hints into loop metadata, keeping other values intact.
  void writeHintsToMetadata(ArrayRef<Hint> HintTypes);

  /// The loop these hints belong to.
  const Loop *TheLoop;
};

/// Checks if this loop is a Tapir loop.  Right now we check that the loop is
/// in a canonical form:
/// 1) The header detaches the body.
/// 2) The loop contains a single latch.
/// 3) The body reattaches to the latch (which is necessary for a valid
///    detached CFG).
/// 4) The loop only branches to the exit block from the header or the latch.
bool isCanonicalTapirLoop(const Loop *L, bool print = false);

//! Identify if a loop could be a DAC loop
bool isDACFor(Loop* L);

}  // end llvm namespace

#endif
