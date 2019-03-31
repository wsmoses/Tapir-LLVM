//===-- Tapir.h - Tapir Transformations -------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes for accessor functions that expose passes
// in the Tapir transformations library.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_TAPIR_H
#define LLVM_TRANSFORMS_TAPIR_H

#include <functional>

namespace llvm {
class FunctionPass;
class ImmutablePass;
class ModulePass;
class Pass;
class TapirTarget;
enum class TapirTargetID;

//===----------------------------------------------------------------------===//
//
// LoopSpawning - Create a loop spawning pass.
//
Pass *createLoopSpawningPass();

//===----------------------------------------------------------------------===//
//
// LoopSpawningTI - Create a loop spawning pass that uses Task Info.
//
Pass *createLoopSpawningTIPass();

//===----------------------------------------------------------------------===//
//
// SmallBlock - Do SmallBlock Pass
//
FunctionPass *createSmallBlockPass();

//===----------------------------------------------------------------------===//
//
// SyncElimination - TODO
//
FunctionPass *createSyncEliminationPass();

//===----------------------------------------------------------------------===//
//
// RedundantSpawn - Do RedundantSpawn Pass
//
FunctionPass *createRedundantSpawnPass();

//===----------------------------------------------------------------------===//
//
// SpawnRestructure - Do SpawnRestructure Pass
//
FunctionPass *createSpawnRestructurePass();

//===----------------------------------------------------------------------===//
//
// SpawnUnswitch - Do SpawnUnswitch Pass
//
FunctionPass *createSpawnUnswitchPass();

//===----------------------------------------------------------------------===//
//
// LowerTapirToTarget - Lower Tapir constructs to a specified parallel runtime.
//
ModulePass *createLowerTapirToTargetPass();

// A wrapper pass around a callback which can be used to produce a Tapir target
// from an external source.
//
// TODO: Determine what arguments if any to pass to this callback
ImmutablePass *createExternalTapirTargetWrapperPass(
    std::function<TapirTarget *(void)> Callback);

//===----------------------------------------------------------------------===//
//
//
FunctionPass *createAnalyzeTapirPass();

//===----------------------------------------------------------------------===//
//
// TaskSimplify - Simplify Tapir tasks
//
FunctionPass *createTaskSimplifyPass();

} // End llvm namespace

#endif
