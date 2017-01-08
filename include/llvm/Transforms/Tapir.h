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

namespace llvm {
class Pass;
class ModulePass;

//===----------------------------------------------------------------------===//
//
// LoopSpawning - Create a loop spawning pass.
//
Pass *createLoopSpawningPass();

//===----------------------------------------------------------------------===//
//
// PromoteDetachToCilk
//
ModulePass *createLowerTapirToCilkPass(bool DisablePostOpts = false,
                                       bool Instrument = false);

} // End llvm namespace

#endif
