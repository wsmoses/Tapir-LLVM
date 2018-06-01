add_llvm_library(LLVMTapirOpts
  CilkABI.cpp
  OpenMPABI.cpp
  QthreadsABI.cpp
  SmallBlock.cpp
  RedundantSpawn.cpp
  SpawnRestructure.cpp
  DetachUnswitch.cpp
  SyncElimination.cpp
  TapirToTarget.cpp
  LoopSpawning.cpp
  Outline.cpp
  Tapir.cpp
  TapirUtils.cpp
  NestedDetachMotion.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Transforms
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Transforms/Tapir

  DEPENDS
  intrinsics_gen
  )
