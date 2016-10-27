add_llvm_library(LLVMTapirOpts
  CilkABI.cpp
  Detach2Cilk.cpp
  Loop2Cilk.cpp
  TapirOutline.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Transforms
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Transforms/Tapir
  )

add_dependencies(LLVMTapirOpts intrinsics_gen)
