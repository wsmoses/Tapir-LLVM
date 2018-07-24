include_directories("${PROJECT_SOURCE_DIR}/tools/polly/include")
include_directories("${PROJECT_BINARY_DIR}/tools/polly/include")

include_directories(BEFORE "${PROJECT_SOURCE_DIR}/tools/polly/lib/External")
include_directories(BEFORE "${PROJECT_BINARY_DIR}/tools/polly/lib/External")

include_directories(BEFORE "${PROJECT_SOURCE_DIR}/tools/polly/lib/External/isl/include")
include_directories(BEFORE "${PROJECT_BINARY_DIR}/tools/polly/lib/External/isl/include")

include_directories(BEFORE "${PROJECT_SOURCE_DIR}/tools/polly/lib/External/pet/include")
include_directories(BEFORE "${PROJECT_BINARY_DIR}/tools/polly/lib/External/pet/include")

if (CMAKE_COMPILER_IS_GNUCXX)
  # FIXME: Turn off exceptions, RTTI:
  # -fno-exceptions -fno-rtti
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-common -Woverloaded-virtual -Wno-long-long -Wall -W -Wno-unused-parameter -Wwrite-strings -fno-exceptions -fno-rtti")
elseif (MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHs-c-")
  add_definitions("-D_HAS_EXCEPTIONS=0")
else ()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-exceptions -fno-rtti")
endif ()

add_llvm_library(LLVMTapirOpts
  CilkABI.cpp
  OpenMPABI.cpp
  PTXABI.cpp
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

  ParallelPollySchedule.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Transforms
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Transforms/Tapir

  DEPENDS
  intrinsics_gen
  Polly
  PollyCore
  )

target_link_libraries(LLVMTapirOpts PRIVATE Polly)