LLVM_CONFIG = llvm-config
#LLVM_CONFIG = ~/llvm/src/build/Debug+Asserts/bin/llvm-config
LLVM_INC = `$(LLVM_CONFIG) --cxxflags` -Wno-cast-qual -Wno-overloaded-virtual
LLVM_LIB = `$(LLVM_CONFIG) --ldflags --libs core mcjit native ipo` -ldl -ltinfo -lz


all:
	g++ llvm.cpp $(LLVM_INC) $(LLVM_LIB) -lpthread -o out -std=c++1y
