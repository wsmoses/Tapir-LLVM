#pragma once
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

namespace llvm {
enum InstrumentationConfigMode { WHITELIST = 0, BLACKLIST = 1 };

enum InstrumentationPoint : int {
  INSTR_INVALID_POINT = 0x0,
  INSTR_FUNCTION_ENTRY = 0x1,
  INSTR_FUNCTION_EXIT = 0x2,
  INSTR_BEFORE_CALL = 0x4,
  INSTR_AFTER_CALL = 0x8
};

inline InstrumentationPoint operator|(const InstrumentationPoint &a,
                                      const InstrumentationPoint &b) {
  return static_cast<InstrumentationPoint>(static_cast<int>(a) |
                                           static_cast<int>(b));
}

inline InstrumentationPoint operator&(const InstrumentationPoint &a,
                                      const InstrumentationPoint &b) {
  return static_cast<InstrumentationPoint>(static_cast<int>(a) &
                                           static_cast<int>(b));
}

inline bool operator==(InstrumentationPoint a, InstrumentationPoint b) {
  return static_cast<int>(a) == static_cast<int>(b);
}

inline InstrumentationPoint &operator|=(InstrumentationPoint &a,
                                        InstrumentationPoint b) {
  return a = a | b;
}

static StringMap<InstrumentationPoint> SurgicalInstrumentationPoints = {
    {"FunctionEntry", INSTR_FUNCTION_ENTRY},
    {
        "FunctionExit",
        INSTR_FUNCTION_EXIT,
    },
    {
        "BeforeCall",
        INSTR_BEFORE_CALL,
    },
    {
        "AfterCall",
        INSTR_AFTER_CALL,
    },
};

InstrumentationPoint
ParseInstrumentationPoint(const StringRef &instrPointString);

class InstrumentationConfig {
public:
  virtual ~InstrumentationConfig() {}

  void SetConfigMode(InstrumentationConfigMode mode) { this->mode = mode; }

  static std::unique_ptr<InstrumentationConfig> GetDefault();

  static std::unique_ptr<InstrumentationConfig>
  ReadFromConfigurationFile(const std::string &filename);

  virtual bool DoesFunctionRequireInstrumentationForPoint(
      const StringRef &functionName, const InstrumentationPoint &point) {
    bool found = targetFunctions.find(functionName) != targetFunctions.end();

    if (found) // The function is in the configuration. Does it specify this
               // instrumentation point?
    {
      InstrumentationPoint &functionPoints = targetFunctions[functionName];

      // INVALID_POINT is interpreted as "all points".
      if (functionPoints != INSTR_INVALID_POINT) {
        if ((targetFunctions[functionName] & point) != point)
          found = false;
      }
    }

    return mode == InstrumentationConfigMode::WHITELIST ? found : !found;
  }

protected:
  InstrumentationConfig(){};
  InstrumentationConfig(const StringMap<InstrumentationPoint> &targetFunctions)
      : targetFunctions(targetFunctions) {}

  StringMap<InstrumentationPoint> targetFunctions;

  InstrumentationConfigMode mode = InstrumentationConfigMode::WHITELIST;
};

class DefaultInstrumentationConfig : public InstrumentationConfig {
public:
  virtual bool DoesFunctionRequireInstrumentationForPoint(
      const StringRef &functionName, const InstrumentationPoint &point) {
    return true;
  }
};
} // namespace llvm