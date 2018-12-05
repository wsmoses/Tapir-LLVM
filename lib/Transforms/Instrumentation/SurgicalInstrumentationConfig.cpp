#include "llvm/Transforms/Instrumentation/SurgicalInstrumentationConfig.h"
#include <vector>
#include <sstream>
#include <memory>


namespace llvm
{
    InstrumentationPoint ParseInstrumentationPoint(const StringRef & instrPointString)
    {
        if (SurgicalInstrumentationPoints.find(instrPointString) == SurgicalInstrumentationPoints.end())
        {
            return InstrumentationPoint::INSTR_INVALID_POINT;
        }
        else
            return SurgicalInstrumentationPoints[instrPointString];
    }


    std::unique_ptr<InstrumentationConfig> llvm::InstrumentationConfig::GetDefault() {
        return std::unique_ptr<DefaultInstrumentationConfig>(new DefaultInstrumentationConfig());
    }

    std::unique_ptr<InstrumentationConfig> InstrumentationConfig::ReadFromConfigurationFile(const std::string & filename)
    {
        auto file = MemoryBuffer::getFile(filename);

        if (!file)
        {
            llvm::report_fatal_error("Instrumentation configuration file could not be opened: " + file.getError().message());
        }

        StringRef contents = file.get()->getBuffer();
        SmallVector<StringRef, 20> lines;

        contents.split(lines, '\n', -1, false);

        StringMap<InstrumentationPoint> functions;

        // One instruction per line.
        for (auto& line : lines)
        {
            auto trimmedLine = line.trim();
            if (trimmedLine.size() == 0 || trimmedLine[0] == '#') // Skip comments or empty lines.
                continue;

            SmallVector<StringRef, 5> tokens;
            trimmedLine.split(tokens, ',', -1, false);

            if (tokens.size() > 0)
            {
                InstrumentationPoint points = InstrumentationPoint::INSTR_INVALID_POINT;
                if (tokens.size() > 1) // This function specifies specific instrumentation points.
                {
                    for (size_t i = 1; i < tokens.size(); ++i)
                    {
                        auto instrPoint = ParseInstrumentationPoint(tokens[i].trim());

                        points |= instrPoint;
                    }
                }

                auto trimmed = tokens[0].trim();
                if (trimmed != "")
                    functions[trimmed] = points;
           }
        }

        // If the configuration file turned out to be empty,
        // instrument everything.
        if (functions.size() == 0)
            return GetDefault();

        return std::unique_ptr<InstrumentationConfig>(new InstrumentationConfig(functions));
    }

}