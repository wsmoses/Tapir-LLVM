/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|*Option Parsing Definitions                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/////////
// Prefixes

#ifdef PREFIX
#define COMMA ,
PREFIX(prefix_0, {0})
PREFIX(prefix_1, {"-" COMMA 0})
PREFIX(prefix_2, {"-" COMMA "--" COMMA 0})
PREFIX(prefix_3, {"/" COMMA "-" COMMA 0})
#undef COMMA
#endif

/////////
// Groups

#ifdef OPTION
OPTION(0, "my group", my_group, Group, INVALID, INVALID, 0, 0, 0, 0, 0)

//////////
// Options

OPTION(prefix_0, "<input>", INPUT, Input, INVALID, INVALID, 0, 0, 0, 0, 0)
OPTION(prefix_0, "<unknown>", UNKNOWN, Unknown, INVALID, INVALID, 0, 0, 0, 0, 0)
OPTION(prefix_1, "A", A, Flag, INVALID, INVALID, 0, OptFlag1, 0,
       "The A option", 0)
OPTION(prefix_1, "B", B, Joined, INVALID, INVALID, 0, OptFlag2, 0,
       "The B option", "B")
OPTION(prefix_2, "C=", Ceq, Joined, INVALID, C, 0, OptFlag1, 0, 0, 0)
OPTION(prefix_1, "C", C, Separate, INVALID, INVALID, 0, OptFlag1, 0,
       "The C option", "C")
OPTION(prefix_3, "C", SLASH_C, Separate, INVALID, INVALID, 0, OptFlag3, 0,
       "The C option", "C")
OPTION(prefix_1, "D", D, CommaJoined, INVALID, INVALID, 0, 0, 0,
       "The D option", "D")
OPTION(prefix_1, "E", E, MultiArg, INVALID, INVALID, 0, OptFlag1 | OptFlag2, 2, 0, 0)
OPTION(prefix_1, "F", F, JoinedOrSeparate, INVALID, INVALID, 0, 0, 0,
       "The F option", "F")
OPTION(prefix_1, "G", G, JoinedAndSeparate, INVALID, INVALID, 0, 0, 0,
       "The G option", "G")
OPTION(prefix_1, "H", H, Flag, INVALID, INVALID, 0, HelpHidden, 0, 0, 0)
OPTION(prefix_1, "I", I, Flag, my_group, H, 0, 0, 0, 0, 0)
OPTION(prefix_1, "Joo", Joo, Flag, INVALID, B, "bar\0", 0, 0, 0, 0)
OPTION(prefix_1, "J", J, Flag, INVALID, B, "foo\0", 0, 0, 0, 0)
OPTION(prefix_1, "slurp", Slurp, RemainingArgs, INVALID, INVALID, 0, 0, 0, 0, 0)
#endif
