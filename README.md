Tapir/LLVM
================================

This directory and its subdirectories contain source code for
Tapir/LLVM, a prototype compiler based on LLVM that implements the
Tapir compiler IR extensions for fork-join parallelism [^fn1].

Tapir/LLVM is under active development.  This directory contains
prototype implementations of compiler technologies that take advantage
of the Tapir compiler IR.  These prototype technologies include the
Rhino extensions to Tapir (unpublished).

Tapir/LLVM is open source software.  You may freely distribute it
under the terms of the license agreement found in LICENSE.txt.

[![CircleCI](https://circleci.com/gh/wsmoses/Parallel-IR.svg?style=svg)](https://circleci.com/gh/wsmoses/Parallel-IR)

[^fn1]: T. B. Schardl, W. S. Moses, C. E. Leiserson.  "Tapir:
Embedding Fork-Join Parallelism into LLVM's Intermediate
Representation."  ACM PPoPP, February 2017, pp. 249-265.  Won Best
Paper Award.  doi:10.1145/3018743.3018758
