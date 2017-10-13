#!/bin/bash
~/git/Parallel-IR/build/bin/clang -ftapir=cilk -O3 vadd.c -o vadd && time ./vadd
