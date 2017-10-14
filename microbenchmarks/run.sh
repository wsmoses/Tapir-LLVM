#!/bin/bash

BENCH=$1
shift
CORES=${1:-8}
shift

echo "============================================================================="

echo "Compiling non-opt version"
../build/bin/clang $@ -fcilkplus -O3 $BENCH/$BENCH.c -o "/tmp/$BENCH"
echo "Compiling opt version"
../build/bin/clang -DOPT $@ -fcilkplus -O3 $BENCH/$BENCH.c -o "/tmp/$BENCH""_opt"

echo "Running on $CORES cores"
echo "Running non-opt version"
NON_OPT_TIME=$(CILK_NWORKERS=$CORES taskset -c 1-$CORES "/tmp/$BENCH" | grep "Time elapsed" | awk '{ print $3 }')
echo "Running opt version"
OPT_TIME=$(CILK_NWORKERS=$CORES taskset -c 1-$CORES "/tmp/$BENCH""_opt" | grep "Time elapsed" | awk '{ print $3 }')

echo "Time for non-opt version: $NON_OPT_TIME ms"
echo "Time for opt version:     $OPT_TIME ms"
echo "Speed up:                 $(echo $OPT_TIME / $NON_OPT_TIME | bc -l)"
echo "Options given:            $@"

echo "============================================================================="
