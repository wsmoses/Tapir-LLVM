main_o3:
	clang -O3 -o main_o3 main.c

main_sync:
	rm *.bc
	clang -c -emit-llvm -S main.c -o main_raw.bc
	opt -O3 main_raw.bc -S -o main_o3.bc
	opt -sync-elimination main_o3.bc -o main_final.bc -S
	clang -O3 main_final.bc -o main_sync