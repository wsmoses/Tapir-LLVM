all: main_o3 main_sync

clean:
	rm main_o3 main_sync

main_o3:
	clang -O3 -o main_o3 main.c
	clang -O3 -S -emit-llvm -o main_o3.ll main.c

main_sync:
	clang -S -emit-llvm -o main_sync_raw.ll main.c
	opt -O3 main_sync_raw.ll -S -o main_sync_o3.ll
	opt -sync-elimination main_sync_o3.ll -S -o main_sync.ll
	clang -O3 main_sync.ll -o main_sync