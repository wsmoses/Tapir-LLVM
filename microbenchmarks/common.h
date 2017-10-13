#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

unsigned long long todval (struct timeval *tp) {
    return tp->tv_sec * 1000 * 1000 + tp->tv_usec;
}

static struct timeval _profile_start, _profile_end;

void profile_start() {
    gettimeofday(&_profile_start, 0);
}

void profile_end() {
    gettimeofday(&_profile_end ,0);
    printf("Time elapsed: %lld ms\n", (todval(&_profile_end) - todval(&_profile_start)) / 1000);
}
