#pragma once
#include <host_defines.h>

template<typename T, size_t size>
struct Stack
{
    __device__ inline void push(const T& e) { elements[++top] = e; }
    __device__ inline T pop() { return elements[top--]; }
    // Removes element on top without returning it
    __device__ inline void dec() { --top; }
    __device__ inline bool isEmpty() const { return top == -1; }
    __device__ inline bool isFull() const { return top == (size - 1); }

    T elements[size];
    int top{ -1 }; // Index of the top element
};
