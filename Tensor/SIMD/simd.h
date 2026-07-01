/*
    This is a C language header file
    When compiling we g++, we should wrap it with extern "C" to avoid name mangling
*/

#ifndef SIMD_H
#define SIMD_H

// Include necessary headers
#include <stddef.h> // size_t

#ifdef __cplusplus
extern "C"
{
#endif

// Function declarations

/**
 * @brief [DEBUG] Gets the current System SIMD vector size in bytes
 * @return size_t The vector size in bytes
 */
size_t simd_get_vecsize(void);

/**
 * @brief Vectorized COPY (any generic type)
 * @param dest pointer to destination array
 * @param src pointer to source array
 * @param length Vector length (in bytes)
 * @note We will help you to do memory alignment and handle the remainder elements
 */
void simd_copy_any(void* dest, const void* src, size_t length);

/**
 * @brief Vectorized FILL (any generic type)
 * @param dest pointer to destination array
 * @param src pointer to source VALUEs, the array should be in length that
 *        VECTOR_BYTES could be divided by.
 *        i.e. if VECTOR_BYTES = 16, src should be in length of 1, 2, 4, 8, or 16 bytes
 *        You could get the size of VECTOR_BYTES by calling simd_get_vecsize()
 * @param dest_length Length of the destination array (in bytes)
 * @param src_length Length of the source array (in bytes)
 * @note We will help you to do memory alignment, the final bits will be wrapped
 *       around to the starting values of the src array
 * @example VECTOR_BYTES = 16, dest_length = 20,
 *          src = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}
 *         dest = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,1,2,3,4}
 */
void simd_fill_any(void* dest, const void* src, size_t dest_length, size_t src_length);

// Integer operations
/**
 * @brief Addition (vectorized)
 * @param a pointer to first array
 * @param b pointer to second array
 * @note The length of the arrays must be the same
 *       We will help you to do memory alignment and handle the remainder elements
 * @param result pointer to the result array
 * @note You can use a or b as result, the operation will then be in-place
 * @param length Vector length
 * @param subtract (unsigned char) 0 for addition (a + b), 1 for subtraction (a - b)
 */
// void simd_add_int(const int* a, const int* b, int* result, size_t length, unsigned char subtract);

#ifdef __cplusplus
}
#endif

#endif // SIMD_H