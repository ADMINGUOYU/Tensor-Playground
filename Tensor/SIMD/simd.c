/*
    This is a C language source file
    When compiling we g++, we should wrap it with extern "C" to avoid name mangling
*/

// include .h header
#include "simd.h"

// include library headers
#include <stddef.h> // size_t

// Automatically determine the maximum hardware-supported vector size
#if defined(__AVX2__) || defined(__AVX512F__)
    #define VECTOR_BYTES 32  // Modern Intel/AMD (AVX2 uses 256 bits)
#elif defined(__ARM_NEON) || defined(__SSE2__)
    #define VECTOR_BYTES 16  // ARM (Apple Silicon/Pi) or older Intel (SSE)
#else
    #define VECTOR_BYTES 16  // Safe fallback size
#endif

// typedef for SIMD vector types [effective only in this compilation unit]
// we also include the unaligned versions
// vector for ANY type (used for copying) - we use unsigned char
typedef unsigned char vany __attribute__((vector_size(VECTOR_BYTES)));
typedef unsigned char vany_unaligned __attribute__((vector_size(VECTOR_BYTES), aligned(1)));
// vector for integers
typedef int vi __attribute__((vector_size(VECTOR_BYTES)));
typedef int vi_unaligned __attribute__((vector_size(VECTOR_BYTES), aligned(1)));
// vector for unsigned integers
typedef unsigned int vu __attribute__((vector_size(VECTOR_BYTES)));
typedef unsigned int vu_unaligned __attribute__((vector_size(VECTOR_BYTES), aligned(1)));
// vector for floats
typedef float vf __attribute__((vector_size(VECTOR_BYTES)));
typedef float vf_unaligned __attribute__((vector_size(VECTOR_BYTES), aligned(1)));
// vector for doubles
typedef double vd __attribute__((vector_size(VECTOR_BYTES)));
typedef double vd_unaligned __attribute__((vector_size(VECTOR_BYTES), aligned(1)));

// define a struct for memory alignment
typedef struct simd_memory_alignment_info
{
    // Starting offset of the memory block (in bytes)
    size_t start_offset;
    // Ending offset of the memory block (in bytes)
    size_t end_offset;
    // Number of bytes that can be processed using SIMD (aligned portion)
    size_t aligned_bytes;
    // NOTE:
    // [<start_offset> | <aligned_bytes> | <end_offset>] <- total length of memory to operate
    
} simd_memory_alignment_info;

/**
 * @brief [STATIC] internal helper to calculate memory alignment
 * @param start Pointer to the start of the memory block
 * @param end Pointer to the end of the memory block
 * @note end should also points to a valid memory address (not one past the end)
 *       i.e. length = end - start + 1
 * @return simd_memory_alignment_info Struct containing alignment information
 */
static simd_memory_alignment_info
calculate_memory_alignment(const void *start, const void *end)
{
    // Calculate the total length of the memory block
    size_t total_length = (const unsigned char *)end - (const unsigned char *)start + 1;

    // Calculate the starting offset for alignment
    // as VECTOR_BYTES is always a power of 2 we use bitwise AND to get the remainder
    size_t start_offset = ((size_t)start) & (VECTOR_BYTES - 1);
    if (start_offset)
        // Adjust to the next aligned address
        start_offset = VECTOR_BYTES - start_offset;
    
    // Calculate the ending offset for alignment
    // Calculate the number of bytes that can be processed using SIMD
    size_t end_offset = 0;
    size_t aligned_bytes = 0;
    if (total_length >= start_offset)
    {
        // get remaining length after the start offset
        size_t remaining_length = total_length - start_offset;
        // calculate the number of bytes that can be processed using SIMD
        aligned_bytes = remaining_length & ~(VECTOR_BYTES - 1);
        // calculate the ending offset for alignment
        end_offset = remaining_length - aligned_bytes;
    }
    else
        // we set start_offset to total_length (we will not have any aligned bytes)
        start_offset = total_length;

    // Return the alignment information
    return (simd_memory_alignment_info)
           { .start_offset = start_offset,
             .end_offset = end_offset,
             .aligned_bytes = aligned_bytes };
}

/**
 * @brief [STATIC inline] internal helper to determine if alignment of two addresses are the same
 * @param ptr_1 Pointer to the first address
 * @param ptr_2 Pointer to the second address
 * @return int 1 if both addresses are aligned to the same boundary, 0 otherwise
 */
static inline int
addresses_aligned(const void *ptr_1, const void *ptr_2)
{
    // Check if both addresses are aligned to the same boundary
    return (((size_t)ptr_1 & (VECTOR_BYTES - 1)) == ((size_t)ptr_2 & (VECTOR_BYTES - 1))) ? 1 : 0;
}

/**
 * @brief [DEBUG] Gets the current System SIMD vector size in bytes
 * @return size_t The vector size in bytes
 */
size_t simd_get_vecsize(void)
{
    return (size_t)VECTOR_BYTES;
}

/**
 * @brief Vectorized COPY (any generic type)
 * @param dest pointer to destination array
 * @param src pointer to source array
 * @param length Vector length (in bytes)
 * @note We will help you to do memory alignment and handle the remainder elements
 */
void simd_copy_any(void *dest, const void *src, size_t length)
{
    // cast the pointers to unsigned char
    unsigned char *d = (unsigned char *)dest;
    const unsigned char *s = (const unsigned char *)src;

    // Nothing to do
    if (d == s || (!length))
        return;

    // Detect overlap that requires reverse copy:
    // [s ............. s+length)
    //        [d ............. d+length)
    // If dest starts inside the source range and is after src, copy backwards.
    const int copy_reversely = (d > s) && (d < (s + length));

    // Forward copy path
    if (!copy_reversely)
    {
        // Check if both pointers share the same relative alignment.
        // If they do, we can safely perform a strictly aligned core loop.
        if (addresses_aligned(dest, src))
        {
            // Calculate memory alignment information
            simd_memory_alignment_info alignment_info = calculate_memory_alignment(d, d + length - 1);

            // Handle the unaligned start portion
            for (size_t i = 0; i < alignment_info.start_offset; ++i)
                d[i] = s[i];

            // Handle the aligned portion using SIMD
            size_t aligned_length = alignment_info.aligned_bytes;
            size_t simd_iterations = aligned_length / VECTOR_BYTES;

            // Update the pointers to point to the aligned portion
            d += alignment_info.start_offset;
            s += alignment_info.start_offset;

            // cast to SIMD vector types -> vany (since we have aligned pointers)
            vany *vd = (vany *)d;
            const vany *vs = (const vany *)s;

            // Perform the SIMD copy
            for (size_t i = 0; i < simd_iterations; ++i)
                vd[i] = vs[i];

            // Update the pointers to point to the end of the aligned portion
            d += aligned_length;
            s += aligned_length;

            // Handle the unaligned end portion
            for (size_t i = 0; i < alignment_info.end_offset; ++i)
                d[i] = s[i];

            // return
            return;
        }

        // Fallback: Pointers are misaligned relative to each other, or the buffer is too small.
        // We execute an entirely unaligned SIMD loop using 'vany_unaligned'.
        size_t simd_iterations = length / VECTOR_BYTES;

        // cast to unaligned SIMD vector types -> vany_unaligned
        vany_unaligned *vd = (vany_unaligned *)d;
        const vany_unaligned *vs = (const vany_unaligned *)s;

        // Perform the unaligned SIMD copy
        for (size_t i = 0; i < simd_iterations; ++i)
            vd[i] = vs[i];

        // Update the pointers to point to the end of the aligned portion
        size_t remaining_bytes = length & (VECTOR_BYTES - 1);
        d += (length - remaining_bytes);
        s += (length - remaining_bytes);

        // Handle the remaining bytes (if any)
        for (size_t i = 0; i < remaining_bytes; ++i)
            d[i] = s[i];

        // return
        return;
    }

    // Backward copy path
    else
    {
        // Get the end pointers (actually points to the last element)
        unsigned char *d_end = d + length - 1;
        const unsigned char *s_end = s + length - 1;

        // Check if both pointers share the same relative alignment.
        // If they do, we can safely perform a strictly aligned core loop.
        if (addresses_aligned(d_end, s_end))
        {
            // Calculate memory alignment information
            simd_memory_alignment_info alignment_info = calculate_memory_alignment(d, d_end);

            // Handle the unaligned end portion
            for (size_t i = 0; i < alignment_info.end_offset; ++i)
                d_end[-i] = s_end[-i];

            // Handle the aligned portion using SIMD
            size_t aligned_length = alignment_info.aligned_bytes;
            size_t simd_iterations = aligned_length / VECTOR_BYTES;

            // Update the pointers to point to the aligned portion
            d_end -= alignment_info.end_offset;
            s_end -= alignment_info.end_offset;

            // cast to SIMD vector types -> vany (since we have aligned pointers)
            vany *vd = (vany *)(d_end - aligned_length + 1);
            const vany *vs = (const vany *)(s_end - aligned_length + 1);

            // Perform the SIMD copy in reverse
            for (size_t i = 0; i < simd_iterations; ++i)
                vd[simd_iterations - 1 - i] = vs[simd_iterations - 1 - i];

            // Update the pointers to point to the start of the aligned portion
            d_end -= aligned_length;
            s_end -= aligned_length;

            // Handle the unaligned start portion
            for (size_t i = 0; i < alignment_info.start_offset; ++i)
                d_end[-i] = s_end[-i];

            return;
        }

        // Fallback: Pointers are misaligned relative to each other, or the buffer is too small.
        // We execute an entirely unaligned SIMD loop using 'vany_unaligned'.
        size_t simd_iterations = length / VECTOR_BYTES;

        // cast to unaligned SIMD vector types -> vany_unaligned
        vany_unaligned *vd = (vany_unaligned *)(d_end - VECTOR_BYTES + 1);
        const vany_unaligned *vs = (const vany_unaligned *)(s_end - VECTOR_BYTES + 1);

        // Perform the unaligned SIMD copy in reverse
        for (size_t i = 0; i < simd_iterations; ++i)
            vd[simd_iterations - 1 - i] = vs[simd_iterations - 1 - i];

        // Update the pointers to point to the start of the aligned portion
        size_t remaining_bytes = length & (VECTOR_BYTES - 1);
        d_end -= (length - remaining_bytes);
        s_end -= (length - remaining_bytes);

        // Handle the remaining bytes (if any)
        for (size_t i = 0; i < remaining_bytes; ++i)
            d_end[-i] = s_end[-i];

        // return
        return;        
    }
}
