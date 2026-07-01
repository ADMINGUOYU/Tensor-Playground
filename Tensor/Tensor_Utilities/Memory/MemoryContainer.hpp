// File: MemoryContainer.hpp
// Description: Class to manage storage and provide efficient
//              operative interface.
//				< Improved version. >
// Date: June. 30, 2026
// @ADMINGUOYU

#ifndef _UTILS_MEMORY_CONTAINER_HPP_
#define _UTILS_MEMORY_CONTAINER_HPP_

#include <cstddef>  // defines: size_t
#include <cstdio>
#include <cstdlib>  // malloc(); free()
#include <utility>  // std::move()
#include <typeinfo> // typeid()

// if not defined (overriden), define these
#ifndef BUFFER_EXPANSION_RATIO
    #define BUFFER_EXPANSION_RATIO 2
#endif
#ifndef BUFFER_SHRINK_THRESHOLD
    #define BUFFER_SHRINK_THRESHOLD 2
#endif

// macro for for allowing threaded operations
// only affects the MemoryContainer's internal operations
#ifdef BUFFER_THREADED_OPERATIONS
    // define maximum number of threads to use
    #ifndef BUFFER_THREADED_OPERATIONS_MAX_THREAD_COUNT
        #define BUFFER_THREADED_OPERATIONS_MAX_THREAD_COUNT 4
    #endif
    // define minimum size per thread (in Byte)
    #ifndef BUFFER_THREADED_OPERATIONS_MIN_SIZE_PER_THREAD
        #define BUFFER_THREADED_OPERATIONS_MIN_SIZE_PER_THREAD (1024 * 1024)  // 1 MB
    #endif
    // include pthread library (C library)
    // NOTE: Windows users, you might need to install:
    //       winpthread (MinGW-w64)
    //       pthreads-win32 (vcpkg)
    // You also have to link with pthread library
    // (add -lpthread to your linker flags)
    #include <pthread.h>
#endif // BUFFER_THREADED_OPERATIONS

// macro for buffer printing, define BUFFER_PRINT_ITEM
// if you want to print the items in the buffer (for debugging)
// this requires <iostream> library
// define BUFFER_PRINT_ITEM_PER_LINE to specify how many items
// to print per line (default is 10)
#ifdef BUFFER_PRINT_ITEM
    // include std library
    #include <iostream>
    // define items per line
    #ifndef BUFFER_PRINT_ITEM_PER_LINE
        #define BUFFER_PRINT_ITEM_PER_LINE 10
    #endif
#endif

// macro for enabling SIMD
// This requires precompiled SIMD library
// We will include "../../SIMD/simd.h" header here
#ifdef BUFFER_ENABLE_SIMD
    #include "../../SIMD/simd.h"
#endif // BUFFER_ENABLE_SIMD

namespace TENSOR_UTILITIES
{

    // Abstract class for interface
    // no data-member included
    class Buffer
    {
    public:
        // constructor
        Buffer(void) = default;
        // virtual destructor
        virtual ~Buffer(void) = default;

        // copy constructor
        Buffer(const Buffer &other) = default;
        // move constructor
        Buffer(Buffer &&other) = default;

        // interface (operations)
    public:
        // initialize all allocated memory blocks
        virtual void init_all(void) = 0;
        // re-generate buffer
        virtual bool allocate(size_t size) = 0;
        // adjust size of buffer (if failed, the memory remains untouched)
        virtual bool re_allocate(size_t size) = 0;
        // erase buffer (clears and de-allocates memory)
        virtual void erase(void) = 0;
        // append to buffer
        virtual bool append(const void * item_ptr, bool expand_buffer = true, size_t expansion_ratio = BUFFER_EXPANSION_RATIO) = 0;
        // shrink (save storage)
        virtual void shrink(size_t shrink_threshold = BUFFER_SHRINK_THRESHOLD, size_t expansion_ratio = BUFFER_EXPANSION_RATIO) = 0;
        // get item from buffer
        virtual void *get(size_t idx) = 0;
        virtual const void *get(size_t idx) const = 0;
        // set item
        virtual bool set(size_t idx, const void * item_ptr) = 0;
        // sets effective range (in count of items)
        virtual bool set_effective_size(size_t item_count) = 0;

        // checks if the given index is in range (buffer size)
        virtual bool idx_in_range(size_t sz) const = 0;
        // gets buffer size (in bytes)
        virtual size_t get_buffer_size(void) const = 0;
        // gets effective buffer size (in bytes)
        virtual size_t get_effective_size(void) const = 0;
        // gets buffer maximum item-count (indexable upper bound, not inclusive)
        virtual size_t get_buffer_item_count(void) const = 0;
        // gets buffer effective item-count (effective upper bound, not inclusive)
        virtual size_t get_effective_item_count(void) const = 0;
        // print function
        virtual void print(void) const = 0;

        // memory & bit-wise operations
    protected:
        // buffer descriptive structure
        struct Memory
        {
            // pointer to memory (address)
            void *ptr{nullptr};
            // total buffer size
            size_t mem_size{0};
            // effective size
            size_t eff_size{0};

        public:
            // constructor
            Memory(void) = default;
            // destructor
            ~Memory(void) = default;

            // allocation & de-allocation
        public:
            // static function for allocation (size in Byte)
            static bool allocate(size_t size, Memory &memory)
            {
                // de-allocate first
                Memory::de_allocate(memory);
                // allocate memory
                memory.ptr = malloc(size);
                // error checking
                if (!memory.ptr)
                    return false;
                // update other attributes
                memory.mem_size = size;
                memory.eff_size = 0;
                // return
                return true;
            }
            // static function for de-allocation
            static void de_allocate(Memory &memory)
            {
                // delete memory
                free(memory.ptr);
                // clear ptr
                memory.ptr = nullptr;
                // update other attributes
                memory.mem_size = 0;
                memory.eff_size = 0;
                // return
                return;
            }
            // static function to reallocate (should be faster)
            static bool re_allocate(size_t size, Memory &memory)
            {
                /*
                NOTE:
                If there is not enough memory, the old memory block
                is not freed and null pointer is returned.
                If ptr is NULL, the behavior is the same as calling malloc(new_size)
                */

                // try reallocate buffer
                void * realloc_ptr = realloc(memory.ptr, size);
                // error checking
                if (!realloc_ptr)
                    return false;
                // update memory pointer
                memory.ptr = realloc_ptr;
                // update other attributes
                memory.mem_size = size;
                if (memory.eff_size > size)
                    memory.eff_size = size;
                    // else, we don't update
                // return
                return true;
            }

            // utilities
        public:
            // comparison result
            struct Cmp_result
            {
                // comparison flag ENUM
                enum class Cmp { EQUAL = 0, NOT_EQUAL = 1 };

                // result flag
                Cmp flag;
                // numbers of identical (blocks) from the begining.
                size_t num_identical;

                Cmp_result(Cmp f, size_t s) : flag(f), num_identical(s) { return; }

                // fast comparison operator
                bool operator==(const bool &other) const
                {
                    bool this_result = (this->flag == Cmp::EQUAL) ? true : false;
                    return (this_result == other);
                }
            };
            // static memory comparison
            //
            //     NOTE:
            //     The block that malloc gives you is guaranteed to be aligned
            //     so that it can hold any type of data. On GNU systems, the address
            //     is always a multiple of eight on 32-bit systems, and a multiple
            //     of 16 on 64-bit systems. 
            //       -- https://sourceware.org/glibc/manual/latest/html_mono/libc.html
            //     since we always stars from the beginning of a buffer, the memory
            //     address should be aligned properly. (should cause no bug)
            static Cmp_result byte_cmp(const Memory &first, const Memory &other)
            {
                // if 'first' and 'other' is the same thing
                if ((&first) == (&other))
                    return Cmp_result{Cmp_result::Cmp::EQUAL, first.mem_size};
                // check if both buffer have value
                if (first.mem_size && other.mem_size)
                {
                    // both buffer have something
                    // calculate check size
                    size_t check_size = (first.mem_size < other.mem_size) ? first.mem_size : other.mem_size;
                    // fetch and convert the pointers
                    const unsigned char *ptr_this = (const unsigned char *)first.ptr;
                    const unsigned char *ptr_othr = (const unsigned char *)other.ptr;
                    // pos where 1st difference appears -> equals to number of identical items (init to 0)
                    size_t dif = 0;
                    
                    // FAST PATH: Compare more bytes at a time using size_t
                    const size_t *words_this = (const size_t *)ptr_this;
                    const size_t *words_othr = (const size_t *)ptr_othr;
                    size_t total_words = check_size / sizeof(size_t);
                    // comparison loop
                    for (; dif < total_words; ++dif)
                        if (words_this[dif] != words_othr[dif])
                            break;
                    // Convert word-index back to byte-index
                    dif *= sizeof(size_t);

                    // SLOW PATH: Clean up remaining unaligned trailing bytes
                    // or narrow down the exact byte offset where the whole word
                    // mismatch occurred
                    // loop through every element (dif is the index of element to check)
                    for (; dif < check_size; ++dif)
                        if (ptr_this[dif] != ptr_othr[dif])
                            break;

                    // if both buffers have the same amount of effective blocks (can equal to each other)
                    if (first.eff_size == other.eff_size)
                    {
                        if (dif >= first.eff_size)
                            // if number of same elements is beyond the effective range -> equal
                            return Cmp_result{Cmp_result::Cmp::EQUAL, dif};
                        // else, not equal
                        else
                            return Cmp_result{Cmp_result::Cmp::NOT_EQUAL, dif};
                    }
                    // not equal
                    return Cmp_result{Cmp_result::Cmp::NOT_EQUAL, dif};
                }
                // if one or both does not have value
                if (first.mem_size == other.mem_size)
                    // both are empty (return equal)
                    return Cmp_result{Cmp_result::Cmp::EQUAL, 0};
                // not both (return not equal)
                else
                    return Cmp_result{Cmp_result::Cmp::NOT_EQUAL, 0};
            }

            // byte copier (will NOT check if the indexes are in range)
            // copy range [start, end)
            // [INTERNAL VERSION] - WILL NOT DO SAFETY CHECK (i.e. start < end)
            static void _byte_copy(size_t start, size_t end, void * dst, const void * src)
            {
                // calculate size to copy
                size_t bytes_to_copy = end - start;

                // convert pointers
                unsigned char *dst_bytes = (unsigned char *)dst + start;
                const unsigned char *src_bytes = (const unsigned char *)src + start;

// [NORMAL] using size_t to perform batch copying
#ifndef BUFFER_ENABLE_SIMD
                // align pointer to offset of (size_t)
                // since dst and src should all be Memory::ptr allocated by malloc
                // they should be aligned by default, adding the same 'start', meaning
                // the same offset (NOTE: (size_t)dst % sizeof(size_t) == 0)
                // NOTE: since sizeof(size_t) is 2^x, we only grab the x bits from the
                //       least significant bit is okay. (& bit-wise AND)
                size_t alignment_offset = start & (sizeof(size_t) - 1);
                // if alignment_offset == 0, we do not want to spend 1 more iteration byte copying
                if (alignment_offset) alignment_offset = sizeof(size_t) - alignment_offset;
                if (alignment_offset > bytes_to_copy)
                    alignment_offset = bytes_to_copy;  // incase that all bytes are misaligned

                // copy alignment part
                if (alignment_offset)
                {
                    // copy
                    for (size_t i = 0; i < alignment_offset; ++i)
                        dst_bytes[i] = src_bytes[i];
                    // update bytes_to_copy
                    bytes_to_copy -= alignment_offset;
                    // advance pointers
                    dst_bytes += alignment_offset;
                    src_bytes += alignment_offset;
                }

                // Calculate the number of whole words to copy (integer division)
                size_t words_to_copy = bytes_to_copy / sizeof(size_t);
                // and the remaining bytes
                size_t rem_bytes = bytes_to_copy & (sizeof(size_t) - 1);

                // FAST COPY
                if (words_to_copy)
                {
                    // Cast to 64-bit word pointers to copy 8 bytes at a time
                    // or 32-bit on 32-bit systems
                    size_t __attribute__((__may_alias__)) *dst_words =
                        (size_t __attribute__((__may_alias__)) *)dst_bytes;
                    const size_t __attribute__((__may_alias__)) *src_words =
                        (const size_t __attribute__((__may_alias__)) *)src_bytes;
                    // copy
                    for (size_t i = 0; i < words_to_copy; ++i)
                        dst_words[i] = src_words[i];
                    // advance pointers
                    dst_bytes = (unsigned char *)(dst_words + words_to_copy);
                    src_bytes = (const unsigned char *)(src_words + words_to_copy);
                }

                // Clean up any remaining trailing bytes
                if (rem_bytes)
                {
                    // copy
                    for (size_t i = 0; i < rem_bytes; ++i)
                        dst_bytes[i] = src_bytes[i];
                }

                // return
                return;

// [SIMD] uses precompiled external C library
#else
                // use SIMD function
                simd_copy_any(dst_bytes, src_bytes, bytes_to_copy);

                // return
                return;
#endif
            }
            // byte copy wrapper (enable multi-threading)
            static void byte_copy(size_t start, size_t end, void * dst, const void * src)
            {
// [NORMAL] single threaded copy
#ifndef BUFFER_THREADED_OPERATIONS
                // directly call our _byte_copy internal function
                _byte_copy(start, end, dst, src);
                // DONE
                return;
// [THREADED] threaded copy
#else
                // if nothing, we return
                if (start >= end) return;
                // calculate total size to copy
                size_t total_size = end - start;
                // calculate thread to use
                // (also make sure we have large enough chunk for each thread)
                size_t thread_count = total_size / BUFFER_THREADED_OPERATIONS_MIN_SIZE_PER_THREAD;
                // make sure we don't exceed maximum thread count
                if (thread_count > BUFFER_THREADED_OPERATIONS_MAX_THREAD_COUNT)
                    thread_count = BUFFER_THREADED_OPERATIONS_MAX_THREAD_COUNT;
                // if thread count is now 0, fall back to single-threaded copy
                if (thread_count <= 1)
                {
                    _byte_copy(start, end, dst, src);
                    return;
                }
                // calculate size per thread
                size_t size_per_thread = total_size / thread_count;
                // calculate remaining size (for the last thread)
                size_t remaining_size = total_size % thread_count;

                // define thread data structure
                struct copier_thread
                {
                    size_t start;
                    size_t end;
                    void *dst;
                    const void *src;
                };
                // create thread data array
                copier_thread * thread_data = (copier_thread *)malloc(thread_count * sizeof(copier_thread));
                // create thread array
                pthread_t * threads = (pthread_t *)malloc(thread_count * sizeof(pthread_t));
                // error checking for malloc
                if (!thread_data || !threads)
                {
                    // if malloc failed, we terminate
                    free(thread_data);
                    free(threads);
                    return;
                }

                // thread function (worker)
                // we use a C++11 lambda function for simplicity
                // pthread accepts a function with type void * (*)(void*)
                void * (*thread_func)(void *) = [](void *arg) -> void *
                {
                    // convert argument to copier_thread pointer
                    copier_thread *data = (copier_thread *)arg;
                    // call the internal byte copy function for this thread's range
                    Memory::_byte_copy(data->start, data->end, data->dst, data->src);
                    // return (nothing to return)
                    return nullptr;
                };

                // assign thread data
                for (size_t i = 0; i < thread_count; ++i)
                {
                    thread_data[i].start = start + i * size_per_thread;
                    thread_data[i].end = (start + (i + 1) * size_per_thread);
                    // add remaining size to the last thread
                    if (i == thread_count - 1)
                        thread_data[i].end += remaining_size;
                    thread_data[i].dst = dst;
                    thread_data[i].src = src;
                }

                // create threads
                for (size_t i = 0; i < thread_count; ++i)
                    pthread_create(&threads[i], nullptr, thread_func, &thread_data[i]);

                // wait for threads to finish
                for (size_t i = 0; i < thread_count; ++i)
                    pthread_join(threads[i], nullptr);

                // free thread data
                free(thread_data);
                // free thread array
                free(threads);

                // return
                return;
#endif
            }
            // clone (full copy)
            static Memory clone(const Memory &memory)
            {
                // create a Memory struct to return
                Memory to_return{};
                // allocate memory
                if (!Memory::allocate(memory.mem_size, to_return))
                    return to_return;
                // copy memory
                Memory::byte_copy(0, memory.eff_size, to_return.ptr, memory.ptr);
                // update other attributes (mem_size updated in allocate)
                to_return.eff_size = memory.eff_size;
                // return
                return to_return;
            }
            // copy from one to another
            // copier will minimize items to copy
            static Memory &byte_copier(Memory &dst, const Memory &src)
            {
                // first, compare before copy
                const Cmp_result &ret = Memory::byte_cmp(dst, src);
                // if equal, do nothing and return
                if (ret.flag == Cmp_result::Cmp::EQUAL)
                    return dst;

                // now, actions have to be taken
                // check if other is empty
                if (src.eff_size == 0)
                {
                    // if so, simply set effective size to 0
                    // not de-allocating memory
                    dst.eff_size = 0;
                    return dst;
                }

                // check if this is empty (while other is not empty)
                // empty means NO memory allocation
                if (!dst.ptr)
                    // allocate just enough memory
                    // error checking (return if request failed)
                    if (!Memory::allocate(src.mem_size, dst))
                        return dst;

                // get number of identical items (items to skip copying)
                size_t identical = ret.num_identical;
                /*
                    Does NOT need copy:
                    identical >= other.effective_size
                    -> also this has enough space

                    NEEDs copy:
                    identical < other.effective_size
                        -> if needs new allocation?
                            (this->buf_size < other.effective_size)
                */
                // identical beyond effective range, no copy needed
                if (identical >= src.eff_size)
                {
                    // set this's effective size and return
                    dst.eff_size = src.eff_size;
                    return dst;
                }
                else
                {
                    // checks if we have enough space (to copy)
                    if (dst.mem_size < src.eff_size)
                    {
                        // we have to re-allocate memory
                        // and remember to set identical to 0
                        identical = 0;
                        // * allocate memory
                        Memory::allocate(src.mem_size, dst);
                    }
                    // now copy (Byte-by_Byte)
                    Memory::byte_copy(identical, src.eff_size, dst.ptr, src.ptr);
                    dst.eff_size = src.eff_size;
                    return dst;
                }
            }
        };
    };

    template <typename T>
    class MemoryContainer : public Buffer
    {
    private:
        // a memory structure
        Buffer::Memory buffer{};
        // records size of data-type
        const size_t dtype_size{};

        // constructor and destructor
    public:
        // constructor
        MemoryContainer(void) : Buffer(), buffer(), dtype_size(sizeof(T)) {}
        MemoryContainer(size_t num_of_item) : MemoryContainer()
        {
            // allocate memory
            if (num_of_item <= 0)
                num_of_item = 1;
            Buffer::Memory::allocate(num_of_item * this->dtype_size, this->buffer);
            // return
            return;
        }
        // copy constructor (same data-type only)
        MemoryContainer(const MemoryContainer &other) : MemoryContainer()
        {
            // copy memory buffer (Byte-by-Byte)
            this->buffer = Buffer::Memory::clone(other.buffer);
            // return
            return;
        }
        // move constructor (same data-type only)
        MemoryContainer(MemoryContainer &&other) : MemoryContainer()
        {
            // move by re-assigning
            this->buffer = other.buffer;
            // reset other's attributes for safe de-allocation
            other.buffer.ptr = nullptr;
            other.buffer.mem_size = 0;
            other.buffer.eff_size = 0;
            // return
            return;
        }
        // destructor
        ~MemoryContainer(void) override
        {
            if (this->buffer.ptr)
                Buffer::Memory::de_allocate(this->buffer);
            return;
        }

        // interface (operations) override
    public:
        // initialize all allocated memory blocks
        void init_all(void) override
        {
            // if no element, return
            if (!this->buffer.ptr)
                return;
            // get pointer
            T *ptr = (T *)this->buffer.ptr;

// [NORMAL] using size_t to perform batch copying
#ifndef BUFFER_ENABLE_SIMD

            // loop and init
            const size_t count = this->buffer.mem_size / this->dtype_size;
            for (size_t i = 0; i < count; ++i)
                ptr[i] = T{};
            // return
            return;

// [SIMD] uses precompiled external C library
#else

            // create a single default init value
            T default_value = T{ };
            // Use SIMD function
            simd_fill_any(ptr, &default_value, this->buffer.mem_size, sizeof(T));
            // return
            return;

#endif
        }
        // re-generate buffer
        bool allocate(size_t num_of_item) override
        {
            if (num_of_item <= 0)
                return false;
            // allocate (Memory will help us de-allocate)
            Buffer::Memory::allocate(num_of_item * this->dtype_size, this->buffer);
            // error checking
            if (this->buffer.ptr)
                return true;
            else
                return false;
        }
        // adjust size of buffer (if failed, the memory remains untouched)
        bool re_allocate(size_t num_of_item) override
        {
            if (num_of_item <= 0)
                return false;
            // reallocate
            if (Buffer::Memory::re_allocate(num_of_item * this->dtype_size, this->buffer))
                return true;
            else
                return false;
        }
        // erase buffer (clears and de-allocates memory)
        void erase(void) override
        {
            // de-allocate memory
            Buffer::Memory::de_allocate(this->buffer);
            // return
            return;
        }
        // append to buffer
        bool append(const void * item_ptr, bool expand_buffer = true, size_t expansion_ratio = BUFFER_EXPANSION_RATIO) override
        {
            // get references of memory attributes
            size_t eff_sz = this->buffer.eff_size / this->dtype_size;
            size_t mem_sz = this->buffer.mem_size / this->dtype_size;
            // we have enough space
            if (eff_sz < mem_sz)
            {
                ((T *)this->buffer.ptr)[eff_sz] = (*(const T *)item_ptr);
                this->buffer.eff_size += this->dtype_size;
                return true;
            }

            // we don't have enough space
            if (!expand_buffer)
                return false;

            // allocate enough memory (with expansion)
            if (mem_sz == 0)
                mem_sz = 1;
            // re-allocate
            if (!Buffer::Memory::re_allocate((mem_sz * expansion_ratio * this->dtype_size), this->buffer))
                return false;
            
            // append the value
            ((T *)this->buffer.ptr)[eff_sz] = (*(const T *)item_ptr);
            // set new effective size
            this->buffer.eff_size += this->dtype_size;

            // return
            return true;
        }
        // shrink (save storage)
        void shrink(size_t shrink_threshold = BUFFER_SHRINK_THRESHOLD, size_t expansion_ratio = BUFFER_EXPANSION_RATIO) override
        {
            // check if we need to shrink
            if (this->buffer.eff_size * shrink_threshold < this->buffer.mem_size)
            {
                // shrink
                if (this->buffer.eff_size == 0) { this->erase(); return; }
                if (!Buffer::Memory::re_allocate(this->buffer.eff_size * expansion_ratio * this->dtype_size, this->buffer))
                    return;  // actually this return does not matter
                // effective size is updated by Buffer::Memory::re_allocate
                // return
                return;
            }
            return;
        }
        // get item from buffer
        void *get(size_t idx) override
        {
            if (!this->idx_in_range(idx))
                return nullptr;
            T *ptr = ((T *)this->buffer.ptr) + idx;
            return ((void *)(ptr));
        }
        const void *get(size_t idx) const override
        {
            if (!this->idx_in_range(idx))
                return nullptr;
            const T *ptr = ((const T *)this->buffer.ptr) + idx;
            return ((const void *)(ptr));
        }
        // set item
        bool set(size_t idx, const void * item_ptr) override
        {
            T *ptr = (T *)this->get(idx);
            if (!ptr)
                return false;
            *ptr = (*((const T *)item_ptr));
            return true;
        }
        // sets effective range (in count of items)
        bool set_effective_size(size_t item_count) override
        {
            if (item_count < 0)
                return false;
            if (item_count > (this->buffer.mem_size / this->dtype_size))
                return false;
            this->buffer.eff_size = item_count * this->dtype_size;
            return true;
        }

        // checks if the given index is in range (buffer size)
        bool idx_in_range(size_t idx) const override
        {
            if (!this->buffer.ptr)
                return false;
            if (idx < 0)
                return false;
            const size_t idx_byte = idx * this->dtype_size;
            if (idx_byte < this->buffer.mem_size)
                return true;
            else
                return false;
        }
        // gets buffer size (in bytes)
        size_t get_buffer_size(void) const override { return this->buffer.mem_size; }
        // gets effective buffer size (in bytes)
        size_t get_effective_size(void) const override { return this->buffer.eff_size; }
        // gets buffer maximum item-count (indexable upper bound, not inclusive)
        size_t get_buffer_item_count(void) const override { return (this->buffer.mem_size / this->dtype_size); }
        // gets buffer effective item-count (effective upper bound, not inclusive)
        size_t get_effective_item_count(void) const override { return (this->buffer.eff_size / this->dtype_size); }
        // print function
        void print(void) const override
        {
            const size_t mem_sz = this->buffer.mem_size / this->dtype_size;
            const size_t eff_sz = this->buffer.eff_size / this->dtype_size;
            printf("Buffer info:\n\t[PTR ADDR] %p\n\t[BUF SIZE] %zu byte(s)\n\t[EFF SIZE] %zu byte(s)\n", this->buffer.ptr, this->buffer.mem_size, this->buffer.eff_size);
            printf("Container info:\n\t[BUF ITEM] %zu\n\t[EFF ITEM] %zu\n", mem_sz, eff_sz);
// print items if needed
#ifdef BUFFER_PRINT_ITEM
            const T *ptr = (const T *)this->buffer.ptr;
            printf("\t[Memory]");
            for (size_t i = 0; i < eff_sz; ++i)
            {
                if (!(i % BUFFER_PRINT_ITEM_PER_LINE))
                    printf("\n\t");
                std::cout << ptr[i];
                putchar(' ');
            }
            putchar('\n');
            printf("\t[Extras]");
            for (size_t i = eff_sz; i < mem_sz; ++i)
            {
                if (!((i - eff_sz) % BUFFER_PRINT_ITEM_PER_LINE))
                    printf("\n\t");
                std::cout << ptr[i];
                putchar(' ');
            }
            putchar('\n');
#endif
            return;
        }

        // casting (friend function declaration)
    public:
        // friend function: copy assignment
        template <typename X, typename Y>
        friend void copy_assign(MemoryContainer<X> &dst, const MemoryContainer<Y> &src);
        // friend function: move assignment (ONLY applicable to same type)
        template <typename X>
        friend void move_assign(MemoryContainer<X> &dst, MemoryContainer<X> &&src);
    };

    /* friend function definition */
    // friend function: copy assignment
    template <typename X, typename Y>
    void copy_assign(MemoryContainer<X> &dst, const MemoryContainer<Y> &src)
    {
        if (typeid(dst) == typeid(src))
        {
            // same type copying
            Buffer::Memory::byte_copier(dst.buffer, src.buffer);
            // return
            return;
        }
        else
        {
            // check if we have enough 'space' (indexable range)
            const size_t src_effective_item_count = src.get_effective_item_count();
            if (dst.get_buffer_item_count() < src_effective_item_count)
            {
                // need to re-allocate (and error checking)
                if (!dst.allocate(src_effective_item_count))
                    return;
            }
            // get the pointers
            X *dst_ptr = (X *)dst.buffer.ptr;
            const Y *src_ptr = (const Y *)src.buffer.ptr;
            // copy using operator= (assume it is usable)
            for (size_t i = 0; i < src_effective_item_count; ++i)
                dst_ptr[i] = src_ptr[i];
            // set effective size
            dst.buffer.eff_size = src_effective_item_count * dst.dtype_size;
            // return
            return;
        }
    }
    // friend function: move assignment (ONLY applicable to same type)
    template <typename X>
    void move_assign(MemoryContainer<X> &dst, MemoryContainer<X> &&src)
    {
        // we have to de-allocate the destination buffer
        if (dst.buffer.ptr)
            Buffer::Memory::de_allocate(dst.buffer);

        // copy assign memory attributes
        dst.buffer = src.buffer;
        // detach source
        src.buffer.ptr = nullptr;
        src.buffer.mem_size = 0;
        src.buffer.eff_size = 0;
        // return
        return;
    }
}

#endif // !_UTILS_MEMORY_CONTAINER_HPP_
