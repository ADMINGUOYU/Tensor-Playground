// File: MemoryContainer.hpp
// Description: Class to manage storage and provide efficient
//              operative interface.
//				< Improved version. >
// Date: June. 19, 2026
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
        // gets buffer maxmum item-count (indexable upper bound, not inclusive)
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
            // static memory comparsion
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

            // byte copier (will NOT check if the indexies are in range)
            static void byte_copy(size_t start, size_t end, void * dst, const void * src)
            {
                // convert pointers
                unsigned char *dst_ptr = (unsigned char *)dst;
                const unsigned char *src_ptr = (const unsigned char *)src;
                // copy loop
                for (size_t i = start; i < end; ++i)
                    dst_ptr[i] = src_ptr[i];
                // return
                return;
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
                // identicals beyond effective range, no copy needed
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

        // constructor and de-structor
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
        // de-structor
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
            // loop and init
            const size_t &count = this->buffer.mem_size / this->dtype_size;
            for (size_t i = 0; i < count; ++i)
                ptr[i] = T{};
            // return
            return;
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
            const size_t &idx_byte = idx * this->dtype_size;
            if (idx_byte < this->buffer.mem_size)
                return true;
            else
                return false;
        }
        // gets buffer size (in bytes)
        size_t get_buffer_size(void) const override { return this->buffer.mem_size; }
        // gets effective buffer size (in bytes)
        size_t get_effective_size(void) const override { return this->buffer.eff_size; }
        // gets buffer maxmum item-count (indexable upper bound, not inclusive)
        size_t get_buffer_item_count(void) const override { return (this->buffer.mem_size / this->dtype_size); }
        // gets buffer effective item-count (effective upper bound, not inclusive)
        size_t get_effective_item_count(void) const override { return (this->buffer.eff_size / this->dtype_size); }
        // print function
        void print(void) const override
        {
            const size_t &mem_sz = this->buffer.mem_size / this->dtype_size;
            const size_t &eff_sz = this->buffer.eff_size / this->dtype_size;
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
            const size_t &src_effective_item_count = src.get_effective_item_count();
            if (dst.get_buffer_item_count() < src_effective_item_count)
            {
                // need to re-allocate (and error checking)
                if (!dst.allocate(src_effective_item_count))
                    return;
            }
            // get the ponters
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
