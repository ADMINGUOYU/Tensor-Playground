// File: Shape.hpp
// Description: Class to store and manage shape
//              AND read sequence. (de-transpose)
//              information.
// Date: Sept. 16, 2025
// @ADMINGUOYU

#ifndef _UTILS_SHAPE_HPP_
#define _UTILS_SHAPE_HPP_

#include <cstdio>
#include <cstddef>  // defines: size_t
#include <utility>  // std::move()
#include "../Memory/MemoryContainer.hpp"

namespace TENSOR_UTILITIES
{
    // class shape (shape manager)
    class Shape
    {
    private:
        // shape info
        MemoryContainer<size_t> m_shape { };
        // read sequence
        MemoryContainer<size_t> m_read { };

    public:
        // constructors and destructor
        Shape (void) = default;
        Shape (const Shape & other) = default;
        Shape (Shape && other) : m_shape(std::move(other.m_shape)), m_read(std::move(other.m_read)) { return; }
        
        ~Shape (void) = default;

        // copy and move assignment
        const Shape & operator= (const Shape & other)
        {
            copy_assign(this->m_shape, other.m_shape);
            copy_assign(this->m_read, other.m_read);
            return (*this);
        }
        const Shape & operator= (Shape && other)
        {
            move_assign(this->m_shape, std::move(other.m_shape));
            move_assign(this->m_read, std::move(other.m_read));
            return (*this);
        }

        // public functions

        /**
         * @brief Sets the shape.
         * @param count Total number of dimensions.
         * @param arr Pointer to dimension buffer.
         * @return Nothing.
         */
        template <typename T>
        void set_shape (const size_t & count, const T * arr)
        {
            // allocate and reset all buffers
            this->m_shape.allocate(count);
            this->m_read.allocate(count);

            // get pointer to buffer
            size_t * ptr_shape = (size_t * )this->m_shape.get(0);
            size_t * ptr_read = (size_t * )this->m_read.get(0);

            // write both buffers
            for (size_t i = 0; i < count; ++i)
            {
                ptr_shape[i] = arr[i];
                ptr_read[i] = i;
            }

            // return
            return;
        }

    };

}

#endif