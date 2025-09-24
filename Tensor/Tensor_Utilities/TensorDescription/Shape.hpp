// File: Shape.hpp
// Description: Class to store and manage shape
//              AND read sequence. (de-transpose)
//              information.
// Date: Sept. 24, 2025
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

    public:
        // constructors and destructor
        Shape (void) = default;
        Shape (const Shape & other) = default;
        Shape (Shape && other) : m_shape(std::move(other.m_shape)) { return; }
        
        ~Shape (void) = default;

        // copy and move assignment
        const Shape & operator= (const Shape & other)
        {
            copy_assign(this->m_shape, other.m_shape);
            return (*this);
        }
        const Shape & operator= (Shape && other)
        {
            move_assign(this->m_shape, std::move(other.m_shape));
            return (*this);
        }

        // public functions (accessors)
        /**
         * @brief Gets total number of dimension within shape.
         * @return Shape count.
         */
        size_t get_shape_count (void) const
        {
            return this->m_shape.get_effective_item_count();
        }

         /**
         * @brief Gets a value of the specified index.
         * @param shape_idx Shape index to retrieve.
         * @return The value of that shape index; 0 if index is invalid.
         */
        size_t get_shape (const size_t & shape_idx)
        {
            // retrieve ptr to shape index
            size_t * ptr_shape = (size_t*)this->m_shape.get(shape_idx);

            // proceed if not nullptr
            if (!ptr_shape)
                return 0;

            // return
            return (*ptr_shape);
        }

        // public functions (mutators)
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
            this->m_shape.set_effective_size(count);

            // get pointer to buffer
            size_t * ptr_shape = (size_t * )this->m_shape.get(0);

            // write both buffers
            for (size_t i = 0; i < count; ++i)
            {
                ptr_shape[i] = arr[i];
            }

            // return
            return;
        }

        /**
         * @brief Allocates memory for shape and sets all dimension to 1
         * @param count Total number of dimensions.
         * @return Nothing.
         */
        void allocate_shape (const size_t & count)
        {
            // allocate and reset all buffers
            this->m_shape.allocate(count);
            this->m_shape.set_effective_size(count);

            // get pointer to buffer
            size_t * ptr_shape = (size_t * )this->m_shape.get(0);

            // write both buffers
            for (size_t i = 0; i < count; ++i)
            {
                ptr_shape[i] = 1;
            }

            // return
            return;
        }

        /**
         * @brief Modifies specific value of a shape index within shape.
         * @param shape_idx Shape index to modify.
         * @param value New shape info.
         * @return Nothing.
         */
        void edit_shape (const size_t & shape_idx, const size_t & value)
        {
            // set using internal function
            this->m_shape.set(shape_idx, &value);

            // return
            return;
        }

        /**
         * @brief Append to given shape index.
         * @param value New shape info.
         * @param end Whether to append at the end
         * @param shape_idx Index of the new item; if end == false; invalid shape_idx will append to the end.
         * @return Nothing.
         */
        void append_at (const size_t & value, bool end = true, const size_t & shape_idx = 0)
        {
            // append new shape to the end
            this->m_shape.append(&value);

            // if end, we're done
            if (end) return;

            // check if shape_idx is valid
            const size_t & effective_size = this->m_shape.get_effective_item_count();
            if ((shape_idx < 0) || (shape_idx >= effective_size)) return;

            // get ptr
            size_t * ptr_shape = (size_t*)this->m_shape.get(0);

            // if not, we shall move the numbers
            for (size_t i = effective_size - 1; i > shape_idx; --i)
                ptr_shape[i] = ptr_shape[i - 1];
                
            ptr_shape[shape_idx] = value;
        
            // return
            return;
        }

         /**
         * @brief Remove a index at given shape index.
         * @param end Whether to delete the end
         * @param shape_idx Index of the item to remove; if end == false; invalid shape_idx will remove the end.
         * @return Nothing.
         */
        void remove_at (bool end = true, const size_t & shape_idx = 0)
        {
            // get current count
            const size_t & effective_size_pre = this->m_shape.get_effective_item_count();

            // set effective size
            this->m_shape.set_effective_size(effective_size_pre - 1);

            // if end, we're done
            if (end)
                return;

            // if not, check shape_idx
            if ((shape_idx < 0) || (shape_idx >= effective_size_pre)) return;

            // we have to move items forward
            // get ptr
            size_t * ptr_shape = (size_t*)this->m_shape.get(0);

            // move the numbers
            for (size_t i = shape_idx; i < effective_size_pre - 1; ++i)
                ptr_shape[i] = ptr_shape[i + 1];

            // return
            return;
        }

        // utilities
        /**
         * @brief Prints shape.
         * @return Nothing.
         */
        void print (void)
        {
            // get ptr and number of items
            size_t * ptr = (size_t*)this->m_shape.get(0);
            const size_t & count = this->m_shape.get_effective_item_count();

            // print
            printf("Shape: ( ");
            for (size_t i = 0; i < count; ++i)
                printf("%zu, ", ptr[i]);
            printf(")\n");

            // flush stdout
            fflush(stdout);

            // return
            return;
        }


    };

}

#endif