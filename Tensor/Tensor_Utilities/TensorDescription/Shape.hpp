// File: Shape.hpp
// Description: Class to store and manage shape
//              AND read sequence. (permutation)
//              information.
// Date: June 10, 2026
// @ADMINGUOYU

#ifndef _UTILS_SHAPE_HPP_
#define _UTILS_SHAPE_HPP_

#include <cstdio>
#include <cstddef>  // defines: size_t
#include <utility>  // std::move()
#include <cstdlib>  // malloc(); free()
#include "../Memory/MemoryContainer.hpp"
#include "Indexer.hpp"

namespace TENSOR_UTILITIES
{
    // predefine broadcast result structure
    // (for friend function return type) - prototype only
    struct Broadcast_result;


    // class shape (shape manager)
    class Shape
    {
    private:
        // shape info
        MemoryContainer<size_t> m_shape { };
        // stride info (this deals with transpose and read sequence)
        MemoryContainer<size_t> m_stride { };

    public:
        // constructors and destructor
        Shape (void) = default;
        Shape (const Shape & other) = default;
        Shape (Shape && other)
            :   m_shape(std::move(other.m_shape)),
                m_stride(std::move(other.m_stride))
            { return; }
        
        ~Shape (void) = default;

        // copy and move assignment
        const Shape & operator= (const Shape & other)
        {
            copy_assign(this->m_shape, other.m_shape);
            copy_assign(this->m_stride, other.m_stride);
            return (*this);
        }
        const Shape & operator= (Shape && other)
        {
            move_assign(this->m_shape, std::move(other.m_shape));
            move_assign(this->m_stride, std::move(other.m_stride));
            return (*this);
        }

        // public functions API
        /**
         * @brief Reset permutation to original (non-permuted) state.
         * @return True if successful, false otherwise.
         * @note This will reset the shape and stride info to the original state
         *       Actually, we just reset the stride record
         *       [Shape info will NOT be changed]
         */
        bool reset_permutation (void)
        {
            // get the number of dimensions (count)
            const size_t count = this->m_shape.get_effective_item_count();

            // allocate stride info
            if (!this->m_stride.allocate(count))
                return false;
            // set effective size (in count of items)
            this->m_stride.set_effective_size(count);

            // re-calculate stride info based on the current shape info
            size_t stride = 1;
            for (size_t i = count; i > 0; --i)
            {
                this->m_stride.set(i - 1, &stride);
                const size_t * shape_ptr = (const size_t*)this->m_shape.get(i - 1);
                stride *= *shape_ptr;
            }

            // return
            return true;
        }
        /**
         * @brief Sets the shape info (using a provided array and count)
         * @return True if successful, false otherwise.
         * @note This will fresh set the shape info, any previous info will be cleared.
         *       Also, the stride info will be re-calculated based on the new shape info.
         */
        bool set_shape (const size_t * shape_ptr, size_t count)
        {
            // set shape info
            if (!this->m_shape.allocate(count))
                return false;
            for (size_t i = 0; i < count; ++i)
            {
                if (shape_ptr[i])
                    this->m_shape.set(i, &shape_ptr[i]);
                else
                    // dimension = 0 is not allowed
                    return false;
            }
            // set effective size (in count of items)
            this->m_shape.set_effective_size(count);
            
            // reset permutation (re-calculate stride info)
            if (!this->reset_permutation())
                return false;

            // return
            return true;
        }
        /**
         * @brief Gets dimension count (number of dimensions) of the shape.
         * @return The dimension count.
         */
        size_t get_dim_count (void) const
        {
            return this->m_shape.get_effective_item_count();
        }
        /**
         * @brief Get shape of a specified dimension.
         * @param dim The dimension to get the shape (0-indexed).
         * @return The shape of the specified dimension if successful,
         *         0 otherwise.
         */
        size_t get_shape (size_t dim) const
        {
            const size_t * shape_ptr = (const size_t*)this->m_shape.get(dim);
            return shape_ptr ? *shape_ptr : 0;
        }
        /**
         * @brief Get stride of a specified dimension.
         * @param dim The dimension to get the stride (0-indexed).
         * @return The stride of the specified dimension if successful,
         *         0 otherwise.
         * @note we always recalculate the stride info based on the current
         *       shape. -> O(n) complexity
         *       this function is meant to assist slicing, so we have to
         *       return based on the current shape
         */
        size_t get_stride (size_t dim) const
        {
            // get the number of dimensions (count)
            const size_t count = this->m_shape.get_effective_item_count();

            // if dim is out of range, return 0
            if (dim >= count)
                return 0;

            // loop calculate the stride
            size_t stride = 1;
            for (size_t i = count; i > 0; --i)
            {
                const size_t * shape_ptr = (const size_t*)this->m_shape.get(i - 1);
                if (i - 1 == dim)
                    return stride;
                stride *= *shape_ptr;
            }

            // we should never reach here, but we return 0 just in case
            return 0;
        }
        /**
         * @brief Get total item count of the shape
         * @return The total item count (product of all dimensions)
         */
        size_t get_item_count (void) const
        {
            // get the number of dimensions (count)
            const size_t count = this->m_shape.get_effective_item_count();

            // if it's empty, return 0
            if (count == 0)
                return 0;

            // loop calculate the total item count
            size_t total_count = 1;
            for (size_t i = 0; i < count; ++i)
            {
                const size_t * shape_ptr = (const size_t*)this->m_shape.get(i);
                total_count *= *shape_ptr;
            }

            // return
            return total_count;
        }
        /**
         * @brief Permutes the shape (updates the shape and stride info accordingly).
         * @return True if successful, false otherwise.
         * @note The permute_ptr have the correct length as the number of dimensions
         *       that contains a permutation of [0, 1, ..., count - 1] 0-indexed.
         *       [YOU HAVE TO MAKE SURE THE LENGTH IS VALID]
         * @note  if original shape is (1, 2, 3)
         *        a permutation of (1, 0, 2) will change the shape to (2, 1, 3)
         *        [YOU HAVE TO MAKE SURE THE PERMUTATION IS VALID]
         */
        bool permute (const size_t * permute_ptr)
        {
            // Get the number of dimensions (count)
            const size_t count = this->m_shape.get_effective_item_count();

            // We create a new shape and stride instead of inplace modification
            MemoryContainer<size_t> new_shape { };
            MemoryContainer<size_t> new_stride { };
            // allocate memory for new shape and stride
            if (!new_shape.allocate(count) || !new_stride.allocate(count))
                return false;
            // Get pointers
            size_t * new_shape_ptr = (size_t*)new_shape.get(count - 1);
            size_t * new_stride_ptr = (size_t*)new_stride.get(count - 1);
            // fill new shape and stride based on the permutation
            // fill it in reversed order to make sure the stride calculation is correct
            for (size_t i = count; i > 0; --i)
            {
                size_t permuted_idx = permute_ptr[i - 1];
                // check if permuted_idx is in valid range
                if (permuted_idx >= count)
                    return false;
                // get old shape and stride based on the permuted index
                const size_t * old_shape_ptr = (const size_t*)this->m_shape.get(permuted_idx);
                const size_t * old_stride_ptr = (const size_t*)this->m_stride.get(permuted_idx);
                // set new shape and stride
                // if shape is 1, we update its stride to its right neighbour's stride (if it exists),
                // otherwise 1 [just to make my design consistent]
                *new_shape_ptr = *old_shape_ptr;
                if (*old_shape_ptr == 1)
                {
                    if (i < count)  // have right neighbour
                        *new_stride_ptr = *(new_stride_ptr + 1);
                    else  // already the end
                        *new_stride_ptr = 1;
                }
                else
                    *new_stride_ptr = *old_stride_ptr;
                // move pointers to the previous position
                --new_shape_ptr;
                --new_stride_ptr;
            }

            // set effective size (in count of items)
            new_shape.set_effective_size(count);
            new_stride.set_effective_size(count);

            // move new shape and stride to current shape and stride
            move_assign(this->m_shape, std::move(new_shape));
            move_assign(this->m_stride, std::move(new_stride));

            // return
            return true;
        }
        /**
         * @brief Get the flattened index from a multi-dimensional index
         *        using the current shape and stride info.
         * @param multi_idx_ptr Pointer to the multi-dimensional index array (length is the number of dimensions).
         * @param allow_broadcasting Whether to allow broadcasting for dimensions with size 1.
         * @return The flattened index if successful, 0 otherwise.
         *         [YOU HAVE TO MAKE SURE THE MULTI-DIMENSIONAL INDEX IS VALID]
         *         [You can test whether we returned 0 but your index is not all 0]
         */
        size_t get_flattened_index
        (const size_t * multi_idx_ptr, bool allow_broadcasting = false) const
        {
            // get the number of dimensions (count)
            const size_t count = this->m_shape.get_effective_item_count();

            // calculate the flattened index
            size_t flattened_idx = 0;
            for (size_t i = 0; i < count; ++i)
            {
                // remember, we use the const version of the getter function
                const size_t * shape_ptr = (const size_t*)this->m_shape.get(i);
                const size_t * stride_ptr = (const size_t*)this->m_stride.get(i);
                // check if the multi-dimensional index is in valid range
                if (multi_idx_ptr[i] >= *shape_ptr)
                {
                    if (allow_broadcasting && *shape_ptr == 1)
                        // this dimension can be broadcasted, treat it as 0
                        continue;
                    else
                        // invalid index, return 0 (you can also choose to throw an error)
                        return 0;
                }
                // accumulate the flattened index
                flattened_idx += multi_idx_ptr[i] * (*stride_ptr);
            }

            // return the calculated flattened index
            return flattened_idx;        
        }

        // View and contiguity
        /**
         * @brief Check if the shape is contiguous
         * @return True if the shape is contiguous, false otherwise
         * @note We consider empty shape as contiguous
         * @note A shape is contiguous if the stride of each dimension is equal to the product of the shapes of the subsequent dimensions.
         *       For example, for a shape (2, 3, 4), the strides should be (12, 4, 1) for it to be contiguous.
         *       If the shape is not contiguous, it means that the data is not stored in a contiguous block of memory in the order defined by the shape.
         */
        bool is_contiguous (void) const
        {
            // get the number of dimensions (count)
            const size_t count = this->m_shape.get_effective_item_count();

            // if it's empty, we consider it as contiguous
            if (count == 0)
                return true;

            // check the stride of each dimension
            size_t expected_stride = 1;
            for (size_t i = count; i > 0; --i)
            {
                const size_t * shape_ptr = (const size_t*)this->m_shape.get(i - 1);
                const size_t * stride_ptr = (const size_t*)this->m_stride.get(i - 1);
                if (*stride_ptr != expected_stride)
                    return false;
                expected_stride *= *shape_ptr;
            }

            // if all strides are correct, it's contiguous
            return true;
        }
        /**
         * @brief Viewable check
         *        checks if we can view the current shape as a new shape
         *        without the need to copy data (for instance)
         * @param shape The new shape to check if viewable.
         * @return The new shape if viewable, an empty shape otherwise.
         * @note For a tensor to be viewed, the new view size must be compatible
         *       with its original size and stride,
         *       i.e., each new view dimension must either be a subspace of an
         *             original dimension, or only span across original dimensions
         *             d, d+1, ... , d+k that satisfy the following contiguity-like
         *             condition that ∀ i = d, ... , d+k-1,
         *             stride[i] = stride[i+1] × size[i+1]
         */
        Shape viewable_as (const Shape & shape) const
        {
            // total item counts must match
            if (this->get_item_count() != shape.get_item_count())
                return Shape { };

            // get counts of the dimensions
            const size_t src_count = this->get_dim_count();
            const size_t tgt_count = shape.get_dim_count();

            // prepare result shape
            // if target_count is 0, the allocation will fail, thus returning empty shape
            Shape result { };
            if ((!result.m_shape.allocate(tgt_count)) || (!result.m_stride.allocate(tgt_count)))
                return Shape { };
            result.m_shape.set_effective_size(tgt_count);
            result.m_stride.set_effective_size(tgt_count);

            // we need to make a copy of this for merging and splitting
            // this acts as a operational buffer
            Shape src_copy = *this;
            if ((!src_copy.m_shape.get_buffer_size()) || (!src_copy.m_stride.get_buffer_size()))
                // if any of these is 0, there is an error
                return Shape { };
            // we define merge start index and end index (for the mergeable lambda function)
            size_t merge_start = 0;
            size_t merge_end = 0;
            // prepare a C++11 lambda function to check if adjacent dimensions can merge
            // capture the buffer and start and end indices as reference 
            // (safe to use within this function call)
            auto mergeable = [&src_copy, &merge_start, &merge_end](void) -> bool
            {
                // check if dimensions from start_dim to end_dim can merge
                // they can merge if they satisfy the contiguity-like condition
                // i will never be equal to end_dim, sine end_dim <= dimension_count - 1
                // everything is OKAY
                // if we return true, it means that dimension[start, end] is a contiguous block
                const size_t * stride_ptr = (const size_t*)src_copy.m_stride.get(merge_start);
                const size_t * next_stride_ptr = stride_ptr + 1;  // directly plus 1 to increase efficiency
                const size_t * next_shape_ptr  = (const size_t*)src_copy.m_shape.get(merge_start + 1);
                for (size_t i = merge_start; i < merge_end; ++i)
                {
                    // check the contiguity-like condition
                    // formula: stride[i] = stride[i+1] × size[i+1]
                    if (*stride_ptr != (*next_stride_ptr) * (*next_shape_ptr))
                        return false;
                    // move to the next dimension
                    ++stride_ptr;
                    ++next_stride_ptr;
                    ++next_shape_ptr;
                }
                // after checking all ONE-BY-ONE, we return true
                return true;
            };

            // walk from last dimension backwards
            // for these iterating pointers, we store 1 larger
            // so we can use 0 to indicate the end of the dimension
            size_t src_i = src_count;
            size_t tgt_i = tgt_count;

            /*
            STRATEGY (start from the last dimension and walk backwards):
            We operate on src_copy (so merge and split could be done in-place)
              a) - if tgt[tgt_i] = 1, we assign and skip
              b) - if src[src_i] = 1, we skip (directly)
              c) - if src[src_i] == tgt[tgt_i], we assign and move both pointers
              d) - if src[src_i] % tgt[tgt_i] == 0, we can split the dimension
              ELSE, we try to merge src[x, ... , src_i] until d) is satisfied
            NOTE: when splitting, we instantly use one split, so the remaining part will not
                  be larger than current src_i (actually, it will be the same in this case)
            STOPPING CONDITION:
                - if we have assigned all target dimensions (tgt_i == 0), we are successful
                - if we have walked through all source dimensions (src_i == 0) but still have
                  target dimensions left, we fail
            */

            // we define a constant for 1, so we can use its address for stride assignment
            const size_t one = 1;

            // Outer loop, just check STOPPING CONDITION
            while (tgt_i > 0 && src_i > 0)
            {
                // define a target_i and source_i real (the -1 version)
                size_t tgt_i_real = tgt_i - 1;
                size_t src_i_real = src_i - 1;

                // get target shape (target stride is not important)
                const size_t * tgt_shape_ptr = (const size_t*)shape.m_shape.get(tgt_i_real);
                
                // case a) - if tgt[tgt_i] = 1, we assign and skip
                if (*tgt_shape_ptr == 1)
                {
                    // assign shape and stride
                    result.m_shape.set(tgt_i_real, tgt_shape_ptr);
                    // stride should be its right neighbour's stride (if it exists), otherwise 1
                    if (tgt_i < tgt_count)  // have right neighbour
                        result.m_stride.set(tgt_i_real, result.m_stride.get(tgt_i));  // tgt_i = tgt_i_real + 1
                    else  // already the end
                        result.m_stride.set(tgt_i_real, &one);
                    // move target pointer
                    --tgt_i;
                    continue;
                }

                // get source shape
                const size_t * src_shape_ptr = (const size_t*)src_copy.m_shape.get(src_i_real);

                // case b) - if src[src_i] = 1, we skip (directly)
                if (*src_shape_ptr == 1)
                {
                    // move source pointer
                    --src_i;
                    continue;
                }

                // case c) - if src[src_i] == tgt[tgt_i], we assign and move both pointers
                if (*src_shape_ptr == *tgt_shape_ptr)
                {
                    // assign shape and stride
                    result.m_shape.set(tgt_i_real, src_shape_ptr);
                    result.m_stride.set(tgt_i_real, src_copy.m_stride.get(src_i_real));
                    // move both pointers
                    --tgt_i;
                    --src_i;
                    continue;
                }

                // case d) - if src[src_i] % tgt[tgt_i] == 0, we can split the dimension
                if (*src_shape_ptr % *tgt_shape_ptr == 0)
                {
                    // get the stride of the split dimension (we will use it for the split part)
                    const size_t * split_stride_ptr = (const size_t*)src_copy.m_stride.get(src_i_real);
                    // assign shape and stride
                    result.m_shape.set(tgt_i_real, tgt_shape_ptr);
                    result.m_stride.set(tgt_i_real, split_stride_ptr);
                    // split the source dimension
                    size_t new_src_shape = *src_shape_ptr / *tgt_shape_ptr;
                    src_copy.m_shape.set(src_i_real, &new_src_shape);
                    // the split (remaining part)'s stride will change
                    // formula: new_stride = old_stride(split_stride_ptr) × new_shape(tgt_shape_ptr)
                    size_t new_src_stride = *split_stride_ptr * (*tgt_shape_ptr);
                    src_copy.m_stride.set(src_i_real, &new_src_stride);
                    // move target pointer only
                    // (source pointer remains the same, since we have split it)
                    --tgt_i;
                    continue;
                }

                // ELSE, we try to merge src[x, ... , src_i] until d) is satisfied
                // each step, we try merge only one more,
                // the merge-end index is src_i_real, we should start from src_i_real - 1
                merge_end = src_i_real;
                // if merge_start is 0, it means we have merged all the way to the beginning,
                // we should stop
                if (src_i_real == 0)
                    // we have merged all the way to the beginning but still cannot satisfy case d),
                    // we fail
                    return Shape { };
                // merge_start is always (merge_end - 1)
                merge_start = merge_end - 1;
                // now we have set the merge start and end, we check if they are mergeable
                if (mergeable())
                {
                    // We merge the dimension
                    // the merged dimension's shape is the product of the merged dimensions' shapes
                    size_t merged_shape = 1;
                    for (size_t i = merge_start; i <= merge_end; ++i)
                    {
                        const size_t * shape_ptr = (const size_t*)src_copy.m_shape.get(i);
                        merged_shape *= *shape_ptr;
                    }
                    // the merged dimension's stride is the stride of the last merged dimension
                    // (merge_end's stride)
                    const size_t * merged_stride_ptr = (const size_t*)src_copy.m_stride.get(merge_end);
                    // we set the merged dimension to the merge start index (just to keep it simple)
                    src_copy.m_shape.set(merge_start, &merged_shape);
                    src_copy.m_stride.set(merge_start, merged_stride_ptr);
                    // we skip the pointers to merge start index
                    src_i = merge_start + 1;  // since we are using 1 larger for the pointers
                    // continue
                    continue;
                }
                else
                {
                    // if we fail to merge, we do not expect we can merge any further
                    // thus, we fail
                    return Shape { };
                }
            }

            // If we landed here we should either hit tgt_i == 0 or src_i == 0
            // if we have assigned all target dimensions (tgt_i == 0), we are successful
            if (tgt_i == 0)
                return result;
            else
                // if we have walked through all source dimensions (src_i == 0) but still
                // have target dimensions left, we fail (actually we should never be here,
                // since her have checked the total item count at the beginning)
                return Shape { };
        }

        // Squeeze and unsqueeze utilities
        /**
         * @brief Squeezes the shape by removing dimensions with size 1.
         *        This un-parameterized version will squeeze all dimensions with size 1.
         * @return True if successful, false otherwise.
         */
        bool squeeze (void)
        {
            // get the number of dimensions (count)
            const size_t count = this->m_shape.get_effective_item_count();

            // we create new shape and stride instead of inplace modification
            MemoryContainer<size_t> new_shape { };
            MemoryContainer<size_t> new_stride { };
            // allocate memory for new shape and stride (worst case, no dimension is squeezed)
            if (!new_shape.allocate(count) || !new_stride.allocate(count))
                return false;
            
            // fill new shape and stride by squeezing dimensions with size 1
            size_t new_idx = 0;
            for (size_t i = 0; i < count; ++i)
            {
                const size_t * shape_ptr = (const size_t*)this->m_shape.get(i);
                const size_t * stride_ptr = (const size_t*)this->m_stride.get(i);
                if (*shape_ptr != 1)
                {
                    // this dimension is not squeezed, copy it to the new shape and stride
                    new_shape.set(new_idx, shape_ptr);
                    new_stride.set(new_idx, stride_ptr);
                    ++new_idx;
                }
                // if *shape_ptr == 1, this dimension is squeezed, we skip it
            }

            // set effective size (in count of items) for the new shape and stride
            new_shape.set_effective_size(new_idx);
            new_stride.set_effective_size(new_idx);

            // move new shape and stride to current shape and stride
            move_assign(this->m_shape, std::move(new_shape));
            move_assign(this->m_stride, std::move(new_stride));

            // return
            return true;
        }
        /**
         * @brief Squeezes the shape by removing dimensions with size 1.
         *        This parameterized version will squeeze the specified dimensions.
         * @param dims The dimensions to squeeze
         *        [SHOULD ALWAYS IN ASCENDING ORDER]
         * @param count The number of dimensions to squeeze (length of the dims array).
         * @return True if successful, false otherwise.
         */
        bool squeeze (const size_t * dims, size_t dims_count)
        {
            // get the number of dimensions (count)
            const size_t count = this->m_shape.get_effective_item_count();

            // if dims_count is larger than count, obviously we cannot squeeze
            if (dims_count > count)
                return false;

            // we create new shape and stride instead of inplace modification
            MemoryContainer<size_t> new_shape { };
            MemoryContainer<size_t> new_stride { };
            // allocate memory for new shape and stride
            // (worst case, no dimension is squeezed)
            // still use the original count
            if (!new_shape.allocate(count) || !new_stride.allocate(count))
                return false;
            
            // fill new shape and stride by squeezing the specified dimension if its size is 1
            size_t new_idx = 0;
            size_t dim_idx = 0;  // index for the dims array
            for (size_t i = 0; i < count; ++i)
            {
                const size_t * shape_ptr = (const size_t*)this->m_shape.get(i);
                const size_t * stride_ptr = (const size_t*)this->m_stride.get(i);
                if (i == *(dims + dim_idx))
                {
                    if (*shape_ptr != 1)
                        // this dimension cannot be squeezed, return false
                        return false;
                    // this dimension is squeezed, we skip it
                    // move to the next dimension in the dims array
                    if (dim_idx < dims_count - 1)
                        ++dim_idx;
                }
                else
                {
                    // this dimension is not squeezed, copy it to the new shape and stride
                    new_shape.set(new_idx, shape_ptr);
                    new_stride.set(new_idx, stride_ptr);
                    ++new_idx;
                }
            }

            // Check if we have squeezed all the specified dimensions
            if (dim_idx != dims_count - 1)
                // we have not squeezed all the specified dimensions, return false
                return false;

            // set effective size (in count of items) for the new shape and stride
            new_shape.set_effective_size(new_idx);
            new_stride.set_effective_size(new_idx);

            // move new shape and stride to current shape and stride
            move_assign(this->m_shape, std::move(new_shape));
            move_assign(this->m_stride, std::move(new_stride));

            // return
            return true;
        }
        /**
         * @brief Squeezes the shape by removing dimensions with size 1.
         *        This parameterized version will squeeze the specified dimension.
         * @param dim The dimension to squeeze.
         * @return True if successful, false otherwise.
         */
        bool squeeze (size_t dim)
        {
            return this->squeeze(&dim, 1);
        }
        /**
         * @brief Unsqueezes the shape by inserting a dimension with size 1 at
         *        specified positions. (relative to old positions)
         * @param dims The positions to insert the new dimension (0-indexed).
         *        [SHOULD ALWAYS IN ASCENDING ORDER]
         * @param dims_count The number of dimensions to unsqueeze
         *        (length of the dims array).
         * @return True if successful, false otherwise.
         * @note If dims has a position equals to the current number of dimensions,
         *       the new dimension will be inserted at the end.
         * @note if current shape is (2, 3), unsqueezing at (1, 2) will give you
         *       (2, 1, 3, 1), relative to old positions, position 1 (0-indexed)
         *       and position 2 (0-indexed)
         *       if current shape is (2, 3, 2), unsqueezing at (0, 1, 1, 3, 3)
         *       will give you (1, 3, 1, 1, 2, 2, 1, 1)
         * @note We don't allow > current number of dimensions,
         *       but we allow equal to current number of dimensions
         * @note Previous strids are preserved while the unsqueezed dimensions
         *       will have stride 1
         */
        bool unsqueeze (const size_t * dims, size_t dims_count)
        {
            // get the number of dimensions (count)
            const size_t count = this->m_shape.get_effective_item_count();

            // if count is 0, we cannot unsqueeze
            if (count == 0)
                return false;

            // we create new shape and stride instead of inplace modification
            MemoryContainer<size_t> new_shape { };
            MemoryContainer<size_t> new_stride { };
            // allocate memory for new shape and stride
            if (!new_shape.allocate(count + dims_count) || !new_stride.allocate(count + dims_count))
                return false;
            
            // one for stride 1
            size_t one = 1;
            // fill new shape and stride by unsqueezing the specified dimensions
            size_t new_idx = 0;
            size_t old_idx = 0;  // index for the old shape and stride
            size_t dim_idx = 0;  // index for the dims array
            // we loop at most count + dims_count times
            // we refer to the old_index when adding dimensions
            // i is a pure counter
            for (size_t i = 0; i < count + dims_count; ++i)
            {
                // if we still have dimensions to unsqueeze and
                // the current position matches the next dimension to unsqueeze
                // referenced to old_idx
                if (dim_idx < dims_count &&
                    old_idx == *(dims + dim_idx))
                {
                    // insert a new dimension with size 1 and
                    // the right neighbour's stride
                    new_shape.set(new_idx, &one);
                    if (old_idx < count)  // have right neighbour
                        // Get right neighbour's stride
                        new_stride.set(new_idx, this->m_stride.get(old_idx));
                    else  // already the end
                        new_stride.set(new_idx, &one);
                    // increment new index
                    ++new_idx;
                    // move to the next dimension in the dims array
                    // if over shoot, we will never enter this branch again
                    ++dim_idx;
                    // continue
                    continue;
                }

                // if old_idx is now equal count,
                // it means we have reached the end of the old shape
                // the previous code has just processed the last dimension
                // we break
                if (old_idx == count)
                    break;

                // copy the current dimension from the old shape and stride
                const size_t * shape_ptr = (const size_t*)this->m_shape.get(old_idx);
                const size_t * stride_ptr = (const size_t*)this->m_stride.get(old_idx);
                new_shape.set(new_idx, shape_ptr);
                new_stride.set(new_idx, stride_ptr);
                // update indices for the next iteration
                ++new_idx;
                ++old_idx;
            }

            // check if we have unsqueezed all the specified dimensions
            if (dim_idx != dims_count)
                // we have not unsqueezed all the specified dimensions, return false
                return false;

            // set effective size (in count of items) for the new shape and stride
            new_shape.set_effective_size(new_idx);
            new_stride.set_effective_size(new_idx);

            // move new shape and stride to current shape and stride
            move_assign(this->m_shape, std::move(new_shape));
            move_assign(this->m_stride, std::move(new_stride));

            // return
            return true;
        }
        /**
         * @brief Unsqueezes the shape by inserting a dimension with size 1 at
         *        specified position. (relative to old positions)
         * @param dim The position to insert the new dimension (0-indexed).
         * @return True if successful, false otherwise.
         * @note If dim is equal to the current number of dimensions,
         *       the new dimension will be inserted at the end.
         * @note if current shape is (2, 3), unsqueezing at position 1 (0-indexed) will give you
         *       (2, 1, 3), relative to old positions, position 1 (0-indexed)
         * @note We don't allow > current number of dimensions, but we allow equal to current number of dimensions
         * @note Previous strids are preserved while the unsqueezed dimension will have stride 1
         */
        bool unsqueeze (size_t dim)
        {
            return this->unsqueeze(&dim, 1);
        }

        // utilities
        /**
         * @brief Prints shape.
         * @return Nothing.
         */
        void print (void) const
        {
            // get ptr and number of items
            const size_t * ptr = (const size_t*)this->m_shape.get(0);
            const size_t count = this->m_shape.get_effective_item_count();

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
        /**
         * @brief Prints stride.
         * @return Nothing.
         */
        void print_stride (void) const
        {
            // get ptr and number of items
            const size_t * ptr = (const size_t*)this->m_stride.get(0);
            const size_t count = this->m_stride.get_effective_item_count();

            // print
            printf("Stride: ( ");
            for (size_t i = 0; i < count; ++i)
                printf("%zu, ", ptr[i]);
            printf(")\n");

            // flush stdout
            fflush(stdout);

            // return
            return;
        }

    // utility: interaction with Indexer
    public:
        /**
         * @brief Generates an Indexer object for the current shape.
         * @return The generated Indexer object.
         */
        Indexer generate_indexer (void) const
        {
            // get the number of dimensions (count)
            const size_t count = this->m_shape.get_effective_item_count();
            // if count is 0, we return an empty Indexer
            if (count == 0)
                return Indexer { };
            // create an Indexer object
            Indexer indexer { };
            // allocate memory for the shape and current_idx in the Indexer object
            indexer.shape = (size_t*)malloc(count * sizeof(size_t));
            indexer.current_idx = (size_t*)malloc(count * sizeof(size_t));
            if (!indexer.shape || !indexer.current_idx)
            {
                // if memory allocation fails, free any allocated memory and return an empty Indexer
                free(indexer.shape);
                free(indexer.current_idx);
                return Indexer { };
            }
            // copy the shape info to the Indexer object
            for (size_t i = 0; i < count; ++i)
                indexer.shape[i] = *(const size_t*)this->m_shape.get(i);
            // initialize current_idx to all zeros
            for (size_t i = 0; i < count; ++i)
                indexer.current_idx[i] = 0;
            // set the dimension count and max_step for the Indexer object
            indexer.dim_count = count;
            indexer.max_step = this->get_item_count();
            // initialize current_idx to all zeros
            for (size_t i = 0; i < count; ++i)
                indexer.current_idx[i] = 0;
            // return the generated Indexer object
            return indexer;
        }
        /**
         * @brief get_flattened_index overload that directly takes an Indexer object
         * @param indexer The Indexer object containing the current indexing state.
         * @param allow_broadcasting Whether to allow broadcasting for dimensions with size 1.
         * @return The flattened index if successful, 0 otherwise.
         *         [YOU HAVE TO MAKE SURE THE INDEXER OBJECT IS VALID]
         *         [You can test whether we returned 0 but your index is not all 0]
         */
        size_t get_flattened_index
        (const Indexer & indexer, bool allow_broadcasting = false) const
        {
            // we simply call the original get_flattened_index function with the current_idx from the Indexer object
            return this->get_flattened_index(indexer.current_idx, allow_broadcasting);
        }

    // Friend functions (utilities)
    public:
        // friend function for broadcasting utility
        friend Broadcast_result get_compatible_shapes(const Shape & shape1, const Shape & shape2);
    };


    // Broadcasting utilities
    /**
     * @brief Broadcasting result structure
     */
    struct Broadcast_result
    {
        // compatible shape
        Shape compatible_shape { };
        // shape 1's broadcasted shape (if compatible)
        // will insert 1s for dimension and automatically corrects strides
        Shape shape1_broadcasted { };
        // shape 2's broadcasted shape (if compatible)
        Shape shape2_broadcasted { };
    };
    /**
     * @brief Compatible shapes (broadcastable):
     *        gets the compatible shapes for two given shapes
     * @return The broadcast result containing compatible and broadcasted shapes
     * @note The shapes are compatible if for each dimension,
     *       the sizes are either equal or one of them is 1.
     *       If Compatible (broadcastable), the shape has the maximum
     *       size for each dimension among the two shapes.
     * @note You can check for an empty shape by checking whether its
     *       dimension count is 0 (get_dim_count() == 0),
     *       check compatible_shape in the result structure
     */
    Broadcast_result get_compatible_shapes
    (const Shape & shape1, const Shape & shape2)
    {
        // get dimension counts of the two shapes
        const size_t count1 = shape1.get_dim_count();
        const size_t count2 = shape2.get_dim_count();

        // make sure no empty shape
        if (count1 == 0 || count2 == 0)
            return Broadcast_result { };

        // the compatible shape will have the maximum dimension count
        size_t max_count = (count1 > count2) ? count1 : count2;
        
        // Set up the broadcast result
        Broadcast_result result { };
        // allocate memory for the compatible shape
        if (!result.compatible_shape.m_shape.allocate(max_count))
            return Broadcast_result { };
        // set effective size (in count of items)
        result.compatible_shape.m_shape.set_effective_size(max_count);

        // fill from the last dimension to the first dimension
        // break out if we find any incompatible dimension
        for (size_t i    = max_count,
                    idx1 = count1 - 1,
                    idx2 = count2 - 1,
                    shape1_done = 0,
                    shape2_done = 0; 
            i > 0;
            --i)
        {
            // get dimensions
            size_t current_dim_1 = (shape1_done) ? 
                                    1 : 
                                    *(const size_t *)shape1.m_shape.get(idx1);
            size_t current_dim_2 = (shape2_done) ? 
                                    1 : 
                                    *(const size_t *)shape2.m_shape.get(idx2);

            // check if the current dimensions are compatible
            if (current_dim_1 == current_dim_2 || 
                current_dim_1 == 1 || 
                current_dim_2 == 1)
            {
                // set the compatible dimension to the maximum of the two dimensions
                size_t compatible_dim = (current_dim_1 > current_dim_2) ? current_dim_1 : current_dim_2;
                result.compatible_shape.m_shape.set(i - 1, &compatible_dim);

                // update indices and done flags for the next iteration
                // we first decrease to 0 and process 0
                // if this iteration is 0, we mark it as done
                if (!shape1_done && idx1 > 0)
                    --idx1;
                else
                    shape1_done = 1;
                if (!shape2_done && idx2 > 0)
                    --idx2;
                else
                    shape2_done = 1;
            }
            else
                // incompatible dimension, return an empty shape
                return Broadcast_result { };
        }

        // reset stride for compatible shape
        if (!result.compatible_shape.reset_permutation())
            return Broadcast_result { };
        
        // Now we sets up the broadcasted shapes for shape1
        // and shape2 based on the compatible shape
        // we first copy and use the unsqueeze utility to insert
        // dimensions with size 1 where necessary
        result.shape1_broadcasted = shape1; // copy assignment
        result.shape2_broadcasted = shape2;
        // construct an array with 0 filled for the unsqueeze utility
        // length is the maximum dimension count
        size_t * unsqueeze_guide = nullptr;
        // we unsqueeze shape1 based on the compatible shape
        size_t to_unsqueeze = max_count - count1;
        if (to_unsqueeze > 0)
        {
            // late initialize the unsqueeze guide
            unsqueeze_guide = (size_t*)calloc(max_count, sizeof(size_t));
            // unsqueeze shape1
            result.shape1_broadcasted.unsqueeze(unsqueeze_guide, to_unsqueeze);
        }
        // print max count
        to_unsqueeze = max_count - count2;
        if (to_unsqueeze > 0)
        {
            // late initialize the unsqueeze guide
            if (!unsqueeze_guide)
                unsqueeze_guide = (size_t*)calloc(max_count, sizeof(size_t));
            result.shape2_broadcasted.unsqueeze(unsqueeze_guide, to_unsqueeze);
        }
        // de-allocate the unsqueeze guide
        free(unsqueeze_guide);

        // return the compatible shape
        return result;
    }

}

#endif