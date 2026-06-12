// File: Indexer.hpp
// Description: A generator class to mange indexing
//              Should ONLY be created by the Shape class
// Date: June 10, 2026
// @ADMINGUOYU

#ifndef _UTILS_INDEXER_HPP_
#define _UTILS_INDEXER_HPP_

#include <cstdio>
#include <cstddef>  // defines: size_t
#include <cstdlib>  // malloc(); free()

namespace TENSOR_UTILITIES
{
    // predefine shape class for friend function declaration
    // (for friend function parameter type) - prototype only
    class Shape;


    // Indexing generator
    /**
     * @brief Indexing structure generator for a given shape
     *        This is a structure with a state if current indexing
     *        This generator provides a next() function to advance
     *        its internal state.
     * 
     * @note You can generate a indexing object from a helper function
     *       in the Shape class. You are hold responsible to make sure the
     *       Indexing object is still valid.
     * 
     * @note The the initial state is step 0, the maximum possible step
     *       is (max_step - 1). You may want to use do-while loop
     *       to not missing the initial state.
     */
    struct Indexer
    {
    public:
        // friend class for accessing internal states
        friend class Shape;

    private:
        // internal states
        /* shape info */
        size_t dim_count{0};
        size_t * shape {nullptr};
        /* current indexing state */
        size_t * current_idx {nullptr};
        /* step */
        size_t step {0};
        size_t max_step {0};

        /* stride */
        size_t stride {1};  // defaults to 1

    private:
        // private constructor (only accessible by friend class Shape)
        Indexer(void) = default;

    public:
        // destructor
        ~Indexer(void)
        {
            // de-allocate memory for shape and current_idx
            free(this->shape);
            free(this->current_idx);
            return;
        }
        // copy is not allowed since this is a stateful generator
        Indexer(const Indexer & other) = delete;
        // move constructor
        Indexer(Indexer && other)
            :   dim_count(other.dim_count),
                shape(other.shape),
                current_idx(other.current_idx),
                step(other.step),
                max_step(other.max_step),
                stride(other.stride)
        {
            // reset the other object to prevent double free
            other.dim_count = 0;
            other.shape = nullptr;
            other.current_idx = nullptr;
            other.step = 0;
            other.max_step = 0;
            return;
        }

        // copy assignment is not allowed
        const Indexer & operator= (const Indexer & other) = delete;
        // move assignment
        const Indexer & operator= (Indexer && other)
        {
            if (this != &other)
            {
                // de-allocate current resources
                free(this->shape);
                free(this->current_idx);
                // move resources from other
                this->dim_count = other.dim_count;
                this->shape = other.shape;
                this->current_idx = other.current_idx;
                this->step = other.step;
                this->max_step = other.max_step;
                this->stride = other.stride;
                // reset the other object to prevent double free
                other.dim_count = 0;
                other.shape = nullptr;
                other.current_idx = nullptr;
                other.step = 0;
                other.max_step = 0;
                other.stride = 1;
            }
            return *this;
        }

    // getter functions
    public:
        /**
         * @brief Gets current step of the indexing.
         * @return The current step.
         */
        size_t get_step(void) const
        {
            return this->step;
        }
        /**
         * @brief Gets the maximum step of the indexing.
         * @return The maximum step.
         */
        size_t get_max_step(void) const
        {
            return this->max_step;
        }
        /**
         * @brief Gets current stride
         */
        size_t get_stride(void) const
        {
            return this->stride;
        }

    // Setter function
    public:
        /**
         * @brief Sets the stride for the indexing.
         * @param stride The stride to set.
         * @note  stride is defaulted to 1
         *        call this without argument to reset stride
         */
        void set_stride(size_t stride = 1)
        {
            // stride 0 is not acceptble
            if (stride > 0)
                this->stride = stride;
            return;
        }

    public:
        // next() function to advance the internal state
        /**
         * @brief Advances the internal state to the next k indexing.
         * @param k The number of steps to advance.
         * @param prev Whether to advance to the previous indexing instead
         *        of the next indexing.
         * @return True if successfully advanced to the next indexing,
         *         false if we have reached the end of indexing.
         */
        bool next(size_t k, bool prev = false)
        {
            // we calculate the actual step to move based on the stride
            k *= this->stride;

            // move forward
            if (!prev)
            {
                // if step + k exceeds max_step, we cannot advance
                if (this->step + k >= this->max_step)
                    return false;

                // advance the current_idx by k steps
                // we add k to the last dimension and
                // carry over if it exceeds the shape
                for (size_t i = this->dim_count; i > 0; --i)
                // we do this since size_t is unsigned
                {
                    size_t dim_idx = i - 1;
                    this->current_idx[dim_idx] += k;
                    if (this->current_idx[dim_idx] < this->shape[dim_idx])
                        // no carry over needed, break
                        break;
                    else
                    {
                        // carry over needed, calculate the carry and update current dimension
                        size_t carry = this->current_idx[dim_idx] / this->shape[dim_idx];
                        this->current_idx[dim_idx] = this->current_idx[dim_idx] % this->shape[dim_idx];
                        // add the carry to the next dimension in the next iteration
                        k = carry;
                    }
                }
                // increase step by k
                this->step += k;
                return true;
            }
            // move backward
            else
            {
                // if step is less than k, we cannot move backward
                if (this->step < k)
                    return false;

                // move backward the current_idx by k steps
                // we subtract k from the last dimension and
                // borrow if it goes below 0
                for (size_t i = this->dim_count; i > 0; --i)
                {
                    size_t dim_idx = i - 1;
                    if (this->current_idx[dim_idx] >= k)
                    // use comparison since size_t is unsigned
                    {
                        // no borrow needed, just subtract and break
                        this->current_idx[dim_idx] -= k;
                        break;
                    }
                    else
                    {
                        // borrow needed, calculate the borrow and update current dimension
                        size_t borrow = (k - this->current_idx[dim_idx] + this->shape[dim_idx] - 1) / this->shape[dim_idx];
                        this->current_idx[dim_idx] = (this->current_idx[dim_idx] + borrow * this->shape[dim_idx]) - k;
                        // add the borrow to the next dimension in the next iteration
                        k = borrow;
                    }
                }
                // decrease step by k
                this->step -= k;
                return true;
            }
        }
        /**
         * @brief Advances the internal state to the next indexing.
         * @return True if successfully advanced to the next indexing,
         *         false if we have reached the end of indexing.
         */
        bool next(void)
        {
            return this->next(1);
        }
        /**
         * @brief Resets the internal state to the initial indexing (all zeros).
         */
        void reset(void)
        {
            // reset current_idx to all zeros
            for (size_t i = 0; i < this->dim_count; ++i)
                this->current_idx[i] = 0;
            // reset step to zero
            this->step = 0;
            return;
        }
        /**
         * @brief Prints the current indexing state.
         */
        void print(void) const
        {
            printf("Current Indexing: ( ");
            for (size_t i = 0; i < this->dim_count; ++i)
                printf("%zu, ", this->current_idx[i]);
            printf(")\n");
            // flush stdout
            fflush(stdout);
            return;
        }
    };
}

#endif  // _UTILS_INDEXER_HPP_
