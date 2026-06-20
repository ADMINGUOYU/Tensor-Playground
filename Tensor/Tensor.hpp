#ifndef _TENSOR_HPP_
#define _TENSOR_HPP_

#include <cstddef>  // defines: size_t
#include "Tensor_Utilities/Tensor_Utilities.hpp"

// This MACRO is the intermediate conversion type to use
#ifndef TENSOR_CONVERSION_INTERMEDIATE_TYPE
    #define TENSOR_CONVERSION_INTERMEDIATE_TYPE double
#endif

namespace ty
{

    /* Base class of Tensor */
    class _Tensor
    {
    // constructors
    public:

        // default constructor
        _Tensor(void) = default;

        // virtual destructor
        virtual ~_Tensor(void) = default;

        // copy constructor
        _Tensor(const _Tensor & other) = default;

        // move constructor
        _Tensor(_Tensor && other) = default;

    // copy and move assignment
    public:

        // no base class / override copy and move operators to
        // avoid problems -> No cross children type copying/moving
        _Tensor & operator= (const _Tensor & other) = delete;
        _Tensor & operator= (_Tensor && other) = delete;

        // copy function
        virtual bool copy_to (_Tensor & dest, bool make_contiguous = true) const = 0;

    // internal APIs
    protected:
        // Get the pointer to memory cell (value peaking)
        virtual const void * data (const size_t flatted_index) const = 0;
        // Get the pointer to memory cell (value editing)
        virtual void * data (const size_t flatted_index) = 0;
        // Also a pair of set (conversion) functions
        virtual bool set_as (const size_t flatted_index,
                             TENSOR_CONVERSION_INTERMEDIATE_TYPE value) = 0;

    // basic public APIs
    public:

        /* Getters */
        // Gets the reference of the shape object
        virtual const TENSOR_UTILITIES::Shape & get_shape (void) const = 0;
        // Gets the reference of the buffer object
        virtual const TENSOR_UTILITIES::Buffer & get_buffer (void) const = 0;
        // Get contiguity of current state
        virtual bool get_contiguity_state (void) const = 0;
        // Get the pointer to memory cell (value peaking)
        virtual const void * data (const size_t * multi_idx_ptr) const = 0;
        virtual const void * data (const TENSOR_UTILITIES::Indexer & indexer) const = 0;


        /* Mutators */
        // squeeze and un-squeeze prototypes
        virtual bool squeeze (void) = 0; 
        virtual bool squeeze (size_t dim) = 0;
        virtual bool squeeze (const size_t * dims, size_t dims_count) = 0;
        virtual bool unsqueeze (size_t dim) = 0;
        virtual bool unsqueeze (const size_t * dims, size_t dims_count) = 0;
        // reshape (provide a new shape array)
        // new shape should have the same total item count
        virtual bool reshape (const size_t * shape, size_t dims_count) = 0;
        virtual bool reshape_like (const TENSOR_UTILITIES::Shape & new_shape) = 0;
        // permutation
        virtual bool permute (const size_t * permute_ptr) = 0;
        // contiguous -> refresh the memory for a continuous memory layout
        virtual bool contiguous (void) = 0;
        // Get the pointer to memory cell (value editing)
        virtual void * data (const size_t * multi_idx_ptr) = 0;
        virtual void * data (const TENSOR_UTILITIES::Indexer & indexer) = 0;
        // Also a pair of set (conversion) functions
        virtual bool set_as (const size_t * multi_idx_ptr,
                             TENSOR_CONVERSION_INTERMEDIATE_TYPE value) = 0;
        virtual bool set_as (const TENSOR_UTILITIES::Indexer & indexer,
                             TENSOR_CONVERSION_INTERMEDIATE_TYPE value) = 0;
        // Allocate the tensor with a given shape (only allocation)
        // should not touch the tensor if failed
        // if success tensor should always be contiguous
        virtual bool allocate (const size_t * shape, size_t dims_count) = 0;
        virtual bool allocate_like (const TENSOR_UTILITIES::Shape & shape) = 0;
        // Initialisation (using buffer's default init function, no guarantee value)
        virtual void init (void) = 0;
        // Erase the tensor data and reset the shape info (void the tensor)
        virtual void erase (void) = 0;

        /* Utilities */
        // print function (outputs tensor in a lovely format)
        virtual void print (unsigned int precision = 6, size_t max_items = 100) const = 0;
    };


    /* Class of Tensor */
    template <typename T = float>
    class Tensor : public _Tensor
    {
    // private datamembers
    private:

        // memory container - we take the entire object here
        TENSOR_UTILITIES::MemoryContainer<T>  m_tensor_buff { };

        // pointer to shape - we take the entire object here
        TENSOR_UTILITIES::Shape m_shape { };

        // memory contiguous flag
        // for empty tensor, this should be true
        // should sync with Shape's is_contiguous() function
        bool m_contiguous { true };

    // constructors and destructor
    public:

        // default constructor
        Tensor (void) = default;

        // copy constructor
        Tensor (const Tensor & other);

        // move constructor
        Tensor (Tensor && other);

        // destructor
        ~Tensor (void) override = default;

    // copy and move assignment
    public:

        // also deletes operator='s default copy and move
        // since we have declared copy and move constructor

        // copy function
        bool copy_to (_Tensor & dest, bool make_contiguous = true) const override;

    // protected APIs
    protected:
        /* Getters */
        const void * data (const size_t flatted_index) const override;
        /* Mutators */
        void * data (const size_t flatted_index) override;
        bool set_as (const size_t flatted_index,
                     TENSOR_CONVERSION_INTERMEDIATE_TYPE value) override;

    // basic public APIs
    public:

        /* Getters */
        const TENSOR_UTILITIES::Shape & get_shape (void) const override;
        const TENSOR_UTILITIES::Buffer & get_buffer (void) const override;
        bool get_contiguity_state (void) const override;
        const void * data (const size_t * multi_idx_ptr) const override;
        const void * data (const TENSOR_UTILITIES::Indexer & indexer) const override;

        /* Mutators */
        bool squeeze (void) override;
        bool squeeze (size_t dim) override;
        bool squeeze (const size_t * dims, size_t dims_count) override;
        bool unsqueeze (size_t dim) override;
        bool unsqueeze (const size_t * dims, size_t dims_count) override;
        bool reshape (const size_t * shape, size_t dims_count) override;
        bool reshape_like (const TENSOR_UTILITIES::Shape & new_shape) override;
        bool permute (const size_t * permute_ptr) override;
        bool contiguous (void) override;
        void * data (const size_t * multi_idx_ptr) override;
        void * data (const TENSOR_UTILITIES::Indexer & indexer) override;
        bool set_as (const size_t * multi_idx_ptr,
                     TENSOR_CONVERSION_INTERMEDIATE_TYPE value) override;
        bool set_as (const TENSOR_UTILITIES::Indexer & indexer,
                     TENSOR_CONVERSION_INTERMEDIATE_TYPE value) override;
        bool allocate (const size_t * shape, size_t dims_count) override;
        bool allocate_like (const TENSOR_UTILITIES::Shape & shape) override;
        void init (void) override;
        void erase (void) override;

        /* Utilities */
        void print (unsigned int precision = 6, size_t max_items = 100) const override;
    };

} // end of namespace

// we implement in here
#include "Tensor.tpp"

#endif