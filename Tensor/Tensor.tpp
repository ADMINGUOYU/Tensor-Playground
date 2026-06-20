/**
 * @file Tensor.tpp
 * @brief Implementation of the Tensor class template
 * @author ADMINGUOYU
 */

// Include the header file for the Tensor class template
#include "Tensor.hpp"
// include used standard libraries
#include <cstddef>  // defines: size_t; ptrdiff_t
#include <utility>  // std::move()
#include <typeinfo> // typeid()
#include <cstdio>   // prinf()
#include <cstdlib>  // malloc(); free()

/**
 * @brief Copy constructor
 * @param other The source tensor to copy from.
 * @note We will make the cloned Tensor contiguous by default
 */
template <typename T>
inline ty::Tensor<T>::Tensor(const Tensor &other)
    :
    Tensor<T>()  // delegate to default constructor (init empty tensor)
{
    // use the copy function to copy data from other to this
    other.copy_to(*this, true);
    // return
    return;
}

/**
 * @brief Move constructor
 * @param other The source tensor to move from.
 * @note We will move everything, so the source tensor will be in an empty state
 *       after this operation.
 * @note Pure move, the contiguity state will be moved as is,
 *       no change will be made.
 */
template <typename T>
inline ty::Tensor<T>::Tensor(Tensor &&other)
{
    // move shape info (this will also move stride info)
    this->m_shape = std::move(other.m_shape);
    // move tensor data
    TENSOR_UTILITIES::move_assign(this->m_tensor_buff, std::move(other.m_tensor_buff));
    // move contiguity state
    this->m_contiguous = other.m_contiguous;
    // the other's contiguity should be true -> empty tensor
    other.m_contiguous = true;
    // return
    return;
}

/**
 * @brief copy function
 * @param dest The destination tensor to copy to.
 * @param make_contiguous If true, the destination tensor will
 *        be made contiguous after copying.
 * @return True if successful, false otherwise.
 * @note If different type, we will still try to copy by assignment
 *       (operator=), but if the type is not compatible, the behavior
 *       is undefined.
 *       [NOTE: for mismatched type, process will fail if make_contiguous
 *       is set to false]
 */
template <typename T>
inline bool ty::Tensor<T>::copy_to(_Tensor &dest, bool make_contiguous) const
{
    // same type flag
    const bool same_type = (typeid(dest) == typeid(*this));

    // check if the destination is the same type
    if (same_type)
    {
        // check if self
        if (&dest == this)
            return true;

        // cast the destination to the correct type
        Tensor<T> & dest_tensor = static_cast<Tensor<T>&>(dest);

        // check if the source is contiguous or we don't want to make it
        // contiguous
        // if so, we call the faster copy function for contiguous memory
        if ((this->m_contiguous) || (!make_contiguous))
        {
            // call the faster copy function for contiguous memory
            TENSOR_UTILITIES::copy_assign(dest_tensor.m_tensor_buff, this->m_tensor_buff);
            // also copy shape info (this will also copy stride info)
            dest_tensor.m_shape = this->m_shape;
            // set contiguity state
            dest_tensor.m_contiguous = this->m_contiguous;
            // return
            return true;
        }
    }
    else
    {
        // type mismatched copy
        // fail if make_contiguous is false
        if (!make_contiguous)
            return false;
    }

    /* General fall-back */
    /* Cases are:
       (same type, not contiguous, make_contiguous is true) - x1
       (different type, make_contiguous is true) - x2
    */
    // first, we collect useful info of the source tensor
    const TENSOR_UTILITIES::Shape & src_shape = this->m_shape;
    // get the item count
    size_t item_count = src_shape.get_item_count();
    // if count is 0, we reset dest to an empty state and return true
    if (item_count == 0) { dest.erase(); return true; }
    // get the indexer for the source tensor
    TENSOR_UTILITIES::Indexer src_indexer = src_shape.generate_indexer();
    // allocate the destination tenser
    if (!dest.allocate_like(src_shape))
    {
        // if allocation failed, return false
        // allocation does not touch the tensor if failed
        return false;
    }
    // if same type
    if (same_type)
    {
        // cast the destination to the correct type
        Tensor<T> & dest_tensor = static_cast<Tensor<T>&>(dest);

        // iterate indexer
        for (size_t i = 0; i < item_count; ++i)
        {
            dest_tensor.m_tensor_buff.set(i,
                this->m_tensor_buff.get(src_shape.get_flattened_index(src_indexer)));
            src_indexer.next();
        }
    }
    // not same type
    else
    {
        // intermediate buffer
        TENSOR_CONVERSION_INTERMEDIATE_TYPE buff { };
        // iterate indexer
        for (size_t i = 0; i < item_count; ++i)
        {
            // get and cast value from T* to
            // the intermidiate type (TENSOR_CONVERSION_INTERMEDIATE_TYPE)
            buff = static_cast<TENSOR_CONVERSION_INTERMEDIATE_TYPE>(
                *((T*)this->m_tensor_buff.get(src_shape.get_flattened_index(src_indexer)))
            );
            // set value
            dest.set_as(i, buff);
            // move to the next index
            src_indexer.next();
        }
    }
    // allocation like has already set the contiguity state
    // return
    return true;
}

/**
 * @brief [INTERNAL] Get data (const version) using a flatted index
 * @param flatted_index The flatted index to access the data.
 * @return A pointer to the data at the specified flatted index.
 *         returns nullptr if the index is out of bounds.
 * @note The flatted index is the index in the contiguous memory layout,
 *       if the tensor is not contiguous, the flatted index might NOT
 *       be what you want. [INTERNAL USE ONLY]
 */
template <typename T>
inline const void *ty::Tensor<T>::data(const size_t flatted_index) const
{
    // using the buffer get function
    return this->m_tensor_buff.get(flatted_index);
}

/**
 * @brief [INTERNAL] Get data (editable version) using a flatted index
 * @param flatted_index The flatted index to access the data.
 * @return A pointer to the data at the specified flatted index.
 *         returns nullptr if the index is out of bounds.
 * @note The flatted index is the index in the contiguous memory layout,
 *       if the tensor is not contiguous, the flatted index might NOT
 *       be what you want. [INTERNAL USE ONLY]
 */
template <typename T>
inline void *ty::Tensor<T>::data(const size_t flatted_index)
{
    // using the buffer get function
    return this->m_tensor_buff.get(flatted_index);
}

/**
 * @brief [INTERNAL] Set data using a flatted index (using a intermediate type)
 * @param flatted_index The flatted index to access the data.
 * @param value The value to set at the specified flatted index.
 * @return True if successful, false otherwise.
 * @note The flatted index is the index in the contiguous memory layout,
 *       if the tensor is not contiguous, the flatted index might NOT
 *       be what you want. [INTERNAL USE ONLY]
 */
template <typename T>
inline bool ty::Tensor<T>::set_as(const size_t flatted_index, TENSOR_CONVERSION_INTERMEDIATE_TYPE value)
{
    // get the memory address first
    T* ptr = (T*)this->data(flatted_index);
    // error checking
    if (!ptr)
        return false;
    // assign the value
    *ptr = static_cast<T>(value);
}

/**
 * @brief Get the shape of the tensor
 * @return A reference to the shape object of the tensor.
 */
template <typename T>
inline const TENSOR_UTILITIES::Shape &ty::Tensor<T>::get_shape(void) const
{
    return this->m_shape;
}

/**
 * @brief Get the buffer of the tensor
 * @return A reference to the buffer object of the tensor.
 */
template <typename T>
inline const TENSOR_UTILITIES::Buffer &ty::Tensor<T>::get_buffer(void) const
{
    return this->m_tensor_buff;
}

/**
 * @brief Get the contiguity state of the tensor
 * @return True if the tensor is contiguous, false otherwise.
 */
template <typename T>
inline bool ty::Tensor<T>::get_contiguity_state(void) const
{
    return this->m_contiguous;
}

/**
 * @brief Get data (const version) using a multi-dimensional index
 * @param multi_idx_ptr Pointer to an array containing the multi-dimensional index.
 * @return A pointer to the data at the specified multi-dimensional index.
 *         returns nullptr if the index is out of bounds.
 */
template <typename T>
inline const void *ty::Tensor<T>::data(const size_t *multi_idx_ptr) const
{
    // get the flatted index using the shape info
    size_t flatted_index = this->m_shape.get_flattened_index(multi_idx_ptr);
    // if the returned value is 0, we further check whether
    // it is really 0 or invalid index
    if (flatted_index == 0)
    {
        const size_t & count = this->m_shape.get_dim_count();
        for (size_t i = 0; i < count; ++i)
            // flatted_index is 0 only when all indexes are 0
            if (multi_idx_ptr != 0)
                return nullptr
    }
    // get the memory location
    return this->data(flatted_index);
}

/**
 * @brief Get data (const version) using a Indexer object
 * @param indexer we will use the current step of the indexer
 */
template <typename T>
inline const void *ty::Tensor<T>::data(const TENSOR_UTILITIES::Indexer &indexer) const
{
    // we still make the out-of-bound check (we do not know if this is the right indexer)
    return this->data(indexer.current_idx)
}

/**
 * @brief Squeeze the tensor (remove ALL dimensions of size 1)
 * @return True if successful, false otherwise
 * @note If the tensor is already squeezed, this function will do
 *       nothing and return true
 */
template <typename T>
inline bool ty::Tensor<T>::squeeze(void)
{
    // we only temper with the shape info
    return this->m_shape.squeeze();
}

/**
 * @brief Squeeze the tensor (remove a specified dimension)
 * @param dim The dimension to remove (0-based index)
 * @return True if successful, false otherwise
 */
template <typename T>
inline bool ty::Tensor<T>::squeeze(size_t dim)
{
    // we only temper with the shape info
    return this->m_shape.squeeze(dim);
}

/**
 * @brief Squeeze the tensor (remove a specified dimensions)
 * @param dims Pointer to an array containing the dimensions to remove
 *        (0-based index)
 * @return True if successful, false otherwise
 */
template <typename T>
inline bool ty::Tensor<T>::squeeze(const size_t *dims, size_t dims_count)
{
    // we only temper with the shape info
    return this->m_shape.squeeze(dims, dims_count);
}

/**
 * @brief Unsqueeze the tensor (add a dimension of size 1 at a specified position)
 * @param dim The position to add the new dimension (0-based index)
 * @return True if successful, false otherwise
 */
template <typename T>
inline bool ty::Tensor<T>::unsqueeze(size_t dim)
{
    // we only temper with the shape info
    return this->m_shape.unsqueeze(dim);
}

/**
 * @brief Unsqueeze the tensor (add dimensions of size 1 at specified positions)
 * @param dims Pointer to an array containing the positions to add the new dimensions
 *        (0-based index)
 * @param dims_count The number of dimensions to add
 * @return True if successful, false otherwise
 */
template <typename T>
inline bool ty::Tensor<T>::unsqueeze(const size_t *dims, size_t dims_count)
{
    // we only temper with the shape info
    return this->m_shape.unsqueeze(dims, dims_count);
}

/**
 * @brief Reshape the tensor to a new shape
 * @param shape Pointer to an array containing the new shape (size of each dimension)
 *        the total number of parameters should match the current shape
 * @param dims_count The number of dimensions in the new shape
 * @return True if successful, false otherwise
 * @note If the reshape operation fails (e.g., due to incompatible shapes), the original tensor
 *       will remain unchanged
 */
template <typename T>
inline bool ty::Tensor<T>::reshape(const size_t *shape, size_t dims_count)
{
    // create a new shape object based on the given new shape
    TENSOR_UTILITIES::Shape new_shape { };
    new_shape.set_shape(shape, dims_count);

    // call the other version of reshape
    return this->reshape_like(new_shape);
}

/**
 * @brief Reshape the tensor to a new shape (using a Shape object)
 * @param new_shape The new shape object to reshape the tensor to.
 * @return True if successful, false otherwise
 * @note If the reshape operation fails (e.g., due to incompatible shapes), the original tensor
 *       will remain unchanged
 */
template <typename T>
inline bool ty::Tensor<T>::reshape_like(const TENSOR_UTILITIES::Shape & shape)
{
    // check if our tensor is contiguous,
    // if so, we can just copy the data and set the new shape
    // faster
    if (this->m_contiguous)
    {
        // set the new shape
        TENSOR_UTILITIES::Shape new_shape = shape;
        // reset stride
        if (!new_shape.reset_permutation())
            return false;
        // set the new shape
        this->m_shape = std::move(new_shape);
        // update contiguity flag (one can ignore this though)
        // this->m_contiguous = this->m_shape.is_contiguous();
        // return
        return true;
    }

    // We first check if we can create a view that does not
    // require copy of data, we don't want to call contiguous so fast
    TENSOR_UTILITIES::Shape viewable = this->m_shape.viewable_as(new_shape);
    // if viewable is not empty, we can just set the new shape to the viewable shape
    if (viewable.get_dim_count() != 0)
    {
        // move the new shape
        this->m_shape = std::move(viewable);
        // update contiguity flag
        this->m_contiguous = this->m_shape.is_contiguous();
        return true;
    }

    // if not viewable, we have to make it contiguous first, then set the new shape
    if (!this->contiguous())
        return false;
    // make a recursive call to this function (we will fall into our first
    // base-case)
    return this->reshape_like(shape);
}

/**
 * @brief Permute the dimensions of the tensor according to a given order
 * @param permute_ptr Pointer to an array containing the new order of dimensions
 *        (0-based index)
 * @return True if successful, false otherwise
 */
template <typename T>
inline bool ty::Tensor<T>::permute(const size_t *permute_ptr)
{
    // we only temper with the shape info
    if (!this->m_shape.permute(permute_ptr))
        return false;
    // update contiguity flag
    this->m_contiguous = this->m_shape.is_contiguous();
    // return
    return true;
}

/**
 * @brief Make the tensor contiguous (refresh the memory for a continuous memory layout)
 * @return True if successful, false otherwise
 * @note If the tensor is already contiguous, this function will do
 *       nothing and return true
 */
template <typename T>
inline bool ty::Tensor<T>::contiguous(void)
{
    // check if already contiguous
    // we do strict check here
    if (this->m_contiguous ^ this->m_shape.is_contiguous())
        return false; // Tensor is problematic [STOP immediately]
    if (this->m_contiguous)
        return true;  // already contiguous, do nothing

    // we create a new tensor and copy
    // copy_to will make sure to set m_contiguous correctly
    Tensor<T> tensor { };
    if (!this->copy_to(tensor, true))
        return false;

    // move back
    *this = std::move(tensor);

    // return true
    return true
}

/**
 * @brief Get data (editable version) using a multi-dimensional index
 * @param multi_idx_ptr Pointer to an array containing the multi-dimensional index.
 * @return A pointer to the data at the specified multi-dimensional index.
 *         returns nullptr if the index is out of bounds.
 */
template <typename T>
inline void *ty::Tensor<T>::data(const size_t *multi_idx_ptr)
{
    // get the flatted index using the shape info
    size_t flatted_index = this->m_shape.get_flattened_index(multi_idx_ptr);
    // if the returned value is 0, we further check whether
    // it is really 0 or invalid index
    if (flatted_index == 0)
    {
        const size_t & count = this->m_shape.get_dim_count();
        for (size_t i = 0; i < count; ++i)
            // flatted_index is 0 only when all indexes are 0
            if (multi_idx_ptr != 0)
                return nullptr
    }
    // get the memory location
    return this->data(flatted_index);
}

/**
 * @brief Get data (editable version) using a Indexer object
 * @param indexer we will use the current step of the indexer
 */
template <typename T>
inline void *ty::Tensor<T>::data(const TENSOR_UTILITIES::Indexer &indexer)
{
    // we still make the out-of-bound check (we do not know if this is the right indexer)
    return this->data(indexer.current_idx)
}

/**
 * @brief Set data using a a multi-dimensional index
 * @param multi_idx_ptr Pointer to an array containing the multi-dimensional index.
 * @param value The value to set at the specified flatted index.
 * @return True if successful, false otherwise.
 */
template <typename T>
inline bool ty::Tensor<T>::set_as(const size_t *multi_idx_ptr, TENSOR_CONVERSION_INTERMEDIATE_TYPE value)
{
    // get the memory address first
    T* ptr = (T*)this->data(multi_idx_ptr);
    // error checking
    if (!ptr)
        return false;
    // assign the value
    *ptr = static_cast<T>(value);
}

/**
 * @brief Set data using a a multi-dimensional index
 * @param indexer we will use the current step of the indexer
 * @param value The value to set at the specified flatted index.
 * @return True if successful, false otherwise.
 */
template <typename T>
inline bool ty::Tensor<T>::set_as(const TENSOR_UTILITIES::Indexer &indexer, TENSOR_CONVERSION_INTERMEDIATE_TYPE value)
{
    // get the memory address first
    T* ptr = (T*)this->data(indexer);
    // error checking
    if (!ptr)
        return false;
    // assign the value
    *ptr = static_cast<T>(value);
}

/**
 * @brief Allocate memory for the tensor based on a new shape
 * @param shape Pointer to an array containing the new shape (size of each dimension)
 * @param dims_count The number of dimensions in the new shape
 * @return True if successful, false otherwise
 * @note This will allocate memory for the tensor based on the new shape,
 *       and set the shape info accordingly. If allocation fails, the original
 *       tensor will remain unchanged.
 * @note After successful allocation, the tensor will be in a contiguous state.
 * @note WE DO NOT INITIALIZE THE VALUES IN THE TENSOR, THEY MAY BE GARBAGE VALUES
 */
template <typename T>
inline bool ty::Tensor<T>::allocate(const size_t *shape, size_t dims_count)
{
    // create a new shape object
    TENSOR_UTILITIES::Shape shape_obj { };
    if (!shape_obj.set_shape(shape, dims_count))
        return false;
    // call allocate_like
    return this->allocate_like(shape_obj);
}

/**
 * @brief Allocate memory for the tensor based on a new shape
 * @param shape A shape object (to reference its shape)
 * @return True if successful, false otherwise
 * @note This will allocate memory for the tensor based on the new shape,
 *       and set the shape info accordingly. If allocation fails, the original
 *       tensor will remain unchanged.
 * @note After successful allocation, the tensor will be in a contiguous state.
 * @note WE DO NOT INITIALIZE THE VALUES IN THE TENSOR, THEY MAY BE GARBAGE VALUES
 */
template <typename T>
inline bool ty::Tensor<T>::allocate_like(const TENSOR_UTILITIES::Shape &shape)
{
    // create a new Tensor object
    Tensor<T> tensor { };
    // get item count
    size_t count = shape.get_item_count();
    // allocate memory
    if (!tensor.m_tensor_buff.allocate(count))
        return false;
    if (!tensor.m_tensor_buff.set_effective_size(count))
        return false;
    // set shape
    tensor.m_shape = shape;
    if (!tensor.m_shape.reset_permutation())
        return false;
    // set contiguity flag
    tensor.m_contiguous = true;

    // move the temp tensor to this
    *this = std::move(tensor);

    // return
    return true;
}

/**
 * @brief Initialize the tensor (using buffer's default init function, no guarantee value)
 */
template <typename T>
inline void ty::Tensor<T>::init(void)
{
    // call buffer initialise function
    this->m_tensor_buff.init_all();
    // return
    return;
}

/**
 * @brief Erase the tensor data and reset the shape info (void the tensor)
 */
template <typename T>
inline void ty::Tensor<T>::erase(void)
{
    // drop buffer and shape info
    this->m_tensor_buff.erase();
    this->m_shape = TENSOR_UTILITIES::Shape { };
    // reset contiguity flag
    this->m_contiguous = true;
    // return
    return;
}

/**
 * @brief Print the tensor
 * @param precision The number of decimal places to display for floating-point types
 * @param max_items The maximum number of items to print (for large tensors)
 * @note We only support tensor in float/double/int
 *       (%f for float, %lf for double, %d for int)
 *       For other types, we will just print the type name and shape info
 * @note You can implement more powerful print function yourself
 */
template <typename T>
inline void ty::Tensor<T>::print(unsigned int precision, size_t max_items) const
{
    // Print shape info and header
    printf("ty::Tensor<%s> with ", typeid(T).name());
    this->m_shape.print(); // contains '\n'

    // check type and set format string
    // this syntax causes runtime overhead
    // (constexpr if-else is available in C++17)
    const char* format_str = nullptr;
    bool integer_type = false;
    if (typeid(T) == typeid(float))
        format_str = "%.*f";
    else if (typeid(T) == typeid(double))
        format_str = "%.*lf";
    else if (typeid(T) == typeid(int))
    {
        format_str = "%d";
        integer_type = true;
    }
    else
        // for unsupported types, return
        return;

    /* ---------- START PRINTING ---------- */

    // Get tensor information
    size_t dimension_count = this->m_shape.get_dim_count();
    TENSOR_UTILITIES::Indexer indexer = this->m_shape.generate_indexer();
    size_t param_count = indexer.get_max_step();

    // calculate number of interest (for indentation)
    size_t * indentation_mark = (size_t *)malloc(dimension_count * sizeof(size_t));
    // first interested should be total number of items
    indentation_mark[0] = param_count;
    // the rest? we divide by (i-1)th dimension
    for (size_t i = 1; i < dimension_count; ++i)
        indentation_mark[i] = indentation_mark[i - 1] / this->m_shape.get_shape(i - 1);

    // starts printing
    size_t total_to_print = (param_count < max_items) ? param_count : max_items;
    size_t indentation_count = 0;
    const char * indentation = "  ";
    // LOOP EACH ELEMENT TO PRINT (i is the indexer location)
    for (size_t i = 0; i < total_to_print; ++i)
    {
        // CHECK IF WE ARE GOING TO INDENT
        for (size_t j = 0; j < dimension_count; ++j)
        {
            if ((i % num_of_interest[j]) == 0)
            {
                // print indentation
                for (int k = 0; k < indentation_count; ++k) printf(indentation);
                // print bracket (if last intrested dimension, use '[')
                if (j == (dimension_count - 1))
                    printf("[ ");
                else
                {
                    // check if next interested number is the same
                    // if so, we do not make a new line
                    if (num_of_interest[j] == num_of_interest[j + 1])
                        printf("{");
                    else
                    {
                        printf("{\n");
                        ++indentation_count;
                    }
                }
            }
        }

        // get ptr of the value to print
        const T * ptr = (const T *)this->data(indexer);
        // advance indexer
        indexer.next();

        // print value
        if (integer_type)
            printf(format_str, ptr)
        else
            printf(format_str, precision, ptr);

        // CHECK IF WE ARE GOING TO INDENT
        for (ptrdiff_t j = ((ptrdiff_t)(dimension_count) - 1);
            j > -1;
            --j)
        {
            if (((i + 1) % num_of_interest[j]) == 0)
            {
                // print bracket (if last intrested dimension, use ']')
                if (j == (dimension_count - 1))
                    printf(" ],");
                else
                {
                    // check if next interested number is the same
                    // if so, we do not make a new line
                    // since we have not made a new line when printing '{'
                    if (num_of_interest[j] != num_of_interest[j + 1])
                    {
                        // prints new line if not
                        printf("\n");
                        // print indentation
                        --indentation_count;
                        for (int k = 0; k < indentation_count; ++k) printf(indentation);
                    }
                    // finally, prints '}'
                    printf("},");
                }
            }
        }
    }

    // print final new line and if we have not printed all items,
    // print a notice
    if (total_to_print < param_count)
        printf("\n... (truncated %zu items)\n", param_count - total_to_print);
    else
        printf("\n");

    /* ---------- END PRINTING ---------- */

    // free indentation_mark
    free(indentation_mark);

    // flush output buffer
    fflush(stdout);
    return;
}
