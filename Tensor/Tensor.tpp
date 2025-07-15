// remove this for compilation
#include "Tensor.hpp"

/* Constructors */
template <typename T>
Tensor<T>::Tensor(int num, T default_val)
    :
    m_data{ new T[num] {  }},
    m_info{ new int {num}, 1 }
{
    if (default_val != T{ })
        this->init_to_value(default_val);
    return; 
}

template <typename T>
Tensor<T>::Tensor(int row, int col, T default_val)
    :
    m_data{ new T[row * col] {  }},
    m_info{ new int[2] {row, col}, 2 }
{
    if (default_val != T{ })
        this->init_to_value(default_val);
    return; 
}

template <typename T>
Tensor<T>::Tensor(int height, int width, int depth, T default_val)
    :
    m_data{ new T[height * width * depth] {  }},
    m_info{ new int[3] {height, width, depth}, 3 }
{
    if (default_val != T{ })
        this->init_to_value(default_val);
    return; 
}

template <typename T>
Tensor<T>::Tensor(int *dims, int num_dims, T default_val, bool delete_flag)
    :
    m_info{ dims, num_dims }
{  
    this->m_data = new T[this->m_info.get_num_of_params()] {  };
    if (default_val != T{ })
        this->init_to_value(default_val);
    return;
}

template <typename T>
Tensor<T>::Tensor(const char *ctrl_str, T default_val)
    :
    m_info{ ctrl_str }
{
    this->m_data = new T[this->m_info.get_num_of_params()] { default_val };
    if (default_val != T{ })
        this->init_to_value(default_val);
    return;
}

template <typename T>
Tensor<T>::Tensor(T *data, const dim_info & info, bool delete_flag)
    :
    Tensor<T>()
{
    /* will perform deep copy */
    if (info.get_num_of_params() <= 0)
    {
        if (delete_flag) delete [] data;
        return;
    }

    this->m_data = new T [info.get_num_of_params()] { };
    for (int i = 0; i < info.get_num_of_params(); ++i)
        this->m_data[i] = data[i];
    
    this->m_info = info;
    
    if (delete_flag) delete [] data;
    return;
}

/* Destructors */
template <typename T>
Tensor<T>::~Tensor(void)
{
    delete [] this->m_data;
    return;
}

/* Copy and Move */
template <typename T>
Tensor<T>::Tensor(const Tensor<T> &other)
    :
    Tensor<T>{ }
{
    (*this) = other;
    return;
}

template <typename T>
Tensor<T>::Tensor(Tensor<T> &&other)
    :
    Tensor<T>{ }
{
    (*this) = std::move(other);
    return;
}

/* Operator */
template <typename T>
Tensor<T> &Tensor<T>::operator=(const Tensor<T> &other)
{
    delete [] this->m_data;
    this->m_info = other.m_info;

    int num_of_params = this->m_info.get_num_of_params();

    if (num_of_params > 0)
        this->m_data = new T[num_of_params] { };
    else
        this->m_data = nullptr;

    for (int i = 0; i < num_of_params; ++i)
        this->m_data[i] = other.m_data[i];
    return (*this);
}

template <typename T>
Tensor<T> &Tensor<T>::operator=(Tensor<T> &&other)
{
    delete [] this->m_data;
    this->m_data = other.m_data;
    this->m_info = std::move(other.m_info);
    other.m_data = nullptr;
    return (*this);
}

/* functions */
template <typename T>
void Tensor<T>::print_as_vector(void) const
{
    if (!this->m_data)
    {
        printf("Tensor is empty.\n");
        return;
    }

    // printing
    printf("Printing as 1D vector: \n");
    putchar('[');
    int num_of_params = this->m_info.get_num_of_params();
    for (int i = 0; i < num_of_params; ++i)
    {
        printf(" %.4f", this->m_data[i]);
        if (i != num_of_params - 1) putchar(',');
    }
    putchar(' ');
    putchar(']');
    putchar('\n');
    this->m_info.print_dim();
    // flush output buffer
    fflush(stdout);
    return;
}

template <typename T>
void Tensor<T>::print(void) const
{
    if (!this->m_data)
    {
        printf("Tensor is empty.\n");
        return;
    }

    // printing
    printf("Printing in Tensor format: \n");
    const int* dims = this->m_info.get_dims();
    int num_of_dims = this->m_info.get_num_of_dims();
    int num_of_params = this->m_info.get_num_of_params();

    // calculate number of interest (for indentation)
    int num_of_interest[num_of_dims] { };
    num_of_interest[num_of_dims - 1] = dims[num_of_dims - 1];
    for (int i = num_of_dims - 2; i > -1; --i)
        num_of_interest[i] = num_of_interest[i + 1] * dims[i];

    // variables
    int table_count = 0;
    for (int i = 0; i < num_of_params; ++i)
    {
        for (int j = 0; j < num_of_dims; ++j)
        {
            if ((i % num_of_interest[j]) == 0)
            {
                for (int k = 0; k < table_count; ++k) putchar('\t');
                if (j == (num_of_dims - 1))
                    printf("[ ");
                else
                {
                    printf("{\n");
                    ++table_count;
                }
            }
        }

        // print value
        printf(" %.4f", this->m_data[i]);

        for (int j = num_of_dims - 1; j > -1; --j)
        {
            if (((i + 1) % num_of_interest[j]) == 0)
            {
                if (j == (num_of_dims - 1))
                    printf(" ],\n");
                else
                {
                    --table_count;
                    putchar('\n');
                    for (int k = 0; k < table_count; ++k) putchar('\t');
                    printf("},\n");
                }
            }
        }
    }
    this->m_info.print_dim();
    printf("Data type: %s.\n", typeid(T).name());
    if (typeid(T) == typeid(int))
        printf("[Warning] Data type is 'int', printing issues might occur.\n");
    // flush output buffer
    fflush(stdout);
    return;
}

template <typename T>
Tensor<T> Tensor<T>::get_val_in_dims(int *dims, int num_dims, bool delete_flag) const
{
    // retrieve index
    const idx_range & range = this->m_info.get_index_range(dims, num_dims, true, delete_flag);
    
    // error checking
    if(!range.valid) return Tensor<T>();

    // return
    return Tensor<T>(&(this->m_data[range.start]), *(range.dim), false);    
}

template <typename T>
Tensor<T> Tensor<T>::mat_mul(const Tensor<T> &other) const
{
    Tensor<T> to_return = (*this);
    if (!this->m_info.multipliable(other.m_info))
    {
        printf("Incompatible for matrix multiplication: ");
        this->m_info.print_dim_only();
        printf(" * ");
        other.m_info.print_dim_only();
        putchar('\n');
        return to_return;
    }
    
    /* incomplete */
    return to_return;
}

template <typename T>
void Tensor<T>::init_to_value(T val)
{
    for (int i = 0; i < this->m_info.get_num_of_params(); ++i)
        this->m_data[i] = val;
    return;
}

template <typename T>
bool Tensor<T>::set_dim(dim_info dim)
{
    if (!this->m_info.compatible(dim))
        return false;
    else
    {
        this->m_info = dim;
        return true;
    }
}

template <typename T>
Tensor<T> &Tensor<T>::set_val(int idx, T val)
{
    if ((idx < 0) || (idx - this->m_info.get_num_of_params() > -1))
        printf("Given index %d is out of range (%d).\n", idx, this->m_info.get_num_of_params() - 1);
    else
        this->m_data[idx] = val;
    
    return (*this);
}

template <typename T>
Tensor<T> &Tensor<T>::set_val(int *dims, int num_dims, T val, bool delete_flag)
{
    // get corresponding address
    const idx_range & range = this->m_info.get_index_range(dims, num_dims, false, delete_flag);

    // set accordingly
    if (range.valid)
        for (int i = range.start; i <= range.end; ++i)
            this->m_data[i] = val;

    // return
    return (*this);
}

template <typename T>
bool Tensor<T>::equal(const Tensor<T> &other) const
{
    if (this == &other) return true;
    if (!this->m_info.equal(other.m_info))
        return false;
    if (this->m_data && other.m_data)
        for (int i = 0; i < this->m_info.get_num_of_params(); ++i)
            if (this->m_data[i] != other.m_data[i]) return false;
    return true;
}

template <typename T>
Tensor<T> Tensor<T>::operator+(const T &val)
{
    Tensor<T> to_return { *this };
    for (int i = 0; i < to_return.m_info.get_num_of_params(); ++i)
        to_return.m_data[i] += val;
    return (to_return);
}

template <typename T>
Tensor<T> & Tensor<T>::operator+=(const T &val)
{
    for (int i = 0; i < this->m_info.get_num_of_params(); ++i)
        this->m_data[i] += val;
    return (*this);
}
