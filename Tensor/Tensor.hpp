#ifndef _TENSOR_HPP_
#define _TENSOR_HPP_

#include <cstdio>
#include <utility>
#include <typeinfo>
#include "dim_info.hpp"

template <typename T = float>
class Tensor
{
private:
    T *m_data {};
    dim_info m_info {};

public:
    /* constructors and destructors */
    Tensor(void) { };
    Tensor(int val, T default_val = { });
    Tensor(int row, int col, T default_val = { });
    Tensor(int height, int width, int depth, T default_val = { });
    Tensor(int * dims, int num_dims, T default_val = { }, bool delete_flag = true);
    Tensor(const char * ctrl_str, T default_val = { });
    Tensor(T *data, const dim_info & info, bool delete_flag = true);
    ~Tensor(void);

    /* copy and move */
    Tensor(const Tensor<T> & other);
    Tensor(Tensor<T> && other);

public:
    /* operator assignment */
    Tensor<T>& operator=(const Tensor<T> & other);
    Tensor<T>& operator=(Tensor<T> && other);

public:
    /* member functions */
    /* accessors / print functions */
    void print_as_vector(void) const;
    void print(void) const;

    const dim_info & get_info(void) const { return this->m_info; }
    

public:
    /* consts */
    Tensor<T> get_val_in_dims(int * dims, int num_dims, bool delete_flag = true) const;
    /* mutators */
    void init_to_value(T val);
    bool set_dim(dim_info dim);
    void squeeze(int dim = -1) { this->m_info.squeeze(dim); }
    void unsqueeze(int dim) { this->m_info.unsqueeze(dim); }
    Tensor<T>& set_val(int idx, T val);
    Tensor<T>& set_val(int * dims, int num_dims, T val, bool delete_flag = true);

private:
    enum Op { ADD = 0x2b, SUB = 0x2d, MUL = 0x2a, DIV = 0x2f };
    Tensor<T> & op_inplace(const T & val, Op op);
    Tensor<T> & op_inplace(const Tensor<T> & val, Op op);
    Tensor<T> op(const T & val, Op op) const;
    Tensor<T> op(const Tensor<T> & val, Op op) const;

public:
    bool equal(const Tensor<T>& other) const;
    bool operator== (const Tensor<T>& other) const { return this->equal(other); }
    Tensor<T> neg_inplace(void);
    Tensor<T> neg(void) const;

    Tensor<T> & operator+=(const T & val) { return (*this).op_inplace(val, Op::ADD); }
    Tensor<T> operator+(const T & val) const { return (*this).op(val, Op::ADD); }
    Tensor<T> & operator+=(const Tensor<T> & val) { return (*this).op_inplace(val, Op::ADD); }
    Tensor<T> operator+(const Tensor<T> & val) const { return (*this).op(val, Op::ADD); }

    Tensor<T> & operator-=(const T & val) { return (*this).op_inplace(val, Op::SUB); }
    Tensor<T> operator-(const T & val) const { return (*this).op(val, Op::SUB); }
    Tensor<T> & operator-=(const Tensor<T> & val) { return (*this).op_inplace(val, Op::SUB); }
    Tensor<T> operator-(const Tensor<T> & val) const { return (*this).op(val, Op::SUB); }

    Tensor<T> & mul_inplace(const T & val) { return (*this).op_inplace(val, Op::MUL); }
    Tensor<T> mul(const T & val) const { return (*this).op(val, Op::MUL); }
    Tensor<T> & mul_inplace(const Tensor<T> & val) { return (*this).op_inplace(val, Op::MUL); }
    Tensor<T> mul(const Tensor<T> & val) const { return (*this).op(val, Op::MUL); }

    Tensor<T> & div_inplace(const T & val) { return (*this).op_inplace(val, Op::DIV); }
    Tensor<T> div(const T & val) const { return (*this).op(val, Op::DIV); }
    Tensor<T> & div_inplace(const Tensor<T> & val) { return (*this).op_inplace(val, Op::DIV); }
    Tensor<T> div(const Tensor<T> & val) const { return (*this).op(val, Op::DIV); }

    // Tensor<T> mat_mul(const Tensor<T> & val);
    // Tensor<T> & mat_mul_inplace(const Tensor<T> & val);
    
};

#include "Tensor.tpp"

#endif