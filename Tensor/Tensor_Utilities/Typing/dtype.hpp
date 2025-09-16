#ifndef _UTILS_DTYPE_HPP_
#define _UTILS_DTYPE_HPP_

#include "../Enum/Enum.hpp"
#include <cstdint>
#include <ostream>

namespace TENSOR_UTILITIES
{
    using grad_t = float;

    template <typename T>
    struct _BaseDataType_
    {
    public:
        T data { };
        grad_t grad { };
    
    public:
        _BaseDataType_ (void) = default;
        _BaseDataType_ (T d) : data(d), grad() { }
    
    public:
        const T & item (void) const { return this->data; }
        _BaseDataType_ clone (void) const
        { return (*this); }

    public:
        template <typename S>
        _BaseDataType_ (const _BaseDataType_<S> & other) : _BaseDataType_()
        { (*this) = other; return; }
    
        template <typename S>
        _BaseDataType_ & operator= (const _BaseDataType_<S> & other)
        {
            this->data = (S)other.data;
            this->grad = other.grad;
            return (*this);
        }

        template <typename S>
        _BaseDataType_ & operator= (const S & other)
        {
            this->data = other;
            return (*this);
        }
    
    public:
        bool operator== (const _BaseDataType_ & other) const
        {
            return (this->data == other.data);
        }

        bool operator!= (const _BaseDataType_ & other) const
        {
            return !this->operator==(other);
        }
    
    public:
        template <typename S, typename P>
        friend _BaseDataType_<S> & op_inplace (_BaseDataType_<S> & current, const _BaseDataType_<P> & other, Op op);

        template <typename S, typename P>
        friend _BaseDataType_<S> & operator+= (_BaseDataType_<S> & current, const _BaseDataType_<P> & other);
        
        template <typename S, typename P>
        friend _BaseDataType_<S> & operator-= (_BaseDataType_<S> & current, const _BaseDataType_<P> & other);
    
        template <typename S, typename P>
        friend _BaseDataType_<S> & operator*= (_BaseDataType_<S> & current, const _BaseDataType_<P> & other); 
    
        template <typename S, typename P>
        friend _BaseDataType_<S> & operator/= (_BaseDataType_<S> & current, const _BaseDataType_<P> & other);

        template <typename S, typename P>
        friend _BaseDataType_<S> & operator+ (const _BaseDataType_<S> & current, const _BaseDataType_<P> & other); 

        template <typename S, typename P>
        friend _BaseDataType_<S> & operator- (const _BaseDataType_<S> & current, const _BaseDataType_<P> & other);
        
        template <typename S, typename P>
        friend _BaseDataType_<S> & mul (const _BaseDataType_<S> & current, const _BaseDataType_<P> & other);

        template <typename S, typename P>
        friend _BaseDataType_<S> & div (const _BaseDataType_<S> & current, const _BaseDataType_<P> & other);
    };

    template <typename S, typename P>
    _BaseDataType_<S> & op_inplace (_BaseDataType_<S> & current, const _BaseDataType_<P> & other, Op op)
    {
        switch (op)
        {
        case Op::ADD:
            current.data += other.data;
            break;
        case Op::SUB:
            current.data -= other.data;
            break;
        case Op::MUL:
            current.data *= other.data;
            break;
        case Op::DIV:
            current.data /= other.data;
            break;
        default:
            break;
        }
        return (current);
    }
    
    template <typename S, typename P>
    _BaseDataType_<S> & operator+= (_BaseDataType_<S> & current, const _BaseDataType_<P> & other) 
    {  return op_inplace(current, other, Op::ADD); }

    template <typename S, typename P>
    _BaseDataType_<S> & operator-= (_BaseDataType_<S> & current, const _BaseDataType_<P> & other) 
    {  return op_inplace(current, other, Op::SUB); }

    template <typename S, typename P>
    _BaseDataType_<S> & operator*= (_BaseDataType_<S> & current, const _BaseDataType_<P> & other) 
    {  return op_inplace(current, other, Op::MUL); }

    template <typename S, typename P>
    _BaseDataType_<S> & operator/= (_BaseDataType_<S> & current, const _BaseDataType_<P> & other) 
    {  return op_inplace(current, other, Op::DIV); }

    template <typename S, typename P>
    _BaseDataType_<S> & operator+ (const _BaseDataType_<S> & current, const _BaseDataType_<P> & other) 
    {  _BaseDataType_<S> cl = current.clone(); return op_inplace(cl, other, Op::ADD); }

    template <typename S, typename P>
    _BaseDataType_<S> & operator- (const _BaseDataType_<S> & current, const _BaseDataType_<P> & other) 
    {  _BaseDataType_<S> cl = current.clone(); return op_inplace(cl, other, Op::SUB); }
    
    template <typename S, typename P>
    _BaseDataType_<S> & mul (const _BaseDataType_<S> & current, const _BaseDataType_<P> & other) 
    {  _BaseDataType_<S> cl = current.clone(); return op_inplace(cl, other, Op::MUL); }

    template <typename S, typename P>
    _BaseDataType_<S> & div (const _BaseDataType_<S> & current, const _BaseDataType_<P> & other) 
    {  _BaseDataType_<S> cl = current.clone(); return op_inplace(cl, other, Op::DIV); }

    template <typename T>
    std::ostream & operator<< (std::ostream & os, const _BaseDataType_<T> & d)
    {
        os << d.data;
        return os;
    }

    typedef _BaseDataType_<float> FLOAT32;
    typedef _BaseDataType_<double> FLOAT64;
    typedef _BaseDataType_<int16_t> INT16;
    typedef _BaseDataType_<int64_t> INT64;

}

#endif