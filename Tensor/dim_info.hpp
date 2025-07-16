#ifndef _DIM_INFO_HPP_
#define _DIM_INFO_HPP_

#include <cstdio>
#include <utility>

class dim_info;

struct idx_range
{
    /* Both index inclusive! */
    int start = -1;
    int end = -1;
    bool valid = false;
    dim_info* dim = nullptr;

    idx_range(void) { return; }
    idx_range(int start, int end, dim_info* dim) : start(start), end(end), valid(true), dim(dim) { return; }
    ~idx_range(void);

    void print(void) const
    {
        if (!this->valid)
            printf("The index range is invalid.\n");
        else
            printf("Index range: start(%d), end(%d).\n", this->start, this->end);
        return;
    }
};

class dim_info
{
private:
    int * m_dims { };
    int m_num_of_dims { };
    int m_num_of_params { };
public:
    /* Constructor */
    dim_info(void) : m_dims{ nullptr }, m_num_of_dims{ 0 }, m_num_of_params{ 0 } { return; }
    dim_info(int * dims, int num_dims, bool delete_flag = true) : dim_info() { this->set_dim(dims, num_dims, delete_flag); }
    dim_info(const char * ctrl_str) : dim_info() { this->set_dim(ctrl_str); }
    dim_info& operator= (const dim_info & other)
    {
        delete [] this->m_dims;
        this->m_num_of_dims = other.m_num_of_dims;
        this->m_num_of_params = other.m_num_of_params;

        if(this->m_num_of_dims > 0)
            this->m_dims = new int[this->m_num_of_dims] { 0 };
        else
            this->m_dims = nullptr;

        for (int i = 0; i < this->m_num_of_dims; ++i)
            this->m_dims[i] = other.m_dims[i];
        return (*this);
    }
    dim_info& operator= (dim_info && other)
    {
        delete [] this->m_dims;
        this->m_dims = other.m_dims; other.m_dims = nullptr;
        this->m_num_of_dims = other.m_num_of_dims;
        this->m_num_of_params = other.m_num_of_params;
        return (*this);
    }
    dim_info(const dim_info & other) : dim_info() { (*this) = other; return; }
    dim_info(dim_info && other) : dim_info() { (*this) = std::move(other); return; }

    /* destructor */
    ~dim_info(void) { delete [] this->m_dims; return; }

private:
    void correct_dim(void)
    {
        for (int i = 0; i < this->m_num_of_dims; ++i)
        {
            if (this->m_dims[i] == 0) this->m_dims[i] = 1;
        }
        return;
    }

    void calculate_num_of_params(void)
    {
        if (this->m_num_of_dims == 0) this->m_num_of_params = 0;

        this->m_num_of_params = 1;
        for (int i = 0; i < this->m_num_of_dims; ++i) this->m_num_of_params *= this->m_dims[i];
    }

public:
    /* functions */
    void set_dim(int * dims, int num_dims, bool delete_flag = true)
    {
        if ((!dims) ) return;
        delete [] this->m_dims;
        this->m_num_of_dims = num_dims;

        if (this->m_num_of_dims > 0)
            this->m_dims = new int[this->m_num_of_dims] { 0 };
        else
            this->m_dims = nullptr;

        for (int i = 0; i < this->m_num_of_dims; ++i)
            this->m_dims[i] = dims[i];
        if (delete_flag) 
            delete [] dims;
        this->correct_dim();
        this->calculate_num_of_params();
        return;
    }
    void set_dim(const char * ctrl_str)
    {
        if ((!ctrl_str) ) return;
        delete [] this->m_dims;

        int index = 0;
        // count input numbers
        for (index = 0; ctrl_str[index] != '\0'; ++index)
            if (ctrl_str[index] == ',') ++this->m_num_of_dims;
        ++this->m_num_of_dims;

        // allocate space
        if (this->m_num_of_dims > 0)
            this->m_dims = new int[this->m_num_of_dims] { 0 };
        else
            this->m_dims = nullptr;
            
        // assign
        --index;
        int loc = this->m_num_of_dims - 1;
        while ((index > -1) && (loc > -1))
        {
            int num_read { };

            for (int pow = 1; index > -1; --index)
            {
                if (ctrl_str[index] == ',') { --index; break; }
                if ((ctrl_str[index] >= '0') && (ctrl_str[index] <= '9'))
                {
                    num_read += (ctrl_str[index] - '0') * pow;
                    pow *= 10;
                }
                continue;
            }

            this->m_dims[loc] += num_read;
            --loc;
        }
        this->correct_dim();
        this->calculate_num_of_params();
        return;
    }

    const int* get_dims(void) const { return this->m_dims; }

    int* get_copy_of_dims(void) const
    {
        if (!this->m_dims) return nullptr;
        int * to_return = new int [this->m_num_of_dims] { };
        for (int i = 0; i < this->m_num_of_dims; ++i) to_return[i] = this->m_dims[i];
        return to_return;
    }

    int get_num_of_dims(void) const { return this->m_num_of_dims; }

    int get_num_of_params(void) const { return this->m_num_of_params; }

    void print_dim_only(void) const
    {
        if (!this->m_dims)
            printf("The dimension info is empty.");
        else
        {
            putchar('[');
            for (int i = 0; i < this->m_num_of_dims; ++i)
            {
                printf(" %d", this->m_dims[i]);
                if (i != this->m_num_of_dims - 1) putchar(',');
            }
            putchar(' ');
            putchar(']');
        }
        // flush output buffer
        fflush(stdout);
        return;
    }

    void print_dim(void) const
    {
        printf("Dimension Info:\n\tDim: %d\n\tDetails: ", this->m_num_of_dims);
        this->print_dim_only();
        putchar('\n');
        printf("\tNumber of parameters: %d\n", this->m_num_of_params);
        // flush output buffer
        fflush(stdout);
        return;
    }

public:
    /* Utilities */
    bool compatible(const dim_info & other) const { return this->m_num_of_params == other.m_num_of_params; }
    
    bool multipliable(const dim_info & other) const
    {
        // check if have enough dimension (also checks if empty)
        if ((this->m_num_of_dims < 2) || (other.m_num_of_dims < 2)) return false;
        // check size
        if ((this->m_dims[this->m_num_of_dims - 1] == other.m_dims[this->m_num_of_dims - 2]) 
            && 
            (this->m_dims[this->m_num_of_dims - 2] == other.m_dims[this->m_num_of_dims - 1]))
            return true;
        else
            return false;
    }
    
    bool equal(const dim_info& other) const
    {
        if (this == &other) return true;
        if ((this->m_num_of_dims != other.m_num_of_dims) || (this->m_num_of_params != other.m_num_of_params))
            return false;
        if (this->m_dims && other.m_dims)
            for (int i = 0; i < this->m_num_of_dims; ++i)
                if (this->m_dims[i] != other.m_dims[i]) return false;
        return true;
    }

    void squeeze(int dim = -1)
    {
        if (!this->m_dims) return;
        if (dim < 0)
        {
            // squeeze all
            int num_to_squeeze = 0;
            for (int i = 0; i < this->m_num_of_dims; ++i)
                if (this->m_dims[i] == 1) ++num_to_squeeze;
            if (!num_to_squeeze) return;
            int * new_dims = new int [this->m_num_of_dims - num_to_squeeze] { };
            for (int i = 0, j = 0; i < this->m_num_of_dims; ++i)
                if (this->m_dims[i] != 1)
                {
                    new_dims[j] = this->m_dims[i];
                    ++j;
                }
            delete [] this->m_dims;
            this->m_dims = new_dims;
            this->m_num_of_dims = this->m_num_of_dims - num_to_squeeze;
        }
        else
        {
            if (dim >= this->m_num_of_dims) return;
            if (this->m_dims[dim] != 1) return;
            int * new_dims = new int [this->m_num_of_dims - 1] { };
            for (int i = 0, j = 0; i < this->m_num_of_dims; ++i)
                if (this->m_dims[i] != 1)
                {
                    new_dims[j] = this->m_dims[i];
                    ++j;
                }
            delete [] this->m_dims;
            this->m_dims = new_dims;
            this->m_num_of_dims = this->m_num_of_dims - 1;
        }
        return;
    }

    void unsqueeze(int dim)
    {
        if (!this->m_dims) return;
        if (dim >= 0)
        {
            int new_num_of_dim = (dim >= this->m_num_of_dims) ? dim + 1 : this->m_num_of_dims + 1;
            int * new_dims = new int [new_num_of_dim] { };
            for (int i = 0, j = 0; i < new_num_of_dim; ++i)
            {
                if (i == dim) { new_dims[i] = 1; continue;}
                if (j < this->m_num_of_dims) { new_dims[i] = this->m_dims[j++]; continue; }
                else new_dims[i] = 1;
            }
            delete [] this->m_dims;
            this->m_dims = new_dims;
            this->m_num_of_dims = new_num_of_dim;
        }
        return;
    }

    idx_range get_index_range(int * dims, int num_dims, bool require_dim = false, bool delete_flag = true) const
    {
        /* this function returns the index range of a specific dimension */
        // not a valid dimension index
        if ((num_dims < 0) || (num_dims - this->m_num_of_dims > 0))
        {
            printf("Given number of dimensions out of range.\n");
            if (delete_flag) delete [] dims;
            return idx_range();
        }

        // get size of the full set (left)
        int size_left = this->m_num_of_params;

        // set starting index and end
        int start = 0;
        int end = 0;

        // iterate through every indexing elements
        for (int i = 0; i < num_dims; ++i)
        {
            // error checking
            if ((dims[i] < 0) || (dims[i] - this->m_dims[i] > -1))
            {
                printf("Given index %d in dimension %d is out of range (%d).\n", dims[i], i, this->m_dims[i] - 1);
                if (delete_flag) delete [] dims;
                return idx_range();
            }
            size_left /= this->m_dims[i];
            start += dims[i] * size_left;
        }
        // calculate end
        end = start + size_left - 1;

        // calculate dimension info
        dim_info * dim {};
        if (require_dim)
        {
            if (this->m_num_of_dims == num_dims)
                dim = new dim_info(new int[1] { 1 }, 1, true);
            else
            {
                int new_num_dims = this->m_num_of_dims - num_dims;
                int* new_dims = new int[new_num_dims + 1] { 1 };
                for (int i = 1; i <= new_num_dims; ++i)
                    new_dims[i] = this->m_dims[num_dims + i - 1];
                dim = new dim_info(new_dims, new_num_dims + 1, false);
                delete [] new_dims;
            }
        }

        // delete if necessary
        if (delete_flag)
            delete [] dims;
        
        // return
        return idx_range(start, end, dim);
    }

public:
    /* operator */
    bool operator==(const dim_info& other) const { return this->equal(other); }
};


// add definition of destructor if class idx_info
idx_range::~idx_range(void)
{
    delete this->dim; 
    return; 
}

#endif