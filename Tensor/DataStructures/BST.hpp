// File: BST.hpp
// Description: Implementation of Binary Search Tree
// Date: Aug. 10, 2025
// @ADMINGUOYU

#ifndef _BST_HPP_
#define _BST_HPP_

#include <cstddef>
#include <utility>

class _BST_
{
public:
    _BST_ (void) = default;
    virtual ~_BST_ (void) = 0;
    virtual _BST_ * clone (void) const = 0;
};

template <typename KEY_t, typename VAL_t>
class BST : public _BST_
{
private:
    struct BSTNode
    {
        KEY_t key { };
        VAL_t val { };
        BSTNode * left { nullptr };
        BSTNode * right { nullptr };

        // Helpful constructor
        BSTNode (const KEY_t & key, const VAL_t & val) : key(key), val(val) { return; }
    };

private:
    BSTNode * root { nullptr };
    size_t count { };

public:
    BST (void) : root(nullptr), count(nullptr) { return; }
    BST (const BST & other) : BST() { this->operator=(other); return; }
    BST (BST && other) : BST() { this->operator=(std::move(other)); return; }
    ~BST (void) override { this->clear(); return; }
    
public:
    BST & operator= (const BST & other)
    {
        this->clear();
        this->root = this->copy(other.root);
        this->count = other.count;
        return (*this);
    }

    BST & operator= (BST && other)
    {
        this->clear();
        this->root = other.root;
        this->count = other.count;
        other.clear();
        return (*this);
    }

public:
    // accessor: get total number of nodes
    size_t get_count (void) const { return this->count; }

public:
    // clear the entire tree
    void clear (void) { if(!this->root) return; this->clear(this->root); this->root = nullptr; this->count = 0; return; }
    // insert a node
    void insert (const KEY_t & key, const VAL_t & val) { this->insert(this->root, key, val); return; }
    // remove a node with key
    void remove (const KEY_t & key) { this->remove(this->root, key); return; }
    // clone (copy) the entire object (new heap allocated object will appear)
    _BST_ * clone (void) const override { return new BST(*this); }
    // finds a value (node) with the given key (return the pointer to the value, nullptr if not found)
    VAL_t * get (const KEY_t & key) { return this->get(this->root, key); }


private:
    // internal clear function (will NOT set node to nullptr, will NOT reset this->count)
    void clear (BSTNode * node)
    {
        if (!node) return;
        // make sure left and right are all deleted
        this->clear(node->left);
        this->clear(node->right);
        // delete this node
        delete node;
        return;
    }

    // internal copy function (will NOT update count)
    BSTNode * copy (const BSTNode * node) const
    {
        // reaches the end or nothing there
        if (!node) return nullptr;
        BSTNode * to_ret = new BSTNode(node->key, node->val);
        // copy left and right
        to_ret->left = this->copy(node->left);
        to_ret->right = this->copy(node->right);
        // return
        return to_ret;
    }

    // internal insert function (WILL update count; WILL set ptr passed in)
    void insert (BSTNode * & node, const KEY_t & key, const VAL_t & val)
    {
        if (!node)
        {
            // find the right spot or there's nothing in the tree
            node = new BSTNode(key, val);
            ++this->count;
            // no need to balance here
            return;
        }
        // find the right spot to add first (recursive approach)
        if (key > node->key)
        {
            // go right
            this->insert(node->right, key, val);
        }
        else if (key < node->key)
        {
            // go left
            this->insert(node->left, key, val);
        }
        else
        {
            // key match (update value)
            node->val = val;
            // no need to balance here
            return;
        }
        // balance the tree
        node = this->balance(node);
        return;
    }

    // internal remove function (WILL update count; WILL set ptr passed in)
    void remove (BSTNode * & node, const KEY_t & key)
    {
        if (!node)
        {
            // not found, do nothing
            return;
        }
        // find the matching node (recursive approach)
        if (key > node->key)
        {
            // go right
            this->remove(node->right, key);
        }
        else if (key < node->key)
        {
            // go left
            this->remove(node->left, key);
        }
        else
        {
            // key match delete that node

            // if the target node is connected to 0 or 1 node
            if (!(node->left || node->right))
            {
                // leaf (specially treated to optimize efficiency)
                delete node;
                node = nullptr;
                --this->count;
                // in this case no need to balance on this node
                return;
            }
            else if (!node->left)
            {
                BSTNode * to_delete = node;
                node = node->right;
                delete to_delete;
                --this->count;
            }
            else if (!node->right)
            {
                BSTNode * to_delete = node;
                node = node->left;
                delete to_delete;
                --this->count;
            }
            // the node is attached to two children
            else
            {
                // Node with two children: Get the inorder successor
                BSTNode * succ = node->right;
                // go to the smallest node that is LARGER than the node to delete
                while (succ->left != nullptr) 
                    succ = succ->left;
                // Copy the inorder successor's content to this node
                node->key = succ->key;
                node->value = succ->value;
                // Delete the inorder successor (but not deleting this node)
                this->remove(node->right, node->key);
            }
            
        }
        // balance the tree
        node = this->balance(node);
        return;
    }

    // internal get function
    VAL_t * get (BSTNode * & node, const KEY_t & key)
    {
        if (!node)
        {
            // not found, return nullptr
            return nullptr;
        }
        // find the node with the given key
        if (key > node->key)
        {
            // go right
            return this->get(node->right, key);
        }
        else if (key < node->key)
        {
            // go left
            return this->get(node->left, key);
        }
        else
        {
            // found, return the address of that value
            return (&node->val);
        }
    }

public:
    // returns the height of a node
    size_t height (const BSTNode * & node) const
    {
        if (!node) return 0;
        size_t l_height = this->height(node->left);
        size_t r_height = this->height(node->right);
        return 1 + ((l_height > r_height) ? l_height : r_height);
    }

    // AVL balancing algorithm (rotate left)
    void rotate_left (BSTNode * & node)
    {
        BSTNode * new_node = node->right;
        node->right = new_node->left;
        new_node->left = node;
        node = new_node;
        return;
    }

    // AVL balancing algorithm (rotate right)
    void rotate_right (BSTNode * & node)
    {
        BSTNode * new_node = node->left;
        node->left = new_node->right;
        new_node->right = node;
        node = new_node;
        return;
    }

    // AVL balancing algorithm
    void balance (BSTNode * & node)
    {
        if (!node) return;

        int ll_height = this->height(node->left->left);
        int lr_height = this->height(node->left->right);
        int rr_height = this->height(node->right->right);
        int rl_height = this->height(node->right->left);
        int l_height = (ll_height > lr_height) ? ll_height + 1 : lr_height + 1;
        int r_height = (rr_height > rl_height) ? rr_height + 1 : rl_height + 1;

        int balance_factor = l_height - r_height;

        if (balance_factor > 1) 
        {
            // left has more nodes
            if (ll_height >= lr_height) 
            {
                this->rotate_right(node);
            } 
            else 
            {
                this->rotate_left(node->left);
                this->rotate_right(node);
            }
        } 
        else if (balance_factor < -1) 
        {
            // right has more nodes
            if (rr_height >= rl_height) 
            {
                this->rotate_left(node);
            } else {
                this->rotate_right(node->right);
                this->rotate_left(node);
            }
        }
        return;
    }

};

#endif