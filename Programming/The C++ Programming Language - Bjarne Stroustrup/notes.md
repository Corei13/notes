---
layout: page
title: The C++ Programming Language
author: Bjarne Stroustrup
category: Programming
tags: C++
---
- **Part I - Introductory Materials**
    - *Abstraction Mechanisms*
        - implementation inheritance (70)
        - unique_ptr (72)
        - move constructors (75)
        - suppressing operations (77)
            Using the default copy or move for a class in a hierarchy is typically a disaster: given only a pointer to a base, we simply don’t know what members the derived class has (§3.2.2), so we can’t know how to copy them. So, the best thing to do is usually to delete the default copy and move operations, that is, to eliminate the default definitions of those two operations:
            
            ```cpp
            class Shape {
            public:
                // no copy operations
                Shape(const Shape&) = delete;
                Shape& operator = (const Shape&) = delete;
                // no move operations
                Shape(Shape&&) = delete;
                Shape& operator = (Shape&&) = delete;
                ~Shape();
            };
            ```
        - variadic templates (82)

            ```cpp
            template <typename T, typename ... Tail>
            void f(T head, Tail... tail) {
                g(head); // do something to head
                f(tail...); // try again with tail
            }
            ```
        - return containers by value (relying on move for efficiency); §3.3.2
        - aliases (84)

            ```cpp
            template <typename Value> using String_map = Map<string,Value>;
            
            String_map <int> m; // m is a Map<str ing,int>
            template <typename T> using Iterator<T> = typename T::iterator;
            ```
        - ::value_type

            ```cpp
            vector <int> v;
            v::value_type a = *v.begin();
            ```

    - *Containers and Algorithms*
        - back_inserter (103)
        - stream iterators (106)

    - *Concurrency and Utilities*

- **Part II - Basic Facilities**
    - *Types and Declarations*
        - function with postfix return

            ```cpp
            auto f(int n) -> int {...}
            ```
        - the type of an expression is never a reference because references are implicitly dereferenced in expressions (§7.7)

            ```cpp
            void g(int& v) {
                auto x = v; // x is an int (not an int&)
                auto& y = v; // y is an int&
            }
            ```
        - usage of decltype

            ```cpp
            template <class T, class U>
            auto operator + (const Matrix <T>& a, const Matrix <U>& b) -> Matrix <decltype(T()+U())>;
            ```
        - lvalue: has identity but not movable
        - xvalue: has identity and movable
        - glvalue: generalized lvalue, has identity
        - prvalue: pure rvalue, does not have identity and movable
        - rvalue: movable

    - *Pointers, Arrays, and References*
        - `char ∗const cp; // const pointer to char`
        - `char const∗ pc; // pointer to const char`
        - `int && &&` is an `int&&`
        - `int &&  &` is an `int&`
        - `int  & &&` is an `int&`
        - `int  &  &` is an `int&`

    - *Structures, Unions, and Enumerations*
        - order members of struct by size to minimize wasted spaces from holes

    - *Statements*
        - declaration inside `if`

            ```cpp
            if (int d = n % 4) {
                d--;
            } else {
                d++;
            }
            ```
        - `for(auto x: obj) { ... }` is equivalent to both
            - `for (auto x = obj.begin(); x != obj.end(); x++) { ... }`
            - `for (auto x = begin(obj); x != end(obj); x++) { ... }`
            using this fact we can design our own iterables

            ```cpp
            struct range {
                int a, b, p;
                range (int a, int b): a(a), b(b) {}
                int operator * () const {
                    return p;
                }
                range& begin () {
                    this->p = a;
                    return *this;
                }
                range& end () {
                    this->p = b;
                    return *this;
                }
                bool operator != (const range& that) const {
                    return p != that.p;
                }
                range& operator ++ () {
                    this->p++;
                    return *this;
                }
            };
            ...
            for (auto x : range(4, 1000)) {
                cout << x << endl;
            }

            ```

    - *Expressions*

    - *Select Operations*

    - *Functions*

    - *Exception Handling*

    - *Namespaces*

    - *Source Files and Programs*

