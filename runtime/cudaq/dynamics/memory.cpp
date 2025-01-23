/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// FILE TO BE DELETED - THERE ARE JUST SOME NOTES/EXPERIMENTS TO CHECK HOW TO AVOID UNNECESSARY COPIES

#include <iostream>
#include <vector>
#include <initializer_list>
#include <type_traits> // enable_if, conjuction

template<class Head, class... Tail>
using are_same = std::conjunction<std::is_same<Head, Tail>...>;

class Foo {
public:
    int i;
    Foo() = default;
    Foo(int i) : i(i) {}
    Foo(const Foo &other) : i(other.i) {
        std::cout << "copy Foo" << std::endl;
    }
};

class Baz {
public:
    Foo f;
    Baz(const Foo &foo) : f(foo) {}

    Baz(const Baz& other) : f(other.f) {
        std::cout << "copy Baz" << std::endl;
    }
};

class Bar {
private:

    void aggregate(const Baz& head) {
        std::cout << "got last " << head.f.i << std::endl;
        items.push_back(head.f);
    }
    
    template <typename ... Args>
    void aggregate(const Baz &head, Args&& ... args)
    {
        std::cout << "got " << head.f.i << std::endl;
        items.push_back(head.f);
        aggregate(std::forward<Args>(args)...);
    }
    
public:
    std::vector<Foo> items;
    Bar() = default;
    Bar(const Foo &foo) {
        items.reserve(1);
        items.push_back(foo);
    }
    Bar(const Bar &other) : items(other.items) {
        std::cout << "copy Bar" << std::endl;
    }
    //Bar(std::initializer_list<Foo> args) : items(args) {}
    
    Bar& operator=(const Bar& other) {
        std::cout << "assignment Bar" << std::endl;
        // Check for self-assignment
        if (this != &other) {
            items = other.items;
            //auto dummy = other.items;
            //other.items;
        }
        return *this;
    }

    template<class... Args, class = std::enable_if_t<are_same<Baz, Args...>::value, void>>
    Bar(const Args&... args) {
        items.reserve(sizeof...(Args));
        std::cout << "create Bar from Baz" << std::endl;
        aggregate(args...);
        std::cout << "done" << std::endl;
    }
};

int main()
{
    Bar bar;
    {
        Foo foo(5);
        Bar dummy(foo); // creates 1 copy of foo
        std::cout << dummy.items[0].i << std::endl;
        bar = Bar(foo); // creates 1 copy to construct bar, 1 copy when assigning
    }
    std::cout << bar.items[0].i << std::endl;
    //std::cout << foo.i << std::endl;

    Baz op1(Foo(1));
    Baz op2(Foo(2));

    Bar bar2(op1, op2);
    std::cout << bar2.items[0].i << " " << bar2.items[1].i << std::endl;

    return 0;
}