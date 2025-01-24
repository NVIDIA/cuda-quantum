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

class Foo {
public:
    int i;
    Foo() = default;
    Foo(int i) : i(i) {}
    Foo(const Foo &other) : i(other.i) {
        std::cout << "copy Foo" << std::endl;
    }

    Foo(Foo &&other) {
        *this = std::forward<Foo>(other);
    }

    Foo& operator=(const Foo& other) {
        std::cout << "assignment Foo" << std::endl;
        // Check for self-assignment
        if (this != &other) {
            i = other.i;
        }
        return *this;
    }

    Foo& operator=(Foo&& other) noexcept {
        std::cout << "move Foo" << std::endl;
        this->i = other.i;
        return *this;
    }
};

class Baz {
public:
    Foo f;
    Baz(Foo&& foo) {
        std::cout << "create Baz" << std::endl;
        f = std::move(foo);
    }

    Baz(const Foo &foo) : f(foo) {
        std::cout << "create Baz" << std::endl;
    }

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

    Bar(Foo&& foo) {
        std::cout << "create Bar from &&" << std::endl;
        items.push_back(std::forward<Foo>(foo));
    }

    Bar(Bar&& other) {
        items = std::move(other.items);
    }

    Bar(const Foo &foo) {
        items.push_back(foo);
    }

    Bar(const Bar &other) : items(other.items) {
        std::cout << "copy Bar" << std::endl;
    }

    //Bar(std::initializer_list<Foo> args) : items(args) {}
    
    Bar& operator=(Bar&& other) noexcept {
        std::cout << "move Bar" << std::endl;
        this->items = std::forward<std::vector<Foo>>(other.items);
        return *this;
    }

    Bar& operator=(const Bar& other) {
        std::cout << "assignment Bar" << std::endl;
        // Check for self-assignment
        if (this != &other) {
            items = other.items;
        }
        return *this;
    }

    template<class... Args, class = std::enable_if_t<std::conjunction<std::is_same<Baz, Args>...>::value, void>>
    Bar(const Args&... args) {
        items.reserve(sizeof...(Args));
        std::cout << "create Bar from Baz" << std::endl;
        aggregate(args...);
        std::cout << "done" << std::endl;
    }
};

class Dummy1 {
protected:
    std::vector<std::vector<std::string>> terms;
    Dummy1() = default;
public:
    Dummy1(std::vector<std::vector<std::string>> data) : terms(data) {}

    std::vector<std::string> get(int i) {
        return terms[i];
    }
};

class Dummy2 : public Dummy1 {
public:
    Dummy2(std::vector<std::string> data) : Dummy1({data}) {}
    
    std::string get(int i) {
        return terms[0][i];
    }
};

class Dummy3 {
public:
    std::vector<std::vector<Bar>> items;
    Dummy3(const std::vector<Bar>& ops) {
        std::cout << "construct Dummy3" << std::endl;
        items.push_back(ops);
    }

    Dummy3(std::vector<Bar>&& ops) {
        std::cout << "construct Dummy3" << std::endl;
        items.push_back(std::move(ops));
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

    // FIXME: AVOID FOO COPY HERE
    Bar bar3(Baz(Foo(3)), Baz(Foo(4)));
    std::cout << bar3.items[0].i << " " << bar3.items[1].i << std::endl;
    
    //Bar(1, Dummy());
    
    std::vector<std::string> data1 = {"op1", "op2"};
    std::vector<std::string> data2 = {"op3", "op4"};

    Dummy1 d1({data1, data2});
    Dummy2 d2(data1);
    
    std::cout << d2.get(0) << " " << d2.get(1) << std::endl;
    std::cout << ((Dummy1)d2).get(0)[0] << " " << ((Dummy1)d2).get(0)[1] << std::endl;

    std::cout << d1.get(0)[0] << " " << d1.get(0)[1] << std::endl;
    std::cout << d1.get(1)[0] << " " << d1.get(1)[1] << std::endl;

    std::vector<Bar> ops = {};
    ops.reserve(2);
    {
        ops.push_back(Bar(Foo(9)));
        Bar op(Foo(10));
        ops.push_back(op);
    }

    Dummy3 d3(std::move(ops));
    std::cout << d3.items[0][0].items[0].i << " " << d3.items[0][1].items[0].i << std::endl;
    //std::cout << ops[0].items[0].i << std::endl;

    {
        std::cout << std::endl;
        // an initializer list will always make an extra copy (by design of the initializer list)
        // d3 = Dummy3({Foo(11), Foo(12)}); -> this will make an extra copy...
        std::vector<Bar> ops2 = {};
        ops2.reserve(2);
        ops2.push_back(Foo(11));
        ops2.push_back(Foo(12));
        d3 = Dummy3(std::move(ops2));
    }
    std::cout << d3.items[0][0].items[0].i << " " << d3.items[0][1].items[0].i << std::endl;
    
    return 0;
}