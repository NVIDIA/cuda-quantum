/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Differential tests of `ASTResultVisitor`'s traversal against clang's
// `RecursiveASTVisitor`. Both visit the same AST and must visit the same
// statements, declarations and constructor initializers in the same order.
//
// Set CUDAQ_AST_DIFF_FILES to a (colon separated) list of C++ source files and
// CUDAQ_AST_DIFF_ARGS to the (space separated) compiler arguments to use for
// them, to run the comparison over real code as well.

#include "gtest/gtest.h"
#include "cudaq/Frontend/nvqpp/ASTResultVisitor.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Config {
  bool implicitCode;
  bool templateInstantiations;
};

// A visited node. Descriptions are only made for nodes that differ.
struct Node {
  enum Kind { Stmt, Decl, CtorInit } kind;
  const void *ptr;
  // Clang may synthesize a function (such as the `__invoke` of a generic
  // lambda) with a type whose parameters are different declarations than the
  // function's parameters. Those are the same parameter, so a parameter is
  // identified by where it is and what it is named.
  bool isParm = false;
  unsigned loc = 0;
  const void *ident = nullptr;

  bool operator==(const Node &o) const {
    if (kind != o.kind || isParm != o.isParm)
      return false;
    return isParm ? (loc == o.loc && ident == o.ident) : ptr == o.ptr;
  }
};

static Node makeNode(const clang::Decl *d) {
  Node n{Node::Decl, d};
  if (auto *parm = llvm::dyn_cast<clang::ParmVarDecl>(d)) {
    n.isParm = true;
    n.loc = parm->getLocation().getRawEncoding();
    n.ident = parm->getIdentifier();
  }
  return n;
}
using NodeList = std::vector<Node>;

// An InitListExpr has a syntactic and a semantic form. A traversal may reach
// one or the other (the RecursiveASTVisitor visits the syntactic form, the
// new visitor the node that its parent has). Record the semantic form.
static const clang::Stmt *canonical(const clang::Stmt *s) {
  if (auto *il = llvm::dyn_cast<clang::InitListExpr>(s))
    if (auto *sem = il->getSemanticForm())
      return sem;
  return s;
}

// Each traversed node is described by one line of the form
//
//   <kind> [<detail>] @<file:line:col>
//
// where <kind> is "Stmt:<StmtClass>", "Decl:<DeclKind> <name>" or "CtorInit",
// and the location is where the node starts. Expressions also give their
// source range and type, and declarations their qualified type when they have
// one. Distinct nodes can still produce identical lines (compare() points that
// out), but the lines are enough to find the construct in the test source.
static std::string describe(const clang::SourceManager &sm,
                            clang::SourceLocation loc, llvm::StringRef kind,
                            llvm::StringRef detail = {}) {
  std::string result;
  llvm::raw_string_ostream os(result);
  os << kind;
  if (!detail.empty())
    os << " [" << detail << ']';
  os << " @";
  loc.print(os, sm);
  return result;
}

static std::string describe(const clang::ASTContext &ctx,
                            const clang::Stmt *s) {
  std::string detail;
  llvm::raw_string_ostream os(detail);
  if (auto *e = llvm::dyn_cast<clang::Expr>(s))
    os << "type=" << e->getType().getAsString() << ' ';
  os << "range=";
  s->getSourceRange().print(os, ctx.getSourceManager());
  return describe(ctx.getSourceManager(), s->getBeginLoc(),
                  std::string("Stmt:") + s->getStmtClassName(), detail);
}

static std::string describe(const clang::ASTContext &ctx,
                            const clang::Decl *d) {
  std::string kind = std::string("Decl:") + d->getDeclKindName();
  std::string detail;
  if (auto *nd = llvm::dyn_cast<clang::NamedDecl>(d))
    kind += " " + nd->getNameAsString();
  if (auto *vd = llvm::dyn_cast<clang::ValueDecl>(d))
    detail = "type=" + vd->getType().getAsString();
  return describe(ctx.getSourceManager(), d->getLocation(), kind, detail);
}

static std::string describe(const clang::ASTContext &ctx,
                            const clang::CXXCtorInitializer *i) {
  return describe(ctx.getSourceManager(), i->getSourceLocation(), "CtorInit");
}

//===----------------------------------------------------------------------===//
// The oracle: RecursiveASTVisitor
//===----------------------------------------------------------------------===//

class RAVCollector : public clang::RecursiveASTVisitor<RAVCollector> {
  using Base = clang::RecursiveASTVisitor<RAVCollector>;

public:
  RAVCollector(clang::ASTContext &ctx, Config cfg) : ctx(ctx), cfg(cfg) {}

  bool shouldVisitImplicitCode() const { return cfg.implicitCode; }
  bool shouldVisitTemplateInstantiations() const {
    return cfg.templateInstantiations;
  }

  bool VisitStmt(clang::Stmt *s) {
    if (typeDepth == 0)
      nodes.push_back({Node::Stmt, canonical(s)});
    return true;
  }
  bool VisitDecl(clang::Decl *d) {
    nodes.push_back(makeNode(d));
    return true;
  }
  bool TraverseConstructorInitializer(clang::CXXCtorInitializer *i) {
    nodes.push_back({Node::CtorInit, i});
    return Base::TraverseConstructorInitializer(i);
  }

  // Expressions embedded in types and attributes are syntax-only. They are not
  // children for the new visitor. A declaration reached from a type (such as a
  // parameter of a function type) resets this.
  bool TraverseTypeLoc(clang::TypeLoc tl, bool traverseQualifier = true) {
    ++typeDepth;
    bool result = Base::TraverseTypeLoc(tl, traverseQualifier);
    --typeDepth;
    return result;
  }
  bool TraverseType(clang::QualType t, bool traverseQualifier = true) {
    ++typeDepth;
    bool result = Base::TraverseType(t, traverseQualifier);
    --typeDepth;
    return result;
  }
  bool TraverseDecl(clang::Decl *d) {
    int saved = typeDepth;
    // The parameters of a function type that is written inside another type
    // (such as the function pointer returned by the conversion operator of a
    // generic lambda, or a function type in a template argument) are
    // syntax-only. They are not children. Those of a function are.
    if (saved > 0 && llvm::isa<clang::ParmVarDecl>(d) &&
        (saved > 1 || !llvm::isa<clang::FunctionDecl>(d->getDeclContext())))
      return true;
    typeDepth = 0;
    bool result = Base::TraverseDecl(d);
    typeDepth = saved;
    return result;
  }
  bool TraverseAttr(clang::Attr *) { return true; }
  // Expressions inside a type are syntax-only, and so is everything in them.
  bool TraverseStmt(clang::Stmt *s, DataRecursionQueue *queue = nullptr) {
    if (typeDepth > 0)
      return true;
    return Base::TraverseStmt(s, queue);
  }

  // Intentional differences for an InitListExpr when implicit code is visited.
  // The RecursiveASTVisitor visits both the syntactic and the semantic form,
  // which visits the same children twice, and it does not visit the array
  // filler (clang has a FIXME about this). The new visitor visits only the
  // semantic form, and its array filler.
  bool TraverseInitListExpr(clang::InitListExpr *s,
                            DataRecursionQueue *queue = nullptr) {
    if (!cfg.implicitCode)
      return Base::TraverseInitListExpr(s, queue);
    auto *sem = s->isSemanticForm() ? s : s->getSemanticForm();
    if (!sem)
      sem = s;
    if (!(s->isSemanticForm() && s->isSyntacticForm()))
      if (!TraverseSynOrSemInitListExpr(sem, queue))
        return false;
    if (s->isSemanticForm() && s->isSyntacticForm())
      if (!Base::TraverseInitListExpr(s, queue))
        return false;
    if (!sem->hasArrayFiller())
      return true;
    // Traverse the filler after the other children.
    if (queue) {
      queue->push_back({sem->getArrayFiller(), false});
      return true;
    }
    return TraverseStmt(sem->getArrayFiller());
  }

  NodeList nodes;

private:
  clang::ASTContext &ctx;
  Config cfg;
  int typeDepth = 0;
};

//===----------------------------------------------------------------------===//
// The visitor under test. Visits are pre-order: record, then walk the children
// using the base's default enumeration.
//===----------------------------------------------------------------------===//

class NewCollector : public cudaq::detail::ASTResultVisitor<NewCollector, int> {
public:
  NewCollector(clang::ASTContext &ctx, Config cfg) : ctx(ctx), cfg(cfg) {}

  bool shouldVisitImplicitCode() { return cfg.implicitCode; }
  bool shouldVisitTemplateInstantiations() {
    return cfg.templateInstantiations;
  }

  // The parameters of a function are its declarations, but the
  // RecursiveASTVisitor only sees a parameter if it is a declaration in the
  // written type of the function. It is not for a function that is declared
  // with a type that is not written as a function type (`__typeof(f)` or a
  // typedef of a function type), or for a function that clang synthesizes (the
  // copy constructor of a lambda).
  static bool hasSyntacticParams(const clang::ParmVarDecl *parm) {
    auto *func = llvm::dyn_cast<clang::FunctionDecl>(parm->getDeclContext());
    if (!func || !func->getTypeSourceInfo())
      return true;
    auto loc = func->getTypeSourceInfo()
                   ->getTypeLoc()
                   .getAsAdjusted<clang::FunctionProtoTypeLoc>();
    if (!loc)
      return false;
    unsigned index = parm->getFunctionScopeIndex();
    return index < loc.getNumParams() && loc.getParam(index);
  }

  template <typename X>
    requires(std::is_base_of_v<clang::Stmt, X> ||
             std::is_base_of_v<clang::Decl, X>)
  Result visit(X *x) {
    if constexpr (std::is_base_of_v<clang::ParmVarDecl, X>)
      if (!hasSyntacticParams(x)) {
        Children kids;
        traverseChildren(x, kids);
        return std::nullopt;
      }
    // Record the pointer to the base class: a derived class may have a
    // different address.
    if constexpr (std::is_base_of_v<clang::Stmt, X>)
      nodes.push_back(
          {Node::Stmt, canonical(static_cast<const clang::Stmt *>(x))});
    else
      nodes.push_back(makeNode(static_cast<const clang::Decl *>(x)));
    Children kids;
    traverseChildren(x, kids);
    return std::nullopt;
  }

  Result visit(clang::CXXCtorInitializer *i) {
    nodes.push_back({Node::CtorInit, i});
    Children kids;
    traverseChildren(i, kids);
    return std::nullopt;
  }

  NodeList nodes;

private:
  clang::ASTContext &ctx;
  Config cfg;
};

//===----------------------------------------------------------------------===//
// Comparison
//===----------------------------------------------------------------------===//

static std::string describe(const clang::ASTContext &ctx, const Node &n) {
  switch (n.kind) {
  case Node::Stmt:
    return describe(ctx, static_cast<const clang::Stmt *>(n.ptr));
  case Node::Decl:
    return describe(ctx, static_cast<const clang::Decl *>(n.ptr));
  case Node::CtorInit:
    return describe(ctx, static_cast<const clang::CXXCtorInitializer *>(n.ptr));
  }
  return {};
}

static std::string compare(const clang::ASTContext &ctx,
                           const NodeList &expectedNodes,
                           const NodeList &actualNodes) {
  if (expectedNodes == actualNodes)
    return {};
  auto strings = [&](const NodeList &nodes) {
    std::vector<std::string> result;
    for (auto &n : nodes)
      result.push_back(describe(ctx, n));
    return result;
  };
  auto expected = strings(expectedNodes);
  auto actual = strings(actualNodes);
  std::ostringstream os;
  auto [e, a] = std::mismatch(expectedNodes.begin(), expectedNodes.end(),
                              actualNodes.begin(), actualNodes.end());
  std::size_t pos = e - expectedNodes.begin();
  // Descriptions can be identical for different nodes. Make that obvious.
  if (pos < expected.size() && pos < actual.size() &&
      expected[pos] == actual[pos]) {
    expected[pos] += " (a different node)";
    actual[pos] += " (a different node)";
  }
  os << "traversals differ at node " << pos << " (RecursiveASTVisitor visited "
     << expected.size() << " nodes, ASTResultVisitor visited " << actual.size()
     << ")\n";
  // Show a window of the traversal around the first difference, marking the
  // node at which they diverge with ">>". The lines are described above.
  auto context = [&](const char *label, const std::vector<std::string> &nodes) {
    os << "  " << label << ":\n";
    std::size_t lo = pos > 3 ? pos - 3 : 0;
    for (std::size_t i = lo; i < nodes.size() && i < pos + 6; ++i)
      os << "    " << (i == pos ? ">> " : "   ") << nodes[i] << '\n';
  };
  context("RecursiveASTVisitor", expected);
  context("ASTResultVisitor", actual);
  // A summary of the node descriptions that only one of the traversals
  // produced, which shows missed or extra nodes even where the order differs.
  // A description is "only" in one if the other has no node with the same
  // description at all, so a node visited a different number of times is not
  // listed. Each list is capped to keep the failure message readable.
  constexpr std::size_t maxShown = 15;
  auto onlyIn = [&](const char *label, const std::vector<std::string> &mine,
                    const std::vector<std::string> &theirs) {
    std::set<std::string> other(theirs.begin(), theirs.end());
    std::size_t total = 0;
    for (auto &n : mine)
      if (!other.count(n)) {
        if (total++ < maxShown)
          os << "  only " << label << ": " << n << '\n';
      }
    if (total > maxShown)
      os << "  only " << label << ": ... and " << total - maxShown << " more\n";
  };
  onlyIn("RecursiveASTVisitor", expected, actual);
  onlyIn("ASTResultVisitor", actual, expected);
  return os.str();
}

static void checkTraversal(clang::ASTUnit &unit, Config cfg) {
  auto &ctx = unit.getASTContext();
  RAVCollector oracle(ctx, cfg);
  oracle.TraverseDecl(ctx.getTranslationUnitDecl());
  NewCollector under(ctx, cfg);
  under.traverse(static_cast<clang::Decl *>(ctx.getTranslationUnitDecl()));
  EXPECT_FALSE(under.hasFailed());
  std::string diff = compare(ctx, oracle.nodes, under.nodes);
  EXPECT_TRUE(diff.empty()) << diff;
  EXPECT_FALSE(under.nodes.empty());
}

static void checkAllConfigs(clang::ASTUnit &unit) {
  for (bool implicitCode : {false, true})
    for (bool instantiations : {false, true}) {
      SCOPED_TRACE(std::string("implicit=") + (implicitCode ? "1" : "0") +
                   " instantiations=" + (instantiations ? "1" : "0"));
      checkTraversal(unit, {implicitCode, instantiations});
    }
}

static void checkSource(llvm::StringRef code) {
  auto unit = clang::tooling::buildASTFromCodeWithArgs(
      code, {"-std=c++20", "-fno-delayed-template-parsing"}, "snippet.cpp");
  ASSERT_TRUE(unit);
  ASSERT_FALSE(unit->getDiagnostics().hasErrorOccurred());
  checkAllConfigs(*unit);
}

//===----------------------------------------------------------------------===//
// The corpus
//===----------------------------------------------------------------------===//

TEST(ASTResultVisitorTraversal, Statements) {
  checkSource(R"(
int g(int);
void f(int n, int *p) {
  int a = 1, b[3] = {1, 2, 3};
  if (n > 1) a = 2; else a = 3;
  if (int c = g(n); c) a += c;
  for (int i = 0; i < n; ++i) { a += i; if (a > 10) break; else continue; }
  for (;;) break;
  while (a < 10) a++;
  do { a--; } while (a > 0);
  switch (n) { case 1: a = 1; break; case 2: case 3: a = 2; break; default: a = 0; }
  for (int x : b) a += x;
  goto done;
  a = 5;
done:
  try { a = g(a); } catch (int e) { a = e; } catch (...) { throw; }
  return;
}
)");
}

TEST(ASTResultVisitorTraversal, RangeForAndStructuredBindings) {
  checkSource(R"(
struct P { int x; int y; };
struct R { int *begin(); int *end(); };
int f(R r, P p) {
  int s = 0;
  for (int v : r) s += v;
  for (auto &&v : r) s += v;
  auto [x, y] = p;
  const auto &[u, w] = p;
  for (int k = 0; auto e : r) s += e + k;
  return s + x + y + u + w;
}
)");
}

TEST(ASTResultVisitorTraversal, ConstructorsAndCopies) {
  checkSource(R"(
struct B { int b; B(int v) : b(v) {} B(const B &) = default; };
struct D : B {
  int x = 3;
  double y;
  int *p = nullptr;
  D() : B(1), y(2.0) {}
  D(int v) : D() { x = v; }
  D(const D &other) : B(other), x(other.x), y(other.y) {}
  D(D &&) = default;
  D &operator=(const D &) = default;
  ~D() {}
};
struct Implicit { B member; int z; };
struct Arr { int a[4]; };
D make(D d) { return d; }
Implicit copy(Implicit i) { Implicit j = i; Implicit k(j); return k; }
Arr copyArr(Arr a) { Arr b = a; b = a; return b; }
void use() { D d(2); D e = d; D f(static_cast<D &&>(e)); e = f; make(f); }
)");
}

TEST(ASTResultVisitorTraversal, DefaultArgumentsAndMemberInit) {
  checkSource(R"(
int h(int);
struct S { int a = h(1); int b{h(2)}; int c = 5; S() {} S(int) : c(6) {} };
int f(int x, int y = h(3), int z = 4) { return x + y + z; }
struct T { void m(int a = 7, S s = S()) {} };
void use() { f(1); f(1, 2); f(1, 2, 3); T t; t.m(); t.m(1); S s; S s2(1); }
)");
}

TEST(ASTResultVisitorTraversal, InitLists) {
  checkSource(R"(
struct P { int x; int y; };
struct Q { P p; int a[3]; };
void use() {
  int a[10] = {1, 2};
  int b[2][2] = {{1, 2}, {3, 4}};
  P p = {1, 2};
  P p2{3};
  Q q = {{1, 2}, {3}};
  Q q2{.p = {1, 2}, .a = {4, 5, 6}};
  int c[] = {1, 2, 3};
  P ps[2] = {{1, 2}, {3, 4}};
  auto l = [] { return P{1, 2}; };
  (void)l;
}
)");
}

TEST(ASTResultVisitorTraversal, Lambdas) {
  checkSource(R"(
int g(int);
void f(int a, int b) {
  int c = 3;
  auto l1 = [] { return 1; };
  auto l2 = [a, &b] { return a + b; };
  auto l3 = [=] { return a + b + c; };
  auto l4 = [&] { b = a; };
  auto l5 = [x = g(a), &y = b](int p, int q = 4) mutable { return x + y + p + q; };
  auto l6 = [](auto v, auto w) { return v + w; };
  auto l7 = [](int p) -> long { return p; };
  auto l8 = []<typename T>(T t) requires (sizeof(T) > 1) { return t; };
  auto l9 = [c]() noexcept { return c; };
  l1(); l2(); l3(); l4(); l5(1); l6(1, 2); l7(1); l8(1); l9();
  auto nested = [&] { return [&] { return a; }(); };
  nested();
}
)");
}

TEST(ASTResultVisitorTraversal, Casts) {
  checkSource(R"(
struct B { virtual ~B() {} };
struct D : B {};
void f(int i, double d, B *pb, D &rd, const int *cp) {
  long l = i;
  double e = (double)i;
  float g = float(d);
  int h = static_cast<int>(d);
  int *p = const_cast<int *>(cp);
  D *pd = dynamic_cast<D *>(pb);
  void *v = reinterpret_cast<void *>(pd);
  B &b = rd;
  bool z = !!i;
  char c = i;
  (void)l; (void)e; (void)g; (void)h; (void)p; (void)v; (void)b; (void)z; (void)c;
}
)");
}

TEST(ASTResultVisitorTraversal, TypeExpressions) {
  checkSource(R"(
struct S { int a; int b; };
void f(int n) {
  unsigned long a = sizeof(int);
  unsigned long b = sizeof(S);
  unsigned long c = sizeof n;
  unsigned long d = alignof(S);
  int *p = new int(3);
  int *q = new int[n];
  S *s = new S{1, 2};
  delete p;
  delete[] q;
  delete s;
  auto t = __builtin_offsetof(S, b);
  (void)a; (void)b; (void)c; (void)d; (void)t;
  bool x = noexcept(f(1));
  decltype(n) y = n;
  (void)x; (void)y;
}
)");
}

TEST(ASTResultVisitorTraversal, ClassesAndMembers) {
  checkSource(R"(
namespace N {
struct A { int x; static int s; int f(int) const; static void g(); operator int() const { return x; } };
int A::f(int v) const { return v + x; }
void A::g() {}
int A::s = 1;
class C : public A { public: using A::f; friend struct A; friend int h(C); int w : 3; enum E { e1, e2 = 3 }; };
union U { int i; float f; };
enum class K : int { a, b = 2 };
typedef int Int;
using Flt = float;
static_assert(sizeof(int) == 4, "size");
}
int N::h(N::C c) { return c.x; }
void use(N::A a, N::C c) { (void)a.f(1); (void)c.f(2); (void)(int)a; N::A::g(); }
)");
}

TEST(ASTResultVisitorTraversal, Templates) {
  checkSource(R"(
template <typename T, int N = 4> struct V { T data[N]; T get(int i) const { return data[i]; } };
template <typename T> struct V<T *, 4> { T *p; };
template <> struct V<char, 4> { char c; };
template <typename T> T twice(T t) { return t + t; }
template <> int twice<int>(int t) { return 2 * t; }
template <typename... Ts> int count(Ts... ts) { return sizeof...(Ts); }
template <typename... Ts> auto sum(Ts... ts) { return (ts + ...); }
template <typename T> concept Small = sizeof(T) < 8;
template <Small T> T id(T t) { return t; }
template <typename T> struct W { template <typename U> U conv(T t) { return U(t); } };
template <typename T> T var = T(1);
template <typename T> using Vec = V<T, 8>;
template struct V<long, 4>;
template long twice<long>(long);
void use() {
  V<int> a; V<int *> b; V<char> c; V<double, 2> d;
  (void)a.get(0); (void)b; (void)c; (void)d;
  (void)twice(1); (void)twice(1.5); (void)twice<float>(1);
  (void)count(1, 2.0, 'c'); (void)sum(1, 2, 3);
  (void)id(1);
  W<int> w; (void)w.conv<double>(1);
  (void)var<int>; (void)var<double>;
  Vec<short> vs; (void)vs;
}
)");
}

TEST(ASTResultVisitorTraversal, FriendsAndOwnedTags) {
  checkSource(R"(
class A { friend class B; friend struct C; friend void fr(A); };
struct S { int a; } s;
typedef struct { int x; } T;
struct W { struct In { int y; } in; enum E { e } en; };
)");
}

TEST(ASTResultVisitorTraversal, RequiresExpressions) {
  checkSource(R"(
template <typename T> concept C2 = true;
template <typename T, typename U> concept Same = true;
template <typename T> concept C = requires(T t, int i) {
  t + t;
  { t.x } -> C2;
  { t * i } noexcept -> Same<int>;
  typename T::type;
  requires sizeof(T) > 1;
};
)");
}

TEST(ASTResultVisitorTraversal, PartialSpecializationsOfVariables) {
  checkSource(R"(
template <typename T, unsigned N> inline constexpr unsigned ext = 0;
template <typename T, unsigned N> inline constexpr unsigned ext<T[N], 0> = N;
template <typename T> struct fn;
template <typename R, typename... A> struct fn<R(A...)> { R (*p)(A...); };
void use() { (void)ext<int[3], 0>; fn<int(int, char)> f; (void)f; }
)");
}

TEST(ASTResultVisitorTraversal, ConstexprIfAndFold) {
  checkSource(R"(
template <typename T> int f(T t) {
  if constexpr (sizeof(T) > 2) return 1; else return 2;
}
template <bool B> int g() { if constexpr (B) { return 1; } return 0; }
void use() { f(1); f('c'); g<true>(); g<false>(); }
)");
}

TEST(ASTResultVisitorTraversal, OperatorsAndCalls) {
  checkSource(R"(
struct S {
  int v;
  S operator+(const S &o) const { return {v + o.v}; }
  S &operator+=(const S &o) { v += o.v; return *this; }
  int operator[](int i) const { return v + i; }
  int operator()(int a) const { return a; }
  bool operator==(const S &o) const { return v == o.v; }
  bool operator<(const S &o) const { return v < o.v; }
};
int f(S a, S b, int *p, int i) {
  a += b;
  S c = a + b;
  int r = c[1] + a(2) + (i ? 1 : 2) + *p + p[i] + (i, 3) + -i + ~i + (i++) + (--i);
  bool t = a == b;
  bool u = a < b;
  return r + t + u;
}
)");
}

// Check coroutines as part of the AST. They are not part of a CUDA-Q kernel.
TEST(ASTResultVisitorTraversal, Coroutines) {
  checkSource(R"(
struct S { int a; };
int f(S s) { return s.a; }
)");
}

// A record that refers to itself (directly, through a pointer, a reference or
// a function type, or through another record) must not make a traversal loop,
// either in the declarations or in the types of its members.
TEST(ASTResultVisitorTraversal, SelfReferentialRecords) {
  const char *code = R"(
struct S {
  S *next;
  S **pp;
  S &ref();
  S (*fn)(S *, S &);
  S *arr[3];
  S *S::*mp;
  static S *inst;
};
template <typename T> struct L { L<T> *n; T v; L<T> *self() { return this; } };
struct A;
struct B { A *a; B *b; };
struct A { B *b; A *self; B (*f)(A); };
L<int> li;
S s;
auto lam = [](S *p) { return p->next; };
)";
  auto unit = clang::tooling::buildASTFromCodeWithArgs(code, {"-std=c++20"},
                                                       "snippet.cpp");
  ASSERT_TRUE(unit);
  ASSERT_FALSE(unit->getDiagnostics().hasErrorOccurred());
  checkAllConfigs(*unit);

  // Also traverse the type of every field and variable directly.
  auto &ctx = unit->getASTContext();
  NewCollector visitor(ctx, {true, true});
  struct Types : clang::RecursiveASTVisitor<Types> {
    bool shouldVisitTemplateInstantiations() const { return true; }
    bool VisitFieldDecl(clang::FieldDecl *d) {
      types.push_back(d->getType());
      return true;
    }
    bool VisitVarDecl(clang::VarDecl *d) {
      types.push_back(d->getType());
      return true;
    }
    bool VisitFunctionDecl(clang::FunctionDecl *d) {
      types.push_back(d->getType());
      return true;
    }
    std::vector<clang::QualType> types;
  } types;
  types.TraverseDecl(ctx.getTranslationUnitDecl());
  EXPECT_GT(types.types.size(), 10u);
  for (auto t : types.types) {
    visitor.traverse(t);
    EXPECT_FALSE(visitor.hasFailed()) << t.getAsString();
  }
}

TEST(ASTResultVisitorTraversal, DepthLimit) {
  auto unit = clang::tooling::buildASTFromCodeWithArgs(
      "int f(int a) { return ((((a + 1) + 2) + 3) + 4); }", {"-std=c++20"},
      "snippet.cpp");
  ASSERT_TRUE(unit);
  auto &ctx = unit->getASTContext();
  {
    NewCollector shallow(ctx, {false, false});
    shallow.setMaxDepth(5);
    shallow.traverse(static_cast<clang::Decl *>(ctx.getTranslationUnitDecl()));
    EXPECT_TRUE(shallow.hasFailed());
    EXPECT_TRUE(shallow.depthExceeded());
  }
  {
    NewCollector deep(ctx, {false, false});
    deep.traverse(static_cast<clang::Decl *>(ctx.getTranslationUnitDecl()));
    EXPECT_FALSE(deep.hasFailed());
    EXPECT_FALSE(deep.depthExceeded());
  }
}

// An expression that is nested more deeply than the native stack can hold. The
// RecursiveASTVisitor does not recurse over it, but the new visitor does, and
// continues on a new stack when it needs to.
TEST(ASTResultVisitorTraversal, DeepExpression) {
  std::string code = "int f(int a) { return a";
  for (int i = 0; i < 40000; ++i)
    code += "+a";
  code += "; }";
  auto unit = clang::tooling::buildASTFromCodeWithArgs(code, {"-std=c++20"},
                                                       "snippet.cpp");
  ASSERT_TRUE(unit);
  ASSERT_FALSE(unit->getDiagnostics().hasErrorOccurred());
  checkTraversal(*unit, {false, false});
}

TEST(ASTResultVisitorTraversal, FilesFromEnvironment) {
  const char *files = std::getenv("CUDAQ_AST_DIFF_FILES");
  if (!files)
    GTEST_SKIP() << "CUDAQ_AST_DIFF_FILES is not set";
  std::vector<std::string> args = {"-std=c++20", "-fsyntax-only"};
  if (const char *extra = std::getenv("CUDAQ_AST_DIFF_ARGS")) {
    std::istringstream is(extra);
    std::string arg;
    while (is >> arg)
      args.push_back(arg);
  }
  std::istringstream list(files);
  std::string file;
  while (std::getline(list, file, ':')) {
    if (file.empty())
      continue;
    SCOPED_TRACE(file);
    auto buffer = llvm::MemoryBuffer::getFile(file);
    ASSERT_TRUE(buffer) << "cannot read " << file;
    auto unit = clang::tooling::buildASTFromCodeWithArgs((*buffer)->getBuffer(),
                                                         args, file);
    ASSERT_TRUE(unit);
    // Real code may contain errors that are expected by the test it comes from.
    checkAllConfigs(*unit);
  }
}

} // namespace
