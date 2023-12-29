
/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "../AnalysisPlugin.h"

#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/Tooling.h"

using namespace clang;

namespace {

/// @brief Find instances of `auto reg = m[x,y,z](q)` in the
/// AST and convert to `auto reg = m[x,y,z](q, "reg")`.
class AddRegisterNameVisitor
    : public clang::RecursiveASTVisitor<AddRegisterNameVisitor> {
protected:
  Rewriter &rewriter;
  // Utility Visitor 1 - Walk the tree and find the first CallExpr you see
  struct FindInitCallExpr : public RecursiveASTVisitor<FindInitCallExpr> {
    CallExpr *callExpr = nullptr;
    bool VisitCallExpr(CallExpr *expr) {
      callExpr = expr;
      // stop the walk
      return false;
    }
  };

  // Utility Visitor 2 - Walk the tree and find DeclRefExpr nodes for
  // the variable name, index, and quantum arg in `VAR[IDX] = MEASURE(QARG)`.
  struct FindVectorVariableNames
      : public RecursiveASTVisitor<FindVectorVariableNames> {
    SmallVector<VarDecl *> variables;
    VarDecl *indexVar = nullptr;
    bool VisitDeclRefExpr(DeclRefExpr *expr) {
      if (auto *varDecl = dyn_cast<VarDecl>(expr->getDecl())) {
        if (varDecl->getType()->isIntegerType() && variables.size() == 1) {
          indexVar = varDecl;
          return true;
        }

        // Only store classical reg and quantum reg here
        variables.push_back(varDecl);
      }
      return true;
    }
    StringRef getClassicalRegisterName() { return variables[0]->getName(); }
    std::optional<StringRef> getIndexName() {
      if (indexVar)
        return indexVar->getName();
      return std::nullopt;
    }

    StringRef getQuantumName() { return variables[1]->getName(); }
  };

  struct GetIntegerLiteral : public RecursiveASTVisitor<GetIntegerLiteral> {
    IntegerLiteral *intLiteral = nullptr;
    bool VisitIntegerLiteral(IntegerLiteral *literal) {
      intLiteral = literal;
      return false;
    }
  };

public:
  AddRegisterNameVisitor(Rewriter &r) : rewriter(r) {}
  bool VisitCXXOperatorCallExpr(clang::CXXOperatorCallExpr *expr) {
    // handle VAR[i] = m[x,y,z](QVAR) ->
    // VAR[i] = m[x,y,z](QVAR, "VAR%i")
    if (expr->getOperator() != OverloadedOperatorKind::OO_Equal)
      return true;

    auto desugared = expr->getType().getCanonicalType();
    auto recordType = dyn_cast<clang::RecordType>(desugared.getTypePtr());
    if (!recordType)
      return true;

    auto recordTypeDeclName = recordType->getDecl()->getName();
    if (!recordTypeDeclName.equals("measure_result"))
      return true;

    auto decl = dyn_cast<CXXMethodDecl>(expr->getCalleeDecl());
    if (!decl)
      return true;
    auto typeName = decl->getParent()->getName();
    if (!typeName.equals("measure_result"))
      return true;

    // Find the measure CallExpr
    if (expr->getNumArgs() != 2)
      return true;
    FindInitCallExpr findCall;
    findCall.TraverseMaterializeTemporaryExpr(
        dyn_cast<MaterializeTemporaryExpr>(expr->getArg(1)));
    // auto qArg = findCall.callExpr->getArg(0);

    // Get the vector register, quantum, and index variable names
    FindVectorVariableNames vis;
    vis.TraverseCXXOperatorCallExpr(expr);
    auto regName = vis.getClassicalRegisterName();
    auto qArgName = vis.getQuantumName();
    auto indexName = vis.getIndexName();
    std::string indexNameStr;
    if (indexName.has_value()) {
      indexNameStr = indexName.value().str();
    } else {
      // The index was constant, get that value.
      GetIntegerLiteral intVis;
      intVis.TraverseCXXOperatorCallExpr(expr);
      indexNameStr =
          std::to_string(intVis.intLiteral->getValue().getLimitedValue());
    }

    if (indexNameStr.empty() || regName.empty() || qArgName.empty())
      return true;

    // Add the register name
    auto insertLoc = findCall.callExpr->getRParenLoc();
    rewriter.InsertTextAfter(insertLoc, ", \"" + regName.str() +
                                            "%\" + std::to_string(" +
                                            indexNameStr + ")");

    return true;
  }

  bool VisitVarDecl(clang::VarDecl *decl) {
    // Convert TYPE var = m[x,y,z](qvar) ->
    // TYPE var = m[x,y,z](qvar, "var")

    // Do nothing if no init expr
    auto initExpr = decl->getInit();
    if (!initExpr)
      return true;

    // It should be a call expression
    FindInitCallExpr findCall;
    findCall.TraverseDecl(decl);
    auto *initCallExpr = findCall.callExpr;
    if (!initCallExpr)
      return true;

    // Do nothing if the user specified the register name,
    // i.e. we have 2 args
    if (initCallExpr->getNumArgs() > 1)
      return true;

    // Get the called function decl
    auto calleeDecl = initCallExpr->getCalleeDecl();
    if (!calleeDecl)
      return true;
    auto funcDecl = dyn_cast<FunctionDecl>(calleeDecl);
    if (!funcDecl)
      return true;

    // Get the function name, should be mx, my, or mz
    auto calledFuncName = funcDecl->getName();
    std::vector<std::string> allowedMeasureNames{"mx", "my", "mz"};
    if (std::find(allowedMeasureNames.begin(), allowedMeasureNames.end(),
                  calledFuncName.str()) == allowedMeasureNames.end())
      return true;

    auto desugared = decl->getType().getCanonicalType();
    if (auto recordType = dyn_cast<clang::RecordType>(desugared.getTypePtr())) {
      auto recordTypeDeclName = recordType->getDecl()->getName();
      if (recordTypeDeclName.equals("measure_result") ||
          recordTypeDeclName.equals("vector")) {
        // Get the quantum argument
        auto qArg = initCallExpr->getArg(0);

        // Get the arg name
        auto qArgName =
            dyn_cast<DeclRefExpr>(qArg)->getNameInfo().getAsString();

        // Get the register name
        auto regName = decl->getName();

        // Get the insertion point
        auto insertLoc = initCallExpr->getRParenLoc();

        // Add the register name
        rewriter.InsertTextAfter(insertLoc, ", \"" + regName.str() + "\"");
      }
    }
    return true;
  }
};

/// @brief This Analysis Plugin will search for measurement result assignment
/// replace with the equivalent measure with register name string provided as
/// the second argument, with the string being the same as the variable name.
class RegisterNamesAnalysis : public cudaq::AnalysisPlugin {
public:
  void traverseTree(const cudaq::AnalysisPlugin::Function &function,
                    clang::DiagnosticsEngine &de, Rewriter &rewriter) override {
    AddRegisterNameVisitor(rewriter).TraverseDecl(function.second);
  }
};
} // namespace

CUDAQ_REGISTER_TYPE(cudaq::AnalysisPlugin, RegisterNamesAnalysis,
                    register_names)