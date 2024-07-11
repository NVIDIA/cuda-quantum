/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// This code is adapted from tweedledum library:
// https://github.com/boschmitt/tweedledum/blob/master/src/Utils/Visualization/string_utf8.cpp

#include "cudaq/algorithms/draw.h"
#include "common/FmtCore.h"
#include <algorithm>
#include <string>
#include <utility>
#include <vector>

using namespace cudaq;

namespace {
enum CharSet : char {
  WIRE_LINE = 0,          // U'─'
  CONTROL_LINE = 1,       // U'│'
  WIRE_CONTROL_CROSS = 2, // U'┼'
  CONTROL = 3,            // U'●'

  BOX_LEFT_WIRE = 4,            // U'┤'
  BOX_RIGHT_WIRE = 5,           // U'├'
  BOX_TOP_CONTROL = 6,          // U'┴'
  BOX_BOTTOM_CONTROL = 7,       // U'┬'
  BOX_TOP_LEFT_CORNER = 8,      // U'╭':
  BOX_TOP_RIGHT_CORNER = 9,     // U'╮':
  BOX_BOTTOM_LEFT_CORNER = 10,  // U'╰'
  BOX_BOTTOM_RIGHT_CORNER = 11, // U'╯':

  SWAP_X = 12 // U'╳'
};
}

inline std::string_view render_char(char c) {
  switch (c) {
  case WIRE_LINE:
    return "─";
  case CONTROL_LINE:
    return "│";
  case WIRE_CONTROL_CROSS:
    return "┼";
  case CONTROL:
    return "●";
  case BOX_LEFT_WIRE:
    return "┤";
  case BOX_RIGHT_WIRE:
    return "├";
  case BOX_TOP_CONTROL:
    return "┴";
  case BOX_BOTTOM_CONTROL:
    return "┬";
  case BOX_TOP_LEFT_CORNER:
    return "╭";
  case BOX_TOP_RIGHT_CORNER:
    return "╮";
  case BOX_BOTTOM_LEFT_CORNER:
    return "╰";
  case BOX_BOTTOM_RIGHT_CORNER:
    return "╯";
  case SWAP_X:
    return "╳";
  default:
    return {};
  }
}

inline std::string render_chars(char *begin, char *end) {
  std::string result;
  while (begin != end) {
    auto rendered_char = render_char(*begin);
    if (!rendered_char.empty())
      result += rendered_char;
    else
      result += *begin;
    ++begin;
  }
  return result;
}

inline void merge_chars(char &c0, char c1) {
  if (c0 == c1)
    return;
  if (c0 == ' ') {
    c0 = c1;
    return;
  }

  if (c1 == CharSet::CONTROL_LINE) {
    switch (c0) {
    case CharSet::CONTROL:
    case CharSet::WIRE_CONTROL_CROSS:
      return;

    case CharSet::WIRE_LINE:
      c0 = CharSet::WIRE_CONTROL_CROSS;
      return;

    default:
      c0 = CharSet::CONTROL_LINE;
      return;
    }
  }

  if (c0 > c1)
    std::swap(c0, c1);

  if (c0 == CharSet::WIRE_LINE) {
    switch (c1) {
    case CharSet::BOX_TOP_LEFT_CORNER:
    case CharSet::BOX_TOP_RIGHT_CORNER:
      c0 = CharSet::BOX_BOTTOM_CONTROL;
      return;

    case CharSet::BOX_BOTTOM_LEFT_CORNER:
    case CharSet::BOX_BOTTOM_RIGHT_CORNER:
      c0 = CharSet::BOX_TOP_CONTROL;
      return;

    default:
      break;
    }
  }
  if (c0 == CharSet::BOX_TOP_LEFT_CORNER &&
      c1 == CharSet::BOX_BOTTOM_LEFT_CORNER) {
    c0 = CharSet::BOX_RIGHT_WIRE;
    return;
  }
  if (c0 == CharSet::BOX_TOP_RIGHT_CORNER &&
      c1 == CharSet::BOX_BOTTOM_RIGHT_CORNER) {
    c0 = CharSet::BOX_LEFT_WIRE;
    return;
  }
  std::swap(c0, c1);
}

const std::map<std::string, std::string> latex_gates_map = {
    {"sdg", R"(S^\dag)"}, {"tdg", R"(T^\dag)"}, {"rx", "R_x"},
    {"ry", "R_y"},        {"rz", "R_z"},        {"r1", "R_1"}};

class Diagram {
public:
  using Wire = int;

  class Operator {
  public:
    Operator(std::vector<Wire> const &wires, int num_targets, int num_controls)
        : wires_(wires), num_targets_(num_targets),
          num_controls_(num_controls) {}

    virtual ~Operator() = default;

    int num_controls() const { return num_controls_; }

    int num_targets() const { return num_targets_; }

    void set_cols(int left_col) {
      left_col_ = left_col;
      right_col_ = left_col_ + width() - 1;
    }

    virtual int width() const = 0;

    virtual void draw(Diagram &diagram) = 0;

  protected:
    std::vector<Wire> wires_;
    int num_targets_;
    int num_controls_;
    int left_col_;
    int right_col_;
  };

  Diagram(int num_qubits)
      : num_qubits_(num_qubits), height_((2 * num_qubits_) + 1) {}

  int num_wires() const { return num_qubits_; }

  int num_qubits() const { return num_qubits_; }

  int height() const { return height_; }

  void width(int width) {
    width_ = width;
    rows_.resize(height_, std::vector<char>(width_, ' '));
    for (int i = 0; i < num_qubits_; ++i) {
      auto &row = rows_.at((2 * i) + 1);
      std::fill(row.begin(), row.end(), CharSet::WIRE_LINE);
    }
  }

  int to_row(Wire wire) const {
    if (wire < num_qubits_)
      return (2u * wire) + 1u;
    return rows_.size() - 2;
  }

  char &at(int row, int col) {
    assert(row < height_);
    assert(col < width_);
    return rows_.at(row).at(col);
  }

  std::vector<char> &row(int row) {
    assert(row < height_);
    return rows_.at(row);
  }

private:
  int num_qubits_;
  int height_;
  int width_;
  std::vector<std::vector<char>> rows_;
};

class Box : public Diagram::Operator {
public:
  using Wire = Diagram::Wire;

  Box(std::string_view label, std::vector<Wire> const &dwires, int num_targets,
      int num_controls)
      : Operator(dwires, num_targets, num_controls), label(label) {}

  virtual int width() const override {
    return label.size() + 2u + (num_controls() > 0);
  }

  virtual void draw(Diagram &diagram) override {
    auto const [min, max] = std::minmax_element(wires_.begin(), wires_.end());
    set_vertical_positions(diagram, *min, *max);
    draw_box(diagram);
    draw_targets(diagram);
    draw_controls(diagram);
    draw_label(diagram);
  }

protected:
  void set_vertical_positions(Diagram const &diagram, Wire top, Wire bot) {
    box_top = diagram.to_row(top) - 1u;
    box_bot = diagram.to_row(bot) + 1u;
    box_mid = (box_top + box_bot) / 2u;
  }

  void draw_box(Diagram &diagram) const {
    // Draw top and bottom
    for (int i = left_col_ + 1; i < right_col_; ++i) {
      merge_chars(diagram.at(box_top, i), CharSet::WIRE_LINE);
      merge_chars(diagram.at(box_bot, i), CharSet::WIRE_LINE);
    }
    // Draw sides
    for (int i = box_top + 1; i < box_bot; ++i) {
      auto left = diagram.row(i).begin() + left_col_;
      auto right = diagram.row(i).begin() + right_col_;
      *left = CharSet::CONTROL_LINE;
      *right = CharSet::CONTROL_LINE;
      std::fill(left + 1, right, ' ');
    }
    // Draw corners
    merge_chars(diagram.at(box_top, left_col_), CharSet::BOX_TOP_LEFT_CORNER);
    merge_chars(diagram.at(box_bot, left_col_),
                CharSet::BOX_BOTTOM_LEFT_CORNER);
    merge_chars(diagram.at(box_top, right_col_), CharSet::BOX_TOP_RIGHT_CORNER);
    merge_chars(diagram.at(box_bot, right_col_),
                CharSet::BOX_BOTTOM_RIGHT_CORNER);
  }

  void draw_targets(Diagram &diagram) const {
    std::for_each(wires_.begin(), wires_.begin() + num_targets(),
                  [&](Wire wire) {
                    int const row = diagram.to_row(wire);
                    diagram.at(row, left_col_) = CharSet::BOX_LEFT_WIRE;
                    diagram.at(row, right_col_) = CharSet::BOX_RIGHT_WIRE;
                    if (num_controls() > 0)
                      diagram.at(row, left_col_ + 1) = '>';
                  });
  }

  virtual void draw_controls(Diagram &diagram) const {
    auto begin = wires_.begin() + num_targets();
    auto end = begin + num_controls();
    std::for_each(begin, end, [&](Wire wire) {
      int const row = diagram.to_row(wire);
      diagram.at(row, left_col_) = CharSet::BOX_LEFT_WIRE;
      diagram.at(row, left_col_ + 1) = CharSet::CONTROL;
      diagram.at(row, right_col_) = CharSet::BOX_RIGHT_WIRE;
    });
  }

  virtual void draw_label(Diagram &diagram) const {
    int const label_start = left_col_ + 1 + (num_controls() > 0);
    std::copy(label.begin(), label.end(),
              diagram.row(box_mid).begin() + label_start);
  }

  int box_top;
  int box_mid;
  int box_bot;
  std::string label;
};

class ControlledBox : public Box {
public:
  ControlledBox(std::string_view label, std::vector<Wire> const &dwires,
                int num_targets, int num_controls)
      : Box(label, dwires, num_targets, num_controls) {}

  virtual int width() const override { return label.size() + 2; }

  virtual void draw(Diagram &diagram) override {
    auto const [min, max] =
        std::minmax_element(wires_.begin(), wires_.begin() + num_targets());
    set_vertical_positions(diagram, *min, *max);
    draw_box(diagram);
    draw_targets(diagram);
    draw_controls(diagram);
    draw_label(diagram);
  }

private:
  virtual void draw_controls(Diagram &diagram) const override {
    int mid_col = (left_col_ + right_col_) / 2;
    auto begin = wires_.begin() + num_targets();
    auto end = begin + num_controls();
    std::for_each(begin, end, [&](Wire wire) {
      int row = diagram.to_row(wire);
      diagram.at(row, mid_col) = CharSet::CONTROL;
      if (row < box_top) {
        for (int i = row + 1; i < box_top; ++i)
          merge_chars(diagram.at(i, mid_col), CharSet::CONTROL_LINE);
        diagram.at(box_top, mid_col) = CharSet::BOX_TOP_CONTROL;
      } else {
        for (int i = box_bot + 1; i < row; ++i)
          merge_chars(diagram.at(i, mid_col), CharSet::CONTROL_LINE);
        diagram.at(box_bot, mid_col) = CharSet::BOX_BOTTOM_CONTROL;
      }
    });
  }

  virtual void draw_label(Diagram &diagram) const override {
    std::copy(label.begin(), label.end(),
              diagram.row(box_mid).begin() + left_col_ + 1);
  }
};

class DiagramSwap : public Diagram::Operator {
public:
  using Wire = Diagram::Wire;

  DiagramSwap(std::vector<Wire> const &dwires, int num_controls)
      : Operator(dwires, 2u, num_controls) {}

  virtual int width() const override { return 3u; }

  virtual void draw(Diagram &diagram) override {
    int mid_col = left_col_ + 1;
    int target_row0 = diagram.to_row(wires_.at(0));
    int target_row1 = diagram.to_row(wires_.at(1));
    diagram.at(target_row0, mid_col) = CharSet::SWAP_X;
    for (int i = target_row0 + 1; i < target_row1; ++i)
      merge_chars(diagram.at(i, mid_col), CharSet::CONTROL_LINE);
    diagram.at(target_row1, mid_col) = CharSet::SWAP_X;
    draw_controls(diagram);
  }

private:
  void draw_controls(Diagram &diagram) const {
    int mid_col = left_col_ + 1;
    int target_row0 = diagram.to_row(wires_.at(0));
    int target_row1 = diagram.to_row(wires_.at(1));
    auto begin = wires_.begin() + num_targets();
    auto end = begin + num_controls();
    std::for_each(begin, end, [&](Wire wire) {
      int row = diagram.to_row(wire);
      diagram.at(row, mid_col) = CharSet::CONTROL;
      if (row < target_row0) {
        for (int i = row + 1; i < target_row0; ++i)
          merge_chars(diagram.at(i, mid_col), CharSet::CONTROL_LINE);
      } else {
        for (int i = target_row1 + 1; i < row; ++i)
          merge_chars(diagram.at(i, mid_col), CharSet::CONTROL_LINE);
      }
    });
  }
};

namespace {
std::vector<int> convertToIDs(const std::vector<QuditInfo> &qudits) {
  std::vector<int> ids;
  ids.reserve(qudits.size());
  std::transform(qudits.cbegin(), qudits.cend(), std::back_inserter(ids),
                 [](auto &info) { return info.id; });
  return ids;
}

using Layer = std::vector<std::size_t>;

// Separate the instructions in layers.  Each layer must contain gates that
// can be drawn in the same diagram layer.  For example, if I have a
// CX(0, 2) and a X(1), then I cannot draw those gates on the same layer
// in the circuit diagram
std::vector<Layer> layers_from_trace(const Trace &trace) {

  std::vector<Layer> layers;
  std::vector<int> wire_layer(trace.getNumQudits(), -1);

  std::size_t ref = 0;
  for (const auto &inst : trace) {
    std::vector<Diagram::Wire> wires = convertToIDs(inst.targets);
    const auto minmax_wires = std::minmax_element(begin(wires), end(wires));
    auto min_dwire = *minmax_wires.first;
    auto max_dwire = *minmax_wires.second;

    if (!inst.controls.empty()) {
      const std::vector<Diagram::Wire> controls = convertToIDs(inst.controls);
      const auto minmax_dwire_controls =
          std::minmax_element(begin(controls), end(controls));
      min_dwire = std::min(*minmax_dwire_controls.first, min_dwire);
      max_dwire = std::max(*minmax_dwire_controls.second, max_dwire);
    }

    int layer = -1;
    for (auto i = min_dwire; i <= max_dwire; ++i)
      layer = std::max(layer, wire_layer.at(i));
    layer += 1;

    if (static_cast<std::size_t>(layer) == layers.size()) {
      layers.emplace_back();
    }
    layers.at(layer).emplace_back(ref);
    // advance wire layer
    for (auto i = min_dwire; i <= max_dwire; ++i)
      wire_layer.at(i) = layer;
    ref += 1;
  }
  return layers;
}

std::vector<std::unique_ptr<Diagram::Operator>>
boxes_from_trace(const Trace &trace) {
  std::vector<std::unique_ptr<Diagram::Operator>> boxes;
  boxes.reserve(std::distance(trace.begin(), trace.end()));

  // same iteration order as in layers_from_trace
  for (const auto &inst : trace) {
    std::vector<Diagram::Wire> wires = convertToIDs(inst.targets);
    std::sort(wires.begin(), wires.end());

    auto min_target = wires.front();
    auto max_target = wires.back();
    bool overlap = false;
    std::vector<Diagram::Wire> controls = convertToIDs(inst.controls);
    for (auto control : controls) {
      if (control > min_target && control < max_target)
        overlap = true;
    }
    wires.insert(wires.end(), controls.begin(), controls.end());

    int padding = 1;
    std::string name = inst.params.empty()
                           ? inst.name
                           : fmt::format("{}({:.4})", inst.name,
                                         fmt::join(inst.params.begin(),
                                                   inst.params.end(), ","));
    std::string label =
        fmt::format("{: ^{}}", name, name.size() + (2 * padding));

    std::unique_ptr<Diagram::Operator> shape = nullptr;
    if (overlap) {
      shape = std::make_unique<Box>(label, wires, inst.targets.size(),
                                    inst.controls.size());
    } else if (name == "swap") {
      shape = std::make_unique<DiagramSwap>(wires, inst.controls.size());
    } else {
      shape = std::make_unique<ControlledBox>(label, wires, inst.targets.size(),
                                              inst.controls.size());
    }
    boxes.push_back(std::move(shape));
  }
  return boxes;
}

std::string string_diagram_from_trace(const Trace &trace,
                                      const std::vector<Layer> &layers) {
  std::vector<std::unique_ptr<Diagram::Operator>> boxes =
      boxes_from_trace(trace);
  std::vector<int> layer_width(layers.size(), 0);
  // set the width of the layers
  for (size_t ref = 0; ref < boxes.size(); ++ref) {
    // find the layer where the box is through ref
    auto layer_it =
        std::find_if(layers.begin(), layers.end(), [&ref](auto &layer) {
          return std::find(layer.begin(), layer.end(), ref) != layer.end();
        });
    auto layer = std::distance(layers.begin(), layer_it);
    layer_width.at(layer) =
        std::max(layer_width.at(layer), boxes[ref]->width());
  }

  Diagram diagram(trace.getNumQudits());
  // Draw labels
  std::size_t prefix_size = 0u;
  std::vector<std::string> prefix(diagram.height(), "");
  for (auto i = 0u; i < trace.getNumQudits(); ++i) {
    auto row = diagram.to_row(i);
    prefix[row] = fmt::format("q{} : ", i);
    prefix_size = std::max(prefix_size, prefix[row].size());
  }

  // Compute diagram width
  auto curr_width = 0u;
  auto acc_width = prefix_size;
  constexpr std::size_t max_columns = 80;
  std::vector<int> cutting_point;
  for (auto layer = 0u; layer < layers.size(); ++layer) {
    for (auto ref : layers.at(layer)) {
      auto &box = boxes.at(ref);
      box->set_cols(curr_width + (layer_width.at(layer) - box->width()) / 2u);
    }
    if ((acc_width + layer_width.at(layer)) >= (max_columns - 1)) {
      cutting_point.push_back(curr_width);
      acc_width = 0u;
    }
    curr_width += layer_width.at(layer);
    acc_width += layer_width.at(layer);
  }
  cutting_point.push_back(curr_width);
  diagram.width(curr_width);

  // Draw boxes
  for (auto const &box : boxes) {
    box->draw(diagram);
  }

  std::string str;
  str.reserve(curr_width * diagram.height() * 4);

  int start = 0;
  for (auto i = 0u; i < cutting_point.size(); ++i) {
    if (i > 0)
      str += fmt::format("\n{:#^{}}\n\n", "", max_columns);

    for (auto row = 0; row < diagram.height(); ++row) {
      auto being = diagram.row(row).data() + start;
      auto end = diagram.row(row).data() + cutting_point.at(i);
      if (i == 0)
        str += fmt::format("{: >{}}", prefix.at(row), prefix_size);

      str += render_chars(being, end);
      if (i + 1 < cutting_point.size())
        str += "»";
      str += '\n';
    }
    start = cutting_point.at(i);
  }
  return str;
}

std::string get_latex_name(const cudaq::Trace::Instruction &inst) {
  std::string latex_gate;
  if (latex_gates_map.find(inst.name) != latex_gates_map.end()) {
    latex_gate = latex_gates_map.at(inst.name);
  } else {
    latex_gate = inst.name;
    // default: name is upper-cased
    std::transform(latex_gate.begin(), latex_gate.end(), latex_gate.begin(),
                   [](unsigned char c) { return std::toupper(c); });
  }
  latex_gate =
      inst.params.empty()
          ? latex_gate
          : fmt::format("{}({:.4})", latex_gate,
                        fmt::join(inst.params.begin(), inst.params.end(), ","));
  return latex_gate;
}

std::string latex_diagram_from_trace(const Trace &trace,
                                     const std::vector<Layer> &layers) {
  // clang-format off
    std::string latex_string = R"(\documentclass{minimal}
\usepackage{quantikz}
\begin{document}
\begin{quantikz}
)";
  // clang-format on
  const std::string sep = " & ";
  std::vector<std::string> latex_lines(trace.getNumQudits());
  for (std::size_t row = 0; row < trace.getNumQudits(); ++row) {
    latex_lines[row] += fmt::format("  \\lstick{{$q_{}$}}", row) + sep;
  }
  for (const auto &layer : layers) {
    // unpack this layer
    for (const auto &ref : layer) {
      auto instruction = trace.begin() + ref;
      auto name = get_latex_name(*instruction);
      // (controlled) swap
      if (name == "SWAP") {
        std::vector<Diagram::Wire> wires = convertToIDs(instruction->targets);
        std::sort(wires.begin(), wires.end());
        auto target_row0 = wires.at(0);
        auto target_row1 = wires.at(1);
        latex_lines[target_row0] +=
            R"(\swap{)" + std::to_string(target_row1 - target_row0) + "}";
        latex_lines[target_row1] += R"(\targX{})";
        std::vector<Diagram::Wire> controls =
            convertToIDs(instruction->controls);
        for (int control : controls) {
          // draw control line to the swap symbol further away
          if (std::abs(control - target_row0) >
              std::abs(control - target_row1)) {
            latex_lines[control] +=
                R"(\ctrl{)" + std::to_string(target_row0 - control) + "}";
          } else {
            latex_lines[control] +=
                R"(\ctrl{)" + std::to_string(target_row1 - control) + "}";
          }
        }
      } else {
        // (controlled) box
        std::vector<Diagram::Wire> wires = convertToIDs(instruction->targets);
        std::sort(wires.begin(), wires.end());
        for (auto wire : wires) {
          latex_lines[wire] += R"(\gate{)" + name + "}";
        }
        std::vector<Diagram::Wire> controls =
            convertToIDs(instruction->controls);
        for (int control : controls) {
          latex_lines[control] +=
              R"(\ctrl{)" + std::to_string(wires.front() - control) + "}";
        }
      }
    }
    for (auto &latex_line : latex_lines) {
      // if nothing happened in this layer, add \qw symbol
      if (latex_line.ends_with(sep)) {
        latex_line += R"(\qw)";
      }
      latex_line += sep;
    }
  }
  for (auto &latex_line : latex_lines) {
    latex_line += "\\qw \\\\\n";
    latex_string += latex_line;
  }
  // clang-format off
  latex_string.append(R"(\end{quantikz}
\end{document}
)");
  // clang-format on
  return latex_string;
}

} // namespace

std::string cudaq::__internal__::getLaTeXString(const Trace &trace) {
  const auto layers = layers_from_trace(trace);
  return latex_diagram_from_trace(trace, layers);
}

std::string cudaq::__internal__::draw(const Trace &trace) {
  if (trace.begin() == trace.end()) {
    return "";
  }
  const auto layers = layers_from_trace(trace);
  return string_diagram_from_trace(trace, layers);
}
