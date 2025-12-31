#include <cute/tensor.hpp>
#include <cute/util/print_latex.hpp>

using namespace cute;

int main() {
    // 1. Define Shape (16 rows, 16 columns)
    auto shape = make_shape(Int<16>{}, Int<16>{});

    // 2. Define Layout (Standard Row Major)
    // LayoutRight automatically generates strides (16, 1) for shape (16, 16)
    auto layout_row_major = make_layout(shape, LayoutRight{});

    // Alternative explicit definition:
    // auto layout_row_major = make_layout(shape, make_stride(Int<16>{}, Int<1>{}));

    // 3. Define Color Functor
    // To visualize the "Row-Major" nature, let's color each row differently.
    auto color_functor = [](int val) -> const char* {
        int row_idx = val / 16;
        
        // Alternating colors for rows to show the "stripes" of contiguous memory
        if (row_idx % 2 == 0) return "blue!20";
        else                  return "cyan!20";
    };

    print_latex(layout_row_major, color_functor);
}