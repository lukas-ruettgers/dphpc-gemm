#include <iostream>

#include "cute/layout.hpp"

#ifndef DPC_M
#define DPC_M 128
#endif
#ifndef DPC_N
#define DPC_N 128
#endif
#ifndef DPC_K
#define DPC_K 64
#endif
constexpr int M = DPC_M;
constexpr int N = DPC_N;
constexpr int K = DPC_K;

using namespace cute;

template <class Shape, class Stride>
void print1D(Layout<Shape,Stride> const& layout)
{
  for (int i = 0; i < size(layout); ++i) {
    printf("%3d  ", layout(i));
  }
}

template <class Shape, class Stride>
void print2D(Layout<Shape,Stride> const& layout)
{
  for (int m = 0; m < size<0>(layout); ++m) {
    for (int n = 0; n < size<1>(layout); ++n) {
      printf("%3d  ", layout(m,n));
    }
    printf("\n");
  }
}

int main(int argc, const char ** args){
    std::cout << "Using     problem shape: 32,32,16 (M,N,K)" << std::endl;
    std::cout << "Using instruction shape: 16,32,8 (M,N,K)" << std::endl;
    cute::Layout a_default = make_layout(Shape<_32,_16>{}, Stride<_1,_16>{});
    
    auto shape  = Shape <Shape<_16,_2>, Shape<_8,_2>>{};
    auto stride = Stride<Stride<_1,_128>, Stride<_16,_256>>{};
    cute::Layout a_blocked = make_layout(shape, stride);
    
    std::cout << "2D blocked: " << std::endl;
    print2D(a_blocked);
    std::cout << std::endl;
    std::cout << "2D default: " << std::endl;
    print2D(a_default);
    std::cout << std::endl;
    return 0;
}

// std::cout << "M: " << M << ", N: " << N << ", K: " << K << std::endl;
// using X = cute::Underscore;

    // std::cout << "Rank: "   <<   rank(a_blocked) << std::endl;
    // std::cout << "Size: "   <<   size(a_blocked) << std::endl;
    // std::cout << "Cosize: " << cosize(a_blocked) << std::endl;
    // std::cout << std::endl;
    // std::cout << "1D: " << std::endl;
    // print1D(a_blocked);
    // std::cout << std::endl;