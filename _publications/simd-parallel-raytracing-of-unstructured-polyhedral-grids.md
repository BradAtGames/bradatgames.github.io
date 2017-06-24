---
title: "SIMD Parallel Raytracing of Unstructured Polyhedral Grids"
link: "/public/papers/egpgv15.pdf"
img: "/public/images/egpgv15.png"
bibtex: |
  @inproceedings{rathke2015simd,
      title={SIMD Parallel Ray Tracing of Homogeneous Polyhedral Grids.},
      author={Rathke, Brad and Wald, Ingo and Chiu, Kenneth and Brownlee, Carson},
      booktitle={EGPGV},
      pages={33--41},
      year={2015}
  }
abstract: |
  Efficient visualization of unstructured data is vital for domain scientists, yet is often impeded by techniques which rely on intermediate representations that consume time and memory, require resampling data, or inefficient implementations of direct ray tracing methods. Previous work to accelerate rendering of unstructured grids have focused on the use of GPUs that are not available in many large-scale computing systems. In this paper, we present a new technique for directly visualizing unstructured grids using a software ray tracer built as a module for the OSPRay ray tracing framework from Intel. Our method is capable of implicit isosurface rendering and direct volume ray casting homogeneous grids of hexahedra, tetrahedra, and multi-level datasets at interactive frame rates on compute nodes without a GPU using an open-source, production-level ray tracing framework that scales with variable SIMD widths and thread counts.
layout: default
---
