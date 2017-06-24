---
title: "The Visualization of Unstructured Polyhedral Grids"
link: "/public/papers/msthesis.pdf"
img: "/public/images/msthesis.png"
bibtex: |
  @mastersthesis{rathke2015visualization,
      title={The visualization of unstructured polyhedral grids},
      author={Rathke, Brad},
      year={2015},
      publisher={STATE UNIVERSITY OF NEW YORK AT BINGHAMTON}
  }
abstract: |
  Efficient visualization of volumetric data sets is an ever present need in the scientific visualization community. While interactive visualization of regularly structured grids is a largely solved problem unstructured grids are a persistent problem. Most attempts at improving the speed at which unstructured grids may be rendered have focused on utilization of GPU hardware which is in general only available in a minority of systems in any given data center. This thesis presents a method of visualizing unstructured grid data sets without the use of GPU hardware using a software ray tracer built as a plug-in module for the Intel OSPRay ray tracing framework. The method is capable of implicit isosurface rendering and direct volume ray casting of homogeneous unstructured grids and multilevel data sets with interactive frame rates and scales with variable SIMD widths and thread counts.
layout: default
---
