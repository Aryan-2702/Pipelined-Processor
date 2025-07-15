# Pipelined-Processor


OpenGL Integration with Vortex GPGPU on Cadence Protium

Introduction to Vortex GPGPU

Vortex is an open-source, FPGA-targeted GPU architecture built on RISC-V cores. It provides a SIMT (single-instruction multiple-thread) execution model where many RISC-V processing elements act as programmable shader cores. The architecture is highly configurable – designers can scale the number of cores, warps, threads, and functional units (ALUs, FPUs, LSU, SFU) per core, as well as cache sizes and pipeline width. Vortex supports standard parallel APIs: its software stack uses the PortableCL (PoCL) framework to provide OpenCL 1.2 compatibility, compiling OpenCL kernels for the RISC-V GPU. Notably, Vortex also implements partial OpenGL support: it can run an OpenGL-ES pipeline by offloading fragment processing to the GPU while doing geometry on the host. In practice, Vortex has been demonstrated as a PCIe-based soft GPU on FPGAs, scaling up to 32 cores on a Stratix 10 FPGA (25.6 GFLOP at 200 MHz). Typical deployments use large FPGAs (Intel Arria/Stratix or Xilinx Alveo/Versal boards) for research and prototyping of GPU designs. Its open-source stack (cores, runtime, and drivers) enables exploration of custom GPU architectures with standard interfaces.

SIMT RISC-V Cores: Vortex uses many RISC-V processor cores as shader units. Each core supports multiple warps and threads, executing GPU instructions in lockstep.

Configurable Microarchitecture: Users can customize the number of cores, ALUs/FPUs per core, threads per warp, cache hierarchy (L1/L2/L3), and other parameters to tailor compute and memory bandwidth.

OpenCL Support (via POCL): The Vortex compiler/runtime is based on PoCL (PortableCL). PoCL’s LLVM/Clang frontend compiles OpenCL C kernels into SPIR-V/LLVM IR and then into Vortex binary code. Vortex thus runs standard OpenCL 1.2 kernels on its RISC-V cores.

3D Graphics Extensions: Vortex includes GPU ISA extensions for graphics. For OpenGL-ES workloads it runs the geometry pipeline on the host CPU and offloads rasterization/fragment shading to the Vortex cores. This split lets the host handle math-heavy transforms while the GPU cores execute pixel shaders.

FPGA Deployment: The design is targeted at FPGAs for prototyping. It has been implemented on Intel (Altera) and Xilinx hardware (e.g. Arria 10, Stratix 10, Alveo U50/U250/U280, Versal VCK5000) as a PCIe soft-GPU. This allows researchers to iterate on full GPU hardware at moderate clock rates and observe real-world software behavior.


Overview of OpenGL Graphics Pipeline

The OpenGL graphics pipeline transforms 3D scene geometry into a 2D image through a series of stages.  In modern OpenGL, the pipeline is programmable with shaders at key points.  First, the Vertex Shader stage takes input vertices (position, normals, etc.) and transforms them into clip-space coordinates.  Next, primitives (points, lines, triangles) are assembled and rasterized: this scan-conversion step generates fragments (potential pixels) by projecting the primitives onto the screen. Each fragment then goes through the Fragment Shader (or pixel shader), which computes its final color by applying lighting, texturing, and other effects. Finally, the pipeline performs per-sample tests (depth, stencil, alpha blending) and writes the result to the framebuffer for display.  In practice, an OpenGL application supplies vertex and fragment shader programs (in GLSL), and the GPU executes these on parallel cores to render images. Modern OpenGL requires at least vertex and fragment shaders to be defined, emphasizing this programmable model.

Key stages in OpenGL rendering include:

Vertex Processing: Each input vertex is run through a vertex shader, which applies model/view/projection transforms and computes per-vertex data.

Primitive Assembly & Rasterization: Vertices are connected into primitives (e.g. triangles), which are then rasterized into a set of fragments (one fragment per covered pixel). This converts geometric data into pixel data.

Fragment Shading: Each fragment is processed by a fragment shader program, which determines the final color (and depth) of that pixel. This stage can include texturing, lighting calculations, and other effects.

Tests and Blending: After shading, fragments undergo depth/stencil testing and blending with existing framebuffer contents. This produces the final pixel output for display.


OpenGL serves as the standard API that orchestrates this pipeline on GPU hardware. Programmers submit geometry data and shaders via OpenGL calls, and the GPU (or hybrid system) carries out these stages in parallel. In modern graphics rendering, this pipeline is how high-level scene descriptions are converted into visible images.

Integration Architecture

To combine OpenGL rendering with the Vortex GPGPU, we envision a hybrid system that partitions the graphics pipeline between the host and the Vortex accelerator. In this model, the host machine (CPU with its OpenGL driver/GPU) executes the early pipeline stages – notably vertex and (optional) geometry processing. Once triangles are assembled, the heavy work of pixel shading is offloaded to Vortex. Concretely, the host runs OpenGL up to the rasterizer, then transfers the relevant fragments or tiles to the Vortex device. The Vortex system on Protium then performs the rasterization and fragment shading as an OpenCL compute task.

Host (OpenGL) Path: On the host side, the OpenGL driver processes vertex arrays, running the vertex shaders and any geometry shaders. It may also perform clipping and primitive assembly. The result is a list of screen-space primitives or tiles to be shaded.

Vortex (GPGPU) Path: The assembled primitives are sent to the Vortex accelerator (via PCIe or shared memory) as input for an OpenCL kernel. Vortex executes the rasterization algorithm and fragment shader code across its many RISC-V cores. This essentially treats each fragment shader invocation as a work-item in an OpenCL kernel on the GPU. For example, in an OpenGL-ES setup, Vortex’s API “runs geometry on the host and the rasterization pipeline… as a kernel on the Vortex parallel architecture”.

Communication: Data exchange occurs over the host-to-device interface. The host maps vertex/index buffers and render targets into shared memory or explicitly sends them. After Vortex finishes shading, it writes the output colors (and optionally depth/stencil) back into a buffer visible to the host.

Display Integration: The host is ultimately responsible for sending the final pixels to the display. This can be done by having the Vortex-rendered frame written into an OpenGL texture or renderbuffer that the host then draws to the screen. In effect, the host’s GPU pipeline is used only for geometry, while the Vortex GPGPU accelerates fragment shading.


Conceptually, the system looks like:

CPU/Host (OpenGL): Vertex Shader → Primitive Assembly
                    │   ▲
            (send primitives/tiles)
                    │   │
Vortex on Protium: Rasterization + Fragment Shaders (as OpenCL compute kernels)19
                    │
              (write color buffer)
                    │
CPU/Host: Blending/Display

This split architecture leverages OpenGL on the host for setup and uses Vortex for the data-parallel shading workload. It avoids implementing a full fixed-function GPU; instead, we “emulate” the fragment stage on a programmable GPGPU.

Implementation Strategy

In practice, OpenGL fragment shaders are implemented as OpenCL kernels running on Vortex. The host’s OpenGL program would submit draw calls as usual, but the fragment shader logic is mirrored by an OpenCL kernel compiled for the Vortex cores. Using the POCL-based toolchain, a GLSL fragment shader (or its SPIR-V bytecode) can be lowered to LLVM IR and then compiled into Vortex instruction binaries. The host then dispatches this kernel across the Vortex cores, passing in vertex attributes, textures, and other inputs as needed.

Data exchange between OpenGL (on the host) and Vortex (device) can be handled in two ways:

Direct OpenGL–OpenCL Interop: If the hardware and drivers support it, we use the cl_khr_gl_sharing extension to map OpenGL objects into OpenCL. This allows an OpenGL texture or buffer to become a shared OpenCL memory object without copying. For example, one could create a cl_mem from an OpenGL texture handle with clCreateFromGLTexture, then acquire it in OpenCL and let the Vortex kernel write pixel values directly. After execution, the host releases the object and can use it normally in OpenGL. This zero-copy approach minimizes overhead.

Manual Copy Method: If interop is unavailable or limited, the data must be copied through host memory. For instance, the host can use glReadPixels or map a Pixel Buffer Object (PBO) to read the current frame into system RAM. That pixel data is then transferred to an OpenCL buffer using clEnqueueWriteBuffer. After the Vortex kernel runs, the results are copied back via clEnqueueReadBuffer and reloaded into an OpenGL texture (e.g. with glTexImage2D or buffer mapping) for display. While less efficient, this method is always compatible.


In both cases, the key is to synchronize access. With cl_khr_gl_sharing, one calls clEnqueueAcquireGLObjects before the kernel and clEnqueueReleaseGLObjects after to ensure coherence. With manual copying, the application must finish OpenGL drawing (e.g. glFinish) before reading, and finish the OpenCL kernel before uploading back. By either path, the fragment processing is effectively realized by OpenCL kernels on Vortex, while geometry and display remain under OpenGL’s control.

Shader Compilation: Use Vortex’s OpenCL toolchain (based on POCL) to turn GLSL/SPIR-V fragment shaders into Vortex kernels.

GL-CL Sharing: Employ cl_khr_gl_sharing to create OpenCL images/buffers directly from OpenGL textures/buffers. This allows Vortex to write results into an OpenGL render target without extra copies.

Fallback Copy: If sharing is not possible, copy framebuffer/texture data on the CPU between GL and CL contexts (e.g. glReadPixels + clEnqueueWriteBuffer, or vice versa).


Prototyping on Protium

Using Cadence’s Protium platform (instead of ad-hoc FPGA boards) offers major benefits for this integration. Protium is an enterprise FPGA prototyping system that provides scalability, speed, and connectivity suitable for early system validation.

Fast Compile and Bring-Up: Protium’s AutoFlow compiler can generate a multi-FPGA prototype in days or weeks rather than the months often required for custom FPGA flows. Its automated partitioning and place-and-route workflows significantly reduce turn-around time. This means Vortex-based designs can be iterated and tested much more quickly.

Huge Capacity: Protium systems scale to billions of ASIC-gate capacity. For example, the X1 system uses blade racks to prototype extremely large designs. This lets one prototype a multi-core GPU (with many Vortex cores) well before tape-out.

At-Speed Performance: Protium runs the design at high clock rates (tens to hundreds of MHz) to deliver nearly “at-speed” operation. It is specifically meant for running real software stacks: it provides enough performance to boot an OS, firmware, and graphics applications. In other words, you can run the actual OpenGL application on the host and drive the Vortex hardware in real time.

Host Interface Integration: Protium boards include high-bandwidth interfaces to the host PC. For example, the Protium X2 board (using a Xilinx VU19P FPGA) offers 2× faster host-to-prototype connection and 2× faster memory bandwidth compared to its predecessor. This ample I/O throughput is ideal for moving large framebuffers or texture data between the CPU and Vortex device. Moreover, Protium provides PCIe and Ethernet interfaces to the host so the prototype can appear as a standard peripheral.

Debug and Visibility: Enterprise prototyping includes rich debug tools and virtual interfaces. Users can capture waveforms, trace memory, and even use existing software debuggers (e.g. GDB) on the CPU while co-running the FPGA design. This aids in validating the OpenGL–Vortex integration end-to-end.


In summary, deploying the Vortex design on Protium yields much faster iteration and higher fidelity. One can prototype a full Vortex-accelerated graphics system with the confidence of running it nearly at its intended speed. It leverages the latest FPGA silicon (Versal/UltraScale+) and automated flows, making complex hybrid GPU experiments practical.

Conclusion

Integrating OpenGL rendering with the Vortex GPGPU on Protium is a viable approach. The underlying technologies already exist: Vortex natively supports hybrid graphics as a soft GPU, and standard interop mechanisms allow OpenGL to cooperate with OpenCL on the device. By letting OpenGL handle scene setup and letting Vortex (via OpenCL) execute the pixel shaders, we can accelerate graphics without a full fixed-function GPU. Prototyping on Cadence Protium makes this especially practical, thanks to its high capacity, rapid compile times, and real-world performance.

In practice, one would compile the desired fragment shader code into a Vortex OpenCL kernel and run it on Protium, either sharing framebuffers via cl_khr_gl_sharing or copying data as needed. The demonstrated performance of Vortex (e.g. tens of GFLOPs with dozens of cores) suggests that even moderately complex shading workloads can be handled. Future work might expand this framework to support more OpenGL features (textures, blending, multi-sampling), integrate with modern APIs (like Vulkan), or explore ASIC implementations of Vortex. The combined open-hardware approach promises a flexible platform for graphics research, enabling new custom GPU architectures to be tested on real workloads before silicon realization.

Sources: The above discussion is based on the Vortex project documentation and papers, as well as Cadence Protium product materials. These sources describe Vortex’s open-source RISC-V GPU design and Protium’s prototyping capabilities in detail.

