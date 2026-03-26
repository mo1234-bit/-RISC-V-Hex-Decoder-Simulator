# RISC-V Hex Decoder & Simulator

A Python-based RISC-V hex decoder and simulator that I built while debugging my **RV32IF Verilog core**.

This tool takes raw machine code in **hex** form, decodes it, simulates execution, and generates a readable report showing:

- static disassembly
- step-by-step execution trace
- integer register state
- floating-point register state
- memory updates

It was built as a lightweight architectural debug companion for RISC-V RTL bring-up.

---

## Why I built this

While debugging my RV32IF pipeline in Verilog, I often needed a fast way to answer questions like:

- What does this hex program actually do?
- What should the architectural state look like after each instruction?
- Is the bug in my RTL, or is my test program doing something else?
- What is happening in the floating-point path?

So I wrote this tool to help me inspect raw hex programs without needing a heavy setup every time.

It became especially useful for:
- debugging instruction behavior
- checking expected register values
- tracing floating-point execution
- comparing architectural intent against RTL results

---

## Features

- Reads RISC-V machine code from a hex file
- Decodes instructions into readable assembly
- Simulates execution step by step
- Tracks integer registers
- Tracks floating-point registers
- Tracks memory writes and loads
- Generates a text report for debugging

---

## ISA Coverage

Current support includes a broad set of instructions from:

- **RV32I**
- **M extension**
- **F extension**
- **D extension**

This tool was built and validated primarily for my own debug workflow around **RV32IF-style bring-up**, so it should be viewed as a practical debug utility rather than a formal ISA reference model.

---

## Input Format

The simulator accepts one 32-bit instruction per line, with or without `0x`.

Example:

```text
00f00293
01b00313
006283b3
```
## Usage
```
python riscv_sim.py input.txt
```
You can also pass an output file:
```
python riscv_sim.py input.txt output.txt 
```
If your tool supports a custom base address, you can use:
```
python riscv_sim.py input.txt output.txt 0x1000
```
## Output

The generated report can include:

- static disassembly table
- execution trace
- integer register dump
- floating-point register dump
- memory side effects

This makes it easier to compare expected architectural behavior against a Verilog core during debug.
