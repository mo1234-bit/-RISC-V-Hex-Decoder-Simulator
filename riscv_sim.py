#!/usr/bin/env python3
"""
=================================================================
  RISC-V Hex Decoder & Simulator  (RV32I + M + F + D)
=================================================================
Supports ALL RV32I + extensions:

  R-type  : ADD, SUB, SLL, SLT, SLTU, XOR, SRL, SRA, OR, AND
  I-type  : ADDI, SLTI, SLTIU, XORI, ORI, ANDI, SLLI, SRLI, SRAI
            LB, LH, LW, LBU, LHU, JALR
  S-type  : SB, SH, SW
  B-type  : BEQ, BNE, BLT, BGE, BLTU, BGEU
  U-type  : LUI, AUIPC
  J-type  : JAL
  System  : ECALL, EBREAK, FENCE
  M-ext   : MUL, MULH, MULHSU, MULHU, DIV, DIVU, REM, REMU
  F-ext   : FLW, FSW
            FADD.S  FSUB.S  FMUL.S  FDIV.S  FSQRT.S
            FSGNJ.S FSGNJN.S FSGNJX.S
            FMIN.S  FMAX.S
            FMADD.S FMSUB.S FNMADD.S FNMSUB.S
            FCVT.W.S  FCVT.WU.S  FCVT.S.W  FCVT.S.WU
            FMV.X.W   FMV.W.X    FCLASS.S
            FEQ.S  FLT.S  FLE.S
  D-ext   : FLD, FSD
            FADD.D  FSUB.D  FMUL.D  FDIV.D  FSQRT.D
            FSGNJ.D FSGNJN.D FSGNJX.D
            FMIN.D  FMAX.D
            FMADD.D FMSUB.D FNMADD.D FNMSUB.D
            FCVT.W.D  FCVT.WU.D  FCVT.D.W  FCVT.D.WU
            FCVT.S.D  FCVT.D.S
            FCLASS.D  FEQ.D  FLT.D  FLE.D

Usage:
  python riscv_sim.py                              (interactive)
  python riscv_sim.py input.txt                    (file -> riscv_output.txt)
  python riscv_sim.py input.txt output.txt         (custom output)
  python riscv_sim.py input.txt output.txt 0x1000  (custom base address)

Input format:
  One 32-bit hex value per line (with or without 0x prefix).
  Lines starting with # or // are comments.
=================================================================
"""

import sys, os, math, struct

# ─────────────────────────────────────────────────────────────
#  Register ABI names
# ─────────────────────────────────────────────────────────────
XREG = [
    "zero","ra","sp","gp","tp","t0","t1","t2",
    "s0","s1","a0","a1","a2","a3","a4","a5",
    "a6","a7","s2","s3","s4","s5","s6","s7",
    "s8","s9","s10","s11","t3","t4","t5","t6"
]
FREG = [
    "ft0","ft1","ft2","ft3","ft4","ft5","ft6","ft7",
    "fs0","fs1","fa0","fa1","fa2","fa3","fa4","fa5",
    "fa6","fa7","fs2","fs3","fs4","fs5","fs6","fs7",
    "fs8","fs9","fs10","fs11","ft8","ft9","ft10","ft11"
]
RM_NAMES = ["rne","rtz","rdn","rup","rmm","???","???","dyn"]

def xn(n):  return XREG[n] if 0 <= n < 32 else f"x{n}"
def fn(n):  return FREG[n] if 0 <= n < 32 else f"f{n}"
def rmn(r): return RM_NAMES[r & 7]

# ─────────────────────────────────────────────────────────────
#  Bit helpers
# ─────────────────────────────────────────────────────────────
def sign_extend(value, bits):
    sb = 1 << (bits - 1)
    return (value & (sb - 1)) - (value & sb)

def u32(v): return int(v) & 0xFFFFFFFF
def s32(v):
    v = u32(v)
    return v if v < 0x80000000 else v - 0x100000000

# ─────────────────────────────────────────────────────────────
#  IEEE-754 helpers
# ─────────────────────────────────────────────────────────────
def bits_to_f32(bits):
    return struct.unpack('<f', struct.pack('<I', u32(bits)))[0]

def f32_to_bits(f):
    try:
        return struct.unpack('<I', struct.pack('<f', float(f)))[0]
    except (OverflowError, struct.error):
        return 0x7FC00000  # canonical NaN

def bits_to_f64(bits):
    return struct.unpack('<d', struct.pack('<Q', int(bits) & 0xFFFFFFFFFFFFFFFF))[0]

def f64_to_bits(f):
    try:
        return struct.unpack('<Q', struct.pack('<d', float(f)))[0]
    except (OverflowError, struct.error):
        return 0x7FF8000000000000  # canonical NaN

def fclass_s(bits):
    sign = (bits >> 31) & 1
    exp  = (bits >> 23) & 0xFF
    man  = bits & 0x7FFFFF
    if exp == 0xFF:
        if man: return 0x200 if (man & 0x400000) else 0x100
        return 0x01 if sign else 0x80
    if exp == 0:
        if man == 0: return 0x02 if sign else 0x40
        return 0x04 if sign else 0x20
    return 0x08 if sign else 0x10

def fclass_d(bits):
    sign = (bits >> 63) & 1
    exp  = (bits >> 52) & 0x7FF
    man  = bits & 0x000FFFFFFFFFFFFF
    if exp == 0x7FF:
        if man: return 0x200 if (man & (1 << 51)) else 0x100
        return 0x01 if sign else 0x80
    if exp == 0:
        if man == 0: return 0x02 if sign else 0x40
        return 0x04 if sign else 0x20
    return 0x08 if sign else 0x10

def fmt_float(v):
    if math.isnan(v):  return "NaN"
    if math.isinf(v):  return "-Inf" if v < 0 else "+Inf"
    if v == 0.0:       return "0.0"
    return f"{v:.8g}"

# ─────────────────────────────────────────────────────────────
#  Field decoders
# ─────────────────────────────────────────────────────────────
def fields_r(inst):
    return (inst>>7)&31,(inst>>12)&7,(inst>>15)&31,(inst>>20)&31,(inst>>25)&127

def fields_i(inst):
    return (inst>>7)&31,(inst>>12)&7,(inst>>15)&31,sign_extend((inst>>20)&0xFFF,12)

def fields_s(inst):
    imm = sign_extend(((inst>>25)<<5)|((inst>>7)&31),12)
    return (inst>>12)&7,(inst>>15)&31,(inst>>20)&31,imm

def fields_b(inst):
    imm = sign_extend(
        ((inst>>31)&1)<<12|((inst>>7)&1)<<11|((inst>>25)&0x3F)<<5|((inst>>8)&0xF)<<1,13)
    return (inst>>12)&7,(inst>>15)&31,(inst>>20)&31,imm

def fields_u(inst):
    return (inst>>7)&31, sign_extend((inst>>12)&0xFFFFF,20)<<12

def fields_j(inst):
    imm = sign_extend(
        ((inst>>31)&1)<<20|((inst>>12)&0xFF)<<12|((inst>>20)&1)<<11|((inst>>21)&0x3FF)<<1,21)
    return (inst>>7)&31, imm

def fields_fp(inst):
    return ((inst>>7)&31,(inst>>12)&7,(inst>>15)&31,
            (inst>>20)&31,(inst>>25)&3,(inst>>27)&31)  # rd,rm,rs1,rs2,fmt,funct5

def fields_r4(inst):
    return ((inst>>7)&31,(inst>>12)&7,(inst>>15)&31,
            (inst>>20)&31,(inst>>25)&3,(inst>>27)&31)  # rd,rm,rs1,rs2,fmt,rs3

# ─────────────────────────────────────────────────────────────
#  Disassembler
# ─────────────────────────────────────────────────────────────
def disassemble(inst, pc=0):
    opcode = inst & 0x7F

    if opcode == 0x33:  # R / M
        rd,f3,rs1,rs2,f7 = fields_r(inst)
        tbl = {
            (0,0x00):"add",  (0,0x20):"sub",  (1,0x00):"sll",
            (2,0x00):"slt",  (3,0x00):"sltu", (4,0x00):"xor",
            (5,0x00):"srl",  (5,0x20):"sra",  (6,0x00):"or",   (7,0x00):"and",
            (0,0x01):"mul",  (1,0x01):"mulh", (2,0x01):"mulhsu",(3,0x01):"mulhu",
            (4,0x01):"div",  (5,0x01):"divu", (6,0x01):"rem",   (7,0x01):"remu",
        }
        return f"{tbl.get((f3,f7),'?R')} {xn(rd)}, {xn(rs1)}, {xn(rs2)}"

    if opcode == 0x13:  # I ALU
        rd,f3,rs1,imm = fields_i(inst)
        sh=(inst>>20)&31; f7=(inst>>25)&127
        if f3==0: return f"addi {xn(rd)}, {xn(rs1)}, {imm}"
        if f3==2: return f"slti {xn(rd)}, {xn(rs1)}, {imm}"
        if f3==3: return f"sltiu {xn(rd)}, {xn(rs1)}, {imm}"
        if f3==4: return f"xori {xn(rd)}, {xn(rs1)}, {imm}"
        if f3==6: return f"ori {xn(rd)}, {xn(rs1)}, {imm}"
        if f3==7: return f"andi {xn(rd)}, {xn(rs1)}, {imm}"
        if f3==1: return f"slli {xn(rd)}, {xn(rs1)}, {sh}"
        if f3==5: return f"{'srli' if f7==0 else 'srai'} {xn(rd)}, {xn(rs1)}, {sh}"
        return "?I-ALU"

    if opcode == 0x03:  # Integer loads
        rd,f3,rs1,imm = fields_i(inst)
        return f"{({0:'lb',1:'lh',2:'lw',4:'lbu',5:'lhu'}).get(f3,'?load')} {xn(rd)}, {imm}({xn(rs1)})"

    if opcode == 0x23:  # Integer stores
        f3,rs1,rs2,imm = fields_s(inst)
        return f"{({0:'sb',1:'sh',2:'sw'}).get(f3,'?store')} {xn(rs2)}, {imm}({xn(rs1)})"

    if opcode == 0x07:  # FP loads
        rd,f3,rs1,imm = fields_i(inst)
        return f"{({2:'flw',3:'fld'}).get(f3,f'?fload')} {fn(rd)}, {imm}({xn(rs1)})"

    if opcode == 0x27:  # FP stores
        f3,rs1,rs2,imm = fields_s(inst)
        return f"{({2:'fsw',3:'fsd'}).get(f3,f'?fstore')} {fn(rs2)}, {imm}({xn(rs1)})"

    if opcode == 0x63:  # Branches
        f3,rs1,rs2,imm = fields_b(inst)
        mn = {0:"beq",1:"bne",4:"blt",5:"bge",6:"bltu",7:"bgeu"}.get(f3,"?br")
        return f"{mn} {xn(rs1)}, {xn(rs2)}, 0x{u32(pc+imm):08X}"

    if opcode == 0x37:
        rd,imm = fields_u(inst); return f"lui {xn(rd)}, 0x{u32(imm)>>12:05X}"
    if opcode == 0x17:
        rd,imm = fields_u(inst); return f"auipc {xn(rd)}, 0x{u32(imm)>>12:05X}"
    if opcode == 0x6F:
        rd,imm = fields_j(inst); return f"jal {xn(rd)}, 0x{u32(pc+imm):08X}"
    if opcode == 0x67:
        rd,f3,rs1,imm = fields_i(inst); return f"jalr {xn(rd)}, {xn(rs1)}, {imm}"

    # R4-type FP
    if opcode in (0x43,0x47,0x4B,0x4F):
        rd,rm,rs1,rs2,fmt,rs3 = fields_r4(inst)
        s = ".s" if fmt==0 else ".d"
        base = {0x43:"fmadd",0x47:"fmsub",0x4B:"fnmsub",0x4F:"fnmadd"}[opcode]
        return f"{base}{s} {fn(rd)}, {fn(rs1)}, {fn(rs2)}, {fn(rs3)}, {rmn(rm)}"

    # All other FP (0x53)
    if opcode == 0x53:
        rd,rm,rs1,rs2,fmt,f5 = fields_fp(inst)
        s = ".s" if fmt==0 else ".d"

        if f5==0x00: return f"fadd{s} {fn(rd)}, {fn(rs1)}, {fn(rs2)}, {rmn(rm)}"
        if f5==0x01: return f"fsub{s} {fn(rd)}, {fn(rs1)}, {fn(rs2)}, {rmn(rm)}"
        if f5==0x02: return f"fmul{s} {fn(rd)}, {fn(rs1)}, {fn(rs2)}, {rmn(rm)}"
        if f5==0x03: return f"fdiv{s} {fn(rd)}, {fn(rs1)}, {fn(rs2)}, {rmn(rm)}"
        if f5==0x0B: return f"fsqrt{s} {fn(rd)}, {fn(rs1)}, {rmn(rm)}"
        if f5==0x04:
            mn = {0:f"fsgnj{s}",1:f"fsgnjn{s}",2:f"fsgnjx{s}"}.get(rm,f"?fsgnj{s}")
            return f"{mn} {fn(rd)}, {fn(rs1)}, {fn(rs2)}"
        if f5==0x05:
            return f"{'fmin' if rm==0 else 'fmax'}{s} {fn(rd)}, {fn(rs1)}, {fn(rs2)}"
        if f5==0x14:
            mn = {0:f"fle{s}",1:f"flt{s}",2:f"feq{s}"}.get(rm,f"?fcmp{s}")
            return f"{mn} {xn(rd)}, {fn(rs1)}, {fn(rs2)}"
        if f5==0x18:
            src="s" if fmt==0 else "d"
            dst="wu" if rs2==1 else "w"
            return f"fcvt.{dst}.{src} {xn(rd)}, {fn(rs1)}, {rmn(rm)}"
        if f5==0x1A:
            dst="s" if fmt==0 else "d"
            src="wu" if rs2==1 else "w"
            return f"fcvt.{dst}.{src} {fn(rd)}, {xn(rs1)}, {rmn(rm)}"
        if f5==0x08:
            if fmt==0: return f"fcvt.s.d {fn(rd)}, {fn(rs1)}, {rmn(rm)}"
            else:      return f"fcvt.d.s {fn(rd)}, {fn(rs1)}, {rmn(rm)}"
        if f5==0x1C:
            if rm==0: return f"fmv.x.w {xn(rd)}, {fn(rs1)}"
            if rm==1: return f"fclass{s} {xn(rd)}, {fn(rs1)}"
        if f5==0x1E: return f"fmv.w.x {fn(rd)}, {xn(rs1)}"
        return f"?fp_f5=0x{f5:02X}_fmt={fmt}"

    if opcode == 0x73:
        f12=(inst>>20)&0xFFF; f3=(inst>>12)&7
        rd=(inst>>7)&31; rs1=(inst>>15)&31; csr=(inst>>20)&0xFFF
        if f3==0:
            return {0:"ecall",1:"ebreak",0x102:"sret",0x302:"mret"}.get(f12,f"system 0x{f12:03X}")
        csr_mn={1:"csrrw",2:"csrrs",3:"csrrc",5:"csrrwi",6:"csrrsi",7:"csrrci"}.get(f3,"?csr")
        return f"{csr_mn} {xn(rd)}, 0x{csr:03X}, {xn(rs1)}"

    if opcode == 0x0F: return "fence"
    return f"unknown  # 0x{inst:08X}"


# ─────────────────────────────────────────────────────────────
#  Simulator
# ─────────────────────────────────────────────────────────────
class RV32Sim:
    def __init__(self, base_addr=0):
        self.xregs    = [0] * 32
        self.fregs    = [0.0] * 32
        self.freg_fmt = ['?'] * 32   # 's', 'd', or '?'
        self.pc       = base_addr
        self.memory   = {}
        self.base_addr= base_addr
        self.halted   = False
        self.halt_why = ""

    def rx(self, n): return self.xregs[n]
    def wx(self, n, v):
        if n != 0: self.xregs[n] = u32(v)
    def rf(self, n): return self.fregs[n]
    def wf(self, n, v, fmt='s'):
        self.fregs[n] = float(v); self.freg_fmt[n] = fmt

    def _rb(self, addr):
        wa=addr&~3; off=addr&3
        return (self.memory.get(wa,0)>>(off*8))&0xFF
    def _wb(self, addr, val):
        wa=addr&~3; off=addr&3
        w=self.memory.get(wa,0)
        self.memory[wa]=(w&~(0xFF<<(off*8)))|((val&0xFF)<<(off*8))
    def _rh(self,a): return self._rb(a)|(self._rb(a+1)<<8)
    def _wh(self,a,v): self._wb(a,v); self._wb(a+1,v>>8)
    def _rw(self,a): return self._rb(a)|(self._rb(a+1)<<8)|(self._rb(a+2)<<16)|(self._rb(a+3)<<24)
    def _ww(self,a,v):
        for i in range(4): self._wb(a+i,(v>>(i*8))&0xFF)
    def _rd(self,a): return self._rw(a)|(self._rw(a+4)<<32)
    def _wd(self,a,v): self._ww(a,v&0xFFFFFFFF); self._ww(a+4,(v>>32)&0xFFFFFFFF)

    def step(self):
        pc=self.pc; inst=self._rw(pc); npc=pc+4; opcode=inst&0x7F
        changes=[]

        def wr(rd,val):
            old=self.rx(rd); self.wx(rd,val)
            if rd!=0: changes.append(('xreg',rd,old,self.rx(rd)))

        def wfr(rd,val,fmt='s'):
            old=self.rf(rd); self.wf(rd,val,fmt)
            changes.append(('freg',rd,old,self.rf(rd),fmt))

        if opcode==0x33:   # R / M
            rd,f3,rs1,rs2,f7=fields_r(inst)
            a,b=self.rx(rs1),self.rx(rs2); sa,sb=s32(a),s32(b)
            if f7==0x01:
                if   f3==0: res=u32(sa*sb)
                elif f3==1: res=u32((sa*sb)>>32)
                elif f3==2: res=u32((sa*a)>>32)
                elif f3==3: res=u32((a*b)>>32)
                elif f3==4: res=u32(int(sa/sb)) if sb!=0 else 0xFFFFFFFF
                elif f3==5: res=u32(int(a/b))   if b!=0  else 0xFFFFFFFF
                elif f3==6: res=u32(sa-int(sa/sb)*sb) if sb!=0 else u32(sa)
                elif f3==7: res=a%b if b!=0 else a
                else: res=0
            else:
                if   f3==0: res=u32(a+b) if f7==0 else u32(a-b)
                elif f3==1: res=u32(a<<(b&31))
                elif f3==2: res=1 if sa<sb else 0
                elif f3==3: res=1 if a<b   else 0
                elif f3==4: res=a^b
                elif f3==5: res=(a>>(b&31)) if f7==0 else u32(sa>>(b&31))
                elif f3==6: res=a|b
                elif f3==7: res=a&b
                else: res=0
            wr(rd,res)

        elif opcode==0x13:  # I ALU
            rd,f3,rs1,imm=fields_i(inst)
            a=self.rx(rs1); sa=s32(a)
            sh=(inst>>20)&31; f7=(inst>>25)&127
            if   f3==0: res=u32(a+imm)
            elif f3==2: res=1 if sa<imm else 0
            elif f3==3: res=1 if a<u32(imm) else 0
            elif f3==4: res=u32(a^imm)
            elif f3==6: res=u32(a|imm)
            elif f3==7: res=u32(a&imm)
            elif f3==1: res=u32(a<<sh)
            elif f3==5: res=(a>>sh) if f7==0 else u32(sa>>sh)
            else: res=0
            wr(rd,res)

        elif opcode==0x03:  # Integer loads
            rd,f3,rs1,imm=fields_i(inst)
            addr=u32(self.rx(rs1)+imm)
            if   f3==0: res=u32(sign_extend(self._rb(addr),8))
            elif f3==1: res=u32(sign_extend(self._rh(addr),16))
            elif f3==2: res=self._rw(addr)
            elif f3==4: res=self._rb(addr)
            elif f3==5: res=self._rh(addr)
            else: res=0
            wr(rd,res)

        elif opcode==0x23:  # Integer stores
            f3,rs1,rs2,imm=fields_s(inst)
            addr=u32(self.rx(rs1)+imm); val=self.rx(rs2)
            sz={0:"byte",1:"half",2:"word"}.get(f3,"?")
            if f3==0: self._wb(addr,val)
            elif f3==1: self._wh(addr,val)
            elif f3==2: self._ww(addr,val)
            changes.append(('mem',addr,sz,val))

        elif opcode==0x07:  # FP loads
            rd,f3,rs1,imm=fields_i(inst)
            addr=u32(self.rx(rs1)+imm)
            if f3==2: wfr(rd, bits_to_f32(self._rw(addr)), 's')
            elif f3==3: wfr(rd, bits_to_f64(self._rd(addr)), 'd')

        elif opcode==0x27:  # FP stores
            f3,rs1,rs2,imm=fields_s(inst)
            addr=u32(self.rx(rs1)+imm); fval=self.rf(rs2)
            if f3==2:
                self._ww(addr, f32_to_bits(fval))
                changes.append(('fmem',addr,'word(F)',fval))
            elif f3==3:
                self._wd(addr, f64_to_bits(fval))
                changes.append(('fmem',addr,'dword(F)',fval))

        elif opcode in (0x43,0x47,0x4B,0x4F):  # R4 FP
            rd,rm,rs1,rs2,fmt,rs3=fields_r4(inst)
            a,b,c=self.rf(rs1),self.rf(rs2),self.rf(rs3)
            if   opcode==0x43: res=a*b+c
            elif opcode==0x47: res=a*b-c
            elif opcode==0x4B: res=-(a*b)+c
            elif opcode==0x4F: res=-(a*b)-c
            else: res=0.0
            wfr(rd, res, 's' if fmt==0 else 'd')

        elif opcode==0x53:  # FP compute
            rd,rm,rs1,rs2,fmt,f5=fields_fp(inst)
            a=self.rf(rs1); b=self.rf(rs2)
            sf='s' if fmt==0 else 'd'

            if f5==0x00: wfr(rd,a+b,sf)
            elif f5==0x01: wfr(rd,a-b,sf)
            elif f5==0x02: wfr(rd,a*b,sf)
            elif f5==0x03:
                if b==0.0: wfr(rd, math.copysign(math.inf,a) if a!=0.0 else float('nan'), sf)
                else: wfr(rd, a/b, sf)
            elif f5==0x0B:
                wfr(rd, math.sqrt(a) if a>=0 else float('nan'), sf)
            elif f5==0x04:  # FSGNJ*
                sa=math.copysign(1.0,a); sb=math.copysign(1.0,b)
                if   rm==0: res=math.copysign(abs(a), sb)
                elif rm==1: res=math.copysign(abs(a),-sb)
                elif rm==2: res=math.copysign(abs(a), sa*sb)
                else: res=a
                wfr(rd,res,sf)
            elif f5==0x05:  # FMIN/FMAX
                nan_a=math.isnan(a); nan_b=math.isnan(b)
                if nan_a: res=b
                elif nan_b: res=a
                else: res=min(a,b) if rm==0 else max(a,b)
                wfr(rd,res,sf)
            elif f5==0x14:  # Comparisons -> integer result
                ok = not math.isnan(a) and not math.isnan(b)
                if   rm==2: wr(rd, 1 if (ok and a==b) else 0)
                elif rm==1: wr(rd, 1 if (ok and a<b)  else 0)
                elif rm==0: wr(rd, 1 if (ok and a<=b) else 0)
                else: wr(rd,0)
            elif f5==0x18:  # FCVT.W/WU from FP
                if rs2==0: wr(rd, u32(int(a)))
                else: wr(rd, u32(max(0,int(a))))
            elif f5==0x1A:  # FCVT.FP from W/WU
                src=float(s32(self.rx(rs1))) if rs2==0 else float(self.rx(rs1))
                wfr(rd, src, sf)
            elif f5==0x08:  # FCVT.S.D / FCVT.D.S
                wfr(rd, float(a), sf)
            elif f5==0x1C:
                if rm==0: wr(rd, f32_to_bits(a))           # FMV.X.W
                elif rm==1:                                  # FCLASS
                    bits = f32_to_bits(a) if fmt==0 else f64_to_bits(a)
                    wr(rd, fclass_s(bits) if fmt==0 else fclass_d(bits))
            elif f5==0x1E:  # FMV.W.X
                wfr(rd, bits_to_f32(self.rx(rs1)), 's')

        elif opcode==0x63:  # Branches
            f3,rs1,rs2,imm=fields_b(inst)
            a,b=self.rx(rs1),self.rx(rs2); sa,sb=s32(a),s32(b)
            taken={0:a==b,1:a!=b,4:sa<sb,5:sa>=sb,6:a<b,7:a>=b}.get(f3,False)
            npc=u32(pc+imm) if taken else npc
            changes.append(('branch',taken,npc))

        elif opcode==0x37:
            rd,imm=fields_u(inst); wr(rd,imm)
        elif opcode==0x17:
            rd,imm=fields_u(inst); wr(rd,u32(pc+imm))
        elif opcode==0x6F:
            rd,imm=fields_j(inst); ret=npc; npc=u32(pc+imm)
            old=self.rx(rd); self.wx(rd,ret)
            if rd!=0: changes.append(('xreg',rd,old,self.rx(rd)))
            changes.append(('jump',rd,ret,npc))
        elif opcode==0x67:
            rd,f3,rs1,imm=fields_i(inst); ret=npc
            target=u32(self.rx(rs1)+imm)&~1; npc=target
            old=self.rx(rd); self.wx(rd,ret)
            if rd!=0: changes.append(('xreg',rd,old,self.rx(rd)))
            changes.append(('jump',rd,ret,npc))
        elif opcode==0x73:
            f12=(inst>>20)&0xFFF; f3=(inst>>12)&7
            if f3==0: self.halted=True; self.halt_why="ebreak" if f12==1 else "ecall"
        elif opcode==0x0F: pass  # FENCE

        self.pc=npc
        return inst,changes

    def xreg_dump(self):
        lines=[]
        for i in range(0,32,4):
            lines.append("  "+"   ".join(f"x{j:02d}({xn(j):>4}): 0x{self.xregs[j]:08X}" for j in range(i,i+4)))
        return "\n".join(lines)

    def freg_dump(self):
        lines=[]
        for i in range(0,32,2):
            parts=[]
            for j in range(i,i+2):
                fmt=self.freg_fmt[j]; v=self.fregs[j]
                if fmt=='s':
                    bits=f32_to_bits(v)
                    tag=f"0x{bits:08X}  ({fmt_float(v)})"
                elif fmt=='d':
                    bits=f64_to_bits(v)
                    tag=f"0x{bits:016X}  ({fmt_float(v)})"
                else:
                    tag="(not written)"
                parts.append(f"  f{j:02d}({fn(j):>4}): {tag}")
            lines.append("".join(parts))
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
#  Input parser
# ─────────────────────────────────────────────────────────────
def parse_hex_input(text):
    result=[]
    for raw in text.splitlines():
        line=raw.split('#')[0].split('//')[0].strip()
        if not line: continue
        line=line.replace('0x','').replace('0X','').replace(',','').replace(' ','').replace('\t','')
        i=0
        while i+8<=len(line):
            chunk=line[i:i+8]
            try: result.append((chunk.upper(),int(chunk,16)))
            except ValueError: pass
            i+=8
    return result


# ─────────────────────────────────────────────────────────────
#  Output generator
# ─────────────────────────────────────────────────────────────
def process(instructions, base_addr=0, output_path="riscv_output.txt"):
    sim=RV32Sim(base_addr)
    out=[]
    def h(s=""): out.append(s)
    BAR="="*76

    h(BAR)
    h("  RISC-V Hex Decoder & Simulator  —  RV32I + M + F + D Extensions")
    h(BAR)
    h(f"  Base Address : 0x{base_addr:08X}")
    h(f"  Instructions : {len(instructions)}")
    h(BAR); h()

    for idx,(_,val) in enumerate(instructions):
        sim._ww(base_addr+idx*4, val)

    # Static disassembly table
    h("┌────────────────────────────────────────────────────────────────────────────┐")
    h("│  STATIC DISASSEMBLY  (decode only, no execution)                          │")
    h("├────────────┬────────────────┬─────────────────────────────────────────────┤")
    h("│  Address   │  Machine Code  │  Assembly                                   │")
    h("├────────────┼────────────────┼─────────────────────────────────────────────┤")
    for idx,(hs,val) in enumerate(instructions):
        addr=base_addr+idx*4
        asm=disassemble(val,addr)
        h(f"│ 0x{addr:08X} │  0x{hs}  │  {asm:<45} │")
    h("└────────────┴────────────────┴─────────────────────────────────────────────┘")
    h()

    # Initial state
    h(BAR); h("  INITIAL INTEGER REGISTER FILE"); h(BAR)
    h(sim.xreg_dump()); h()

    # Simulation
    h(BAR); h("  SIMULATION TRACE"); h(BAR); h()
    sim.pc=base_addr
    visited={}; steps=0
    MAX=len(instructions)*20+100

    for _ in range(MAX):
        if sim.halted: break
        pc=sim.pc; off=pc-base_addr
        if off<0 or off>=len(instructions)*4:
            h(f"  [!] PC 0x{pc:08X} outside loaded range — stopping."); break
        visited[pc]=visited.get(pc,0)+1
        if visited[pc]>50:
            h(f"  [!] PC 0x{pc:08X} visited {visited[pc]} times — loop guard."); break
        idx=off//4; hs=instructions[idx][0]
        inst_val,changes=sim.step()
        asm=disassemble(inst_val,pc); steps+=1
        h(f"  ┌─ Step {steps:04d}  PC=0x{pc:08X}  │  0x{hs}  │  {asm}")
        xr=[c for c in changes if c[0]=='xreg']
        fr=[c for c in changes if c[0]=='freg']
        me=[c for c in changes if c[0]=='mem']
        fm=[c for c in changes if c[0]=='fmem']
        br=[c for c in changes if c[0]=='branch']
        jp=[c for c in changes if c[0]=='jump']
        if not (xr or fr or me or fm or br or jp):
            h("  │  (no visible side-effects)")
        for _,rd,old,new in xr:
            h(f"  │  XREG  x{rd:02d} ({xn(rd):>4}) : 0x{old:08X} ({s32(old):>12d})  →  0x{new:08X} ({s32(new):>12d})")
        for _,rd,old,new,fmt in fr:
            if fmt=='s':
                ob=f32_to_bits(old); nb=f32_to_bits(new)
                h(f"  │  FREG  f{rd:02d} ({fn(rd):>4}) [single] : 0x{ob:08X} ({fmt_float(old)})  →  0x{nb:08X} ({fmt_float(new)})")
            else:
                ob=f64_to_bits(old); nb=f64_to_bits(new)
                h(f"  │  FREG  f{rd:02d} ({fn(rd):>4}) [double] : 0x{ob:016X} ({fmt_float(old)})  →  0x{nb:016X} ({fmt_float(new)})")
        for _,addr,sz,val in me:
            h(f"  │  MEM   [{sz}] @ 0x{addr:08X}  ←  0x{val:08X}  ({val})")
        for _,addr,sz,fval in fm:
            h(f"  │  MEM   [{sz}] @ 0x{addr:08X}  ←  {fmt_float(fval)}")
        for _,taken,npc2 in br:
            h(f"  │  BRANCH  {'TAKEN  ->  0x'+f'{npc2:08X}' if taken else 'NOT TAKEN (fall-through)'}")
        for _,rd,ret,tgt in jp:
            h(f"  │  JUMP   target=0x{tgt:08X}  ret_addr=0x{ret:08X}  (saved in {xn(rd)})")
        h(f"  └{'─'*72}"); h()

    if sim.halted:
        h(f"  >>> Simulation halted by: {sim.halt_why.upper()}"); h()
    if steps==0:
        h("  (no instructions executed)"); h()

    # Final integer registers
    h(BAR); h("  FINAL INTEGER REGISTER FILE"); h(BAR)
    h(sim.xreg_dump()); h()

    # Final FP registers (only if any FP was present)
    fp_ops={0x07,0x27,0x43,0x47,0x4B,0x4F,0x53}
    if any((v&0x7F) in fp_ops for _,v in instructions):
        h(BAR); h("  FINAL FLOATING-POINT REGISTER FILE"); h(BAR)
        h(sim.freg_dump()); h()

    # Data memory
    code_words={(base_addr+i*4)&~3 for i in range(len(instructions))}
    data_mem={a:v for a,v in sim.memory.items() if a not in code_words and v!=0}
    if data_mem:
        h(BAR); h("  DATA MEMORY  (written during simulation)"); h(BAR)
        h(f"  {'Address':<14} {'Hex Value':<18} {'Unsigned':>12}  {'Signed':>12}")
        h(f"  {'-------':<14} {'---------':<18} {'--------':>12}  {'------':>12}")
        for addr in sorted(data_mem):
            v=data_mem[addr]
            h(f"  0x{addr:08X}   0x{v:08X}          {v:>12d}   {s32(v):>12d}")
        h()

    h(BAR); h(f"  Total steps executed : {steps}"); h(BAR)
    text="\n".join(out)
    with open(output_path,"w",encoding="utf-8") as f: f.write(text)
    return text


# ─────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────
def main():
    print("="*56)
    print("  RISC-V Hex Decoder & Simulator  (RV32I+M+F+D)")
    print("="*56)
    input_path  = sys.argv[1] if len(sys.argv)>1 else None
    output_path = sys.argv[2] if len(sys.argv)>2 else "riscv_output.txt"
    base_addr   = int(sys.argv[3],16) if len(sys.argv)>3 else 0

    if input_path:
        if not os.path.exists(input_path):
            print(f"[ERROR] File not found: {input_path}"); sys.exit(1)
        with open(input_path,"r",encoding="utf-8") as f: raw=f.read()
        print(f"  Input  : {input_path}")
    else:
        print("  Interactive mode — enter hex instructions (8 hex digits each).")
        print("  Type 'done' or Ctrl-D to finish.\n")
        lines=[]
        while True:
            try:
                ln=input("  > ")
                if ln.strip().lower() in ("done","quit","exit"): break
                lines.append(ln)
            except EOFError: break
        raw="\n".join(lines)

    print(f"  Output : {output_path}")
    print(f"  Base   : 0x{base_addr:08X}\n")
    instructions=parse_hex_input(raw)
    if not instructions:
        print("[ERROR] No valid 32-bit hex instructions found."); sys.exit(1)
    print(f"  Parsed {len(instructions)} instruction(s) — simulating...\n")
    result=process(instructions, base_addr, output_path)
    print(result)
    print(f"\n    Results saved to: {output_path}")

if __name__=="__main__":
    main()
