"""The emitter: accumulates PTX lines and symbol tables, then assembles the final text."""

PTX_VERSION = "8.0"          # sm_89 needs 7.8 or later
PTX_TARGET = "sm_89"         # the problems are scheduled for it; it runs unchanged on sm_120
STATIC_SMEM_LIMIT = 48 * 1024


class PtxEmitter:
    """Collects body lines and hands out fresh virtual registers; the .reg
    declarations are only known once the walk is done, so render() adds them.

    Register classes (see dtypes.py): p (predicate), f (f32), h (b16, f16 storage), d (f64),
    r (b32), rd (b64).
    """

    def __init__(self):
        self.lines = []
        self.shared = []                                # .shared declarations
        self.reg_counts = {"p": 0, "f": 0, "h": 0, "d": 0, "r": 0, "rd": 0}
        self.buffers = {}                               # IR buffer name -> global pointer reg
        self.vars = {}                                  # IR axis / SSA name -> register or immediate
        self.smem_base = {}                             # shared array name -> u32 base address reg
        self.smem_shapes = {}                           # shared array name -> extents
        self.smem_types = {}                            # shared array name -> Ty of its elements
        self.types = {}                                 # SSA value name -> Ty
        self.frags = {}                                 # mma fragment name -> registers (a list, or a list per family member)
        self.target = PTX_TARGET                        # raised by statements that need a newer architecture
        self.tma_params = {}                            # tensor-map name -> its .param symbol
        self.allow_fma = False                          # tolerance permits ptxas to fuse a multiply and an add
        self.cse = {}                                   # (opcode, operands) -> register already holding that value
        self.label_count = 0
        self.dynamic_smem = False                       # shared arrays live in one extern array
        self.smem_used = 0                              # bytes handed out of it so far

    def reg(self, kind: str) -> str:
        """A fresh virtual register of the given kind, e.g. '%f3'."""
        self.reg_counts[kind] += 1
        return f"%{kind}{self.reg_counts[kind]}"

    def bind(self, name: str, reg: str, t) -> None:
        """Record that SSA value `name` lives in `reg` and has type `t`."""
        self.vars[name] = reg
        self.types[name] = t

    def type_of(self, name: str):
        return self.types[name]

    def emit(self, line: str) -> None:
        self.lines.append("        " + line)

    def label(self, name: str) -> None:
        self.lines.append(f"{name}:")
        self.cse.clear()                                # control can reach here from elsewhere: nothing is known

    def cached(self, key: tuple, emit) -> str:
        """The register holding `key`'s value, emitting it with `emit()` only the first time.
        Valid within straight-line code; any label forgets everything."""
        if key not in self.cse:
            self.cse[key] = emit()
        return self.cse[key]

    def new_label(self, prefix: str) -> str:
        """A label name that is unique within the kernel."""
        self.label_count += 1
        return f"{prefix}_{self.label_count}"


def is_imm(operand: str) -> bool:
    """True when an integer operand is an immediate rather than a register."""
    return operand.lstrip("-").isdigit()


def render(entry: str, arg_order: list[str], em: PtxEmitter) -> str:
    """Assemble header, signature, register declarations and body into PTX text."""
    params = ",\n".join(
        f"        .param .align 64 .b8 param_{name}[128]" if name in em.tma_params else f"        .param .u64 param_{name}"
        for name in arg_order)
    regs = [f"        .reg .{ty} %{kind}<{em.reg_counts[kind] + 1}>;"
            for kind, ty in (("p", "pred"), ("f", "f32"), ("h", "b16"), ("d", "f64"),
                             ("r", "b32"), ("rd", "b64"))]
    return "\n".join([
        f".version {PTX_VERSION}",
        f".target {em.target}",
        ".address_size 64",
        "",
        *em.shared,
        *([".extern .shared .align 16 .b8 smem_dyn[];"] if em.dynamic_smem else []),
        f".visible .entry {entry}(",
        params,
        ")",
        "{",
        *regs,
        *em.lines,
        "DONE:",
        "        ret;",
        "}",
        "",
    ])
