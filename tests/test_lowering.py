import math
import unittest

from emmy.compiler.ir.expr import BinaryExpr, Literal, TernaryExpr, Var

from tests.helpers import ROOT, lowered, run_integer_ptx
from ptxgen.dtypes import BOOL, F16, F32, F64, I32
from ptxgen.elementwise import apply_op
from ptxgen.emitter import PtxEmitter, render
from ptxgen.expr import emit_flat_offset, emit_int_expr, emit_pred, fold, int_op
from ptxgen.memory import emit_swizzle, swizzle_params, vector_suffix


def var(name):
    return Var(name)


def lit(value):
    return Literal(value, "int")


class IndexExpressions(unittest.TestCase):
    def test_literals_and_variables_need_no_instruction(self):
        em = PtxEmitter()
        em.vars["a0"] = "%r1"
        self.assertEqual(emit_int_expr(em, lit(7)), "7")
        self.assertEqual(emit_int_expr(em, var("a0")), "%r1")
        self.assertEqual(em.lines, [])

    def test_binary_expression_lowers_recursively(self):
        em = PtxEmitter()
        em.vars["a0"] = "%r1"
        out = emit_int_expr(em, BinaryExpr("+", BinaryExpr("*", var("a0"), lit(4)), lit(1)))
        lines = [line.strip() for line in em.lines]
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[0].startswith("mul.lo.s32") and lines[0].endswith("%r1, 4;"))
        self.assertTrue(lines[1].startswith("add.s32") and lines[1].endswith(", 1;"))
        self.assertEqual(out, lines[1].split()[1].rstrip(","))

    def test_ternary_becomes_selp_on_a_predicate(self):
        em = PtxEmitter()
        em.vars["k"] = "%r1"
        emit_int_expr(em, TernaryExpr(BinaryExpr("<", var("k"), lit(8)), var("k"), lit(0)))
        text = "\n".join(em.lines)
        self.assertIn("setp.lt.s32", text)
        self.assertIn("selp.s32", text)

    def test_logical_predicates_combine(self):
        em = PtxEmitter()
        em.vars["a"] = "%r1"
        emit_pred(em, BinaryExpr("&&", BinaryExpr(">", var("a"), lit(0)), BinaryExpr("<", var("a"), lit(9))))
        self.assertIn("and.pred", "\n".join(em.lines))

    def test_row_major_flat_offset_scales_by_trailing_extents(self):
        em = PtxEmitter()
        em.vars["i"], em.vars["j"] = "%r1", "%r2"
        emit_flat_offset(em, (var("i"), var("j")), [4, 16])
        text = "\n".join(em.lines)
        self.assertIn("mul.lo.s32", text)        # i * 16
        self.assertIn("16;", text)

    def test_index_rank_must_match_the_buffer(self):
        with self.assertRaises(NotImplementedError):
            emit_flat_offset(PtxEmitter(), (lit(0),), [4, 4])


class IndexOptimizations(unittest.TestCase):
    """Constant folding, algebraic identities and common-subexpression elimination of index math."""

    def test_immediates_fold_with_c_division_semantics(self):
        self.assertEqual(fold("*", 6, 7), 42)
        self.assertEqual(fold("/", -7, 2), -3)          # truncates toward zero, like the GPU's div.s32
        self.assertEqual(fold("%", -7, 2), -1)
        self.assertIsNone(fold("/", 1, 0))

    def test_identities_emit_nothing(self):
        em = PtxEmitter()
        self.assertEqual(int_op(em, "+", "%r1", "0"), "%r1")
        self.assertEqual(int_op(em, "*", "1", "%r1"), "%r1")
        self.assertEqual(int_op(em, "*", "%r1", "0"), "0")
        self.assertEqual(int_op(em, "*", "6", "7"), "42")
        self.assertEqual(em.lines, [])

    def test_a_repeated_operation_is_emitted_once_and_commutes(self):
        em = PtxEmitter()
        first = int_op(em, "*", "%r1", "4")
        second = int_op(em, "*", "4", "%r1")
        self.assertEqual(first, second)
        self.assertEqual(len(em.lines), 1)

    def test_a_label_forgets_everything(self):
        em = PtxEmitter()
        int_op(em, "+", "%r1", "%r2")
        em.label("L")
        int_op(em, "+", "%r1", "%r2")
        self.assertEqual(sum("add.s32" in line for line in em.lines), 2)


class LoopUnrolling(unittest.TestCase):
    def kernel_loops(self, problem, kernel):
        from emmy.compiler.ir.stmt.blocks import Loop, StridedLoop
        from ptxgen.loader import find_kernel, find_tile, load_ir
        graph, _ = load_ir(str(ROOT / problem / f"{kernel}.ir.json"))
        return find_tile(find_kernel(graph)).body.iter_of_type(Loop, StridedLoop)

    def test_only_innermost_static_loops_unroll(self):
        from ptxgen.control import UNROLL_FACTOR, static_range, unroll_factor
        loops = self.kernel_loops("problems/m01_matmul_512", "k_matmul_2967be")
        factors = {loop.axis.name: unroll_factor(loop, static_range(loop)) for loop in loops}
        self.assertEqual(factors["_ki"], UNROLL_FACTOR)     # innermost, 64 trips
        self.assertEqual(factors["_ks"], 1)                 # contains the _ki loop

    def test_a_loop_with_a_runtime_start_is_left_alone(self):
        from ptxgen.control import static_range, unroll_factor
        loops = self.kernel_loops("problems/r04_row_sum", "k_sum_1_reduce")
        self.assertTrue(all(unroll_factor(loop, static_range(loop)) == 1 for loop in loops))


class FusedMultiplyAdd(unittest.TestCase):
    def multiply(self, allow_fma):
        em = PtxEmitter()
        em.allow_fma = allow_fma
        a, b = em.reg("f"), em.reg("f")
        apply_op(em, "multiply", [(a, F32), (b, F32)])
        return em.lines[0].strip()

    def test_strict_by_default(self):
        self.assertTrue(self.multiply(False).startswith("mul.rn.f32"))

    def test_contractable_when_the_tolerance_allows(self):
        self.assertTrue(self.multiply(True).startswith("mul.f32"))

    def accumulate(self, base):
        from ptxgen.control import emit_accum
        from types import SimpleNamespace
        em = PtxEmitter()
        em.allow_fma = True
        for name in ("acc", "value", "scaled"):
            em.bind(name, em.reg("f"), F32)
        emit_accum(em, SimpleNamespace(name="acc", base=base, value="value", op=SimpleNamespace(name="add")))
        return em.lines[-1].strip()

    def test_a_plain_accumulator_may_fuse(self):
        self.assertTrue(self.accumulate(None).startswith("add.f32"))

    def test_a_rescaled_accumulator_stays_strict(self):
        self.assertTrue(self.accumulate("scaled").startswith("add.rn.f32"))


class TensorCoreStores(unittest.TestCase):
    """A lane's column pair may be one 2-wide store only when it is aligned and never split by a bound."""

    def store(self, n_bound=None):
        from types import SimpleNamespace
        guard = None if n_bound is None else (lit(0), n_bound)
        return SimpleNamespace(n_guard=guard)

    def test_even_stride_without_a_mask_pairs(self):
        from ptxgen.tensorcore import pair_store_ok
        self.assertTrue(pair_store_ok(self.store(), 64, F16))

    def test_odd_stride_never_pairs(self):
        from ptxgen.tensorcore import pair_store_ok
        self.assertFalse(pair_store_ok(self.store(), 65, F16))      # odd rows start on a 2-byte boundary

    def test_a_masked_pair_needs_a_known_even_bound(self):
        from ptxgen.tensorcore import pair_store_ok
        self.assertTrue(pair_store_ok(self.store(lit(64)), 64, F16))
        self.assertFalse(pair_store_ok(self.store(lit(65)), 66, F16))   # the last column would be lost
        self.assertFalse(pair_store_ok(self.store(Var("n")), 64, F16))  # unknown at compile time


class LoadGrouping(unittest.TestCase):
    """Consecutive-element loads of an unrolled loop become one 16-byte vector load, when provably safe."""

    def inner_loop(self, problem, kernel):
        from emmy.compiler.ir.stmt.blocks import Loop, StridedLoop
        from ptxgen.loader import find_kernel, find_tile, get_arg_types, load_ir
        graph, meta = load_ir(str(ROOT / problem / f"{kernel}.ir.json"))
        kernel_op = find_kernel(graph)
        loops = find_tile(kernel_op).body.iter_of_type(Loop, StridedLoop)
        loop = next(l for l in loops if l.axis.name == "a2" and not any(
            isinstance(s, (Loop, StridedLoop)) for s in l.body.iter() if s is not l))
        return loop, get_arg_types(meta, kernel_op.name)

    def test_only_the_contiguous_operand_is_grouped(self):
        from ptxgen.memory import grouped_loads
        loop, types = self.inner_loop("problems/a01_attention", "k_matmul_softmax_reduce_b55365")
        grouped = grouped_loads(PtxEmitter(), loop, types, 4)
        loads = {s.input: id(s) for s in loop.body if hasattr(s, "input")}
        self.assertEqual(grouped, {loads["q"]})             # q[row, k]; kt[k, j] is strided

    def test_a_row_length_that_is_not_a_multiple_of_the_group_is_left_alone(self):
        from ptxgen.memory import grouped_loads
        loop, types = self.inner_loop("problems/a01_attention", "k_matmul_softmax_reduce_b55365")
        types = {**types, "q": (types["q"][0], [64, 63])}   # rows would no longer start 16-byte aligned
        self.assertEqual(grouped_loads(PtxEmitter(), loop, types, 4), set())

    def test_no_grouping_without_unrolling(self):
        from ptxgen.memory import grouped_loads
        loop, types = self.inner_loop("problems/a01_attention", "k_matmul_softmax_reduce_b55365")
        self.assertEqual(grouped_loads(PtxEmitter(), loop, types, 1), set())

    def test_the_vector_load_is_in_the_ptx(self):
        from compile import compile_ir
        text = compile_ir(str(ROOT / "problems/a01_attention/k_matmul_softmax_reduce_b55365.ir.json"))
        self.assertEqual(text.count("ld.global.v4.f32"), 4)


class Swizzle(unittest.TestCase):
    def reference(self, element, shift, mask):
        return element ^ (((element >> shift) & mask) << 3)

    def test_emitted_sequence_matches_the_definition(self):
        for mode, (shift, mask) in (("B128", (6, 7)), ("B64", (6, 3)), ("B32", (6, 1))):
            em = PtxEmitter()
            em.vars["e"] = "%r0"
            result = emit_swizzle(em, "%r0", mode)
            for element in range(0, 2048, 3):
                registers = run_integer_ptx([l.strip() for l in em.lines], {"%r0": element})
                self.assertEqual(registers[result], self.reference(element, shift, mask), (mode, element))

    def test_none_is_the_identity(self):
        em = PtxEmitter()
        self.assertEqual(emit_swizzle(em, "%r3", "NONE"), "%r3")
        self.assertEqual(em.lines, [])

    def test_shift_override_and_unknown_modes(self):
        self.assertEqual(swizzle_params("B64@5"), (5, 3))
        with self.assertRaises(NotImplementedError):
            swizzle_params("B7")


class Vectors(unittest.TestCase):
    def test_widths_depend_on_the_element_size(self):
        self.assertEqual(vector_suffix(4, 4), ".v4")
        self.assertEqual(vector_suffix(2, 2), ".v2")
        with self.assertRaises(NotImplementedError):
            vector_suffix(3, 4)
        self.assertEqual(vector_suffix(1, 8), "")
        with self.assertRaises(NotImplementedError):
            vector_suffix(4, 8)


class Arithmetic(unittest.TestCase):
    def apply(self, op, args, result=None):
        em = PtxEmitter()
        resolved = []
        for t in args:
            reg = em.reg(t.cls)
            resolved.append((reg, t))
        out = apply_op(em, op, resolved, result)
        return [line.strip() for line in em.lines], out

    def test_f32_arithmetic_pins_the_rounding_mode(self):
        lines, _ = self.apply("multiply", [F32, F32])
        self.assertTrue(lines[0].startswith("mul.rn.f32"))      # .rn also forbids fma contraction

    def test_f16_results_are_rounded_once(self):
        lines, (_, t) = self.apply("add", [F16, F16])
        self.assertIs(t, F16)
        self.assertEqual([l.split()[0] for l in lines], ["add.rn.f32", "cvt.rn.f16.f32", "cvt.f32.f16"])

    def test_an_f32_operand_of_an_f16_op_is_not_rounded(self):
        lines, _ = self.apply("multiply", [F16, F32], result=F32)
        self.assertFalse(any("cvt" in line for line in lines))

    def test_integer_and_f64_classes_use_their_own_opcodes(self):
        self.assertTrue(self.apply("add", [I32, I32])[0][0].startswith("add.s32"))
        self.assertTrue(self.apply("multiply", [I32, I32])[0][0].startswith("mul.lo.s32"))
        self.assertTrue(self.apply("maximum", [F64, F64])[0][0].startswith("max.f64"))

    def test_comparison_yields_a_predicate(self):
        lines, (reg, t) = self.apply("greater", [F32, F32])
        self.assertIs(t, BOOL)
        self.assertTrue(reg.startswith("%p"))
        self.assertTrue(lines[0].startswith("setp.gt.f32"))

    def test_unknown_op_is_reported(self):
        with self.assertRaises(NotImplementedError):
            self.apply("frobnicate", [F32])

    def test_exp_goes_through_ex2(self):
        lines, _ = self.apply("exp", [F32])
        self.assertTrue(any(l.startswith("ex2.approx.f32") for l in lines))


class Rendering(unittest.TestCase):
    def test_register_counts_are_declared_with_one_to_spare(self):
        em = PtxEmitter()
        em.reg("f"), em.reg("f"), em.reg("rd")
        ptx = render("k", [], em)
        self.assertIn(".reg .f32 %f<3>;", ptx)
        self.assertIn(".reg .b64 %rd<2>;", ptx)

    def test_header_and_signature(self):
        em = PtxEmitter()
        ptx = render("k_demo", ["x", "out"], em)
        self.assertTrue(ptx.startswith(".version 8.0\n.target sm_89\n.address_size 64"))
        self.assertIn(".visible .entry k_demo(", ptx)
        self.assertIn(".param .u64 param_x,", ptx)

    def test_tensor_maps_are_128_byte_aligned_params_and_raise_the_target(self):
        em = PtxEmitter()
        em.tma_params["desc"] = "param_desc"
        em.target = "sm_90"
        ptx = render("k", ["x", "desc"], em)
        self.assertIn(".target sm_90", ptx)
        self.assertIn(".param .align 64 .b8 param_desc[128]", ptx)


if __name__ == "__main__":
    unittest.main()
