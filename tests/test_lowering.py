import math
import unittest

from emmy.compiler.ir.expr import BinaryExpr, Literal, TernaryExpr, Var

from tests.helpers import ROOT, lowered, run_integer_ptx
from ptxgen.dtypes import BOOL, F16, F32, F64, I32
from ptxgen.elementwise import apply_op
from ptxgen.emitter import PtxEmitter, render
from ptxgen.expr import emit_flat_offset, emit_int_expr, emit_pred
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
