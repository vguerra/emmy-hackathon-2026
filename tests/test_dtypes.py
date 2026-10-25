import unittest

from tests.helpers import lowered
from ptxgen.dtypes import (F16, F32, F64, I32, I64, emit_convert, f32_imm, f64_imm, identity_imm, imm,
                           ty)


class Immediates(unittest.TestCase):
    def test_f32_and_f64_hex(self):
        self.assertEqual(f32_imm(1.0), "0f3F800000")
        self.assertEqual(f32_imm(-2.0), "0fC0000000")
        self.assertEqual(f64_imm(1.0), "0d3FF0000000000000")

    def test_f16_immediates_hold_the_rounded_value(self):
        # 0.1 is not representable in f16; the stored literal is the nearest half
        self.assertEqual(imm(F16, 0.1), f32_imm(0.0999755859375))

    def test_f16_overflow_becomes_infinity(self):
        self.assertEqual(imm(F16, -1e30), f32_imm(float("-inf")))

    def test_integer_identities_for_max_and_min(self):
        self.assertEqual(identity_imm(I32, "maximum", -1e30), str(-(2 ** 31)))
        self.assertEqual(identity_imm(I64, "minimum", 1e30), str(2 ** 63 - 1))
        self.assertEqual(identity_imm(F32, "add", 0.0), f32_imm(0.0))


class Types(unittest.TestCase):
    def test_names_and_shared_spellings(self):
        self.assertIs(ty("f16"), F16)
        self.assertIs(ty("__half"), F16)
        self.assertIs(ty("float"), F32)
        self.assertIs(ty("unsigned long long"), I64)

    def test_unsupported_type_is_reported(self):
        with self.assertRaises(NotImplementedError):
            ty("bf16")


class Conversions(unittest.TestCase):
    def convert(self, source, target):
        def build(em):
            em.bind("x", em.reg(source.cls), source)
            emit_convert(em, em.vars["x"], source, target)
        return lowered(build)

    def test_same_type_needs_no_instruction(self):
        self.assertEqual(self.convert(F32, F32), [])

    def test_f16_widens_for_free_and_rounds_on_the_way_back(self):
        self.assertEqual(self.convert(F16, F32), [])
        lines = self.convert(F32, F16)
        self.assertTrue(lines[0].startswith("cvt.rn.f16.f32"))
        self.assertTrue(lines[1].startswith("cvt.f32.f16"))

    def test_between_register_classes(self):
        self.assertTrue(self.convert(F32, F64)[0].startswith("cvt.f64.f32"))
        self.assertTrue(self.convert(F64, F32)[0].startswith("cvt.rn.f32.f64"))
        self.assertTrue(self.convert(I32, F32)[0].startswith("cvt.rn.f32.s32"))
        self.assertTrue(self.convert(F32, I32)[0].startswith("cvt.rzi.s32.f32"))
        self.assertTrue(self.convert(I32, I64)[0].startswith("cvt.s64.s32"))


if __name__ == "__main__":
    unittest.main()
