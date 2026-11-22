//! Shortest round-trip spelling of binary32, as frozen for artifact vectors.
//!
//! `+0` and `-0` are both `0`. The digits are the shortest sequence that parses
//! back to the same bits with round-ties-to-even. Acceptance is
//! `text.parse::<f64>() as f32`, which is the conversion the importer applies.

pub(super) fn format_f32(value: f32) -> String {
    // The artifact grammar uses one spelling for both IEEE zeroes. Test equality
    // before extracting the sign so a negative zero cannot acquire a leading
    // minus.
    if value == 0.0 {
        return "0".to_owned();
    }
    let bits = value.to_bits();
    let negative = bits >> 31 == 1;
    // Separate the sign first and calculate decimal digits from the magnitude.
    // Normal and subnormal inputs both reduce to an integer significand times an
    // exact power of two.
    let (mantissa, exponent) = significand(bits);
    let (digits, scale) = decimal_digits(mantissa, exponent);
    // Try increasing numbers of significant digits, rounding the exact value anew
    // at each width. The first candidate accepted by both checks gives a
    // reproducible vector spelling across runs.
    for width in 1..=digits.len().max(1) {
        let (significand_digits, exponent_k) = round_significant(&digits, scale, width);
        let body = spell(&significand_digits, exponent_k);
        let text = if negative {
            let mut spelled = String::with_capacity(body.len() + 1);
            spelled.push('-');
            spelled.push_str(&body);
            spelled
        } else {
            body
        };
        // Validate the same two-step conversion used by the importer. Parsing the
        // candidate directly as f32 could bypass a rounding decision at the
        // intermediate f64 value.
        if text.parse::<f64>().ok().map(|parsed| parsed as f32) != Some(value) {
            continue;
        }
        // The product parser rejects an inexact integer such as `1e23`.
        // The next wider spelling that names the same bits is the one import accepts.
        // Wrap the candidate as one array element to exercise the same JSON number
        // grammar as artifact vectors. A spelling that round-trips numerically can
        // still fail this format check.
        if crate::json::parse(&format!("[{text}]")).is_ok() {
            return text;
        }
    }
    // A finite value always round-trips at its exact decimal width.
    "0".to_owned()
}

fn significand(bits: u32) -> (u32, i32) {
    let biased = ((bits >> 23) & 0xff) as i32;
    let fraction = bits & 0x7f_ffff;
    // A subnormal has no implicit leading bit. Every such fraction uses the same
    // 2^-149 step, down to a one-bit fraction for the smallest positive f32.
    if biased == 0 {
        (fraction, -149)
    } else {
        // Restore the hidden leading one for a normal input. Subtract the exponent
        // bias and the 23 fraction bits because the returned mantissa is an
        // integer.
        (fraction | 0x80_0000, biased - 127 - 23)
    }
}

/// Little-endian decimal digits of `mantissa * 2^exponent`, times `10^scale`.
fn decimal_digits(mantissa: u32, exponent: i32) -> (Vec<u8>, i32) {
    // The returned digit integer D and scale satisfy D * 10^scale = mantissa *
    // 2^exponent. Digits are ordered from least to most significant.
    let mut digits = Vec::new();
    let mut rest = mantissa;
    if rest == 0 {
        return (vec![0], 0);
    }
    // Store decimal digits least-significant first. Multiplication can then
    // propagate carry through increasing indices without prepending new high
    // digits.
    while rest > 0 {
        digits.push((rest % 10) as u8);
        rest /= 10;
    }
    if exponent >= 0 {
        for _ in 0..exponent {
            mul_small(&mut digits, 2);
        }
        trim_trailing_zeros(digits, 0)
    } else {
        // Use 2^-n = 5^n * 10^-n. Repeated multiplication by five gives an exact
        // integer digit sequence, and the negative decimal scale places its
        // decimal point.
        let shift = -exponent;
        for _ in 0..shift {
            mul_small(&mut digits, 5);
        }
        trim_trailing_zeros(digits, -shift)
    }
}

fn mul_small(digits: &mut Vec<u8>, factor: u8) {
    let mut carry = 0u16;
    // Consume digits from low to high so the carry goes into the next place. The
    // only multipliers used here are two and five, and intermediate products fit
    // in u16.
    for digit in digits.iter_mut() {
        let value = u16::from(*digit) * u16::from(factor) + carry;
        *digit = (value % 10) as u8;
        carry = value / 10;
    }
    while carry > 0 {
        digits.push((carry % 10) as u8);
        carry /= 10;
    }
}

fn trim_trailing_zeros(mut digits: Vec<u8>, mut scale: i32) -> (Vec<u8>, i32) {
    // Index zero is the units digit in this internal order. Removing a factor of
    // ten increases the decimal scale by one to preserve the same value, and a
    // single zero digit is retained.
    while digits.len() > 1 && digits[0] == 0 {
        digits.remove(0);
        scale += 1;
    }
    (digits, scale)
}

/// Most-significant digit first, together with the scientific exponent `k`.
fn round_significant(digits: &[u8], scale: i32, width: usize) -> (Vec<u8>, i32) {
    let len = digits.len();
    // The highest digit is at position len - 1 in the little-endian integer. Add
    // its power of ten to the shared scale to obtain the scientific exponent.
    let exponent_k = scale + len as i32 - 1;
    // If the caller kept every digit, no rounding is needed. Reverse once for the
    // most-significant-first spelling stage.
    if len <= width {
        let mut out = digits.to_vec();
        out.reverse();
        return (out, exponent_k);
    }
    // The first discarded digit is adjacent to the kept suffix. The remaining
    // discarded digits distinguish an exact halfway value from one strictly above
    // it.
    let least_kept = len - width;
    let round_digit = digits[least_kept - 1];
    let rest_nonzero = digits[..least_kept - 1].iter().any(|digit| *digit != 0);
    let mut kept = digits[least_kept..].to_vec();
    // Round to the nearest value of this width. A discarded 5 plus any nonzero
    // tail rounds upward. An exact 5 with a zero tail selects an even lowest kept
    // digit.
    let round_up = if rest_nonzero {
        round_digit >= 5
    } else if round_digit > 5 {
        true
    } else if round_digit < 5 {
        false
    } else {
        kept[0] % 2 == 1
    };
    // Propagate carry from the least significant kept digit. The digits stay
    // little-endian until rounding is complete, so a run of nines is contiguous
    // from index zero.
    if round_up {
        let mut index = 0;
        let mut overflow = true;
        while index < kept.len() {
            if kept[index] < 9 {
                kept[index] += 1;
                overflow = false;
                break;
            }
            kept[index] = 0;
            index += 1;
        }
        // A carry out of the entire kept width changes the scientific exponent.
        // For example, rounding 9.99 to two significant digits yields digits 1,0
        // at exponent 1.
        if overflow {
            let mut out = Vec::with_capacity(width);
            out.push(1);
            out.resize(width, 0);
            return (out, exponent_k + 1);
        }
    }
    // Return the rounded most-significant digits, preserving the requested width
    // for the caller's round-trip search.
    kept.reverse();
    (kept, exponent_k)
}

// Spell a rounded digit sequence under the artifact format's fixed/scientific
// rule. Keep the sign outside this function so it cannot affect decimal placement.
fn spell(digits: &[u8], exponent_k: i32) -> String {
    let mut body = String::with_capacity(digits.len() + 8);
    // Select fixed notation for scientific exponents from -4 through 15 inclusive.
    // The boundary is a serialization choice, so changing it can alter artifact
    // bytes for the same vector.
    if (-4..=15).contains(&exponent_k) {
        if exponent_k >= 0 {
            // In the fixed integer case, exponent plus one counts positions before
            // the point. Place the point only when the digits extend beyond those
            // positions.
            let integer_len = exponent_k as usize + 1;
            for (index, digit) in digits.iter().enumerate() {
                if index == integer_len {
                    body.push('.');
                }
                body.push(char::from(b'0' + *digit));
            }
            // If there are fewer significant digits than integer positions, append
            // zeros to preserve place value. The suffix needs no decimal point in
            // this case.
            if digits.len() < integer_len {
                for _ in digits.len()..integer_len {
                    body.push('0');
                }
            }
        } else {
            // A negative fixed exponent puts all significant digits after the
            // point. Add the leading zero and exactly the skipped fractional
            // positions before copying the digits.
            body.push_str("0.");
            for _ in 0..(-exponent_k - 1) {
                body.push('0');
            }
            for digit in digits {
                body.push(char::from(b'0' + *digit));
            }
        }
    } else {
        // Scientific output keeps one digit before the point. A one-digit
        // significand needs neither a trailing point nor a fractional zero.
        body.push(char::from(b'0' + digits[0]));
        if digits.len() > 1 {
            body.push('.');
            for digit in &digits[1..] {
                body.push(char::from(b'0' + *digit));
            }
        }
        // Write a lowercase exponent marker followed by the ordinary signed
        // integer spelling. Nonnegative exponents have no plus prefix.
        body.push('e');
        body.push_str(&exponent_k.to_string());
    }
    body
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_signs_and_common_values_use_the_frozen_spelling() {
        assert_eq!(format_f32(0.0), "0");
        assert_eq!(format_f32(-0.0), "0");
        assert_eq!(format_f32(1.0), "1");
        assert_eq!(format_f32(-1.0), "-1");
        assert_eq!(format_f32(0.5), "0.5");
        assert_eq!(format_f32(10.0), "10");
        assert_eq!(format_f32(0.1), "0.1");
        assert_eq!(format_f32(-0.0001), "-0.0001");
    }

    #[test]
    fn random_bit_patterns_round_trip_through_f64_then_f32() {
        let mut state = 0x1234_5678_u32;
        for _ in 0..2_000 {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let value = f32::from_bits(state);
            if !value.is_finite() {
                continue;
            }
            let spelled = format_f32(value);
            assert!(
                !spelled.contains('+') && !spelled.contains("NaN") && !spelled.contains("inf"),
                "{spelled}"
            );
            let back = spelled.parse::<f64>().unwrap() as f32;
            assert_eq!(back.to_bits(), value.to_bits(), "{value:?} -> {spelled}");
        }
        for bits in [
            0x0000_0001,
            0x007f_ffff,
            0x7f7f_ffff,
            0xff7f_ffff,
            0x8000_0001,
        ] {
            let value = f32::from_bits(bits);
            let spelled = format_f32(value);
            let back = spelled.parse::<f64>().unwrap() as f32;
            assert_eq!(back.to_bits(), value.to_bits(), "{spelled}");
        }
    }
}
