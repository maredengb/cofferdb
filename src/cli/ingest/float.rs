//! Shortest round-trip spelling of binary32, as frozen for artifact vectors.
//!
//! `+0` and `-0` are both `0`. The digits are the shortest sequence that parses
//! back to the same bits with round-ties-to-even. Acceptance is
//! `text.parse::<f64>() as f32`, which is the conversion the importer applies.

pub(super) fn format_f32(value: f32) -> String {
    if value == 0.0 {
        return "0".to_owned();
    }
    let bits = value.to_bits();
    let negative = bits >> 31 == 1;
    let (mantissa, exponent) = significand(bits);
    let (digits, scale) = decimal_digits(mantissa, exponent);
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
        if text.parse::<f64>().ok().map(|parsed| parsed as f32) != Some(value) {
            continue;
        }
        // The product parser rejects an inexact integer such as `1e23`.
        // The next wider spelling that names the same bits is the one import accepts.
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
    if biased == 0 {
        (fraction, -149)
    } else {
        (fraction | 0x80_0000, biased - 127 - 23)
    }
}

/// Little-endian decimal digits of `mantissa * 2^exponent`, times `10^scale`.
fn decimal_digits(mantissa: u32, exponent: i32) -> (Vec<u8>, i32) {
    let mut digits = Vec::new();
    let mut rest = mantissa;
    if rest == 0 {
        return (vec![0], 0);
    }
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
        let shift = -exponent;
        for _ in 0..shift {
            mul_small(&mut digits, 5);
        }
        trim_trailing_zeros(digits, -shift)
    }
}

fn mul_small(digits: &mut Vec<u8>, factor: u8) {
    let mut carry = 0u16;
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
    while digits.len() > 1 && digits[0] == 0 {
        digits.remove(0);
        scale += 1;
    }
    (digits, scale)
}

/// Most-significant digit first, together with the scientific exponent `k`.
fn round_significant(digits: &[u8], scale: i32, width: usize) -> (Vec<u8>, i32) {
    let len = digits.len();
    let exponent_k = scale + len as i32 - 1;
    if len <= width {
        let mut out = digits.to_vec();
        out.reverse();
        return (out, exponent_k);
    }
    let least_kept = len - width;
    let round_digit = digits[least_kept - 1];
    let rest_nonzero = digits[..least_kept - 1].iter().any(|digit| *digit != 0);
    let mut kept = digits[least_kept..].to_vec();
    let round_up = if rest_nonzero {
        round_digit >= 5
    } else if round_digit > 5 {
        true
    } else if round_digit < 5 {
        false
    } else {
        kept[0] % 2 == 1
    };
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
        if overflow {
            let mut out = Vec::with_capacity(width);
            out.push(1);
            out.resize(width, 0);
            return (out, exponent_k + 1);
        }
    }
    kept.reverse();
    (kept, exponent_k)
}

fn spell(digits: &[u8], exponent_k: i32) -> String {
    let mut body = String::with_capacity(digits.len() + 8);
    if (-4..=15).contains(&exponent_k) {
        if exponent_k >= 0 {
            let integer_len = exponent_k as usize + 1;
            for (index, digit) in digits.iter().enumerate() {
                if index == integer_len {
                    body.push('.');
                }
                body.push(char::from(b'0' + *digit));
            }
            if digits.len() < integer_len {
                for _ in digits.len()..integer_len {
                    body.push('0');
                }
            }
        } else {
            body.push_str("0.");
            for _ in 0..(-exponent_k - 1) {
                body.push('0');
            }
            for digit in digits {
                body.push(char::from(b'0' + *digit));
            }
        }
    } else {
        body.push(char::from(b'0' + digits[0]));
        if digits.len() > 1 {
            body.push('.');
            for digit in &digits[1..] {
                body.push(char::from(b'0' + *digit));
            }
        }
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
