//! api/app.py 와 같은 HTTP 계약 (POST /eta, GET /health).
//!
//! axum 의 `Json` 추출기는 형식 오류에 400·415 를 돌려주므로 쓰지 않는다.
//! 본문을 `Bytes` 로 받아 직접 파싱하고, 모든 입력 오류를 FastAPI 처럼 422 로 돌려준다.
//!
//! 입력 변환 규칙은 FastAPI/pydantic(lax 모드)의 실제 동작을 따른다 (api-go/eta.go 와 같다):
//!   - distance_km: 숫자, 숫자 문자열(" 2 ", "1e3", "1_000"), bool(1/0). ge=0
//!   - pending_orders: 정수, 정수형 float(3.0), 정수 문자열("3", "3.0", "+3", "1_0"), bool. ge=0
//!   - 모르는 필드는 무시, 중복 키는 마지막 값
//!   - Content-Type 이 없거나 JSON 이 아니면 422 (application/json, application/*+json 만 JSON 으로 읽는다)
//!   - 422 본문은 {"detail": [{"type", "loc", "msg", "input"}]}
//!
//! 다른 점: Python 은 inf/NaN/오버플로 입력에서 minutes:null 이나 500 을 내는 알려진 결함이 있다
//! (api/tests 의 xfail). 이 서비스는 그 경우 테스트가 의도한 대로 422 를 돌려준다.

use std::sync::Arc;

use axum::{
    Router,
    body::Bytes,
    extract::{DefaultBodyLimit, State},
    http::{HeaderMap, StatusCode, Uri, header},
    response::{IntoResponse, Redirect, Response},
    routing::{get, post},
};
use serde_json::{Map, Value, json};

const MAX_BODY_BYTES: usize = 1 << 20;

/// (거리 km, 대기 주문 수) → 예상 시간(분).
pub trait Predictor: Send + Sync + 'static {
    fn predict(&self, distance_km: f32, pending_orders: f32) -> Result<f32, String>;
}

struct AppState {
    predictor: Box<dyn Predictor>,
    model: String,
}

/// 라우팅(404/405, 끝 슬래시 리다이렉트 포함)까지 담은 Router 를 만든다.
pub fn router(predictor: Box<dyn Predictor>, model: &str) -> Router {
    let state = Arc::new(AppState { predictor, model: model.to_string() });
    Router::new()
        .route("/health", get(health).fallback(method_not_allowed))
        .route("/eta", post(eta).fallback(method_not_allowed))
        .fallback(not_found)
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
        .with_state(state)
}

fn json_response(status: StatusCode, body: Value) -> Response {
    (status, [(header::CONTENT_TYPE, "application/json")], body.to_string()).into_response()
}

async fn health(State(s): State<Arc<AppState>>) -> Response {
    json_response(StatusCode::OK, json!({"status": "ok", "model": s.model}))
}

async fn method_not_allowed() -> Response {
    json_response(StatusCode::METHOD_NOT_ALLOWED, json!({"detail": "Method Not Allowed"}))
}

async fn not_found(uri: Uri) -> Response {
    // Starlette 처럼 /health/ → /health 로 보낸다
    let path = uri.path();
    if let Some(trimmed) = path.strip_suffix('/')
        && matches!(trimmed, "/health" | "/eta")
    {
        let target = match uri.query() {
            Some(q) => format!("{trimmed}?{q}"),
            None => trimmed.to_string(),
        };
        return Redirect::temporary(&target).into_response();
    }
    json_response(StatusCode::NOT_FOUND, json!({"detail": "Not Found"}))
}

async fn eta(State(s): State<Arc<AppState>>, headers: HeaderMap, body: Bytes) -> Response {
    let content_type = headers.get(header::CONTENT_TYPE).map(|v| String::from_utf8_lossy(v.as_bytes()).into_owned());
    let (distance, orders) = match parse_eta_request(content_type.as_deref(), &body) {
        Ok(v) => v,
        Err(errs) => return unprocessable(errs),
    };

    let minutes = match s.predictor.predict(distance as f32, orders as f32) {
        Ok(m) => f64::from(m),
        Err(e) => {
            eprintln!("예측 실패: {e}");
            return json_response(StatusCode::INTERNAL_SERVER_ERROR, json!({"detail": "Internal Server Error"}));
        }
    };
    if !minutes.is_finite() {
        // float32 범위를 넘는 입력 등. JSON 은 inf/NaN 을 표현할 수 없으므로 null 대신 422.
        let input = serde_json::from_slice::<Value>(&body).unwrap_or(Value::Null);
        return unprocessable(vec![FieldError::new(
            "value_error",
            json!(["body"]),
            "Value error, prediction is not a finite number (input out of model range)",
            input,
        )]);
    }
    json_response(StatusCode::OK, json!({"minutes": minutes, "model": s.model}))
}

#[derive(Debug, Clone, PartialEq)]
pub struct FieldError {
    pub kind: &'static str,
    pub loc: Value,
    pub msg: &'static str,
    pub input: Value,
}

impl FieldError {
    fn new(kind: &'static str, loc: Value, msg: &'static str, input: Value) -> Self {
        Self { kind, loc, msg, input }
    }
    fn field(kind: &'static str, field: &str, msg: &'static str, input: &Value) -> Self {
        Self::new(kind, json!(["body", field]), msg, input.clone())
    }
    fn to_json(&self) -> Value {
        json!({"type": self.kind, "loc": self.loc, "msg": self.msg, "input": self.input})
    }
}

fn unprocessable(errs: Vec<FieldError>) -> Response {
    let detail: Vec<Value> = errs.iter().map(FieldError::to_json).collect();
    json_response(StatusCode::UNPROCESSABLE_ENTITY, json!({ "detail": detail }))
}

const DICT_MSG: &str = "Input should be a valid dictionary or object to extract fields from";

/// FastAPI 와 같다: application/json 또는 application/*+json (대소문자 무시, 매개변수 허용).
fn is_json_content_type(ct: &str) -> bool {
    let media = ct.split(';').next().unwrap_or("").trim().to_ascii_lowercase();
    match media.split_once('/') {
        Some(("application", sub)) => sub == "json" || sub.ends_with("+json"),
        _ => false,
    }
}

/// 본문을 검증해 (distance_km, pending_orders) 를 돌려준다.
pub fn parse_eta_request(content_type: Option<&str>, body: &[u8]) -> Result<(f64, f64), Vec<FieldError>> {
    if body.is_empty() {
        return Err(vec![FieldError::new("missing", json!(["body"]), "Field required", Value::Null)]);
    }
    if !content_type.is_some_and(is_json_content_type) {
        let input = Value::String(String::from_utf8_lossy(body).into_owned());
        return Err(vec![FieldError::new("model_attributes_type", json!(["body"]), DICT_MSG, input)]);
    }

    let value: Value = serde_json::from_slice(body).map_err(|e| vec![json_invalid(&e, body)])?;
    let Value::Object(obj) = value else {
        return Err(vec![FieldError::new("model_attributes_type", json!(["body"]), DICT_MSG, value)]);
    };

    let distance = parse_distance(&obj);
    let orders = parse_pending_orders(&obj);
    match (distance, orders) {
        (Ok(d), Ok(o)) => Ok((d, o)),
        (d, o) => Err([d.err(), o.err()].into_iter().flatten().collect()),
    }
}

/// 깨진 JSON: loc 는 ["body", 문제가 된 문자의 바이트 위치] (Python json 과 같은 기준).
fn json_invalid(e: &serde_json::Error, body: &[u8]) -> FieldError {
    let pos = if e.is_eof() {
        body.len()
    } else {
        // serde_json 은 1 부터 세는 줄·열을 준다. 열은 그 줄 안의 바이트 수다.
        let line_start: usize =
            body.split(|&b| b == b'\n').take(e.line().saturating_sub(1)).map(|l| l.len() + 1).sum();
        (line_start + e.column().saturating_sub(1)).min(body.len())
    };
    FieldError::new("json_invalid", json!(["body", pos]), "JSON decode error", json!({}))
}

/// Python float() 문자열 문법 (밑줄 구분자 허용, 16진수 불가). inf/nan 은 따로 처리한다.
fn is_float_str(s: &str) -> bool {
    let s = s.strip_prefix(['+', '-']).unwrap_or(s);
    let (mantissa, exponent) = match s.find(['e', 'E']) {
        Some(i) => (&s[..i], Some(&s[i + 1..])),
        None => (s, None),
    };
    let mantissa_ok = match mantissa.split_once('.') {
        Some(("", frac)) => is_digits(frac),
        Some((int, "")) => is_digits(int),
        Some((int, frac)) => is_digits(int) && is_digits(frac),
        None => is_digits(mantissa),
    };
    let exponent_ok = exponent.is_none_or(|e| is_digits(e.strip_prefix(['+', '-']).unwrap_or(e)));
    mantissa_ok && exponent_ok
}

/// 숫자열. 밑줄은 숫자 사이에 하나씩만 올 수 있다 ("1_000" 은 되고 "_1", "1__0", "1_" 은 안 된다).
fn is_digits(s: &str) -> bool {
    !s.is_empty()
        && !s.starts_with('_')
        && !s.ends_with('_')
        && !s.contains("__")
        && s.bytes().all(|b| b.is_ascii_digit() || b == b'_')
}

/// Python float() 처럼 "inf", "-Infinity", "nan" 을 읽는다 (대소문자 무시).
fn parse_non_finite_str(s: &str) -> Option<f64> {
    let (sign, rest) = match s.strip_prefix('-') {
        Some(rest) => (-1.0, rest),
        None => (1.0, s.strip_prefix('+').unwrap_or(s)),
    };
    match rest.to_ascii_lowercase().as_str() {
        "inf" | "infinity" => Some(sign * f64::INFINITY),
        "nan" => Some(f64::NAN),
        _ => None,
    }
}

/// 정수 문자열: [+-]숫자열, 뒤에 ".0", ".00" 같은 0 소수부 허용.
fn is_int_str(s: &str) -> bool {
    let s = s.strip_prefix(['+', '-']).unwrap_or(s);
    match s.split_once('.') {
        Some((int, zeros)) => is_digits(int) && !zeros.is_empty() && zeros.bytes().all(|b| b == b'0'),
        None => is_digits(s),
    }
}

/// Python str.strip() 처럼 앞뒤 공백을 지운다.
fn py_strip(s: &str) -> &str {
    s.trim_matches(char::is_whitespace)
}

const FLOAT_PARSING: &str = "Input should be a valid number, unable to parse string as a number";

fn parse_distance(obj: &Map<String, Value>) -> Result<f64, FieldError> {
    const NAME: &str = "distance_km";
    let Some(raw) = obj.get(NAME) else {
        return Err(FieldError::field("missing", NAME, "Field required", &Value::Object(obj.clone())));
    };
    let f = match raw {
        Value::Bool(b) => f64::from(u8::from(*b)),
        // arbitrary_precision: 원래 표기를 그대로 갖고 있다. 범위를 넘으면 inf 가 된다.
        Value::Number(n) => n
            .to_string()
            .parse::<f64>()
            .map_err(|_| FieldError::field("float_parsing", NAME, FLOAT_PARSING, raw))?,
        Value::String(s) => {
            let s = py_strip(s);
            if let Some(f) = parse_non_finite_str(s) {
                f
            } else if !is_float_str(s) {
                return Err(FieldError::field("float_parsing", NAME, FLOAT_PARSING, raw));
            } else {
                s.replace('_', "").parse::<f64>().unwrap_or(f64::INFINITY)
            }
        }
        Value::Null | Value::Array(_) | Value::Object(_) => {
            return Err(FieldError::field("float_type", NAME, "Input should be a valid number", raw));
        }
    };
    // pydantic 처럼 ge=0 을 먼저 본다: NaN, -inf 는 greater_than_equal.
    if f.is_nan() || f < 0.0 {
        return Err(FieldError::field(
            "greater_than_equal",
            NAME,
            "Input should be greater than or equal to 0",
            raw,
        ));
    }
    // +inf 는 Python 에서 minutes:null 이나 500 이 되는 알려진 결함이다. 여기서는 422.
    if f.is_infinite() {
        return Err(FieldError::field("finite_number", NAME, "Input should be a finite number", raw));
    }
    Ok(f)
}

/// 정수 리터럴(부호 포함)의 값을 f64 로. 크기 제한 없음 (Python int 와 같다). 음수면 None.
fn non_negative_int_to_f64(digits: &str) -> Option<f64> {
    let digits = digits.replace('_', "");
    let (neg, body) = match digits.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, digits.strip_prefix('+').unwrap_or(&digits)),
    };
    let is_zero = body.bytes().all(|b| b == b'0');
    if neg && !is_zero {
        return None;
    }
    Some(body.parse::<f64>().unwrap_or(f64::INFINITY))
}

fn parse_pending_orders(obj: &Map<String, Value>) -> Result<f64, FieldError> {
    const NAME: &str = "pending_orders";
    let Some(raw) = obj.get(NAME) else {
        return Err(FieldError::field("missing", NAME, "Field required", &Value::Object(obj.clone())));
    };
    let negative = || {
        FieldError::field("greater_than_equal", NAME, "Input should be greater than or equal to 0", raw)
    };
    match raw {
        Value::Bool(b) => Ok(f64::from(u8::from(*b))),
        Value::Number(n) => {
            let s = n.to_string();
            if !s.contains(['.', 'e', 'E']) {
                return non_negative_int_to_f64(&s).ok_or_else(negative);
            }
            let f = s.parse::<f64>().unwrap_or(f64::INFINITY);
            if !f.is_finite() || f.abs() >= 9.223_372_036_854_775_808e18 {
                return Err(FieldError::field(
                    "int_parsing_size",
                    NAME,
                    "Unable to parse input string as an integer, exceeded maximum size",
                    raw,
                ));
            }
            if f != f.trunc() {
                return Err(FieldError::field(
                    "int_from_float",
                    NAME,
                    "Input should be a valid integer, got a number with a fractional part",
                    raw,
                ));
            }
            if f < 0.0 { Err(negative()) } else { Ok(f.abs()) }
        }
        Value::String(s) => {
            let s = py_strip(s);
            if !is_int_str(s) {
                return Err(FieldError::field(
                    "int_parsing",
                    NAME,
                    "Input should be a valid integer, unable to parse string as an integer",
                    raw,
                ));
            }
            let int_part = s.split_once('.').map_or(s, |(i, _)| i);
            non_negative_int_to_f64(int_part).ok_or_else(negative)
        }
        Value::Null | Value::Array(_) | Value::Object(_) => {
            Err(FieldError::field("int_type", NAME, "Input should be a valid integer", raw))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(body: &str) -> Result<(f64, f64), Vec<FieldError>> {
        parse_eta_request(Some("application/json"), body.as_bytes())
    }

    fn kinds(body: &str) -> Vec<&'static str> {
        parse(body).unwrap_err().iter().map(|e| e.kind).collect()
    }

    #[test]
    fn accepts_numbers_and_lenient_coercions() {
        assert_eq!(parse(r#"{"distance_km": 2, "pending_orders": 3}"#), Ok((2.0, 3.0)));
        assert_eq!(parse(r#"{"distance_km": " 2 ", "pending_orders": "+3"}"#), Ok((2.0, 3.0)));
        assert_eq!(parse(r#"{"distance_km": "1_000", "pending_orders": "3.00"}"#), Ok((1000.0, 3.0)));
        assert_eq!(parse(r#"{"distance_km": true, "pending_orders": false}"#), Ok((1.0, 0.0)));
        assert_eq!(parse(r#"{"distance_km": 1, "pending_orders": 3.0, "foo": 1}"#), Ok((1.0, 3.0)));
        assert_eq!(parse(r#"{"distance_km": 1, "pending_orders": 1, "pending_orders": 4}"#), Ok((1.0, 4.0)));
        assert_eq!(parse(r#"{"distance_km": -0.0, "pending_orders": -0}"#), Ok((0.0, 0.0)));
        assert_eq!(parse(r#"{"distance_km": 0, "pending_orders": 1000000000000000000000000000000}"#), Ok((0.0, 1e30)));
    }

    #[test]
    fn rejects_bad_fields_with_pydantic_types() {
        assert_eq!(kinds(r#"{"distance_km": -1, "pending_orders": 0}"#), ["greater_than_equal"]);
        assert_eq!(kinds(r#"{"distance_km": 1, "pending_orders": -1}"#), ["greater_than_equal"]);
        assert_eq!(kinds(r#"{"distance_km": "abc", "pending_orders": 0}"#), ["float_parsing"]);
        assert_eq!(kinds(r#"{"distance_km": "1__0", "pending_orders": 0}"#), ["float_parsing"]);
        assert_eq!(kinds(r#"{"distance_km": "inf", "pending_orders": 0}"#), ["finite_number"]);
        assert_eq!(kinds(r#"{"distance_km": "-Infinity", "pending_orders": 0}"#), ["greater_than_equal"]);
        assert_eq!(kinds(r#"{"distance_km": "nan", "pending_orders": 0}"#), ["greater_than_equal"]);
        assert_eq!(kinds(r#"{"distance_km": 1e400, "pending_orders": 0}"#), ["finite_number"]);
        assert_eq!(kinds(r#"{"distance_km": 1, "pending_orders": 1.5}"#), ["int_from_float"]);
        assert_eq!(kinds(r#"{"distance_km": 1, "pending_orders": "many"}"#), ["int_parsing"]);
        assert_eq!(kinds(r#"{"distance_km": 1, "pending_orders": 1e19}"#), ["int_parsing_size"]);
        assert_eq!(kinds(r#"{"distance_km": null, "pending_orders": null}"#), ["float_type", "int_type"]);
        assert_eq!(kinds("{}"), ["missing", "missing"]);
    }

    #[test]
    fn rejects_malformed_bodies() {
        assert_eq!(kinds("{bad"), ["json_invalid"]);
        assert_eq!(kinds("[1, 2]"), ["model_attributes_type"]);
        assert_eq!(parse("").unwrap_err()[0].kind, "missing");
        let text = parse_eta_request(Some("text/plain"), br#"{"distance_km":1,"pending_orders":1}"#);
        assert_eq!(text.unwrap_err()[0].kind, "model_attributes_type");
        // Content-Type 이 없으면 FastAPI 처럼 422
        let none = parse_eta_request(None, br#"{"distance_km":1,"pending_orders":1}"#);
        assert_eq!(none.unwrap_err()[0].kind, "model_attributes_type");
        assert!(parse_eta_request(Some("application/vnd.api+json; charset=utf-8"), br#"{"distance_km":1,"pending_orders":1}"#).is_ok());
    }

    #[test]
    fn json_invalid_points_at_bad_byte() {
        assert_eq!(parse("{bad").unwrap_err()[0].loc, json!(["body", 1]));
        assert_eq!(parse("{\n  x").unwrap_err()[0].loc, json!(["body", 4]));
    }
}
